"""
claude_xai_predict_l535_n35.py
────────────────────────────────
Du doan L535:
  - 35 model nhi phan cho N1-N5 (5 so chinh, range 1-35) → chon top K
  - N6 (so phu): bo qua

Kien truc:
  - XGBoost / LightGBM / CatBoost binary classifier (35 model)
  - Walk-forward CV: precision toi uu qua F-beta(0.5) threshold tuning
  - No data leak: train (ky 0→N-2), predict_row (ky N-1)
  - Sliding window + time decay weight
  - Imbalance: compute_sample_weight("balanced") * decay
  - Co-occurrence vectorized
  - pwn1-pwn7 (Power 655 truoc ky) dung lam feature
  - Ensemble 5 configs, weight theo softmax(mean_prec), loai config beat_ratio < 14%
  - Majority-vote filter
  - Brier score chi de monitor
  - Save predict → l535kqpredict

Usage:
    python claude_xai_predict_l535_n35.py --source db --lags 3 --top 8
    python claude_xai_predict_l535_n35.py --source db --top 10 --ensemble --filter auto --save
    python claude_xai_predict_l535_n35.py --update_result
"""

import argparse, os, warnings
import numpy as np
import pandas as pd
from collections import Counter
from datetime import datetime, timezone

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import precision_score, recall_score, brier_score_loss
from sklearn.utils.class_weight import compute_sample_weight

warnings.filterwarnings("ignore")

DATABASE_URL = os.environ.get(
    "DATABASE_URL", "postgresql://postgres:bin@localhost:5432/katalott"
)

ALL_NUMS = list(range(1, 36))
N_MAIN   = 5   # so chinh N1-N5
N_POOL   = 35

VERSION     = "v1.1.0"
DESCRIPTION = (
    "35 binary classifier (XGB/LGBM/CatBoost), no-leak split, vectorized features, "
    "balanced sample_weight, F-beta lag-1 threshold, "
    "hypergeometric baseline (5/35), weighted ensemble softmax, "
    "majority-vote filter, pwn1-7 features, save to l535kqpredict"
)

ENSEMBLE_MIN_BEAT_RATIO = 6 / 35   # ~17.1%: config bi loai neu beat_ratio < nguong


# ─────────────────────────────────────────────
# 0. BASELINE
# ─────────────────────────────────────────────

def hypergeometric_baseline(top_k: int,
                             n_drawn: int = N_MAIN,
                             pool: int = N_POOL) -> float:
    return n_drawn / pool   # E[hits]/top_k = 5/35 ≈ 0.1429


# ─────────────────────────────────────────────
# 1. LOAD
# ─────────────────────────────────────────────

def load_from_db(n_ky: int = 0, qh_ky: int = 20) -> pd.DataFrame:
    import psycopg2
    limit = f"LIMIT {n_ky}" if n_ky > 0 else ""
    sql = f"""
        SELECT * FROM public.l535kqdetail
        WHERE qh_ky = {qh_ky}
        ORDER BY ky DESC {limit}
    """
    conn = psycopg2.connect(DATABASE_URL)
    df   = pd.read_sql(sql, conn)
    conn.close()
    return df.sort_values("ky").reset_index(drop=True)


def load_from_csv(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath)
    return df.sort_values("ky").reset_index(drop=True)


# ─────────────────────────────────────────────
# 2. HELPERS
# ─────────────────────────────────────────────

THU_MAP = {"T2": 2, "T3": 3, "T4": 4, "T5": 5, "T6": 6, "T7": 7, "CN": 8}
DD_MAP  = {"CC": 0, "CL": 1, "LC": 2, "LL": 3}

def _safe_int(s) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")

def _encode_thu(series) -> pd.Series:
    return series.astype(str).str.strip().map(THU_MAP)

def _encode_dd(n_first, n_last) -> pd.Series:
    def _cl(n): return "C" if n % 2 == 0 else "L"
    result = []
    for f, l in zip(n_first, n_last):
        if pd.isna(f) or pd.isna(l):
            result.append(np.nan)
        else:
            result.append(DD_MAP.get(_cl(int(f)) + _cl(int(l)), np.nan))
    return pd.Series(result, index=n_first.index)

def _ngay_diff(d1: pd.Series, d2: pd.Series) -> pd.Series:
    return (pd.to_datetime(d1, errors="coerce") -
            pd.to_datetime(d2, errors="coerce")).dt.days

def get_drawn_set(row) -> set:
    """Chi lay N1-N5 (so chinh)."""
    return {
        int(row[c]) for c in ["n1", "n2", "n3", "n4", "n5"]
        if pd.notna(row[c]) and row[c] != 0
    }


# ─────────────────────────────────────────────
# 3. HIT MATRIX & CO-OCCURRENCE (vectorized)
#    Chi tinh tren N1-N5 (so chinh), bo N6
# ─────────────────────────────────────────────

def build_hit_matrix(df: pd.DataFrame) -> np.ndarray:
    """Hit matrix chi cho so chinh N1-N5."""
    n   = len(df)
    mat = np.zeros((n, N_POOL + 1), dtype=np.int8)
    for col in ["n1", "n2", "n3", "n4", "n5"]:   # bo n6
        if col not in df.columns:
            continue
        vals = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int).values
        for i, v in enumerate(vals):
            if 1 <= v <= N_POOL:
                mat[i, v] = 1
    return mat


def build_cooccurrence_matrix(hit_mat: np.ndarray) -> np.ndarray:
    h  = hit_mat[:, 1:].astype(np.float32)
    co = h.T @ h
    np.fill_diagonal(co, 0)
    return co


# ─────────────────────────────────────────────
# 4. GLOBAL FEATURES
# ─────────────────────────────────────────────

def make_global_features(df: pd.DataFrame, lags: int = 3) -> pd.DataFrame:
    d = df.copy()

    # dd encode
    d["dd_enc"] = d["dd"].apply(
        lambda v: {"CC": 0, "CL": 1, "LC": 2, "LL": 3}.get(str(v).strip(), -1)
    )

    # cl5: 5 bit chan/le cua 5 so chinh
    s = d["cl5"].astype(str).str.strip()
    for i in range(5):
        d[f"cl5_b{i}"] = s.str[i].map({"C": 0, "L": 1}).fillna(-1).astype(int)
    d["cl5_pattern"] = s.apply(
        lambda v: int("".join("0" if c == "C" else "1" for c in v), 2)
        if len(v) == 5 else np.nan
    )
    d["cl5_changed"] = (d["cl5_pattern"] != d["cl5_pattern"].shift(1)).astype(float)

    # stats so chinh N1-N5
    num_cols = ["n1", "n2", "n3", "n4", "n5"]
    nums_df  = df[num_cols].apply(pd.to_numeric, errors="coerce")
    d["drawn_min"]    = nums_df.min(axis=1)
    d["drawn_max"]    = nums_df.max(axis=1)
    d["drawn_range"]  = d["drawn_max"] - d["drawn_min"]
    d["drawn_consec"] = nums_df.apply(
        lambda r: sum(
            1 for i in range(len(r) - 1)
            if pd.notna(r.iloc[i]) and pd.notna(r.iloc[i+1])
            and abs(r.iloc[i+1] - r.iloc[i]) == 1
        ), axis=1
    ).astype(float)

    # dec0-dec3 (pool 35: thap phan 0-3)
    for di in range(4):
        col = f"dec{di}"
        if col in d.columns:
            d[col] = _safe_int(d[col])

    d["thu_enc"] = _encode_thu(d["thu"])

    if "jpck" in d.columns:
        d["jpck"] = _safe_int(d["jpck"]).fillna(0)



    # Power 655 truoc ky (pwn1-pwn6, bo pwn7) → features
    pw_cols  = [f"pwn{i}" for i in range(1, 7)]
    pw_avail = [c for c in pw_cols if c in d.columns]
    if pw_avail:
        pw_nums = df[pw_avail].apply(pd.to_numeric, errors="coerce")
        d["pw_min"]     = pw_nums.min(axis=1)
        d["pw_max"]     = pw_nums.max(axis=1)
        d["pw_range"]   = d["pw_max"] - d["pw_min"]
        d["pw_sc"]      = (pw_nums % 2 == 0).sum(axis=1).astype(float)
        d["pw_sum"]     = pw_nums.sum(axis=1)
        d["pw_in_pool"] = (pw_nums <= N_POOL).sum(axis=1).astype(float)
        l535_main = df[["n1","n2","n3","n4","n5"]].apply(pd.to_numeric, errors="coerce")
        def _pw_overlap(row_idx):
            pw_set = set(pw_nums.iloc[row_idx].dropna().astype(int).tolist())
            l5_set = set(l535_main.iloc[row_idx].dropna().astype(int).tolist())
            return float(len(pw_set & l5_set))
        d["pw_l535_overlap"] = [_pw_overlap(i) for i in range(len(d))]

    # Mega (mgn1-mgn6) → features
    mg_cols  = [f"mgn{i}" for i in range(1, 7)]
    mg_avail = [c for c in mg_cols if c in d.columns]
    if mg_avail:
        mg_nums = df[mg_avail].apply(pd.to_numeric, errors="coerce")
        d["mg_min"]     = mg_nums.min(axis=1)
        d["mg_max"]     = mg_nums.max(axis=1)
        d["mg_range"]   = d["mg_max"] - d["mg_min"]
        d["mg_sc"]      = (mg_nums % 2 == 0).sum(axis=1).astype(float)
        d["mg_sum"]     = mg_nums.sum(axis=1)
        d["mg_in_pool"] = (mg_nums <= N_POOL).sum(axis=1).astype(float)
        if "mgthu" in d.columns:
            d["mg_thu_enc"]   = _encode_thu(d["mgthu"])
        if "mgngay" in d.columns and "ngay" in d.columns:
            d["mg_ngay_diff"] = _ngay_diff(d["ngay"], d["mgngay"])
        l535_main = df[["n1","n2","n3","n4","n5"]].apply(pd.to_numeric, errors="coerce")
        def _mg_overlap(row_idx):
            mg_set = set(mg_nums.iloc[row_idx].dropna().astype(int).tolist())
            l5_set = set(l535_main.iloc[row_idx].dropna().astype(int).tolist())
            return float(len(mg_set & l5_set))
        d["mg_l535_overlap"] = [_mg_overlap(i) for i in range(len(d))]

    # Build feat DataFrame
    feat = pd.DataFrame(index=d.index)
    feat["thu_enc"] = d["thu_enc"]

    prev_cols = [
        "dd_enc", "sc", "sum", "ke",
        "cl5_b0", "cl5_b1", "cl5_b2", "cl5_b3", "cl5_b4",
        "cl5_pattern", "cl5_changed",
        "dec0", "dec1", "dec2", "dec3",
        "drawn_min", "drawn_max", "drawn_range", "drawn_consec",
    ]
    for c in ["jpck"]:
        if c in d.columns:
            prev_cols.append(c)
    # pw features (Power pwn1-pwn6)
    for c in ["pw_min", "pw_max", "pw_range", "pw_sc", "pw_sum",
              "pw_in_pool", "pw_l535_overlap"]:
        if c in d.columns:
            prev_cols.append(c)
    # mg features (Mega mgn1-mgn6)
    for c in ["mg_min", "mg_max", "mg_range", "mg_sc", "mg_sum",
              "mg_in_pool", "mg_l535_overlap", "mg_thu_enc", "mg_ngay_diff"]:
        if c in d.columns:
            prev_cols.append(c)

    prev_cols = [c for c in prev_cols if c in d.columns]
    for col in prev_cols:
        feat[f"{col}_prev"] = d[col].shift(1)

    lag_cols = ["dd_enc", "sc", "cl5_pattern", "drawn_range"]
    lag_cols = [c for c in lag_cols if c in d.columns]
    for lag in range(2, lags + 1):
        for col in lag_cols:
            feat[f"{col}_lag{lag}"] = d[col].shift(lag)

    for col in ["sc", "drawn_range"]:
        if col not in d.columns:
            continue
        base = d[col].shift(1)
        for w in [3, 5]:
            feat[f"{col}_roll{w}_mean"] = base.rolling(w).mean()
            feat[f"{col}_roll{w}_std"]  = base.rolling(w).std()

    if "sc" in d.columns:
        sc_s = d["sc"].shift(1)
        feat["sc_trend"] = sc_s.rolling(5).mean() - sc_s.rolling(10).mean()

    # Cross Power vs L535
    if "pw_sc_prev" in feat.columns and "sc_prev" in feat.columns:
        feat["cross_sc_pw"] = feat["sc_prev"] * feat["pw_sc_prev"]
    if "pw_in_pool_prev" in feat.columns and "drawn_range_prev" in feat.columns:
        feat["cross_pw_pool_range"] = feat["pw_in_pool_prev"] * feat["drawn_range_prev"]
    if "pw_l535_overlap_prev" in feat.columns and "sc_prev" in feat.columns:
        feat["cross_pw_overlap_sc"] = feat["pw_l535_overlap_prev"] * feat["sc_prev"]

    # Cross Mega vs L535
    if "mg_sc_prev" in feat.columns and "sc_prev" in feat.columns:
        feat["cross_sc_mg"] = feat["sc_prev"] * feat["mg_sc_prev"]
    if "mg_in_pool_prev" in feat.columns and "drawn_range_prev" in feat.columns:
        feat["cross_mg_pool_range"] = feat["mg_in_pool_prev"] * feat["drawn_range_prev"]
    if "mg_l535_overlap_prev" in feat.columns and "sc_prev" in feat.columns:
        feat["cross_mg_overlap_sc"] = feat["mg_l535_overlap_prev"] * feat["sc_prev"]

    # Cross Power vs Mega
    if "pw_sc_prev" in feat.columns and "mg_sc_prev" in feat.columns:
        feat["cross_pw_mg_sc"] = feat["pw_sc_prev"] * feat["mg_sc_prev"]
    if "pw_l535_overlap_prev" in feat.columns and "mg_l535_overlap_prev" in feat.columns:
        feat["cross_pw_mg_overlap"] = feat["pw_l535_overlap_prev"] * feat["mg_l535_overlap_prev"]

    return feat


# ─────────────────────────────────────────────
# 5. PER-NUMBER FEATURES (vectorized)
# ─────────────────────────────────────────────

def make_per_number_features(df: pd.DataFrame,
                              num: int,
                              hit_mat: np.ndarray,
                              co_mat: np.ndarray) -> pd.DataFrame:
    feat = pd.DataFrame(index=df.index)
    n    = len(df)
    hit  = hit_mat[:, num].astype(float)

    feat["n_hit_prev"] = pd.Series(hit, index=df.index).shift(1)

    last_seen = np.full(n, np.nan)
    last_idx  = -1
    for i in range(n):
        if last_idx >= 0:
            last_seen[i] = i - last_idx
        if hit[i] == 1:
            last_idx = i
    feat["n_last_seen"] = pd.Series(last_seen, index=df.index).shift(1)

    streak = np.zeros(n)
    cur    = 0
    for i in range(n):
        streak[i] = cur
        cur = 0 if hit[i] == 1 else cur + 1
    feat["n_streak"] = pd.Series(streak, index=df.index).shift(1)

    hit_s = pd.Series(hit, index=df.index).shift(1)
    feat["n_freq_10"]    = hit_s.rolling(10).mean()
    feat["n_freq_20"]    = hit_s.rolling(20).mean()
    feat["n_freq_50"]    = hit_s.rolling(50).mean()
    feat["n_freq_trend"] = hit_s.rolling(10).mean() - hit_s.rolling(30).mean()

    gap_mean = np.full(n, np.nan)
    gap_std  = np.full(n, np.nan)
    gaps     = []
    last_i   = -1
    for i in range(n):
        if last_i >= 0:
            gaps.append(i - last_i)
        if hit[i] == 1:
            last_i = i
        if len(gaps) >= 2:
            gap_mean[i] = np.mean(gaps)
            gap_std[i]  = np.std(gaps)
    feat["n_gap_mean"] = pd.Series(gap_mean, index=df.index).shift(1)
    feat["n_gap_std"]  = pd.Series(gap_std,  index=df.index).shift(1)

    # Position frequency trong N1-N5
    for pos_i, col in enumerate(["n1", "n2", "n3", "n4", "n5"], 1):
        if col in df.columns:
            pos_hit = (pd.to_numeric(df[col], errors="coerce") == num).astype(float)
            feat[f"n_pos{pos_i}_freq"] = pos_hit.shift(1).rolling(50).mean()

    co_row   = co_mat[num - 1, :]
    prev_mat = np.vstack([np.zeros((1, N_POOL + 1), dtype=np.int8), hit_mat[:-1, :]])
    co_scores = np.full(n, np.nan)
    for i in range(1, n):
        co_scores[i] = float(co_row @ prev_mat[i, 1:]) / max(N_MAIN, 1)
    feat["n_cooccur_score"] = pd.Series(co_scores, index=df.index)

    if "qhl" in df.columns:
        def _in_qhl(v):
            if pd.isna(v) or v == "":
                return 0.0
            try:
                return float(num in [int(x) for x in str(v).split(",") if x.strip()])
            except:
                return 0.0
        feat["n_is_qhl"] = df["qhl"].shift(1).apply(_in_qhl)

    # Power 655: so num co xuat hien trong pwn1-pwn7 ky truoc khong
    pw_cols = [f"pwn{i}" for i in range(1, 8)]
    pw_avail = [c for c in pw_cols if c in df.columns]
    if pw_avail:
        pw_nums = df[pw_avail].apply(pd.to_numeric, errors="coerce")
        pw_hit  = pw_nums.apply(lambda row: float(num in row.dropna().astype(int).tolist()), axis=1)
        feat["n_in_pw_prev"]      = pw_hit.shift(1)
        feat["n_in_pw_freq10"]    = pw_hit.shift(1).rolling(10).mean()

    if "dot" in df.columns:
        feat["dot_prev"] = _safe_int(df["dot"]).shift(1)

    feat["n_is_even"]  = float(num % 2 == 0)
    feat["n_dec"]      = float(num // 10)
    feat["n_val_norm"] = float(num / N_POOL)

    return feat


# ─────────────────────────────────────────────
# 6. BUILD DATASET
# ─────────────────────────────────────────────

def build_dataset(df: pd.DataFrame,
                  num: int,
                  global_feat: pd.DataFrame,
                  hit_mat: np.ndarray,
                  co_mat: np.ndarray) -> tuple:
    per_feat = make_per_number_features(df, num, hit_mat, co_mat)
    all_feat = pd.concat([global_feat, per_feat], axis=1).fillna(-1)
    target   = pd.Series(hit_mat[:, num].astype(int), index=df.index)

    combined = pd.concat(
        [all_feat, target.rename("y"), df["ky"]], axis=1
    ).dropna(subset=["n_freq_50"]).reset_index(drop=True)

    feature_cols   = list(all_feat.columns)
    combined_train = combined.iloc[:-1].reset_index(drop=True)
    predict_row    = combined.iloc[[-1]]
    return combined_train, predict_row, feature_cols


# ─────────────────────────────────────────────
# 7. MODEL BUILDERS
# ─────────────────────────────────────────────

def make_xgb(params: dict = None) -> XGBClassifier:
    defaults = dict(
        n_estimators=300, max_depth=3, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.7, min_child_weight=3,
        gamma=0.1, reg_alpha=0.05, reg_lambda=1.0,
        objective="binary:logistic", eval_metric="logloss",
        random_state=42, n_jobs=-1, early_stopping_rounds=30,
    )
    if params:
        defaults.update(params)
    return XGBClassifier(**defaults)


def make_lgbm(params: dict = None) -> LGBMClassifier:
    defaults = dict(
        n_estimators=300, max_depth=3, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.7, min_child_leaves=3,
        reg_alpha=0.05, reg_lambda=1.0,
        objective="binary", random_state=42, n_jobs=-1, verbose=-1,
    )
    if params:
        defaults.update(params)
    return LGBMClassifier(**defaults)


def make_catboost(params: dict = None) -> CatBoostClassifier:
    defaults = dict(
        iterations=300, depth=3, learning_rate=0.05,
        subsample=0.8, loss_function="Logloss",
        random_seed=42, thread_count=-1, verbose=0,
    )
    if params:
        defaults.update(params)
    return CatBoostClassifier(**defaults)


def _make_model(model_type: str, params: dict = None):
    if model_type == "lgbm":
        return make_lgbm(params)
    if model_type == "catboost":
        return make_catboost(params)
    return make_xgb(params)


# ─────────────────────────────────────────────
# 8. WALK-FORWARD UTILS
# ─────────────────────────────────────────────

def _walk_forward_splits(n: int, min_train: int, val_size: int):
    splits = []
    start  = min_train
    while start + val_size <= n:
        splits.append((np.arange(0, start), np.arange(start, start + val_size)))
        start += val_size
    return splits


def _make_decay_weights(n: int, decay: float = 0.995) -> np.ndarray:
    idx = np.arange(n)
    w   = decay ** (n - 1 - idx)
    return w / w.mean()


# ─────────────────────────────────────────────
# 9. THRESHOLD TUNING
# ─────────────────────────────────────────────

def _best_threshold_fbeta(y_true: np.ndarray, y_prob: np.ndarray,
                           beta: float = 0.5) -> float:
    if y_true.sum() == 0:
        return 0.5
    candidates  = np.unique(y_prob)
    best_t, best_f = 0.5, -1.0
    b2 = beta ** 2
    for t in candidates:
        pred = (y_prob >= t).astype(int)
        tp   = float((pred * y_true).sum())
        fp   = float((pred * (1 - y_true)).sum())
        fn   = float(((1 - pred) * y_true).sum())
        denom = (1 + b2) * tp + b2 * fn + fp
        if denom == 0:
            continue
        f = (1 + b2) * tp / denom
        if f > best_f:
            best_f, best_t = f, t
    return float(best_t)


# ─────────────────────────────────────────────
# 10. TRAIN ONE NUMBER
# ─────────────────────────────────────────────

def train_one(combined: pd.DataFrame,
              feature_cols: list,
              val_size: int,
              min_train: int,
              train_window: int = 0,
              decay: float = 0.995,
              model_type: str = "xgb",
              xgb_params: dict = None) -> tuple:
    X = combined[feature_cols].values
    y = combined["y"].values

    splits      = _walk_forward_splits(len(X), min_train, val_size)
    prec_list   = []
    rec_list    = []
    brier_list  = []
    thresh_list = []

    prev_thresh = 0.5

    for tr_idx, va_idx in splits:
        X_tr, X_va = X[tr_idx], X[va_idx]
        y_tr, y_va = y[tr_idx], y[va_idx]
        if y_tr.sum() == 0:
            continue

        if train_window > 0 and len(tr_idx) > train_window:
            tr_idx = tr_idx[-train_window:]
            X_tr   = X[tr_idx]
            y_tr   = y[tr_idx]

        sw_tr = compute_sample_weight("balanced", y_tr) * _make_decay_weights(len(y_tr), decay)

        mdl = _make_model(model_type, xgb_params)
        if model_type == "xgb":
            mdl.fit(X_tr, y_tr, sample_weight=sw_tr,
                    eval_set=[(X_va, y_va)], verbose=False)
        else:
            mdl.fit(X_tr, y_tr, sample_weight=sw_tr)

        prob_va = mdl.predict_proba(X_va)[:, 1]

        pred = (prob_va >= prev_thresh).astype(int)
        prec_list.append(precision_score(y_va, pred, zero_division=0))
        rec_list.append(recall_score(y_va, pred, zero_division=0))
        brier_list.append(brier_score_loss(y_va, prob_va))

        prev_thresh = _best_threshold_fbeta(y_va, prob_va, beta=0.5)
        thresh_list.append(prev_thresh)

    # Final model
    if train_window > 0 and len(X) > train_window:
        X_fin, y_fin = X[-train_window:], y[-train_window:]
    else:
        X_fin, y_fin = X, y

    sw_fin  = compute_sample_weight("balanced", y_fin) * _make_decay_weights(len(y_fin), decay)
    final_p = dict(xgb_params or {})
    if model_type == "xgb":
        final_p["early_stopping_rounds"] = None
    final = _make_model(model_type, final_p)
    if model_type == "catboost":
        final.fit(X_fin, y_fin, sample_weight=sw_fin, verbose=False)
    else:
        final.fit(X_fin, y_fin, sample_weight=sw_fin)

    mean_prec  = float(np.mean(prec_list))  if prec_list  else 0.0
    mean_rec   = float(np.mean(rec_list))   if rec_list   else 0.0
    mean_brier = float(np.mean(brier_list)) if brier_list else 1.0
    best_thresh = float(thresh_list[-1])    if thresh_list else 0.5

    return final, best_thresh, mean_prec, mean_rec, mean_brier


# ─────────────────────────────────────────────
# 11. TRAIN ALL (35 models)
# ─────────────────────────────────────────────

def train_all(df: pd.DataFrame,
              global_feat: pd.DataFrame,
              hit_mat: np.ndarray,
              co_mat: np.ndarray,
              lags: int,
              val_size: int,
              min_train: int,
              train_window: int = 0,
              decay: float = 0.995,
              model_type: str = "xgb",
              xgb_params: dict = None,
              show_importance: bool = False) -> dict:

    results  = {}
    baseline = hypergeometric_baseline(top_k=8)

    print(f"\n  [N1-N5] Hypergeometric baseline precision: {baseline:.4f} ({baseline:.1%})")
    print(f"  {'Num':>4}  {'CV Prec':>8}  {'CV Rec':>8}  {'Brier':>8}  {'Thresh':>7}  {'Flag'}")
    print(f"  {'─'*58}")

    for num in ALL_NUMS:
        combined_train, predict_row, feature_cols = build_dataset(
            df, num, global_feat, hit_mat, co_mat
        )
        model, thresh, prec, rec, brier = train_one(
            combined_train, feature_cols,
            val_size, min_train, train_window, decay,
            model_type, xgb_params
        )
        results[num] = dict(
            model=model, thresh=thresh, prec=prec, rec=rec, brier=brier,
            feature_cols=feature_cols,
            combined=combined_train,
            predict_row=predict_row,
        )
        flag = "✅" if prec > baseline else "⚠️ "
        print(f"  {num:>4}  {prec:>8.3f}  {rec:>8.3f}  {brier:>8.4f}  {thresh:>7.3f}  {flag}")

    precs  = [results[n]["prec"]  for n in ALL_NUMS]
    briers = [results[n]["brier"] for n in ALL_NUMS]
    print(f"\n  Mean CV Precision : {np.mean(precs):.4f}")
    print(f"  Mean Brier Score  : {np.mean(briers):.4f}  (monitor only, naive={N_MAIN/N_POOL*(1-N_MAIN/N_POOL):.4f})")
    print(f"  Beat baseline     : {sum(1 for p in precs if p > baseline)}/35")

    if show_importance:
        _show_top_importance(results)

    return results


# ─────────────────────────────────────────────
# 12. IMPORTANCE
# ─────────────────────────────────────────────

def _show_top_importance(results: dict, top_n: int = 15):
    imp_sum = {}
    for num in ALL_NUMS:
        r = results[num]
        try:
            imps = r["model"].feature_importances_
        except AttributeError:
            continue
        for c, v in zip(r["feature_cols"], imps):
            imp_sum[c] = imp_sum.get(c, 0.0) + v

    imp_avg = {k: v / 35 for k, v in imp_sum.items()}
    top     = sorted(imp_avg.items(), key=lambda x: -x[1])[:top_n]
    print(f"\n  Top {top_n} features (trung binh 35 model):")
    for f, s in top:
        print(f"    {f:<44} {s:.4f}  {'█' * int(s * 500)}")


# ─────────────────────────────────────────────
# 13. ENSEMBLE
# ─────────────────────────────────────────────

ENSEMBLE_CONFIGS = [
    {"lags": 3, "decay": 0.995, "train_window": 0,   "model_type": "xgb",      "xgb_params": {"max_depth": 3},                     "label": "M1(xgb,d3,lag3,all)"},
    {"lags": 3, "decay": 0.995, "train_window": 0,   "model_type": "xgb",      "xgb_params": {"max_depth": 5, "n_estimators": 500}, "label": "M2(xgb,d5,n500,lag3,all)"},
    {"lags": 3, "decay": 0.995, "train_window": 0,   "model_type": "lgbm",     "xgb_params": {},                                    "label": "M3(lgbm,lag3,all)"},
    {"lags": 3, "decay": 0.995, "train_window": 0,   "model_type": "lgbm",     "xgb_params": {"max_depth": 5, "n_estimators": 500}, "label": "M4(lgbm,d5,n500,lag3,all)"},
    {"lags": 3, "decay": 0.990, "train_window": 300, "model_type": "xgb",      "xgb_params": {"max_depth": 3, "n_estimators": 500}, "label": "M5(xgb,d3,n500,w300)"},
    {"lags": 3, "decay": 0.995, "train_window": 0,   "model_type": "catboost", "xgb_params": {},                                    "label": "M6(catboost,d3,lag3,all)"},
    {"lags": 3, "decay": 0.990, "train_window": 300, "model_type": "catboost", "xgb_params": {"depth": 5, "iterations": 500},       "label": "M7(catboost,d5,n500,w300)"},
    {"lags": 3, "decay": 0.995, "train_window": 0,   "model_type": "catboost", "xgb_params": {"depth": 5, "iterations": 500},       "label": "M8(catboost,d5,n500,lag3,all)"},
]


def train_ensemble(df: pd.DataFrame,
                   hit_mat: np.ndarray,
                   co_mat: np.ndarray,
                   val_size: int,
                   min_train: int) -> list:
    all_results = []
    for cfg in ENSEMBLE_CONFIGS:
        print(f"\n  ── Ensemble {cfg['label']} ──")
        gf = make_global_features(df, lags=cfg["lags"])
        r  = train_all(
            df, gf, hit_mat, co_mat,
            lags=cfg["lags"], val_size=val_size, min_train=min_train,
            train_window=cfg["train_window"], decay=cfg["decay"],
            model_type=cfg.get("model_type", "xgb"),
            xgb_params=cfg.get("xgb_params", None),
            show_importance=False,
        )
        all_results.append(r)
    return all_results


def _ensemble_weights(all_results: list, baseline: float) -> np.ndarray:
    weights = []
    for r in all_results:
        precs      = [r[n]["prec"] for n in ALL_NUMS]
        mean_prec  = float(np.mean(precs))
        beat_ratio = sum(1 for p in precs if p > baseline) / len(ALL_NUMS)
        weights.append(mean_prec if beat_ratio >= ENSEMBLE_MIN_BEAT_RATIO else 0.0)

    arr   = np.array(weights)
    total = arr.sum()
    if total == 0:
        beat_ratios = [
            sum(1 for p in [r[n]["prec"] for n in ALL_NUMS] if p > baseline) / len(ALL_NUMS)
            for r in all_results
        ]
        arr[int(np.argmax(beat_ratios))] = 1.0
        total = 1.0
    return arr / total


def predict_top_k_ensemble(all_results: list,
                            top_k: int = 8,
                            filter_threshold: float = None) -> list:
    baseline = hypergeometric_baseline(top_k=top_k)
    thresh   = filter_threshold if filter_threshold is not None else baseline
    weights  = _ensemble_weights(all_results, baseline)

    print(f"\n  Ensemble weights:")
    for i, (w, cfg) in enumerate(zip(weights, ENSEMBLE_CONFIGS)):
        precs      = [all_results[i][n]["prec"] for n in ALL_NUMS]
        beat_ratio = sum(1 for p in precs if p > baseline) / len(ALL_NUMS)
        status     = "✅" if w > 0 else f"❌ (beat={beat_ratio:.0%}<{ENSEMBLE_MIN_BEAT_RATIO:.0%})"
        print(f"    {cfg['label']:<42} w={w:.4f}  {status}")

    n_active = sum(1 for w in weights if w > 0)
    maj_min  = max(1, n_active // 2 + 1) if n_active > 1 else 1

    proba = []
    for num in ALL_NUMS:
        prob_avg   = 0.0
        cvp_avg    = 0.0
        beat_count = 0
        for i, (w, r) in enumerate(zip(weights, all_results)):
            res    = r[num]
            last_X = res["predict_row"][res["feature_cols"]].values
            prob   = res["model"].predict_proba(last_X)[0][1]
            if w > 0:
                prob_avg += w * prob
                cvp_avg  += w * res["prec"]
            if res["prec"] > baseline:
                beat_count += 1
        proba.append((num, float(prob_avg), float(cvp_avg), beat_count))

    before     = len(proba)
    proba_filt = [(n, p, c, b) for n, p, c, b in proba
                  if c >= thresh and b >= maj_min]
    removed    = before - len(proba_filt)
    print(f"\n  🔍 Filter: CV_prec >= {thresh:.4f} AND beat_configs >= {maj_min}/{len(all_results)}"
          f"  (active={n_active})  → loai {removed} so, con {len(proba_filt)} so")

    if len(proba_filt) < top_k and maj_min > 1:
        proba_filt = [(n, p, c, b) for n, p, c, b in proba if c >= thresh]
        print(f"  ⚠️  Fallback: bo majority filter, con {len(proba_filt)} so")

    proba_filt.sort(key=lambda x: -x[1])
    return [(n, p, c) for n, p, c, _ in proba_filt]


def predict_top_k_single(results: dict,
                          top_k: int = 8,
                          filter_threshold: float = None) -> list:
    baseline = hypergeometric_baseline(top_k=top_k)
    thresh   = filter_threshold if filter_threshold is not None else baseline

    proba = []
    for num in ALL_NUMS:
        r      = results[num]
        last_X = r["predict_row"][r["feature_cols"]].values
        prob   = r["model"].predict_proba(last_X)[0][1]
        proba.append((num, prob, r["prec"]))

    before  = len(proba)
    proba   = [(n, p, c) for n, p, c in proba if c >= thresh]
    removed = before - len(proba)
    if removed > 0:
        print(f"\n  🔍 Filter CV_prec >= {thresh:.4f}: loai {removed} so, con {len(proba)} so")

    proba.sort(key=lambda x: -x[1])
    return proba


# ─────────────────────────────────────────────
# 14. SHOW PREDICTION
# ─────────────────────────────────────────────

def show_prediction(df: pd.DataFrame, proba: list, top_k: int = 8):
    last_ky  = str(df["ky"].iloc[-1] or "").strip()
    top      = proba[:top_k]
    baseline = hypergeometric_baseline(top_k=top_k)

    print(f"\n{'═'*62}")
    print(f"📌 Input  : ky {last_ky}")
    print(f"🎯 Du doan cho ky {int(last_ky) + 1}:")
    print(f"  Baseline (hypergeometric 5/35): {baseline:.4f} ({baseline:.1%})")
    print(f"{'═'*62}")
    print(f"\n  📌 Top {top_k} so chinh (N1-N5):")

    nums = []
    for rank, (num, prob, cvp) in enumerate(top, 1):
        bar  = "█" * int(prob * 25)
        flag = "✅" if cvp > baseline else "⚠️ "
        print(f"    #{rank:>2}  So {num:>2}  P={prob:.4f}  CV_prec={cvp:.4f}  {flag}  {bar}")
        nums.append(num)

    n_even  = sum(1 for n in nums if n % 2 == 0)
    dec_cnt = Counter(n // 10 for n in nums)
    print(f"\n  → Chon    : {sorted(nums)}")
    print(f"  → Chan/Le : {n_even}C / {top_k - n_even}L")
    print(f"  → Chuc    : " + "  ".join(
        f"{d*10}x:{dec_cnt.get(d, 0)}" for d in range(4) if dec_cnt.get(d, 0) > 0
    ))
    print(f"{'═'*62}\n")
    return nums


# ─────────────────────────────────────────────
# 15. SAVE PREDICT
# ─────────────────────────────────────────────

def save_predict(args, df: pd.DataFrame, proba: list, cv_prec: float):
    import psycopg2

    input_ky = str(df["ky"].iloc[-1] or "").strip()
    try:
        predict_ky = str(int(input_ky) + 1).zfill(len(input_ky))
    except:
        predict_ky = None

    score_mode = "ensemble" if getattr(args, "ensemble", False) else "pure_p"

    row = {}
    for i in range(1, 19):
        if i <= len(proba):
            num, prob, cvp = proba[i - 1]
            row[f"n{i}"]   = int(num)
            row[f"p{i}"]   = float(prob)
            row[f"cvp{i}"] = float(cvp)
        else:
            row[f"n{i}"] = row[f"p{i}"] = row[f"cvp{i}"] = None

    sql = """
    INSERT INTO public.l535kqpredict (
        run_at, version, description,
        source, lags, train_window, decay, top_k, score_mode,
        cv_prec_mean, input_ky, predict_ky,
        n1,p1,cvp1, n2,p2,cvp2, n3,p3,cvp3, n4,p4,cvp4,
        n5,p5,cvp5, n6,p6,cvp6, n7,p7,cvp7, n8,p8,cvp8,
        n9,p9,cvp9, n10,p10,cvp10, n11,p11,cvp11, n12,p12,cvp12,
        n13,p13,cvp13, n14,p14,cvp14, n15,p15,cvp15, n16,p16,cvp16,
        n17,p17,cvp17, n18,p18,cvp18
    ) VALUES (
        %(run_at)s, %(version)s, %(description)s,
        %(source)s, %(lags)s, %(train_window)s, %(decay)s,
        %(top_k)s, %(score_mode)s, %(cv_prec_mean)s,
        %(input_ky)s, %(predict_ky)s,
        %(n1)s,%(p1)s,%(cvp1)s, %(n2)s,%(p2)s,%(cvp2)s,
        %(n3)s,%(p3)s,%(cvp3)s, %(n4)s,%(p4)s,%(cvp4)s,
        %(n5)s,%(p5)s,%(cvp5)s, %(n6)s,%(p6)s,%(cvp6)s,
        %(n7)s,%(p7)s,%(cvp7)s, %(n8)s,%(p8)s,%(cvp8)s,
        %(n9)s,%(p9)s,%(cvp9)s, %(n10)s,%(p10)s,%(cvp10)s,
        %(n11)s,%(p11)s,%(cvp11)s, %(n12)s,%(p12)s,%(cvp12)s,
        %(n13)s,%(p13)s,%(cvp13)s, %(n14)s,%(p14)s,%(cvp14)s,
        %(n15)s,%(p15)s,%(cvp15)s, %(n16)s,%(p16)s,%(cvp16)s,
        %(n17)s,%(p17)s,%(cvp17)s, %(n18)s,%(p18)s,%(cvp18)s
    )
    ON CONFLICT (predict_ky, version, score_mode) DO UPDATE SET
        run_at=EXCLUDED.run_at, description=EXCLUDED.description,
        source=EXCLUDED.source, lags=EXCLUDED.lags,
        train_window=EXCLUDED.train_window, decay=EXCLUDED.decay,
        top_k=EXCLUDED.top_k, cv_prec_mean=EXCLUDED.cv_prec_mean,
        input_ky=EXCLUDED.input_ky,
        n1=EXCLUDED.n1,p1=EXCLUDED.p1,cvp1=EXCLUDED.cvp1,
        n2=EXCLUDED.n2,p2=EXCLUDED.p2,cvp2=EXCLUDED.cvp2,
        n3=EXCLUDED.n3,p3=EXCLUDED.p3,cvp3=EXCLUDED.cvp3,
        n4=EXCLUDED.n4,p4=EXCLUDED.p4,cvp4=EXCLUDED.cvp4,
        n5=EXCLUDED.n5,p5=EXCLUDED.p5,cvp5=EXCLUDED.cvp5,
        n6=EXCLUDED.n6,p6=EXCLUDED.p6,cvp6=EXCLUDED.cvp6,
        n7=EXCLUDED.n7,p7=EXCLUDED.p7,cvp7=EXCLUDED.cvp7,
        n8=EXCLUDED.n8,p8=EXCLUDED.p8,cvp8=EXCLUDED.cvp8,
        n9=EXCLUDED.n9,p9=EXCLUDED.p9,cvp9=EXCLUDED.cvp9,
        n10=EXCLUDED.n10,p10=EXCLUDED.p10,cvp10=EXCLUDED.cvp10,
        n11=EXCLUDED.n11,p11=EXCLUDED.p11,cvp11=EXCLUDED.cvp11,
        n12=EXCLUDED.n12,p12=EXCLUDED.p12,cvp12=EXCLUDED.cvp12,
        n13=EXCLUDED.n13,p13=EXCLUDED.p13,cvp13=EXCLUDED.cvp13,
        n14=EXCLUDED.n14,p14=EXCLUDED.p14,cvp14=EXCLUDED.cvp14,
        n15=EXCLUDED.n15,p15=EXCLUDED.p15,cvp15=EXCLUDED.cvp15,
        n16=EXCLUDED.n16,p16=EXCLUDED.p16,cvp16=EXCLUDED.cvp16,
        n17=EXCLUDED.n17,p17=EXCLUDED.p17,cvp17=EXCLUDED.cvp17,
        n18=EXCLUDED.n18,p18=EXCLUDED.p18,cvp18=EXCLUDED.cvp18,
        updated_at=now();
    """
    params = {
        "run_at": datetime.now(timezone.utc),
        "version": VERSION, "description": DESCRIPTION,
        "source": getattr(args, "source", "db"),
        "lags": getattr(args, "lags", 3),
        "train_window": getattr(args, "train_window", 0),
        "decay": getattr(args, "decay", 0.995),
        "top_k": getattr(args, "top", 8),
        "score_mode": score_mode,
        "cv_prec_mean": float(cv_prec),
        "input_ky": input_ky, "predict_ky": predict_ky,
        **row,
    }
    conn = psycopg2.connect(DATABASE_URL)
    conn.autocommit = False
    cur  = conn.cursor()
    try:
        cur.execute(sql, params)
        conn.commit()
        print(f"\n💾 Saved → l535kqpredict"
              f"  (input_ky={input_ky}, predict_ky={predict_ky},"
              f"  version={VERSION}, mode={score_mode})")
    except Exception as e:
        conn.rollback()
        print(f"\n⚠️  Save failed: {e}")
    finally:
        cur.close()
        conn.close()


# ─────────────────────────────────────────────
# 16. UPDATE RESULT
# ─────────────────────────────────────────────

def update_result():
    import psycopg2, psycopg2.extras

    conn = psycopg2.connect(DATABASE_URL)
    conn.autocommit = False
    cur  = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

    try:
        cur.execute("""
            SELECT id, predict_ky, top_k, n1,n2,n3,n4,n5,n6,n7,n8
            FROM public.l535kqpredict
            WHERE actual_n1 IS NULL AND predict_ky IS NOT NULL
            ORDER BY predict_ky
        """)
        pending  = cur.fetchall()
        baseline = hypergeometric_baseline(top_k=8)
        print(f"📋 Tim thay {len(pending)} ky chua co ket qua thuc te")
        updated  = 0

        for row in pending:
            predict_ky = str(row["predict_ky"] or "").strip()
            # Chi lay N1-N5 (so chinh) de doi chieu
            cur.execute("""
                SELECT n1,n2,n3,n4,n5 FROM public.l535kqdetail
                WHERE ky = %s LIMIT 1
            """, (predict_ky,))
            actual = cur.fetchone()
            if actual is None:
                print(f"  ky {predict_ky}: chua co ket qua, bo qua")
                continue

            actual_nums = {
                int(actual[f"n{i}"])
                for i in range(1, 6)   # chi N1-N5
                if actual[f"n{i}"] is not None and actual[f"n{i}"] != 0
            }
            top_k     = int(row["top_k"] or 8)
            pred_nums = [int(row[f"n{i}"]) for i in range(1, 19)
                         if row.get(f"n{i}") is not None
                         and len([x for x in [row.get(f"n{j}") for j in range(1, i)]
                                  if x is not None]) < top_k][:top_k]

            hits     = set(pred_nums) & actual_nums
            hit_cnt  = len(hits)
            prec_val = hit_cnt / top_k if top_k > 0 else 0.0

            cur.execute("""
                UPDATE public.l535kqpredict SET
                    actual_n1=%(an1)s, actual_n2=%(an2)s, actual_n3=%(an3)s,
                    actual_n4=%(an4)s, actual_n5=%(an5)s, actual_n6=%(an6)s,
                    hit_cnt=%(hit_cnt)s, precision_val=%(prec)s, updated_at=%(upd)s
                WHERE id=%(id)s
            """, {
                "an1": actual["n1"], "an2": actual["n2"], "an3": actual["n3"],
                "an4": actual["n4"], "an5": actual["n5"], "an6": None,  # so phu khong predict
                "hit_cnt": hit_cnt, "prec": prec_val,
                "upd": datetime.now(timezone.utc), "id": row["id"],
            })
            flag = "✅" if prec_val > baseline else ("🟡" if prec_val > 0 else "⚠️ ")
            print(f"  ky {predict_ky}: pred={sorted(pred_nums)}  actual={sorted(actual_nums)}"
                  f"  hit={hit_cnt}/{top_k}  prec={prec_val:.1%}  {flag}")
            updated += 1

        conn.commit()
        print(f"\n✅ Da update {updated}/{len(pending)} ky")
    except Exception as e:
        conn.rollback()
        print(f"[ERROR] {e}")
        raise
    finally:
        cur.close()
        conn.close()


# ─────────────────────────────────────────────
# 17. MAIN
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="XGBoost predict L535 v1.0.0")
    parser.add_argument("--update_result", action="store_true")
    parser.add_argument("--source",        choices=["db", "csv"], default="db")
    parser.add_argument("--file",          default="l535kqdetail.csv")
    parser.add_argument("--n_ky",          type=int,   default=0)
    parser.add_argument("--qh_ky",         type=int,   default=20)
    parser.add_argument("--lags",          type=int,   default=3)
    parser.add_argument("--top",           type=int,   default=8)
    parser.add_argument("--val_size",      type=int,   default=50)
    parser.add_argument("--min_train",     type=int,   default=400)
    parser.add_argument("--train_window",  type=int,   default=0)
    parser.add_argument("--decay",         type=float, default=0.995)
    parser.add_argument("--importance",    action="store_true")
    parser.add_argument("--ensemble",      action="store_true")
    parser.add_argument("--filter",        type=str,   default="auto",
                        help="CV prec filter: float hoac 'auto' (default: hypergeometric baseline)")
    parser.add_argument("--save",          action="store_true")
    args = parser.parse_args()

    if args.update_result:
        print("🔄 Update ket qua thuc te...")
        update_result()
        return

    # Parse --filter
    if isinstance(args.filter, str) and args.filter.lower() == "auto":
        filter_thresh = None
    else:
        try:
            filter_thresh = float(args.filter)
        except (ValueError, TypeError):
            print(f"⚠️  --filter '{args.filter}' khong hop le, dung auto")
            filter_thresh = None

    print(f"📌 Version : {VERSION}")
    print(f"📝 Desc    : {DESCRIPTION[:90]}...")
    print("📥 Loading data...")
    df = (load_from_csv(args.file) if args.source == "csv"
          else load_from_db(args.n_ky, args.qh_ky))
    print(f"   {len(df)} ky")
    print(f"   Mega  : {'✅' if 'mgn1' in df.columns else '⚠️'}")
    print(f"   JP    : {'✅' if 'jpck' in df.columns else '⚠️'}")
    print(f"   QHL   : {'✅' if 'qhl'  in df.columns else '⚠️'}")
    print(f"   Power : {'✅' if 'pwn1' in df.columns else '⚠️'}")

    print("\n⚙️  Precomputing hit matrix & co-occurrence matrix (N1-N5 only)...")
    hit_mat = build_hit_matrix(df)
    co_mat  = build_cooccurrence_matrix(hit_mat)
    print(f"   hit_mat: {hit_mat.shape}  co_mat: {co_mat.shape}")
    baseline = hypergeometric_baseline(top_k=args.top)
    print(f"   Baseline (hypergeometric 5/35, top={args.top}): {baseline:.4f} ({baseline:.1%})")

    if args.ensemble:
        print(f"\n🚀 Training ensemble ({len(ENSEMBLE_CONFIGS)} configs × 35 so)...")
        all_results = train_ensemble(df, hit_mat, co_mat,
                                     val_size=args.val_size, min_train=args.min_train)
        print(f"\n🔮 Predicting (ensemble, weighted by CV prec)...")
        proba   = predict_top_k_ensemble(all_results, top_k=args.top,
                                          filter_threshold=filter_thresh)
        cv_prec = np.mean([all_results[0][n]["prec"] for n in ALL_NUMS])
    else:
        print("\n⚙️  Global feature engineering...")
        global_feat = make_global_features(df, lags=args.lags)
        print(f"   {global_feat.shape[1]} global features")
        print(f"\n🚀 Training 35 model (top={args.top})...")
        results = train_all(
            df, global_feat, hit_mat, co_mat,
            lags=args.lags, val_size=args.val_size, min_train=args.min_train,
            train_window=args.train_window, decay=args.decay,
            show_importance=args.importance,
        )
        print(f"\n🔮 Predicting...")
        proba   = predict_top_k_single(results, top_k=args.top,
                                        filter_threshold=filter_thresh)
        cv_prec = np.mean([results[n]["prec"] for n in ALL_NUMS])

    show_prediction(df, proba, top_k=args.top)

    if args.save:
        save_predict(args, df, proba, cv_prec)


if __name__ == "__main__":
    main()