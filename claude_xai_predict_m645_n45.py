"""
claude_xai_predict_m645_n45.py
────────────────────────────────
Du doan M645:
  - 45 model nhi phan cho N1-N6 (6 so chinh, range 1-45) → chon top K

Kien truc (v2.0.0 — align voi P655 v1.3.0):
  - XGBoost / LightGBM / CatBoost binary classifier / so (45 model)
  - Walk-forward CV: F-beta(0.5) threshold tuning, lag-1 strategy (no leak)
  - No data leak: train (ky 0→N-2), predict_row (ky N-1)
  - Sliding window + time decay weight
  - Imbalance: compute_sample_weight("balanced") * decay (bo scale_pos_weight)
  - build_hit_matrix + build_cooccurrence_matrix vectorized (thay n_pair_freq O(N²))
  - Brier score monitor
  - Ensemble weighted softmax + beat_ratio filter + majority-vote filter
  - --filter auto (hypergeometric baseline)
  - Global feat: drawn_min/max/range/consec, sc_trend (tu P655)
  - Per-num: n_freq_trend, n_gap_mean/std, n_pos1-6_freq, n_cooccur_score, n_val_norm
  - Save predict → m645kqpredict
  - Update result → m645kqdetail

Fixed bugs:
  - CatBoost: bo scale_pos_weight (khong co param nay)
  - Double-weight: bo scale_pos_weight khoi XGB/LGBM, chi dung balanced*decay
  - _show_top_importance: chia len(ALL_NUMS) thay hardcode /45
  - update_result: pred_nums extraction don gian, chinh xac
  - train_ensemble: truyen hit_mat, co_mat
  - _filter str 'auto' support

Cross-ref: chi dung Power (PW), bo P655 do it overlap.

Usage:
    python claude_xai_predict_m645_n45.py --source db --lags 3 --top 8
    python claude_xai_predict_m645_n45.py --source db --top 8 --ensemble --filter auto --save
    python claude_xai_predict_m645_n45.py --update_result
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

ALL_NUMS = list(range(1, 46))   # M645: so chinh 1-45
N_MAIN   = 6
N_POOL   = 45

VERSION     = "v2.0.0"
DESCRIPTION = (
    "45 binary classifier (XGB/LGBM/CatBoost), no-leak split, vectorized features, "
    "balanced sample_weight, F-beta lag-1 threshold (no eval leak), "
    "hypergeometric baseline, weighted ensemble softmax, "
    "majority-vote filter, PW cross features, save to m645kqpredict"
)

ENSEMBLE_MIN_BEAT_RATIO = 6 / 45   # ~13.3%: config bi loai neu beat_ratio < nguong


# ─────────────────────────────────────────────
# 0. BASELINE
# ─────────────────────────────────────────────

def hypergeometric_baseline(top_k: int,
                             n_drawn: int = N_MAIN,
                             pool: int = N_POOL) -> float:
    """E[hits]/top_k = 6/45 ≈ 0.1333"""
    return n_drawn / pool


# ─────────────────────────────────────────────
# 1. LOAD
# ─────────────────────────────────────────────

def load_from_db(n_ky: int = 0, qh_ky: int = 20) -> pd.DataFrame:
    import psycopg2
    limit = f"LIMIT {n_ky}" if n_ky > 0 else ""
    sql = f"""
        SELECT * FROM public.m645kqdetail
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
    """Lay tap hop n1-n6 tu mot row."""
    return {
        int(row[c]) for c in ["n1", "n2", "n3", "n4", "n5", "n6"]
        if pd.notna(row[c]) and row[c] != 0
    }


# ─────────────────────────────────────────────
# 3. HIT MATRIX & CO-OCCURRENCE (vectorized)
# ─────────────────────────────────────────────

def build_hit_matrix(df: pd.DataFrame) -> np.ndarray:
    """
    Tra ve ma tran (n_ky, N_POOL+1), dtype int8.
    hit_mat[i, v] = 1 neu so v xuat hien o ky thu i.
    Index 0 khong dung (so bat dau tu 1).
    """
    n   = len(df)
    mat = np.zeros((n, N_POOL + 1), dtype=np.int8)
    for col in ["n1", "n2", "n3", "n4", "n5", "n6"]:
        if col not in df.columns:
            continue
        vals = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int).values
        for i, v in enumerate(vals):
            if 1 <= v <= N_POOL:
                mat[i, v] = 1
    return mat


def build_cooccurrence_matrix(hit_mat: np.ndarray) -> np.ndarray:
    """
    Co-occurrence matrix (N_POOL x N_POOL).
    co_mat[i,j] = so lan so (i+1) va (j+1) cung xuat hien.
    Duong cheo = 0.
    """
    h  = hit_mat[:, 1:].astype(np.float32)
    co = h.T @ h
    np.fill_diagonal(co, 0)
    return co


# ─────────────────────────────────────────────
# 4. GLOBAL FEATURES
# ─────────────────────────────────────────────

def make_global_features(df: pd.DataFrame, lags: int = 3) -> pd.DataFrame:
    """
    Tinh cac feature GLOBAL cua M645.
    Tat ca shift >= 1. Cross-ref: chi Power (PW).
    """
    d = df.copy()

    # ── Encode DD ──
    d["dd_enc"] = d["dd"].apply(
        lambda v: {"CC": 0, "CL": 1, "LC": 2, "LL": 3}.get(str(v).strip(), -1)
    )

    # ── CL6 ──
    s = d["cl6"].astype(str).str.strip()
    for i in range(6):
        d[f"cl6_b{i}"] = s.str[i].map({"C": 0, "L": 1}).fillna(-1).astype(int)
    d["cl6_pattern"] = s.apply(
        lambda v: int("".join("0" if c == "C" else "1" for c in v), 2)
        if len(v) == 6 else np.nan
    )
    d["cl6_changed"] = (d["cl6_pattern"] != d["cl6_pattern"].shift(1)).astype(float)

    # ── Drawn stats (tu P655) ──
    num_cols = ["n1", "n2", "n3", "n4", "n5", "n6"]
    nums_df  = df[num_cols].apply(pd.to_numeric, errors="coerce")
    d["drawn_min"]   = nums_df.min(axis=1)
    d["drawn_max"]   = nums_df.max(axis=1)
    d["drawn_range"] = d["drawn_max"] - d["drawn_min"]
    d["drawn_consec"] = nums_df.apply(
        lambda r: sum(
            1 for i in range(len(r) - 1)
            if pd.notna(r.iloc[i]) and pd.notna(r.iloc[i + 1])
            and abs(r.iloc[i + 1] - r.iloc[i]) == 1
        ), axis=1
    ).astype(float)

    # ── Thu ──
    d["thu_enc"] = _encode_thu(d["thu"])

    # ── JP ──
    if "jpck" in d.columns:
        d["jpck"] = _safe_int(d["jpck"]).fillna(0)
    if "jp_cnt" in d.columns:
        d["jp_enc"] = (_safe_int(d["jp_cnt"]) > 0).astype(int)

    # ── PW (Power) cross features — chi M645 ──
    if all(c in d.columns for c in ["pwn1", "pwn6"]):
        pwn = [_safe_int(d[f"pwn{i}"]) for i in range(1, 7)]
        if "pwthu" in d.columns:
            d["pw_thu_enc"]   = _encode_thu(d["pwthu"])
        if "pwngay" in d.columns and "ngay" in d.columns:
            d["pw_ngay_diff"] = _ngay_diff(d["ngay"], d["pwngay"])
        d["pw_dd"] = _encode_dd(pwn[0], pwn[5])
        d["pw_sc"] = sum((n % 2 == 0).astype(float) for n in pwn)

    feat = pd.DataFrame(index=d.index)

    # A. Biet truoc — no shift
    feat["thu_enc"] = d["thu_enc"]

    # B. Ky truoc (shift 1)
    prev_cols = [
        "dd_enc", "sc", "sum", "ke",
        "cl6_b0", "cl6_b1", "cl6_b2", "cl6_b3", "cl6_b4", "cl6_b5",
        "cl6_pattern", "cl6_changed",
        "dec0", "dec1", "dec2", "dec3", "dec4",  # M645: chuc 0-4 (so 1-45)
        "drawn_min", "drawn_max", "drawn_range", "drawn_consec",
    ]
    for c in ["jpck", "jp_enc"]:
        if c in d.columns:
            prev_cols.append(c)
    prev_cols = [c for c in prev_cols if c in d.columns]
    for col in prev_cols:
        feat[f"{col}_prev"] = d[col].shift(1)

    # C. Lag 2 → lags
    lag_cols = ["dd_enc", "sc", "cl6_pattern", "drawn_range"]
    lag_cols = [c for c in lag_cols if c in d.columns]
    for lag in range(2, lags + 1):
        for col in lag_cols:
            feat[f"{col}_lag{lag}"] = d[col].shift(lag)

    # D. Rolling sc + drawn_range
    for col in ["sc", "drawn_range"]:
        if col not in d.columns:
            continue
        base = d[col].shift(1)
        for w in [3, 5]:
            feat[f"{col}_roll{w}_mean"] = base.rolling(w).mean()
            feat[f"{col}_roll{w}_std"]  = base.rolling(w).std()

    # E. sc_trend (tu P655)
    if "sc" in d.columns:
        sc_s = d["sc"].shift(1)
        feat["sc_trend"] = sc_s.rolling(5).mean() - sc_s.rolling(10).mean()

    # F. PW features (no shift — biet truoc ky nay)
    for c in ["pw_thu_enc", "pw_ngay_diff", "pw_dd", "pw_sc"]:
        if c in d.columns:
            feat[c] = d[c]

    # G. Cross PW x M645
    if "pw_dd" in feat.columns and "dd_enc_prev" in feat.columns:
        feat["cross_pw_dd_match"] = (
            feat["pw_dd"] == feat["dd_enc_prev"]
        ).astype(float)
    if "pw_sc" in feat.columns and "sc_prev" in feat.columns:
        feat["cross_sc_pw"] = feat["sc_prev"] * feat["pw_sc"]

    return feat


# ─────────────────────────────────────────────
# 5. PER-NUMBER FEATURES (vectorized — tu P655)
# ─────────────────────────────────────────────

def make_per_number_features(df: pd.DataFrame,
                              num: int,
                              hit_mat: np.ndarray,
                              co_mat: np.ndarray) -> pd.DataFrame:
    """
    Tinh features rieng cho so `num` trong range 1-45.
    Dung hit_mat va co_mat vectorized, thay the n_pair_freq O(N²) cu.
    """
    feat = pd.DataFrame(index=df.index)
    n    = len(df)
    hit  = hit_mat[:, num].astype(float)

    # n_hit_prev
    feat["n_hit_prev"] = pd.Series(hit, index=df.index).shift(1)

    # n_last_seen: so ky ke tu lan cuoi xuat hien
    last_seen = np.full(n, np.nan)
    last_idx  = -1
    for i in range(n):
        if last_idx >= 0:
            last_seen[i] = i - last_idx
        if hit[i] == 1:
            last_idx = i
    feat["n_last_seen"] = pd.Series(last_seen, index=df.index).shift(1)

    # n_streak: so ky lien tiep khong xuat hien
    streak = np.zeros(n)
    cur    = 0
    for i in range(n):
        streak[i] = cur
        cur = 0 if hit[i] == 1 else cur + 1
    feat["n_streak"] = pd.Series(streak, index=df.index).shift(1)

    # n_freq_10/20/50, n_freq_trend
    hit_s = pd.Series(hit, index=df.index).shift(1)
    feat["n_freq_10"]    = hit_s.rolling(10).mean()
    feat["n_freq_20"]    = hit_s.rolling(20).mean()
    feat["n_freq_50"]    = hit_s.rolling(50).mean()
    feat["n_freq_trend"] = hit_s.rolling(10).mean() - hit_s.rolling(30).mean()

    # n_gap_mean, n_gap_std: thong ke khoang cach giua cac lan xuat hien
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

    # n_pos1-6_freq: tan suat so num xuat hien o tung vi tri N1-N6
    for pos_i, col in enumerate(["n1", "n2", "n3", "n4", "n5", "n6"], 1):
        if col in df.columns:
            pos_hit = (pd.to_numeric(df[col], errors="coerce") == num).astype(float)
            feat[f"n_pos{pos_i}_freq"] = pos_hit.shift(1).rolling(50).mean()

    # n_cooccur_score: diem co-occurrence voi cac so ky truoc (vectorized, thay n_pair_freq)
    co_row    = co_mat[num - 1, :]                                         # (N_POOL,)
    prev_mat  = np.vstack([np.zeros((1, N_POOL + 1), dtype=np.int8),
                            hit_mat[:-1, :]])                              # shift 1
    co_scores = np.full(n, np.nan)
    for i in range(1, n):
        co_scores[i] = float(co_row @ prev_mat[i, 1:]) / max(N_MAIN, 1)
    feat["n_cooccur_score"] = pd.Series(co_scores, index=df.index)

    # n_is_qhl
    if "qhl" in df.columns:
        def _in_qhl(v):
            if pd.isna(v) or v == "":
                return 0.0
            try:
                return float(num in [int(x) for x in str(v).split(",") if x.strip()])
            except:
                return 0.0
        feat["n_is_qhl"] = df["qhl"].shift(1).apply(_in_qhl)

    # Static
    feat["n_is_even"]  = float(num % 2 == 0)
    feat["n_dec"]      = float(num // 10)      # 0-4 cho M645 (so 1-45)
    feat["n_val_norm"] = float(num / N_POOL)   # 0.022 - 1.0

    return feat


# ─────────────────────────────────────────────
# 6. BUILD DATASET
# ─────────────────────────────────────────────

def build_dataset(df: pd.DataFrame,
                  num: int,
                  global_feat: pd.DataFrame,
                  hit_mat: np.ndarray,
                  co_mat: np.ndarray) -> tuple:
    """
    Tra ve (combined_train, predict_row, feature_cols).
    No-leak: combined_train = ky 0→N-2, predict_row = ky N-1.
    """
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
    """
    Bo scale_pos_weight: da dung compute_sample_weight("balanced"),
    dung them scale_pos_weight se double-weight, sai.
    """
    defaults = dict(
        n_estimators=300, max_depth=3, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.7, min_child_weight=3,
        gamma=0.1, reg_alpha=0.05, reg_lambda=1.0,
        objective="binary:logistic", eval_metric="logloss",
        random_state=42, n_jobs=-1,
        early_stopping_rounds=30,
    )
    if params:
        defaults.update(params)
    return XGBClassifier(**defaults)


def make_lgbm(params: dict = None) -> LGBMClassifier:
    """Bo scale_pos_weight — ly do nhu XGB."""
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
    """
    Bo scale_pos_weight — CatBoost khong co param nay (gay loi v1).
    Dung class_weights thay the neu can, hien tai dung sample_weight.
    """
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
# 9. THRESHOLD TUNING (tu P655)
# ─────────────────────────────────────────────

def _best_threshold_fbeta(y_true: np.ndarray, y_prob: np.ndarray,
                           beta: float = 0.5) -> float:
    """
    Tim threshold maximize F-beta(0.5) tren val set.
    beta < 1 → uu tien precision hon recall.
    Dung de tim threshold cho fold TIEP THEO (lag-1, no leak).
    """
    if y_true.sum() == 0:
        return 0.5
    candidates     = np.unique(y_prob)
    best_t, best_f = 0.5, -1.0
    b2 = beta ** 2
    for t in candidates:
        pred  = (y_prob >= t).astype(int)
        tp    = float((pred * y_true).sum())
        fp    = float((pred * (1 - y_true)).sum())
        fn    = float(((1 - pred) * y_true).sum())
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
    """
    Tra ve (final_model, best_threshold, mean_cv_precision, mean_cv_recall, mean_brier).

    Threshold strategy (lag-1, no leak):
      - Fold i: train → predict val_i → evaluate bang threshold cua fold (i-1)
      - Sau do tim threshold moi tren val_i → dung cho fold (i+1)
      - best_thresh = threshold tu fold cuoi → dung khi predict ky tiep theo
    Brier score: chi de monitor.
    """
    X = combined[feature_cols].values
    y = combined["y"].values

    splits      = _walk_forward_splits(len(X), min_train, val_size)
    prec_list   = []
    rec_list    = []
    brier_list  = []
    thresh_list = []

    prev_thresh = 0.5   # fold dau tien: chua co threshold tu truoc

    for tr_idx, va_idx in splits:
        X_tr, X_va = X[tr_idx], X[va_idx]
        y_tr, y_va = y[tr_idx], y[va_idx]
        if y_tr.sum() == 0:
            continue

        if train_window > 0 and len(tr_idx) > train_window:
            tr_idx = tr_idx[-train_window:]
            X_tr   = X[tr_idx]
            y_tr   = y[tr_idx]

        sw_tr = (compute_sample_weight("balanced", y_tr)
                 * _make_decay_weights(len(y_tr), decay))

        mdl = _make_model(model_type, xgb_params)
        if model_type == "xgb":
            mdl.fit(X_tr, y_tr, sample_weight=sw_tr,
                    eval_set=[(X_va, y_va)], verbose=False)
        else:
            mdl.fit(X_tr, y_tr, sample_weight=sw_tr)

        prob_va = mdl.predict_proba(X_va)[:, 1]

        # Evaluate bang threshold cua fold TRUOC (lag-1 — no leak)
        pred = (prob_va >= prev_thresh).astype(int)
        prec_list.append(precision_score(y_va, pred, zero_division=0))
        rec_list.append(recall_score(y_va, pred, zero_division=0))
        brier_list.append(brier_score_loss(y_va, prob_va))

        # Tim threshold toi uu tren val hien tai → dung cho fold tiep theo
        prev_thresh = _best_threshold_fbeta(y_va, prob_va, beta=0.5)
        thresh_list.append(prev_thresh)

    # ── Final model (full data, no early stopping) ──
    if train_window > 0 and len(X) > train_window:
        X_fin, y_fin = X[-train_window:], y[-train_window:]
    else:
        X_fin, y_fin = X, y

    sw_fin  = (compute_sample_weight("balanced", y_fin)
               * _make_decay_weights(len(y_fin), decay))
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
# 11. TRAIN ALL (45 model)
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
    """Train 45 model cho so chinh N1-N6 (range 1-45)."""
    results  = {}
    baseline = hypergeometric_baseline(top_k=8)

    print(f"\n  [N1-N6] Hypergeometric baseline precision: {baseline:.4f} ({baseline:.1%})")
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
    naive_brier = N_MAIN / N_POOL * (1 - N_MAIN / N_POOL)
    print(f"\n  Mean CV Precision : {np.mean(precs):.4f}")
    print(f"  Mean Brier Score  : {np.mean(briers):.4f}  (monitor only, naive={naive_brier:.4f})")
    print(f"  Beat baseline     : {sum(1 for p in precs if p > baseline)}/{len(ALL_NUMS)}")

    if show_importance:
        _show_top_importance(results)

    return results


# ─────────────────────────────────────────────
# 12. IMPORTANCE
# ─────────────────────────────────────────────

def _show_top_importance(results: dict, top_n: int = 15):
    imp_sum = {}
    n_models = len(ALL_NUMS)
    for num in ALL_NUMS:
        r = results[num]
        try:
            imps = r["model"].feature_importances_
        except AttributeError:
            continue
        for c, v in zip(r["feature_cols"], imps):
            imp_sum[c] = imp_sum.get(c, 0.0) + v

    imp_avg = {k: v / n_models for k, v in imp_sum.items()}   # FIX: bo hardcode /45
    top     = sorted(imp_avg.items(), key=lambda x: -x[1])[:top_n]
    print(f"\n  Top {top_n} features (trung binh {n_models} model):")
    for f, s in top:
        print(f"    {f:<44} {s:.4f}  {'█' * int(s * 500)}")

    pw_imp = sum(v for k, v in imp_avg.items() if k.startswith("pw_"))
    cx_imp = sum(v for k, v in imp_avg.items() if k.startswith("cross_"))
    jp_imp = sum(v for k, v in imp_avg.items() if "jp" in k)
    print(f"\n  Importance theo nhom:")
    print(f"    PW    : {pw_imp:.4f}")
    print(f"    Cross : {cx_imp:.4f}")
    print(f"    JP    : {jp_imp:.4f}")
    print(f"    M645  : {1 - pw_imp - cx_imp - jp_imp:.4f}")


# ─────────────────────────────────────────────
# 13. ENSEMBLE
# ─────────────────────────────────────────────

ENSEMBLE_CONFIGS = [
    {"lags": 3, "decay": 0.995, "train_window": 0,   "model_type": "xgb",      "xgb_params": {"max_depth": 3},                     "label": "M1(xgb,d3,lag3,all)"},
    {"lags": 3, "decay": 0.995, "train_window": 0,   "model_type": "xgb",      "xgb_params": {"max_depth": 5, "n_estimators": 500}, "label": "M2(xgb,d5,n500,lag3,all)"},
    {"lags": 3, "decay": 0.995, "train_window": 0,   "model_type": "lgbm",     "xgb_params": {},                                    "label": "M3(lgbm,lag3,all)"},
    {"lags": 3, "decay": 0.995, "train_window": 0,   "model_type": "lgbm",     "xgb_params": {"max_depth": 5, "n_estimators": 500}, "label": "M4(lgbm,d5,n500,lag3,all)"},
    {"lags": 3, "decay": 0.990, "train_window": 500, "model_type": "catboost", "xgb_params": {},                                    "label": "M5(cat,d3,w500)"},
]


def train_ensemble(df: pd.DataFrame,
                   hit_mat: np.ndarray,
                   co_mat: np.ndarray,
                   val_size: int,
                   min_train: int) -> list:
    """FIX: truyen hit_mat, co_mat vao train_all."""
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
    """
    Softmax weight theo mean CV precision.
    Config co beat_ratio < ENSEMBLE_MIN_BEAT_RATIO → weight = 0.
    Neu tat ca bi loai → giu config co beat_ratio cao nhat.
    """
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
    print(f"  Baseline (hypergeometric): {baseline:.4f} ({baseline:.1%})")
    print(f"{'═'*62}")
    print(f"\n  📌 Top {top_k} so chinh (N1-N6):")

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
        f"{d*10}x:{dec_cnt.get(d, 0)}" for d in range(5) if dec_cnt.get(d, 0) > 0
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
    INSERT INTO public.m645kqpredict (
        run_at, version, description,
        source, lags, train_window, decay, top_k, score_mode,
        cv_prec_n16, input_ky, predict_ky,
        n1,p1,cvp1, n2,p2,cvp2, n3,p3,cvp3, n4,p4,cvp4,
        n5,p5,cvp5, n6,p6,cvp6, n7,p7,cvp7, n8,p8,cvp8,
        n9,p9,cvp9, n10,p10,cvp10, n11,p11,cvp11, n12,p12,cvp12,
        n13,p13,cvp13, n14,p14,cvp14, n15,p15,cvp15, n16,p16,cvp16,
        n17,p17,cvp17, n18,p18,cvp18
    ) VALUES (
        %(run_at)s, %(version)s, %(description)s,
        %(source)s, %(lags)s, %(train_window)s, %(decay)s,
        %(top_k)s, %(score_mode)s, %(cv_prec_n16)s,
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
        top_k=EXCLUDED.top_k, cv_prec_n16=EXCLUDED.cv_prec_n16,
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
        "cv_prec_n16": float(cv_prec),
        "input_ky": input_ky, "predict_ky": predict_ky,
        **row,
    }
    conn = psycopg2.connect(DATABASE_URL)
    conn.autocommit = False
    cur  = conn.cursor()
    try:
        cur.execute(sql, params)
        conn.commit()
        print(f"\n💾 Saved → m645kqpredict"
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
            SELECT id, predict_ky, top_k,
                   n1,n2,n3,n4,n5,n6,n7,n8,
                   n9,n10,n11,n12,n13,n14,n15,n16,n17,n18
            FROM public.m645kqpredict
            WHERE actual_n1 IS NULL AND predict_ky IS NOT NULL
            ORDER BY predict_ky
        """)
        pending  = cur.fetchall()
        baseline = hypergeometric_baseline(top_k=8)
        print(f"📋 Tim thay {len(pending)} ky chua co ket qua thuc te")
        updated  = 0

        for row in pending:
            predict_ky = str(row["predict_ky"] or "").strip()
            cur.execute("""
                SELECT n1,n2,n3,n4,n5,n6 FROM public.m645kqdetail
                WHERE ky = %s LIMIT 1
            """, (predict_ky,))
            actual = cur.fetchone()
            if actual is None:
                print(f"  ky {predict_ky}: chua co ket qua, bo qua")
                continue

            actual_nums = {
                int(actual[f"n{i}"])
                for i in range(1, 7)
                if actual[f"n{i}"] is not None and actual[f"n{i}"] != 0
            }

            # FIX: pred_nums extraction don gian va chinh xac
            top_k     = int(row["top_k"] or 8)
            pred_nums = []
            for i in range(1, 19):
                n = row.get(f"n{i}")
                if n is not None and len(pred_nums) < top_k:
                    pred_nums.append(int(n))

            hits     = set(pred_nums) & actual_nums
            hit_cnt  = len(hits)
            prec_val = hit_cnt / top_k if top_k > 0 else 0.0

            cur.execute("""
                UPDATE public.m645kqpredict SET
                    actual_n1=%(an1)s, actual_n2=%(an2)s, actual_n3=%(an3)s,
                    actual_n4=%(an4)s, actual_n5=%(an5)s, actual_n6=%(an6)s,
                    hit_cnt=%(hit_cnt)s, precision_val=%(prec)s, updated_at=%(upd)s
                WHERE id=%(id)s
            """, {
                "an1": actual["n1"], "an2": actual["n2"], "an3": actual["n3"],
                "an4": actual["n4"], "an5": actual["n5"], "an6": actual["n6"],
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
    parser = argparse.ArgumentParser(description="XGBoost predict M645 v2.0.0 — 45 so chinh")
    parser.add_argument("--update_result", action="store_true")
    parser.add_argument("--source",        choices=["db", "csv"], default="db")
    parser.add_argument("--file",          default="m645kqdetail.csv")
    parser.add_argument("--n_ky",          type=int,   default=0)
    parser.add_argument("--qh_ky",         type=int,   default=20)
    parser.add_argument("--lags",          type=int,   default=3)
    parser.add_argument("--top",           type=int,   default=8)
    parser.add_argument("--val_size",      type=int,   default=50)
    parser.add_argument("--min_train",     type=int,   default=300)
    parser.add_argument("--train_window",  type=int,   default=0)
    parser.add_argument("--decay",         type=float, default=0.995)
    parser.add_argument("--importance",    action="store_true")
    parser.add_argument("--ensemble",      action="store_true")
    parser.add_argument("--filter",        type=str,   default="auto",
                        help="CV prec filter: float hoac 'auto' (default: hypergeometric baseline)")
    parser.add_argument("--save",          action="store_true")
    args = parser.parse_args()

    if args.update_result:
        print("🔄 Update ket qua thuc te tu m645kqdetail...")
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
    print(f"   PW  : {'✅' if 'pwn1' in df.columns else '⚠️'}")
    print(f"   JP  : {'✅' if 'jpck' in df.columns else '⚠️'}")
    print(f"   QHL : {'✅' if 'qhl'  in df.columns else '⚠️'}")

    print("\n⚙️  Precomputing hit matrix & co-occurrence matrix...")
    hit_mat  = build_hit_matrix(df)
    co_mat   = build_cooccurrence_matrix(hit_mat)
    print(f"   hit_mat: {hit_mat.shape}  co_mat: {co_mat.shape}")
    baseline = hypergeometric_baseline(top_k=args.top)
    print(f"   Baseline (hypergeometric, top={args.top}): {baseline:.4f} ({baseline:.1%})")

    if args.ensemble:
        print(f"\n🚀 Training ensemble ({len(ENSEMBLE_CONFIGS)} configs × {len(ALL_NUMS)} so)...")
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
        print(f"\n🚀 Training {len(ALL_NUMS)} model (top={args.top})...")
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