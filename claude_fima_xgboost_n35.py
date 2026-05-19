"""
claude_fima_xgboost_n35.py
──────────────────────────
Du doan 35 so cua L535 (n1-n5), chon top 8 (hoac top 18) theo xac suat.

Kien truc:
  - 35 XGBoost binary classifier song song (1 model / so)
  - Moi model predict P(so n xuat hien trong n1-n5 ky tiep theo)
  - Rank 35 xac suat, chon top K (Kieu A — pure top K)
  - No data leak: tach train (ky 1→N-1) va predict_row (ky N)

Features (no-leak, shift >= 1):
  A. Lich: thu, dot (biet truoc)
  B. Per-number history: last_seen, freq_20, freq_50, is_qhl
  C. Global L535: sc, sum, dd_enc, cl5_b0..b4, cl5_pattern, cl5_changed,
                  dec0, dec3 (lag 1-lags)
     [v1.4.0] Bo cl5_l_count (= 5 - sc, redundant). Them cl5_b0..b4 (5 bit
              rieng), cl5_changed (pattern doi so voi ky truoc).
  D. Rolling global: sc (window 3, 5)
     [v1.4.0] Bo cl5_l_count_roll* (redundant).
  E. Mega/Power: mg_dd, pw_dd, pw_sc, cross features
     [v1.4.0] Them pw_sc (dem so chan cua ky Power gan nhat).
  F. JP: jpck, is_jp
  G. Sliding window + Time decay weight
  H. [v1.4.0] scale_pos_weight=7 trong XGB (5/35 ~ 14.3% positive).

Changelog:
  v1.4.0 — Bo cl5_l_count (redundant voi sc). Them cl5_b0..b4, cl5_changed,
            pw_sc. scale_pos_weight=7. Cross: them cross_sc_pw.

Usage:
    python claude_fima_xgboost_n35.py --source db --lags 3 --top 8
    python claude_fima_xgboost_n35.py --source db --lags 3 --top 8 --importance
    python claude_fima_xgboost_n35.py --source db --lags 3 --top 8 --train_window 300 --decay 0.995
    python claude_fima_xgboost_n35.py --update_result
"""

import argparse, os, warnings
import joblib
import numpy as np
import pandas as pd
from collections import Counter
from datetime import datetime, timezone

from xgboost import XGBClassifier
from sklearn.metrics import precision_score, recall_score
from sklearn.utils.class_weight import compute_sample_weight
from scipy.stats import mode

warnings.filterwarnings("ignore")

DATABASE_URL = os.environ.get(
    "DATABASE_URL", "postgresql://postgres:bin@localhost:5432/katalott"
)

ALL_NUMS = list(range(1, 36))   # 1-35

# ── Version ───────────────────────────────────────────────────────────────────
VERSION     = "v1.4.0"
DESCRIPTION = (
    "35 binary XGBoost (n1-n5), no-leak train/predict split, "
    "sliding window + time decay, Mega/Power cross+pw_sc features, "
    "cl5 bit-encode, scale_pos_weight, walk-forward CV, save to l535kqpredict"
)

# ─────────────────────────────────────────────
# 1. LOAD
# ─────────────────────────────────────────────

def load_from_db(n_ky=0, qh_ky=20) -> pd.DataFrame:
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

def load_from_csv(filepath) -> pd.DataFrame:
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

def _ngay_diff(d1: pd.Series, d2: pd.Series) -> pd.Series:
    return (pd.to_datetime(d1, errors="coerce") -
            pd.to_datetime(d2, errors="coerce")).dt.days

def _encode_dd(n_first, n_last) -> pd.Series:
    def _cl(n): return "C" if n % 2 == 0 else "L"
    result = []
    for f, l in zip(n_first, n_last):
        if pd.isna(f) or pd.isna(l):
            result.append(np.nan)
        else:
            result.append(DD_MAP.get(_cl(int(f)) + _cl(int(l)), np.nan))
    return pd.Series(result, index=n_first.index)

def get_drawn_set(row) -> set:
    """Lay tap hop n1-n5 tu mot row."""
    return {
        int(row[c]) for c in ["n1","n2","n3","n4","n5"]
        if pd.notna(row[c]) and row[c] != 0
    }


# ─────────────────────────────────────────────
# 3. FEATURE ENGINEERING
# ─────────────────────────────────────────────

def make_global_features(df: pd.DataFrame, lags: int = 3) -> pd.DataFrame:
    """
    Tinh cac feature GLOBAL (khong phu thuoc vao so cu the).
    Tat ca shift >= 1 de tranh data leak.
    Tra ve DataFrame co cung index voi df.

    v1.4.0 changes:
      - Bo cl5_l_count (= 5 - sc, du thua tuyen tinh).
      - Them cl5_b0..b4: 5 bit C/L rieng cho tung vi tri (port tu M645).
      - Them cl5_changed: pattern cl5 co doi so voi ky truoc khong.
      - Them pw_sc: dem so chan trong ky Power gan nhat (port tu M645).
      - Rolling D: chi giu sc_roll*, bo cl5_l_count_roll* (du thua).
      - Cross G: them cross_sc_pw (sc_prev * pw_sc).
    """
    d = df.copy()

    # ── Encode DD (N1 va N5) ──
    d["dd_enc"] = d["dd"].apply(
        lambda v: {"CC": 0, "CL": 1, "LC": 2, "LL": 3}.get(str(v).strip(), -1)
    )

    # ── CL5: 5 bit rieng + pattern + changed ──
    s = d["cl5"].astype(str).str.strip()
    for i in range(5):
        d[f"cl5_b{i}"] = s.str[i].map({"C": 0, "L": 1}).fillna(-1).astype(int)
    d["cl5_pattern"] = s.apply(
        lambda v: int("".join("0" if c == "C" else "1" for c in v), 2)
        if len(v) == 5 else np.nan
    )
    d["cl5_changed"] = (d["cl5_pattern"] != d["cl5_pattern"].shift(1)).astype(float)

    # ── Thu / dot ──
    d["thu_enc"] = _encode_thu(d["thu"])
    d["dot"]     = pd.to_numeric(d["dot"], errors="coerce").fillna(0).astype(int)

    # ── JP ──
    if "jpck" in d.columns:
        d["jpck"] = pd.to_numeric(d["jpck"], errors="coerce").fillna(0)
    if "is_jp" in d.columns:
        d["is_jp_enc"] = d["is_jp"].astype(int)

    # ── Power sc (dem so chan trong ky Power) ──
    if all(c in d.columns for c in [f"pwn{i}" for i in range(1, 6)]):
        pn = [_safe_int(d[f"pwn{i}"]) for i in range(1, 6)]
        d["pw_sc"] = sum((n % 2 == 0).astype(float) for n in pn)

    feat = pd.DataFrame(index=d.index)

    # A. Biet truoc (lich) — no shift
    feat["thu_enc"] = d["thu_enc"]
    feat["dot"]     = d["dot"]

    # B. Global L535 ky truoc (shift 1)
    prev_cols = [
        "dd_enc", "sc", "sum",
        "cl5_b0", "cl5_b1", "cl5_b2", "cl5_b3", "cl5_b4",
        "cl5_pattern", "cl5_changed",
        "dec0", "dec3",
    ]
    for c in ["jpck", "is_jp_enc"]:
        if c in d.columns:
            prev_cols.append(c)
    prev_cols = [c for c in prev_cols if c in d.columns]
    for col in prev_cols:
        feat[f"{col}_prev"] = d[col].shift(1)

    # C. Lag 2 → lags (chi dd_enc, sc, cl5_pattern — bo cl5_l_count)
    lag_cols = ["dd_enc", "sc", "cl5_pattern"]
    lag_cols = [c for c in lag_cols if c in d.columns]
    for lag in range(2, lags + 1):
        for col in lag_cols:
            feat[f"{col}_lag{lag}"] = d[col].shift(lag)

    # D. Rolling window 3, 5 — chi sc (bo cl5_l_count_roll*, redundant)
    for w in [3, 5]:
        if "sc" in d.columns:
            base = d["sc"].shift(1)
            feat[f"sc_roll{w}_mean"] = base.rolling(w).mean()
            feat[f"sc_roll{w}_std"]  = base.rolling(w).std()

    # E. Mega features
    if all(c in d.columns for c in ["mgn1", "mgn6"]):
        mn = [_safe_int(d[f"mgn{i}"]) for i in range(1, 7)]
        if "mgthu" in d.columns:
            feat["mg_thu_enc"]   = _encode_thu(d["mgthu"])
        if "mgngay" in d.columns:
            feat["mg_ngay_diff"] = _ngay_diff(d["ngay"], d["mgngay"])
        feat["mg_dd"] = _encode_dd(mn[0], mn[5])

    # F. Power features
    if all(c in d.columns for c in ["pwn1", "pwn6"]):
        pn = [_safe_int(d[f"pwn{i}"]) for i in range(1, 7)]
        if "pwthu" in d.columns:
            feat["pw_thu_enc"]   = _encode_thu(d["pwthu"])
        if "pwngay" in d.columns:
            feat["pw_ngay_diff"] = _ngay_diff(d["ngay"], d["pwngay"])
        feat["pw_dd"] = _encode_dd(pn[0], pn[5])
    if "pw_sc" in d.columns:
        feat["pw_sc"] = d["pw_sc"]

    # G. Cross features
    if "mg_dd" in feat.columns and "dd_enc_prev" in feat.columns:
        feat["cross_mg_dd_match"] = (feat["mg_dd"] == feat["dd_enc_prev"]).astype(float)
    if "pw_dd" in feat.columns and "dd_enc_prev" in feat.columns:
        feat["cross_pw_dd_match"] = (feat["pw_dd"] == feat["dd_enc_prev"]).astype(float)
    if "mg_ngay_diff" in feat.columns:
        feat["cross_dot_mg_diff"] = feat["dot"] * feat["mg_ngay_diff"]
    if "pw_ngay_diff" in feat.columns:
        feat["cross_dot_pw_diff"] = feat["dot"] * feat["pw_ngay_diff"]
    if "mg_dd" in feat.columns and "sc_prev" in feat.columns:
        feat["cross_sc_mg_dd"] = feat["sc_prev"] * feat["mg_dd"]
    if "pw_dd" in feat.columns and "sc_prev" in feat.columns:
        feat["cross_sc_pw_dd"] = feat["sc_prev"] * feat["pw_dd"]
    if "pw_sc" in feat.columns and "sc_prev" in feat.columns:
        feat["cross_sc_pw"] = feat["sc_prev"] * feat["pw_sc"]

    return feat


def make_per_number_features(df: pd.DataFrame, num: int) -> pd.DataFrame:
    """
    Tinh features rieng cho so `num` (1-35).
    Tat ca deu shift >= 1.

    Features:
      - n_hit_prev   : so `num` co xuat hien ky truoc khong (0/1)
      - n_last_seen  : cach bao nhieu ky lan cuoi so `num` xuat hien
      - n_streak     : so ky lien tiep KHONG xuat hien (reset ve 0 khi xuat hien)
      - n_freq_10    : tan suat trong 10 ky gan nhat (ngan han)
      - n_freq_20    : tan suat trong 20 ky gan nhat
      - n_freq_50    : tan suat trong 50 ky gan nhat (dai han)
      - n_is_qhl     : so `num` co trong danh sach QHL khong
      - n_pair_freq  : tan suat so `num` xuat hien cung cac so ky truoc (30 ky)
      - dot_prev     : dot (1/2) cua ky truoc
      - n_is_even    : so chan (1) hay le (0)
      - n_dec        : chuc cua so (0-3)
    """
    feat = pd.DataFrame(index=df.index)

    hit = df.apply(lambda r: int(num in get_drawn_set(r)), axis=1)

    # n_hit_prev
    feat["n_hit_prev"] = hit.shift(1)

    # n_last_seen
    last_seen = []
    last = np.nan
    for i, h in enumerate(hit):
        if i == 0:
            last_seen.append(np.nan)
            if h == 1: last = 0
        else:
            last_seen.append(np.nan if pd.isna(last) else i - last)
            if h == 1: last = i
    feat["n_last_seen"] = pd.Series(last_seen, index=df.index).shift(1)

    # n_streak: so ky lien tiep khong xuat hien
    streak = []
    cur_streak = 0
    for i, h in enumerate(hit):
        if i == 0:
            streak.append(np.nan)
        else:
            streak.append(float(cur_streak))
        if h == 1:
            cur_streak = 0
        else:
            cur_streak += 1
    feat["n_streak"] = pd.Series(streak, index=df.index).shift(1)

    # n_freq_10, n_freq_20, n_freq_50
    hit_shifted = hit.shift(1)
    feat["n_freq_10"] = hit_shifted.rolling(10).mean()
    feat["n_freq_20"] = hit_shifted.rolling(20).mean()
    feat["n_freq_50"] = hit_shifted.rolling(50).mean()

    # n_is_qhl
    if "qhl" in df.columns:
        def _in_qhl(v):
            if pd.isna(v) or v == "": return 0.0
            try:
                return float(num in [int(x) for x in str(v).split(",") if x.strip()])
            except: return 0.0
        feat["n_is_qhl"] = df["qhl"].shift(1).apply(_in_qhl)

    # n_pair_freq: tan suat so `num` xuat hien cung cac so ky truoc (30 ky)
    # Vectorized: precompute hit_matrix 1 lan ben ngoai neu co, o day dung
    # rolling sum tren hit cua `num` va prev_has_overlap de tranh double loop.
    #
    # Dinh nghia: trong 30 ky gan nhat (shift 1), so ky nao co ca `num` XA
    # va co it nhat 1 so chung voi tap so ky truoc (anchor)?
    #
    # Logic tuong duong voi code cu nhung dung pandas thuan:
    #   hit_num[j]       = (num in drawn_j)          — da co san o tren
    #   prev_set[i]      = drawn_set(ky i-1)          — shift(1) cua drawn_sets
    #   overlap[j]       = len(prev_set[i] & drawn_j) > 0
    # Thay vi vong lap, ta tinh overlap tai moi j so voi prev_set cua chinh j
    # (bang cach shift drawn_sets 1), roi lay rolling(30).mean() tren
    # (hit_num & overlap_with_prev).
    drawn_sets  = df.apply(get_drawn_set, axis=1)          # Series of sets
    prev_sets   = drawn_sets.shift(1)                       # set cua ky truoc

    # overlap_flag[j] = 1 neu drawn_sets[j] co chung voi prev_sets[j]
    overlap_flag = pd.Series([
        1.0 if (isinstance(ps, set) and len(drawn_sets.iloc[j] & ps) > 0) else 0.0
        for j, ps in enumerate(prev_sets)
    ], index=df.index)

    # joint[j] = 1 neu so `num` xuat hien tai j VA co overlap voi anchor
    joint = hit.astype(float) * overlap_flag

    # rolling(30).mean() tren joint, shift 1 de tranh leak
    pair_raw = joint.shift(1).rolling(30, min_periods=30).mean()
    # Cac ky < 31 se la NaN tu dong (min_periods=30)
    feat["n_pair_freq"] = pair_raw

    # dot_prev: dot (1/2) cua ky truoc
    if "dot" in df.columns:
        feat["dot_prev"] = pd.to_numeric(df["dot"], errors="coerce").shift(1)

    # Static
    feat["n_is_even"] = float(num % 2 == 0)
    feat["n_dec"]     = float(num // 10)

    return feat


def build_dataset(df: pd.DataFrame, num: int, global_feat: pd.DataFrame,
                  lags: int = 3) -> tuple:
    """
    Gop global_feat + per_number_feat, tao target cho so `num`.

    Logic no-leak:
      - combined_train: ky 0 → N-2 (co target y thuc te)
        Train model tren tap nay.
      - predict_row   : ky N-1 (ky moi nhat, chua co target tuong lai)
        Dung features cua ky nay de predict ky tiep theo.

    Tra ve (combined_train, predict_row, feature_cols).
    """
    per_feat = make_per_number_features(df, num)
    all_feat = pd.concat([global_feat, per_feat], axis=1)

    # Target: so `num` co xuat hien trong n1-n5 ky hien tai khong
    target = df.apply(lambda r: int(num in get_drawn_set(r)), axis=1)

    combined = pd.concat([all_feat, target.rename("y"), df["ky"]], axis=1)
    combined = combined.dropna().reset_index(drop=True)

    feature_cols = list(all_feat.columns)

    # Tach train (bo ky cuoi) va predict_row (ky cuoi)
    combined_train = combined.iloc[:-1].reset_index(drop=True)
    predict_row    = combined.iloc[[-1]]   # ky moi nhat: dung features, bo target

    return combined_train, predict_row, feature_cols


# ─────────────────────────────────────────────
# 4. XGB BUILDER
# ─────────────────────────────────────────────

def make_xgb(params: dict = None):
    defaults = dict(
        n_estimators          = 400,
        max_depth             = 2,
        learning_rate         = 0.05,
        subsample             = 0.7,
        colsample_bytree      = 0.6,
        min_child_weight      = 5,
        gamma                 = 1,
        reg_alpha             = 0.1,
        reg_lambda            = 2.0,
        scale_pos_weight      = 7,        # L535: ~5/35 ~ 14.3% positive
        objective             = "binary:logistic",
        eval_metric           = "logloss",
        random_state          = 42,
        n_jobs                = -1,
        early_stopping_rounds = 50,
    )
    if params:
        defaults.update(params)
    return XGBClassifier(**defaults)


# ─────────────────────────────────────────────
# 5. WALK-FORWARD
# ─────────────────────────────────────────────

def _walk_forward_splits(n: int, min_train: int, val_size: int):
    splits = []
    start  = min_train
    while start + val_size <= n:
        splits.append((np.arange(0, start), np.arange(start, start + val_size)))
        start += val_size
    return splits


# ─────────────────────────────────────────────
# 6. TRAIN ONE NUMBER
# ─────────────────────────────────────────────

def _make_decay_weights(n: int, decay: float = 0.995) -> np.ndarray:
    """
    Tao weight giam dan theo thoi gian (exponential decay).
    Row cuoi (moi nhat) = 1.0, row dau = decay^(n-1).
    decay=0.995: sau 100 ky weight = 0.995^100 ~ 0.60
    decay=0.990: sau 100 ky weight = 0.990^100 ~ 0.37
    """
    idx = np.arange(n)
    w   = decay ** (n - 1 - idx)   # row cuoi = decay^0 = 1.0
    return w / w.mean()             # normalize de tong khong doi


def train_one(combined: pd.DataFrame, feature_cols: list,
              val_size: int, min_train: int,
              train_window: int = 0,
              decay: float = 0.995,
              xgb_params: dict = None) -> tuple:
    """
    Train walk-forward cho 1 so.

    Tham so moi:
      train_window : chi dung N ky gan nhat de train final model
                     (0 = dung tat ca, sliding window)
      decay        : he so giam weight theo thoi gian (0.99-1.0)
                     (1.0 = tat decay, dung sample_weight binh thuong)

    Tra ve (final_model, mean_prec, mean_rec).
    """
    X = combined[feature_cols].values
    y = combined["y"].values

    splits    = _walk_forward_splits(len(X), min_train, val_size)
    prec_list, rec_list = [], []

    for tr_idx, va_idx in splits:
        X_tr, X_va = X[tr_idx], X[va_idx]
        y_tr, y_va = y[tr_idx], y[va_idx]

        if y_tr.sum() == 0:
            continue

        # Sliding window trong CV: lay toi da train_window ky cuoi cua fold
        if train_window > 0 and len(tr_idx) > train_window:
            tr_idx = tr_idx[-train_window:]
            X_tr   = X[tr_idx]
            y_tr   = y[tr_idx]

        # Time decay weight
        sw_bal   = compute_sample_weight("balanced", y_tr)
        sw_decay = _make_decay_weights(len(y_tr), decay)
        sw_tr    = sw_bal * sw_decay

        mdl = make_xgb(xgb_params)
        mdl.fit(
            X_tr, y_tr,
            sample_weight=sw_tr,
            eval_set=[(X_va, y_va)],
            verbose=False,
        )
        pred = mdl.predict(X_va)
        if pred.sum() > 0:
            prec_list.append(precision_score(y_va, pred, zero_division=0))
            rec_list.append(recall_score(y_va, pred, zero_division=0))

    # ── Final model ──
    # Sliding window: chi dung train_window ky gan nhat
    if train_window > 0 and len(X) > train_window:
        X_fin = X[-train_window:]
        y_fin = y[-train_window:]
    else:
        X_fin = X
        y_fin = y

    sw_bal_f   = compute_sample_weight("balanced", y_fin)
    sw_decay_f = _make_decay_weights(len(y_fin), decay)
    sw_fin     = sw_bal_f * sw_decay_f

    final_p = dict(xgb_params or {})
    final_p["early_stopping_rounds"] = None
    final = make_xgb(final_p)
    final.fit(X_fin, y_fin, sample_weight=sw_fin, verbose=False)

    mean_prec = float(np.mean(prec_list)) if prec_list else 0.0
    mean_rec  = float(np.mean(rec_list))  if rec_list  else 0.0
    return final, mean_prec, mean_rec


# ─────────────────────────────────────────────
# 7. TRAIN ALL 35
# ─────────────────────────────────────────────

def train_all(df: pd.DataFrame, global_feat: pd.DataFrame,
              lags: int, val_size: int, min_train: int,
              train_window: int = 0, decay: float = 0.995,
              xgb_params: dict = None,
              show_importance: bool = False) -> dict:
    """
    Train 35 model, tra ve dict: num -> {model, prec, rec, feature_cols, predict_row}.
    train_window: chi dung N ky gan nhat (0 = tat ca)
    decay       : he so giam weight theo thoi gian
    """
    results = {}
    baseline_prec = 5 * 8 / 35 / 8   # = 14.3%

    print(f"\n  Baseline precision (random 8/35): {baseline_prec:.1%}")
    if train_window > 0:
        print(f"  Sliding window : {train_window} ky gan nhat")
    print(f"  Decay weight   : {decay} (1.0 = tat)")
    print(f"  {'Num':>4}  {'CV Prec':>8}  {'CV Rec':>8}  {'Flag'}")
    print(f"  {'─'*40}")

    for num in ALL_NUMS:
        combined_train, predict_row, feature_cols = build_dataset(
            df, num, global_feat, lags
        )
        model, prec, rec = train_one(
            combined_train, feature_cols,
            val_size, min_train,
            train_window=train_window,
            decay=decay,
            xgb_params=xgb_params,
        )
        results[num] = dict(
            model        = model,
            prec         = prec,
            rec          = rec,
            feature_cols = feature_cols,
            combined     = combined_train,
            predict_row  = predict_row,
        )
        flag = "✅" if prec > baseline_prec else "⚠️ "
        print(f"  {num:>4}  {prec:>8.3f}  {rec:>8.3f}  {flag}")

    precs = [results[n]["prec"] for n in ALL_NUMS]
    recs  = [results[n]["rec"]  for n in ALL_NUMS]
    print(f"\n  Mean CV Precision : {np.mean(precs):.3f}")
    print(f"  Mean CV Recall    : {np.mean(recs):.3f}")
    print(f"  So beat baseline  : {sum(1 for p in precs if p > baseline_prec)}/35")

    if show_importance:
        _show_top_importance(results)

    return results


# ─────────────────────────────────────────────
# 8. IMPORTANCE (tong hop)
# ─────────────────────────────────────────────

def _show_top_importance(results: dict, top_n: int = 15):
    """Tong hop importance trung binh tren 35 model."""
    imp_sum = {}
    for num, r in results.items():
        mdl  = r["model"]
        cols = r["feature_cols"]
        for c, v in zip(cols, mdl.feature_importances_):
            imp_sum[c] = imp_sum.get(c, 0.0) + v

    # Trung binh
    imp_avg = {k: v / 35 for k, v in imp_sum.items()}
    top = sorted(imp_avg.items(), key=lambda x: -x[1])[:top_n]

    print(f"\n  Top {top_n} features (trung binh 35 model):")
    for f, s in top:
        bar = "█" * int(s * 600)
        print(f"    {f:<42} {s:.4f}  {bar}")


# ─────────────────────────────────────────────
# 10. ENSEMBLE + FILTER
# ─────────────────────────────────────────────

ENSEMBLE_CONFIGS = [
    {"lags": 3, "decay": 0.995, "train_window": 0,   "label": "M1(lag3,d0.995,all)"},
    {"lags": 5, "decay": 0.990, "train_window": 0,   "label": "M2(lag5,d0.990,all)"},
    {"lags": 3, "decay": 0.995, "train_window": 300, "label": "M3(lag3,d0.995,w300)"},
]

def train_ensemble(df: pd.DataFrame, val_size: int, min_train: int,
                   xgb_params: dict = None) -> list:
    """
    Train 3 bo model voi config khac nhau.
    Tra ve list cac results dict.
    """
    all_results = []
    for cfg in ENSEMBLE_CONFIGS:
        lags         = cfg["lags"]
        decay        = cfg["decay"]
        train_window = cfg["train_window"]
        label        = cfg["label"]
        print(f"\n  ── Ensemble {label} ──")

        global_feat = make_global_features(df, lags=lags)
        results = train_all(
            df, global_feat,
            lags=lags,
            val_size=val_size,
            min_train=min_train,
            train_window=train_window,
            decay=decay,
            xgb_params=xgb_params,
            show_importance=False,
        )
        all_results.append(results)
    return all_results


def predict_top_k_ensemble(all_results: list, top_k: int = 8,
                            filter_threshold: float = 0.12) -> list:
    """
    Average xac suat tu nhieu bo model.
    Filter: loai so co CV_prec trung binh < filter_threshold.
    Tra ve list (num, avg_prob, avg_cvprec) sap xep giam dan.
    """
    proba_list = []
    for num in ALL_NUMS:
        probs  = []
        cvprecs = []
        for results in all_results:
            r           = results[num]
            feat_cols   = r["feature_cols"]
            model       = r["model"]
            predict_row = r["predict_row"]
            last_X      = predict_row[feat_cols].values
            prob        = model.predict_proba(last_X)[0][1]
            probs.append(prob)
            cvprecs.append(r["prec"])

        avg_prob   = float(np.mean(probs))
        avg_cvprec = float(np.mean(cvprecs))
        proba_list.append((num, avg_prob, avg_cvprec))

    # Filter: loai so co avg_cvprec < threshold
    n_before = len(proba_list)
    if filter_threshold > 0:
        proba_list = [
            (n, p, c) for n, p, c in proba_list
            if c >= filter_threshold
        ]
    n_after = len(proba_list)
    if n_before != n_after:
        print(f"\n  🔍 Filter threshold={filter_threshold:.1%}: "
              f"loai {n_before - n_after} so, con lai {n_after} so")

    proba_list.sort(key=lambda x: -x[1])
    return proba_list

def predict_top_k(df: pd.DataFrame, global_feat: pd.DataFrame,
                  results: dict, top_k: int = 8) -> list:
    """
    Voi moi so 1-35, lay xac suat tu model dung predict_row (ky moi nhat).
    predict_row chua duoc dung de train → no-leak thuc su.
    Tra ve list (num, proba) sap xep giam dan.
    """
    proba_list = []
    for num in ALL_NUMS:
        r           = results[num]
        feat_cols   = r["feature_cols"]
        model       = r["model"]
        predict_row = r["predict_row"]

        last_X = predict_row[feat_cols].values
        prob   = model.predict_proba(last_X)[0][1]
        proba_list.append((num, prob))

    proba_list.sort(key=lambda x: -x[1])
    return proba_list


def show_prediction(df: pd.DataFrame, proba_list: list,
                    top_k: int = 8, results: dict = None):
    """
    proba_list co the la:
      - [(num, prob)]         — single model
      - [(num, prob, cvprec)] — ensemble
    """
    last_ky  = (df["ky"].iloc[-1] or "").strip()
    next_ky  = f"(sau {last_ky})"
    top      = proba_list[:top_k]

    print(f"\n{'═'*56}")
    print(f"📌 Input : ky {last_ky} (features cua ky nay)")
    print(f"🎯 Du doan {top_k} so cho ky {next_ky}:")
    print(f"{'═'*56}")

    nums_chosen = []
    for rank, item in enumerate(top, 1):
        num  = item[0]
        prob = item[1]
        cvp  = item[2] if len(item) > 2 else (
            results[num]["prec"] if results else 0.0
        )
        bar  = "█" * int(prob * 30)
        print(f"  #{rank:>2}  So {num:>2}  P={prob:.3f}  CV_prec={cvp:.3f}  {bar}")
        nums_chosen.append(num)

    print(f"\n  → Chon: {sorted(nums_chosen)}")
    n_even  = sum(1 for n in nums_chosen if n % 2 == 0)
    dec_cnt = Counter(n // 10 for n in nums_chosen)
    print(f"  → Chan/Le: {n_even}C / {top_k - n_even}L")
    print(f"  → Chuc   : " + "  ".join(
        f"{d*10}x:{dec_cnt.get(d,0)}" for d in range(4)
    ))
    print(f"{'═'*56}\n")
    return nums_chosen


# ─────────────────────────────────────────────
# 10. EVALUATE HISTORY (back-test)
# ─────────────────────────────────────────────

def evaluate_history(df: pd.DataFrame, results: dict,
                     top_k: int = 8, last_n: int = 50,
                     lags: int = 3, train_window: int = 0,
                     decay: float = 0.995, min_train: int = 200,
                     val_size: int = 20, xgb_params: dict = None):
    """
    Back-test KHONG look-ahead bias: tai moi ky i, retrain model chi
    dung data tu dau den i (khong nhin tuong lai), sau do predict ky i+1.

    Tham so them so voi phien ban cu:
      lags, train_window, decay, min_train, val_size, xgb_params
      — truyen thang vao train_one, dam bao nhat quan voi final model.

    Chi phi: cham hon phien ban cu (moi ky retrain 35 model) nhung
    metric trung thuc hon — dung cho bao cao hieu nang that su.
    Goi y: dung last_n <= 30 de kiem soat thoi gian chay.
    """
    print(f"\n📊 Back-test KHONG look-ahead bias — {last_n} ky cuoi (top {top_k}):")
    print(f"  ⚠️  Moi ky retrain 35 model → co the chay ~{last_n*2}-{last_n*5} giay")
    print(f"  {'Ky':<8}  {'Chon':^35}  {'Trung':>5}  {'Prec':>6}")
    print(f"  {'─'*65}")

    prec_list = []
    hit_list  = []

    n_total = len(df)
    # Ky bat dau backtest: dam bao du min_train ky de train
    start = max(min_train, n_total - last_n - 1)
    if start >= n_total - 1:
        print(f"  ⚠️  Khong du data de backtest (can it nhat {min_train+1} ky)")
        return

    for i in range(start, n_total - 1):
        ky_str = (df["ky"].iloc[i] or "").strip()
        actual = get_drawn_set(df.iloc[i + 1])   # ket qua ky tiep theo (chua biet luc train)

        # Subset data: chi dung tu dau den ky i (inclusive) de train
        df_sub = df.iloc[: i + 1].reset_index(drop=True)

        # Build global features cho subset
        gf_sub = make_global_features(df_sub, lags=lags)

        proba_i = []
        for num in ALL_NUMS:
            # Build dataset: combined_train = iloc[:-1], predict_row = iloc[-1]
            combined_sub, predict_row_sub, feat_cols = build_dataset(
                df_sub, num, gf_sub, lags
            )
            if len(combined_sub) < min_train:
                proba_i.append((num, 0.0))
                continue

            # Train chi tren data den ky i — khong nhin ky i+1
            model_i, _, _ = train_one(
                combined_sub, feat_cols,
                val_size=val_size,
                min_train=min_train,
                train_window=train_window,
                decay=decay,
                xgb_params=xgb_params,
            )
            # predict_row_sub la features cua ky i → predict ky i+1
            X_pred = predict_row_sub[feat_cols].values
            prob   = model_i.predict_proba(X_pred)[0][1]
            proba_i.append((num, prob))

        proba_i.sort(key=lambda x: -x[1])
        chosen = set(n for n, _ in proba_i[:top_k])
        hits   = chosen & actual
        prec   = len(hits) / top_k
        prec_list.append(prec)
        hit_list.append(len(hits))

        if (i - start) % 5 == 0 or i == n_total - 2:
            chosen_str = " ".join(f"{n:>2}" for n in sorted(chosen))
            print(
                f"  {ky_str:<8}  [{chosen_str}]"
                f"  {len(hits):>5}  {prec:>6.1%}"
            )

    if not prec_list:
        print("  ⚠️  Khong co ky nao du dieu kien backtest")
        return

    mean_prec = np.mean(prec_list)
    mean_hits = np.mean(hit_list)
    baseline  = 5 / 35   # = 14.3%
    lift      = mean_prec - baseline
    flag      = "✅" if lift > 0.02 else ("🟡" if lift > 0 else "⚠️ ")

    print(f"\n  Mean precision : {mean_prec:.3f}  (baseline={baseline:.3f})  "
          f"lift={lift:+.3f}  {flag}")
    print(f"  Mean hits/ky   : {mean_hits:.2f} / {top_k}")


# ─────────────────────────────────────────────
# 11. OPTUNA TUNING
# ─────────────────────────────────────────────

def optuna_tune(df: pd.DataFrame, global_feat: pd.DataFrame,
                lags: int, val_size: int, min_train: int,
                n_trials: int = 30) -> dict:
    """
    Tune hyperparameter tren 5 so ngau nhien (nhanh hon tune ca 35).
    """
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        print("  ⚠️  Optuna chua cai: pip install optuna")
        return None

    sample_nums = [7, 14, 21, 28, 35]   # 5 so dai dien

    def objective(trial):
        params = dict(
            max_depth        = trial.suggest_int("max_depth", 2, 4),
            learning_rate    = trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            n_estimators     = trial.suggest_int("n_estimators", 200, 600),
            subsample        = trial.suggest_float("subsample", 0.5, 0.9),
            colsample_bytree = trial.suggest_float("colsample_bytree", 0.4, 0.9),
            min_child_weight = trial.suggest_int("min_child_weight", 3, 20),
            gamma            = trial.suggest_float("gamma", 0.0, 3.0),
            reg_alpha        = trial.suggest_float("reg_alpha", 0.0, 1.0),
            reg_lambda       = trial.suggest_float("reg_lambda", 0.5, 4.0),
            early_stopping_rounds = 40,
        )
        prec_all = []
        for num in sample_nums:
            combined, feature_cols = build_dataset(df, num, global_feat, lags)
            _, prec, _ = train_one(combined, feature_cols, val_size, min_train, params)
            prec_all.append(prec)
        return np.mean(prec_all)

    print(f"\n🔍 Optuna tuning ({n_trials} trials, sample nums={sample_nums})...")
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best = study.best_params
    print(f"\n  Best mean precision : {study.best_value:.3f}")
    print("  Best params :")
    for k, v in best.items():
        print(f"    {k:<25} = {v}")
    return best


# ─────────────────────────────────────────────
# 12a. SAVE / LOAD MODEL (joblib)
# ─────────────────────────────────────────────

def _model_path(path: str, version: str) -> str:
    """
    Tra ve duong dan file .pkl.
    Neu path khong co extension, tu dong them '<version>.pkl'.
    Vi du: 'models/' -> 'models/v1.3.0.pkl'
    """
    import os
    if os.path.isdir(path) or path.endswith("/") or path.endswith(os.sep):
        os.makedirs(path, exist_ok=True)
        return os.path.join(path, f"{version}.pkl")
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    return path


def save_model(results: dict, df: pd.DataFrame, args, path: str):
    """
    Luu toan bo results (35 model + metadata) ra file .pkl bang joblib.

    Noi dung luu:
      - results  : dict num -> {model, prec, rec, feature_cols, ...}
      - input_ky : ky cuoi cua data dung de train
      - version  : VERSION string
      - args_snapshot: cac tham so train chinh (lags, decay, train_window, ...)
      - saved_at : timestamp UTC

    Khong luu 'combined' (DataFrame lon) de giam kich thuoc file.
    """
    import joblib

    fpath = _model_path(path, VERSION)

    # Loai 'combined' (DataFrame) khoi moi num truoc khi luu
    results_slim = {}
    for num, r in results.items():
        results_slim[num] = {k: v for k, v in r.items() if k != "combined"}

    payload = {
        "results"      : results_slim,
        "input_ky"     : (df["ky"].iloc[-1] or "").strip(),
        "n_rows"       : len(df),
        "version"      : VERSION,
        "saved_at"     : datetime.now(timezone.utc).isoformat(),
        "args_snapshot": {
            "lags"        : getattr(args, "lags", 3),
            "decay"       : getattr(args, "decay", 0.995),
            "train_window": getattr(args, "train_window", 0),
            "min_train"   : getattr(args, "min_train", 200),
            "val_size"    : getattr(args, "val_size", 20),
            "top_k"       : getattr(args, "top", 8),
            "ensemble"    : getattr(args, "ensemble", False),
        },
    }

    joblib.dump(payload, fpath, compress=3)
    size_mb = os.path.getsize(fpath) / 1024 / 1024
    print(f"\n💾 Da luu model → {fpath}  ({size_mb:.1f} MB)")
    print(f"   input_ky={payload['input_ky']}  version={VERSION}")


def load_model(path: str) -> tuple:
    """
    Load model tu file .pkl.

    Tra ve (results, payload) trong do:
      - results : dict num -> {model, prec, rec, feature_cols, predict_row}
                  Luu y: 'combined' KHONG co (da loai khi save).
                  evaluate_history() can retrain nen khong can 'combined'.
      - payload : toan bo dict goc (co metadata)

    Raise FileNotFoundError neu file khong ton tai.
    """
    import joblib, os

    if not os.path.exists(path):
        raise FileNotFoundError(f"Khong tim thay model file: {path}")

    payload = joblib.load(path)
    results = payload["results"]

    print(f"\n📂 Da load model ← {path}")
    print(f"   version={payload.get('version')}  "
          f"input_ky={payload.get('input_ky')}  "
          f"saved_at={payload.get('saved_at', '')[:19]}")
    snap = payload.get("args_snapshot", {})
    print(f"   lags={snap.get('lags')}  decay={snap.get('decay')}  "
          f"train_window={snap.get('train_window')}  "
          f"min_train={snap.get('min_train')}")

    precs = [results[n]["prec"] for n in ALL_NUMS if n in results]
    print(f"   Mean CV prec (35 model): {np.mean(precs):.3f}")
    return results, payload


# ─────────────────────────────────────────────
# 12. SAVE PREDICT TO DB
# ─────────────────────────────────────────────

def save_predict(args, df: pd.DataFrame, proba_list: list,
                 results: dict, cv_prec_mean: float):
    """
    Luu ket qua du doan vao table l535kqpredict.
    proba_list: list (num, prob) sap xep giam dan (tat ca 35 so).
    Chi luu top 18 (hoac it hon neu top_k < 18).
    """
    import psycopg2

    input_ky  = (df["ky"].iloc[-1] or "").strip()

    # Uoc tinh predict_ky: ky tiep theo sau input_ky (khong biet chinh xac)
    # De null, user tu dien sau khi biet ky nao duoc xo tiep
    # Tuy nhien co the uoc tinh don gian = int(input_ky) + 1
    try:
        predict_ky = str(int(input_ky) + 1).zfill(len(input_ky))
    except:
        predict_ky = None

    # Chuan bi 18 so (pad bang None neu top_k < 18)
    top18 = proba_list[:18]
    row   = {}
    for i in range(1, 19):
        if i <= len(top18):
            item = top18[i - 1]
            num  = item[0]
            prob = item[1]
            cvp  = item[2] if len(item) > 2 else (results[num]["prec"] if results else 0.0)
            row[f"n{i}"]   = int(num)
            row[f"p{i}"]   = float(prob)
            row[f"cvp{i}"] = float(cvp)
        else:
            row[f"n{i}"]   = None
            row[f"p{i}"]   = None
            row[f"cvp{i}"] = None

    sql = """
    INSERT INTO public.l535kqpredict (
        run_at, version, description,
        source, lags, train_window, decay, top_k, score_mode,
        cv_prec_mean, input_ky, predict_ky,
        n1,  p1,  cvp1,  n2,  p2,  cvp2,  n3,  p3,  cvp3,
        n4,  p4,  cvp4,  n5,  p5,  cvp5,  n6,  p6,  cvp6,
        n7,  p7,  cvp7,  n8,  p8,  cvp8,  n9,  p9,  cvp9,
        n10, p10, cvp10, n11, p11, cvp11, n12, p12, cvp12,
        n13, p13, cvp13, n14, p14, cvp14, n15, p15, cvp15,
        n16, p16, cvp16, n17, p17, cvp17, n18, p18, cvp18
    ) VALUES (
        %(run_at)s, %(version)s, %(description)s,
        %(source)s, %(lags)s, %(train_window)s, %(decay)s,
        %(top_k)s, %(score_mode)s, %(cv_prec_mean)s,
        %(input_ky)s, %(predict_ky)s,
        %(n1)s,  %(p1)s,  %(cvp1)s,  %(n2)s,  %(p2)s,  %(cvp2)s,
        %(n3)s,  %(p3)s,  %(cvp3)s,  %(n4)s,  %(p4)s,  %(cvp4)s,
        %(n5)s,  %(p5)s,  %(cvp5)s,  %(n6)s,  %(p6)s,  %(cvp6)s,
        %(n7)s,  %(p7)s,  %(cvp7)s,  %(n8)s,  %(p8)s,  %(cvp8)s,
        %(n9)s,  %(p9)s,  %(cvp9)s,  %(n10)s, %(p10)s, %(cvp10)s,
        %(n11)s, %(p11)s, %(cvp11)s, %(n12)s, %(p12)s, %(cvp12)s,
        %(n13)s, %(p13)s, %(cvp13)s, %(n14)s, %(p14)s, %(cvp14)s,
        %(n15)s, %(p15)s, %(cvp15)s, %(n16)s, %(p16)s, %(cvp16)s,
        %(n17)s, %(p17)s, %(cvp17)s, %(n18)s, %(p18)s, %(cvp18)s
    )
    ON CONFLICT (predict_ky, version, score_mode) DO UPDATE SET
        run_at       = EXCLUDED.run_at,
        description  = EXCLUDED.description,
        source       = EXCLUDED.source,
        lags         = EXCLUDED.lags,
        train_window = EXCLUDED.train_window,
        decay        = EXCLUDED.decay,
        top_k        = EXCLUDED.top_k,
        cv_prec_mean = EXCLUDED.cv_prec_mean,
        input_ky     = EXCLUDED.input_ky,
        n1=EXCLUDED.n1,   p1=EXCLUDED.p1,   cvp1=EXCLUDED.cvp1,
        n2=EXCLUDED.n2,   p2=EXCLUDED.p2,   cvp2=EXCLUDED.cvp2,
        n3=EXCLUDED.n3,   p3=EXCLUDED.p3,   cvp3=EXCLUDED.cvp3,
        n4=EXCLUDED.n4,   p4=EXCLUDED.p4,   cvp4=EXCLUDED.cvp4,
        n5=EXCLUDED.n5,   p5=EXCLUDED.p5,   cvp5=EXCLUDED.cvp5,
        n6=EXCLUDED.n6,   p6=EXCLUDED.p6,   cvp6=EXCLUDED.cvp6,
        n7=EXCLUDED.n7,   p7=EXCLUDED.p7,   cvp7=EXCLUDED.cvp7,
        n8=EXCLUDED.n8,   p8=EXCLUDED.p8,   cvp8=EXCLUDED.cvp8,
        n9=EXCLUDED.n9,   p9=EXCLUDED.p9,   cvp9=EXCLUDED.cvp9,
        n10=EXCLUDED.n10, p10=EXCLUDED.p10, cvp10=EXCLUDED.cvp10,
        n11=EXCLUDED.n11, p11=EXCLUDED.p11, cvp11=EXCLUDED.cvp11,
        n12=EXCLUDED.n12, p12=EXCLUDED.p12, cvp12=EXCLUDED.cvp12,
        n13=EXCLUDED.n13, p13=EXCLUDED.p13, cvp13=EXCLUDED.cvp13,
        n14=EXCLUDED.n14, p14=EXCLUDED.p14, cvp14=EXCLUDED.cvp14,
        n15=EXCLUDED.n15, p15=EXCLUDED.p15, cvp15=EXCLUDED.cvp15,
        n16=EXCLUDED.n16, p16=EXCLUDED.p16, cvp16=EXCLUDED.cvp16,
        n17=EXCLUDED.n17, p17=EXCLUDED.p17, cvp17=EXCLUDED.cvp17,
        n18=EXCLUDED.n18, p18=EXCLUDED.p18, cvp18=EXCLUDED.cvp18;
    """

    params = {
        "run_at"      : datetime.now(timezone.utc),
        "version"     : VERSION,
        "description" : DESCRIPTION,
        "source"      : getattr(args, "source", "db"),
        "lags"        : getattr(args, "lags", 3),
        "train_window": getattr(args, "train_window", 0),
        "decay"       : getattr(args, "decay", 0.995),
        "top_k"       : getattr(args, "top", 8),
        "score_mode"  : "ensemble" if getattr(args, "ensemble", False) else "pure_p",
        "cv_prec_mean": float(cv_prec_mean),
        "input_ky"    : input_ky,
        "predict_ky"  : predict_ky,
        **row,
    }

    conn = psycopg2.connect(DATABASE_URL)
    conn.autocommit = False
    cur  = conn.cursor()
    try:
        cur.execute(sql, params)
        conn.commit()
        print(f"\n💾 Da luu du doan vao l535kqpredict"
              f"  (input_ky={input_ky}, predict_ky={predict_ky},"
              f"  version={VERSION})")
    except Exception as e:
        conn.rollback()
        print(f"\n⚠️  Luu that bai: {e}")
    finally:
        cur.close()
        conn.close()


# ─────────────────────────────────────────────
# 13. UPDATE RESULT FROM l535kqdetail
# ─────────────────────────────────────────────

def update_result():
    """
    Tim cac row trong l535kqpredict co actual_n1 IS NULL,
    query l535kqdetail theo predict_ky de lay ket qua thuc te,
    tinh hit_cnt, precision_val va update.
    """
    import psycopg2
    import psycopg2.extras

    conn = psycopg2.connect(DATABASE_URL)
    conn.autocommit = False
    cur  = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

    try:
        # 1. Lay cac row chua co ket qua
        cur.execute("""
            SELECT id, predict_ky, top_k,
                   n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,
                   n11,n12,n13,n14,n15,n16,n17,n18
            FROM public.l535kqpredict
            WHERE actual_n1 IS NULL AND predict_ky IS NOT NULL
            ORDER BY predict_ky
        """)
        pending = cur.fetchall()
        print(f"📋 Tim thay {len(pending)} ky chua co ket qua thuc te")

        updated = 0
        for row in pending:
            predict_ky = (row["predict_ky"] or "").strip()

            # 2. Lay ket qua thuc te tu l535kqdetail
            cur.execute("""
                SELECT n1,n2,n3,n4,n5,n6
                FROM public.l535kqdetail
                WHERE ky = %s
                LIMIT 1
            """, (predict_ky,))
            actual = cur.fetchone()

            if actual is None:
                print(f"  ky {predict_ky}: chua co ket qua trong l535kqdetail, bo qua")
                continue

            # 3. Lay tap so thuc te n1-n5
            actual_nums = {
                int(actual[f"n{i}"])
                for i in range(1, 6)
                if actual[f"n{i}"] is not None and actual[f"n{i}"] != 0
            }
            actual_n6 = actual["n6"]

            # 4. Lay tap so du doan (top_k so dau)
            top_k    = int(row["top_k"] or 8)
            pred_nums = []
            for i in range(1, 19):
                n = row[f"n{i}"]
                if n is not None and len(pred_nums) < top_k:
                    pred_nums.append(int(n))

            # 5. Tinh hit
            hits      = set(pred_nums) & actual_nums
            hit_cnt   = len(hits)
            prec_val  = hit_cnt / top_k if top_k > 0 else 0.0

            # 6. Update
            cur.execute("""
                UPDATE public.l535kqpredict SET
                    actual_n1     = %s,
                    actual_n2     = %s,
                    actual_n3     = %s,
                    actual_n4     = %s,
                    actual_n5     = %s,
                    actual_n6     = %s,
                    hit_cnt       = %s,
                    precision_val = %s,
                    updated_at    = %s
                WHERE id = %s
            """, (
                actual["n1"], actual["n2"], actual["n3"],
                actual["n4"], actual["n5"], actual_n6,
                hit_cnt, prec_val,
                datetime.now(timezone.utc),
                row["id"],
            ))

            flag = "✅" if prec_val > 0.143 else ("🟡" if prec_val > 0 else "⚠️ ")
            print(
                f"  ky {predict_ky}: "
                f"du doan={sorted(pred_nums[:top_k])}  "
                f"thuc te={sorted(actual_nums)}  "
                f"trung={sorted(hits)}  "
                f"hit={hit_cnt}/{top_k}  prec={prec_val:.1%}  {flag}"
            )
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
# 14. MAIN
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="XGBoost predict 35 so L535")
    parser.add_argument("--update_result", action="store_true",
                        help="Cap nhat ket qua thuc te tu l535kqdetail")
    parser.add_argument("--source",      choices=["db", "csv"], default="db")
    parser.add_argument("--file",        default="l535kqdetail_dat.csv")
    parser.add_argument("--n_ky",        type=int, default=0)
    parser.add_argument("--qh_ky",       type=int, default=20)
    parser.add_argument("--lags",        type=int, default=3)
    parser.add_argument("--top",         type=int, default=8,
                        help="So luong so chon (mac dinh: 8)")
    parser.add_argument("--val_size",    type=int, default=20,
                        help="So ky moi fold walk-forward (mac dinh: 20)")
    parser.add_argument("--min_train",   type=int, default=200,
                        help="So ky toi thieu de bat dau train (mac dinh: 200)")
    parser.add_argument("--importance",  action="store_true",
                        help="In feature importance trung binh 35 model")
    parser.add_argument("--backtest",    type=int, default=0,
                        help="Back-test N ky cuoi (0 = tat)")
    parser.add_argument("--train_window", type=int,   default=0,
                        help="Sliding window: chi dung N ky gan nhat (0 = tat ca)")
    parser.add_argument("--decay",        type=float, default=0.995,
                        help="Time decay weight (0.99-1.0, 1.0 = tat decay)")
    parser.add_argument("--save",         action="store_true",
                        help="Luu ket qua du doan vao l535kqpredict")
    parser.add_argument("--ensemble",    action="store_true",
                        help="Dung ensemble 3 model (M1/M2/M3) thay vi 1 model")
    parser.add_argument("--filter",      type=float, default=0.12,
                        help="Filter threshold CV_prec (mac dinh 0.12, 0=tat filter)")
    parser.add_argument("--tune",        action="store_true")
    parser.add_argument("--tune_trials", type=int, default=30)
    parser.add_argument("--save_model",  default="",
                        help="Luu 35 model ra file .pkl sau khi train. "
                             "Vi du: --save_model models/  hoac --save_model my.pkl")
    parser.add_argument("--load_model",  default="",
                        help="Load model da train tu file .pkl, bo qua buoc train. "
                             "Vi du: --load_model models/v1.3.0.pkl")
    args = parser.parse_args()

    # ── Mode: update result ──
    if args.update_result:
        print(f"🔄 Update ket qua thuc te tu l535kqdetail...")
        update_result()
        return

    # ── Mode: load model (bo qua train) ──
    if args.load_model:
        try:
            results, payload = load_model(args.load_model)
            snap = payload.get("args_snapshot", {})
            # Lay lags tu snapshot de build global_feat dung
            load_lags = snap.get("lags", args.lags)
            global_feat = make_global_features(df, lags=load_lags)

            # Rebuild predict_row cho tung so neu chua co
            # (predict_row duoc luu san trong results_slim)
            print(f"\n🔮 Predicting top {args.top} (tu model da load)...")
            proba_list = predict_top_k(df, global_feat, results, top_k=args.top)

            if args.filter > 0:
                before = len(proba_list)
                proba_list = [
                    (n, p) for n, p in proba_list
                    if results[n]["prec"] >= args.filter
                ]
                if len(proba_list) < before:
                    print(f"  🔍 Filter {args.filter:.0%}: loai {before - len(proba_list)} so")

            show_prediction(df, proba_list, top_k=args.top, results=results)
            cv_prec_mean = np.mean([results[n]["prec"] for n in ALL_NUMS])

            if args.save:
                save_predict(args, df, proba_list, results, cv_prec_mean)
            return
        except FileNotFoundError as e:
            print(f"  ⚠️  {e} — se train moi thay the.")

    # ── Mode: train + predict ──
    print(f"📌 Version : {VERSION}")
    print(f"📝 Desc    : {DESCRIPTION[:80]}...")
    print("📥 Loading data...")
    df = (
        load_from_csv(args.file)
        if args.source == "csv"
        else load_from_db(args.n_ky, args.qh_ky)
    )
    print(f"   {len(df)} ky")

    has_mega  = "mgn1" in df.columns
    has_power = "pwn1" in df.columns
    has_jp    = "jpck" in df.columns
    has_qhl   = "qhl"  in df.columns
    print(f"   Mega  : {'✅' if has_mega  else '⚠️'}")
    print(f"   Power : {'✅' if has_power else '⚠️'}")
    print(f"   JP    : {'✅' if has_jp    else '⚠️'}")
    print(f"   QHL   : {'✅' if has_qhl   else '⚠️'}")

    print("\n⚙️  Global feature engineering...")
    if not args.ensemble:
        global_feat = make_global_features(df, lags=args.lags)
        print(f"   {global_feat.shape[1]} global features")

    # Optuna tuning
    xgb_params = None
    if args.tune:
        global_feat = make_global_features(df, lags=args.lags)
        xgb_params = optuna_tune(
            df, global_feat,
            lags=args.lags,
            val_size=args.val_size,
            min_train=args.min_train,
            n_trials=args.tune_trials,
        )

    if args.ensemble:
        # ── Ensemble mode ──
        print(f"\n🚀 Training ensemble ({len(ENSEMBLE_CONFIGS)} models)...")
        all_results = train_ensemble(
            df,
            val_size=args.val_size,
            min_train=args.min_train,
            xgb_params=xgb_params,
        )

        if args.backtest > 0:
            cfg0 = ENSEMBLE_CONFIGS[0]
            evaluate_history(
                df, all_results[0], top_k=args.top, last_n=args.backtest,
                lags=cfg0["lags"], train_window=cfg0["train_window"],
                decay=cfg0["decay"], min_train=args.min_train,
                val_size=args.val_size, xgb_params=xgb_params,
            )

        print(f"\n🔮 Predicting top {args.top} (ensemble + filter={args.filter:.0%})...")
        proba_list   = predict_top_k_ensemble(
            all_results, top_k=args.top, filter_threshold=args.filter
        )
        show_prediction(df, proba_list, top_k=args.top)

        # CV prec mean tu model dau tien
        cv_prec_mean = np.mean([all_results[0][n]["prec"] for n in ALL_NUMS])
        results      = all_results[0]   # dung cho save

        if args.save_model:
            save_model(results, df, args, args.save_model)

    else:
        # ── Single model mode ──
        global_feat = make_global_features(df, lags=args.lags)
        print(f"   {global_feat.shape[1]} global features")

        print(f"\n🚀 Training 35 model (top={args.top})...")
        results = train_all(
            df, global_feat,
            lags=args.lags,
            val_size=args.val_size,
            min_train=args.min_train,
            train_window=args.train_window,
            decay=args.decay,
            xgb_params=xgb_params,
            show_importance=args.importance,
        )

        if args.backtest > 0:
            evaluate_history(
                df, results, top_k=args.top, last_n=args.backtest,
                lags=args.lags, train_window=args.train_window,
                decay=args.decay, min_train=args.min_train,
                val_size=args.val_size, xgb_params=xgb_params,
            )

        print(f"\n🔮 Predicting top {args.top}...")
        proba_list = predict_top_k(df, global_feat, results, top_k=args.top)

        # Apply filter neu co
        if args.filter > 0:
            before = len(proba_list)
            proba_list = [
                (n, p) for n, p in proba_list
                if results[n]["prec"] >= args.filter
            ]
            if len(proba_list) < before:
                print(f"  🔍 Filter {args.filter:.0%}: loai {before - len(proba_list)} so")

        show_prediction(df, proba_list, top_k=args.top, results=results)
        cv_prec_mean = np.mean([results[n]["prec"] for n in ALL_NUMS])

        if args.save_model:
            save_model(results, df, args, args.save_model)

    # Save to DB
    if args.save:
        # Rebuild proba_list dang (num, prob) neu can
        save_predict(args, df, proba_list, results, cv_prec_mean)


if __name__ == "__main__":
    main()