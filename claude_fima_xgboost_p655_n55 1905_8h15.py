"""
claude_fima_xgboost_p655_n55.py
────────────────────────────────
Du doan P655:
  - 55 model nhi phan cho N1-N6 (6 so chinh, range 1-55) → chon top 8

Kien truc:
  - 1 XGBoost binary classifier / so (55 model)
  - Walk-forward CV voi precision metric
  - No data leak: tach train (ky 1→N-1) va predict_row (ky N)
  - Sliding window + Time decay weight
  - Ensemble 3 configs (M1/M2/M3)
  - Filter threshold CV_prec
  - Save predict → p655kqpredict
  - Update result → tu dong lay tu p655kqdetail

Cross-ref: chi dung Mega M645 (bo Lotto do it overlap).

Usage:
    python claude_fima_xgboost_p655_n55.py --source db --lags 3 --top 8
    python claude_fima_xgboost_p655_n55.py --source db --top 8 --ensemble --filter 0.10
    python claude_fima_xgboost_p655_n55.py --source db --top 8 --ensemble --filter 0.10 --save
    python claude_fima_xgboost_p655_n55.py --update_result
"""

import argparse, os, warnings
import numpy as np
import pandas as pd
from collections import Counter
from datetime import datetime, timezone

from xgboost import XGBClassifier
from sklearn.metrics import precision_score, recall_score
from sklearn.utils.class_weight import compute_sample_weight

warnings.filterwarnings("ignore")

DATABASE_URL = os.environ.get(
    "DATABASE_URL", "postgresql://postgres:bin@localhost:5432/katalott"
)

ALL_NUMS = list(range(1, 56))   # P655: so chinh 1-55

# ── Version ───────────────────────────────────────────────────────────────────
VERSION     = "v1.1.0"
DESCRIPTION = (
    "55 binary XGBoost (n1-n6), no-leak train/predict split, "
    "sliding window + time decay, Mega cross features, "
    "walk-forward CV precision, ensemble 3 configs, save to p655kqpredict"
)

# ─────────────────────────────────────────────
# 1. LOAD
# ─────────────────────────────────────────────

def load_from_db(n_ky=0, qh_ky=20) -> pd.DataFrame:
    import psycopg2
    limit = f"LIMIT {n_ky}" if n_ky > 0 else ""
    sql = f"""
        SELECT * FROM public.p655kqdetail
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

THU_MAP = {"T2":2,"T3":3,"T4":4,"T5":5,"T6":6,"T7":7,"CN":8}
DD_MAP  = {"CC":0,"CL":1,"LC":2,"LL":3}

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
    """Lay tap hop n1-n6 (so chinh) tu mot row P655."""
    return {
        int(row[c]) for c in ["n1","n2","n3","n4","n5","n6"]
        if pd.notna(row[c]) and row[c] != 0
    }


# ─────────────────────────────────────────────
# 3. GLOBAL FEATURE ENGINEERING
# ─────────────────────────────────────────────

def make_global_features(df: pd.DataFrame, lags: int = 3) -> pd.DataFrame:
    """
    Tinh cac feature GLOBAL cua P655.
    Tat ca shift >= 1. Cross-ref: chi Mega M645.
    cl6_l_count da bo (redundant voi sc).
    """
    d = df.copy()

    # ── Encode DD (N1 va N6) ──
    d["dd_enc"] = d["dd"].apply(
        lambda v: {"CC":0,"CL":1,"LC":2,"LL":3}.get(str(v).strip(), -1)
    )

    # ── CL6 ──
    s = d["cl6"].astype(str).str.strip()
    for i in range(6):
        d[f"cl6_b{i}"] = s.str[i].map({"C":0,"L":1}).fillna(-1).astype(int)
    d["cl6_pattern"] = s.apply(
        lambda v: int("".join("0" if c=="C" else "1" for c in v), 2)
        if len(v)==6 else np.nan
    )
    d["cl6_changed"] = (d["cl6_pattern"] != d["cl6_pattern"].shift(1)).astype(float)

    # ── Thu ──
    d["thu_enc"] = _encode_thu(d["thu"])

    # ── JP ──
    if "jpck" in d.columns:
        d["jpck"] = _safe_int(d["jpck"]).fillna(0)
    if "jp1_cnt" in d.columns:
        d["jp1_enc"] = (_safe_int(d["jp1_cnt"]) > 0).astype(int)

    # ── Mega features ──
    if all(c in d.columns for c in ["mgn1","mgn6"]):
        mn = [_safe_int(d[f"mgn{i}"]) for i in range(1,7)]
        if "mgthu" in d.columns:
            d["mg_thu_enc"]   = _encode_thu(d["mgthu"])
        if "mgngay" in d.columns and "ngay" in d.columns:
            d["mg_ngay_diff"] = _ngay_diff(d["ngay"], d["mgngay"])
        d["mg_dd"] = _encode_dd(mn[0], mn[5])
        d["mg_sc"] = sum((n % 2 == 0).astype(float) for n in mn)

    feat = pd.DataFrame(index=d.index)

    # A. Biet truoc — no shift
    feat["thu_enc"] = d["thu_enc"]

    # B. Ky truoc (shift 1)
    prev_cols = [
        "dd_enc","sc","sum","ke",
        "cl6_b0","cl6_b1","cl6_b2","cl6_b3","cl6_b4","cl6_b5",
        "cl6_pattern","cl6_changed",
        "dec0","dec1","dec2","dec3","dec4","dec5",
    ]
    for c in ["jpck","jp1_enc"]:
        if c in d.columns: prev_cols.append(c)
    prev_cols = [c for c in prev_cols if c in d.columns]
    for col in prev_cols:
        feat[f"{col}_prev"] = d[col].shift(1)

    # C. Lag 2 → lags
    lag_cols = ["dd_enc","sc","cl6_pattern"]
    lag_cols = [c for c in lag_cols if c in d.columns]
    for lag in range(2, lags+1):
        for col in lag_cols:
            feat[f"{col}_lag{lag}"] = d[col].shift(lag)

    # D. Rolling (window 3, 5) — chi dung sc
    for w in [3, 5]:
        base = d["sc"].shift(1)
        feat[f"sc_roll{w}_mean"] = base.rolling(w).mean()
        feat[f"sc_roll{w}_std"]  = base.rolling(w).std()

    # E. Mega (no shift)
    for c in ["mg_thu_enc","mg_ngay_diff","mg_dd","mg_sc"]:
        if c in d.columns:
            feat[c] = d[c]

    # F. Cross Mega x P655
    if "mg_dd" in feat.columns and "dd_enc_prev" in feat.columns:
        feat["cross_mg_dd_match"] = (
            feat["mg_dd"] == feat["dd_enc_prev"]
        ).astype(float)
    if "mg_sc" in feat.columns and "sc_prev" in feat.columns:
        feat["cross_sc_mg"] = feat["sc_prev"] * feat["mg_sc"]

    return feat


# ─────────────────────────────────────────────
# 4. PER-NUMBER FEATURES
# ─────────────────────────────────────────────

def make_per_number_features(df: pd.DataFrame, num: int) -> pd.DataFrame:
    """Tinh features rieng cho so `num` trong N1-N6 (range 1-55)."""
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

    # n_streak
    streak = []
    cur_streak = 0
    for i, h in enumerate(hit):
        if i == 0:
            streak.append(np.nan)
        else:
            streak.append(float(cur_streak))
        cur_streak = 0 if h == 1 else cur_streak + 1
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

    # n7_last_lag: cach bao nhieu ky thi `num` xuat hien o vi tri N7
    n7_hit = (df["n7"] == num).astype(int)
    n7_last = []
    last_n7 = np.nan
    for i, h in enumerate(n7_hit):
        if i == 0:
            n7_last.append(np.nan)
        else:
            n7_last.append(np.nan if pd.isna(last_n7) else i - last_n7)
        if h == 1:
            last_n7 = i
    feat["n7_last_lag"] = pd.Series(n7_last, index=df.index).shift(1)

    # n_pair_freq: tan suat so `num` xuat hien cung cac so ky truoc (30 ky)
    prev_nums = df.apply(get_drawn_set, axis=1).shift(1)
    pair_vals = []
    for i in range(len(df)):
        if i < 31:
            pair_vals.append(np.nan)
            continue
        pn = prev_nums.iloc[i]
        if not pn:
            pair_vals.append(0.0)
            continue
        count = sum(
            1 for j in range(i-30, i)
            if num in get_drawn_set(df.iloc[j])
            and len(get_drawn_set(df.iloc[j]) & pn) > 0
        )
        pair_vals.append(count / 30.0)
    feat["n_pair_freq"] = pd.Series(pair_vals, index=df.index)

    # dot_prev
    if "dot" in df.columns:
        feat["dot_prev"] = _safe_int(df["dot"]).shift(1)

    # Static
    feat["n_is_even"] = float(num % 2 == 0)
    feat["n_dec"]     = float(num // 10)

    return feat


# ─────────────────────────────────────────────
# 5. BUILD DATASET
# ─────────────────────────────────────────────

def build_dataset(df: pd.DataFrame, num: int,
                  global_feat: pd.DataFrame) -> tuple:
    """
    Tra ve (combined_train, predict_row, feature_cols).
    No-leak: combined_train = ky 0→N-2, predict_row = ky N-1.
    """
    per_feat = make_per_number_features(df, num)
    all_feat = pd.concat([global_feat, per_feat], axis=1)
    all_feat = all_feat.fillna(-1)

    target = df.apply(lambda r: int(num in get_drawn_set(r)), axis=1)

    combined = pd.concat(
        [all_feat, target.rename("y"), df["ky"]], axis=1
    ).dropna(subset=["n_freq_50"]).reset_index(drop=True)

    feature_cols   = list(all_feat.columns)
    combined_train = combined.iloc[:-1].reset_index(drop=True)
    predict_row    = combined.iloc[[-1]]

    return combined_train, predict_row, feature_cols


# ─────────────────────────────────────────────
# 6. XGB BUILDER
# ─────────────────────────────────────────────

def make_xgb(params: dict = None):
    defaults = dict(
        n_estimators          = 300,
        max_depth             = 3,
        learning_rate         = 0.05,
        subsample             = 0.8,
        colsample_bytree      = 0.7,
        min_child_weight      = 3,
        gamma                 = 0.1,
        reg_alpha             = 0.05,
        reg_lambda            = 1.0,
        scale_pos_weight      = 8,
        objective             = "binary:logistic",
        eval_metric           = "logloss",
        random_state          = 42,
        n_jobs                = -1,
        early_stopping_rounds = 30,
    )
    if params:
        defaults.update(params)
    return XGBClassifier(**defaults)


# ─────────────────────────────────────────────
# 7. WALK-FORWARD
# ─────────────────────────────────────────────

def _walk_forward_splits(n: int, min_train: int, val_size: int):
    splits = []
    start  = min_train
    while start + val_size <= n:
        splits.append((np.arange(0, start), np.arange(start, start+val_size)))
        start += val_size
    return splits

def _make_decay_weights(n: int, decay: float = 0.995) -> np.ndarray:
    idx = np.arange(n)
    w   = decay ** (n - 1 - idx)
    return w / w.mean()


# ─────────────────────────────────────────────
# 8. TRAIN ONE NUMBER
# ─────────────────────────────────────────────

def train_one(combined: pd.DataFrame, feature_cols: list,
              val_size: int, min_train: int,
              train_window: int = 0, decay: float = 0.995,
              xgb_params: dict = None) -> tuple:
    X = combined[feature_cols].values
    y = combined["y"].values

    splits    = _walk_forward_splits(len(X), min_train, val_size)
    prec_list, rec_list = [], []

    for tr_idx, va_idx in splits:
        X_tr, X_va = X[tr_idx], X[va_idx]
        y_tr, y_va = y[tr_idx], y[va_idx]
        if y_tr.sum() == 0: continue

        if train_window > 0 and len(tr_idx) > train_window:
            tr_idx = tr_idx[-train_window:]
            X_tr   = X[tr_idx]
            y_tr   = y[tr_idx]

        sw_bal   = compute_sample_weight("balanced", y_tr)
        sw_decay = _make_decay_weights(len(y_tr), decay)
        sw_tr    = sw_bal * sw_decay

        mdl = make_xgb(xgb_params)
        mdl.fit(X_tr, y_tr, sample_weight=sw_tr,
                eval_set=[(X_va, y_va)], verbose=False)
        pred = mdl.predict(X_va)
        prec_list.append(precision_score(y_va, pred, zero_division=0))
        rec_list.append(recall_score(y_va, pred, zero_division=0))

    # Final model
    if train_window > 0 and len(X) > train_window:
        X_fin, y_fin = X[-train_window:], y[-train_window:]
    else:
        X_fin, y_fin = X, y

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
# 9. TRAIN ALL (55 model)
# ─────────────────────────────────────────────

def train_all(df: pd.DataFrame, global_feat: pd.DataFrame,
              lags: int, val_size: int, min_train: int,
              train_window: int = 0, decay: float = 0.995,
              xgb_params: dict = None,
              show_importance: bool = False) -> dict:
    """Train 55 model cho so chinh N1-N6."""
    results  = {}
    baseline = 6 * 8 / 55 / 8   # = 10.9%

    print(f"\n  [N1-N6] Baseline precision (random 8/55): {baseline:.1%}")
    print(f"  {'Num':>4}  {'CV Prec':>8}  {'CV Rec':>8}  {'Flag'}")
    print(f"  {'─'*40}")

    for num in ALL_NUMS:
        combined_train, predict_row, feature_cols = build_dataset(
            df, num, global_feat
        )
        model, prec, rec = train_one(
            combined_train, feature_cols,
            val_size, min_train, train_window, decay, xgb_params
        )
        results[num] = dict(
            model=model, prec=prec, rec=rec,
            feature_cols=feature_cols,
            combined=combined_train,
            predict_row=predict_row,
        )
        flag = "✅" if prec > baseline else "⚠️ "
        print(f"  {num:>4}  {prec:>8.3f}  {rec:>8.3f}  {flag}")

    precs = [results[n]["prec"] for n in ALL_NUMS]
    print(f"\n  Mean CV Precision: {np.mean(precs):.3f}")
    print(f"  So beat baseline : {sum(1 for p in precs if p > baseline)}/55")

    if show_importance:
        _show_top_importance(results)

    return results


# ─────────────────────────────────────────────
# 10. IMPORTANCE
# ─────────────────────────────────────────────

def _show_top_importance(results: dict, top_n: int = 15):
    imp_sum = {}
    for num in ALL_NUMS:
        r = results[num]
        for c, v in zip(r["feature_cols"], r["model"].feature_importances_):
            imp_sum[c] = imp_sum.get(c, 0.0) + v
    imp_avg = {k: v/55 for k, v in imp_sum.items()}
    top = sorted(imp_avg.items(), key=lambda x: -x[1])[:top_n]
    print(f"\n  Top {top_n} features (trung binh 55 model):")
    for f, s in top:
        print(f"    {f:<42} {s:.4f}  {'█'*int(s*500)}")

    mg_imp = sum(v for k,v in imp_avg.items() if k.startswith("mg_"))
    cx_imp = sum(v for k,v in imp_avg.items() if k.startswith("cross_"))
    jp_imp = sum(v for k,v in imp_avg.items() if "jp" in k)
    print(f"\n  Importance theo nhom:")
    print(f"    Mega  : {mg_imp:.4f}")
    print(f"    Cross : {cx_imp:.4f}")
    print(f"    JP    : {jp_imp:.4f}")
    print(f"    P655  : {1-mg_imp-cx_imp-jp_imp:.4f}")


# ─────────────────────────────────────────────
# 11. ENSEMBLE
# ─────────────────────────────────────────────

ENSEMBLE_CONFIGS = [
    {"lags":3, "decay":0.995, "train_window":0,   "label":"M1(lag3,d0.995,all)"},
    {"lags":5, "decay":0.990, "train_window":0,   "label":"M2(lag5,d0.990,all)"},
    {"lags":3, "decay":0.995, "train_window":500, "label":"M3(lag3,d0.995,w500)"},
]

def train_ensemble(df: pd.DataFrame, val_size: int, min_train: int,
                   xgb_params: dict = None) -> list:
    all_results = []
    for cfg in ENSEMBLE_CONFIGS:
        print(f"\n  ── Ensemble {cfg['label']} ──")
        gf = make_global_features(df, lags=cfg["lags"])
        r  = train_all(
            df, gf,
            lags=cfg["lags"], val_size=val_size, min_train=min_train,
            train_window=cfg["train_window"], decay=cfg["decay"],
            xgb_params=xgb_params, show_importance=False,
        )
        all_results.append(r)
    return all_results


def predict_top_k_ensemble(all_results: list,
                            top_k: int = 8,
                            filter_threshold: float = 0.10) -> list:
    """Average xac suat, filter, chon top_k cho N16."""
    proba = []
    for num in ALL_NUMS:
        probs, cvps = [], []
        for r in all_results:
            res    = r[num]
            last_X = res["predict_row"][res["feature_cols"]].values
            prob   = res["model"].predict_proba(last_X)[0][1]
            probs.append(prob)
            cvps.append(res["prec"])
        proba.append((num, float(np.mean(probs)), float(np.mean(cvps))))

    if filter_threshold > 0:
        before = len(proba)
        proba  = [(n,p,c) for n,p,c in proba if c >= filter_threshold]
        removed = before - len(proba)
        if removed > 0:
            print(f"\n  🔍 Filter {filter_threshold:.0%}: loai {removed} so, con {len(proba)} so")

    proba.sort(key=lambda x: -x[1])
    return proba


def predict_top_k_single(results: dict, top_k: int = 8,
                          filter_threshold: float = 0.10) -> list:
    """Single model predict."""
    proba = []
    for num in ALL_NUMS:
        r      = results[num]
        last_X = r["predict_row"][r["feature_cols"]].values
        prob   = r["model"].predict_proba(last_X)[0][1]
        proba.append((num, prob, r["prec"]))

    if filter_threshold > 0:
        before  = len(proba)
        proba   = [(n,p,c) for n,p,c in proba if c >= filter_threshold]
        removed = before - len(proba)
        if removed > 0:
            print(f"\n  🔍 Filter {filter_threshold:.0%}: loai {removed} so, con {len(proba)} so")

    proba.sort(key=lambda x: -x[1])
    return proba


# ─────────────────────────────────────────────
# 12. SHOW PREDICTION
# ─────────────────────────────────────────────

def show_prediction(df: pd.DataFrame, proba: list, top_k: int = 8):
    last_ky = (df["ky"].iloc[-1] or "").strip()
    top     = proba[:top_k]

    print(f"\n{'═'*58}")
    print(f"📌 Input : ky {last_ky} (features cua ky nay)")
    print(f"🎯 Du doan cho ky (sau {last_ky}):")
    print(f"{'═'*58}")
    print(f"\n  📌 6 so chinh (N1-N6) — Top {top_k}:")

    nums = []
    for rank, (num, prob, cvp) in enumerate(top, 1):
        bar = "█" * int(prob * 25)
        print(f"    #{rank:>2}  So {num:>2}  P={prob:.3f}  CV_prec={cvp:.3f}  {bar}")
        nums.append(num)

    n_even  = sum(1 for n in nums if n % 2 == 0)
    dec_cnt = Counter(n // 10 for n in nums)
    print(f"\n  → Chon    : {sorted(nums)}")
    print(f"  → Chan/Le : {n_even}C / {top_k - n_even}L")
    print(f"  → Chuc    : " + "  ".join(
        f"{d*10}x:{dec_cnt.get(d,0)}" for d in range(6) if dec_cnt.get(d,0) > 0
    ))
    print(f"{'═'*58}\n")
    return nums


# ─────────────────────────────────────────────
# 13. SAVE PREDICT
# ─────────────────────────────────────────────

def save_predict(args, df: pd.DataFrame,
                 proba: list, cv_prec: float):
    import psycopg2

    input_ky = (df["ky"].iloc[-1] or "").strip()
    try:
        predict_ky = str(int(input_ky) + 1).zfill(len(input_ky))
    except:
        predict_ky = None

    score_mode = "ensemble" if getattr(args, "ensemble", False) else "pure_p"

    # Top 18 so chinh
    row = {}
    for i in range(1, 19):
        if i <= len(proba):
            num, prob, cvp = proba[i-1]
            row[f"n{i}"]   = int(num)
            row[f"p{i}"]   = float(prob)
            row[f"cvp{i}"] = float(cvp)
        else:
            row[f"n{i}"] = row[f"p{i}"] = row[f"cvp{i}"] = None

    sql = """
    INSERT INTO public.p655kqpredict (
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
        "source": getattr(args,"source","db"),
        "lags": getattr(args,"lags",3),
        "train_window": getattr(args,"train_window",0),
        "decay": getattr(args,"decay",0.995),
        "top_k": getattr(args,"top",8),
        "score_mode": score_mode,
        "cv_prec_n16": float(cv_prec),
        "input_ky": input_ky,
        "predict_ky": predict_ky,
        **row,
    }

    conn = psycopg2.connect(DATABASE_URL)
    conn.autocommit = False
    cur  = conn.cursor()
    try:
        cur.execute(sql, params)
        conn.commit()
        print(f"\n💾 Da luu du doan vao p655kqpredict"
              f"  (input_ky={input_ky}, predict_ky={predict_ky},"
              f"  version={VERSION}, mode={score_mode})")
    except Exception as e:
        conn.rollback()
        print(f"\n⚠️  Luu that bai: {e}")
    finally:
        cur.close()
        conn.close()


# ─────────────────────────────────────────────
# 14. UPDATE RESULT
# ─────────────────────────────────────────────

def update_result():
    import psycopg2, psycopg2.extras

    conn = psycopg2.connect(DATABASE_URL)
    conn.autocommit = False
    cur  = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

    try:
        cur.execute("""
            SELECT id, predict_ky, top_k,
                   n1,n2,n3,n4,n5,n6,n7,n8
            FROM public.p655kqpredict
            WHERE actual_n1 IS NULL AND predict_ky IS NOT NULL
            ORDER BY predict_ky
        """)
        pending = cur.fetchall()
        print(f"📋 Tim thay {len(pending)} ky chua co ket qua thuc te")

        updated = 0
        for row in pending:
            predict_ky = (row["predict_ky"] or "").strip()

            cur.execute("""
                SELECT n1,n2,n3,n4,n5,n6
                FROM public.p655kqdetail
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

            top_k = int(row["top_k"] or 8)
            pred_nums = []
            for i in range(1, 19):
                n = row.get(f"n{i}")
                if n is not None and len(pred_nums) < top_k:
                    pred_nums.append(int(n))

            hits     = set(pred_nums) & actual_nums
            hit_cnt  = len(hits)
            prec_val = hit_cnt / top_k if top_k > 0 else 0.0

            cur.execute("""
                UPDATE public.p655kqpredict SET
                    actual_n1=%(an1)s, actual_n2=%(an2)s, actual_n3=%(an3)s,
                    actual_n4=%(an4)s, actual_n5=%(an5)s, actual_n6=%(an6)s,
                    hit_cnt=%(hit_cnt)s, precision_val=%(prec)s,
                    updated_at=%(upd)s
                WHERE id=%(id)s
            """, {
                "an1":actual["n1"], "an2":actual["n2"], "an3":actual["n3"],
                "an4":actual["n4"], "an5":actual["n5"], "an6":actual["n6"],
                "hit_cnt":hit_cnt, "prec":prec_val,
                "upd":datetime.now(timezone.utc),
                "id":row["id"],
            })

            flag = "✅" if prec_val > 0.109 else ("🟡" if prec_val > 0 else "⚠️ ")
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
# 15. MAIN
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="XGBoost predict P655 — 55 so chinh")
    parser.add_argument("--update_result", action="store_true")
    parser.add_argument("--source",       choices=["db","csv"], default="db")
    parser.add_argument("--file",         default="p655kqdetail.csv")
    parser.add_argument("--n_ky",         type=int,   default=0)
    parser.add_argument("--qh_ky",        type=int,   default=20)
    parser.add_argument("--lags",         type=int,   default=3)
    parser.add_argument("--top",          type=int,   default=8)
    parser.add_argument("--val_size",     type=int,   default=50)
    parser.add_argument("--min_train",    type=int,   default=300)
    parser.add_argument("--train_window", type=int,   default=0)
    parser.add_argument("--decay",        type=float, default=0.995)
    parser.add_argument("--importance",   action="store_true")
    parser.add_argument("--ensemble",     action="store_true")
    parser.add_argument("--filter",       type=float, default=0.10)
    parser.add_argument("--save",         action="store_true")
    args = parser.parse_args()

    if args.update_result:
        print("🔄 Update ket qua thuc te tu p655kqdetail...")
        update_result()
        return

    print(f"📌 Version : {VERSION}")
    print(f"📝 Desc    : {DESCRIPTION[:80]}...")
    print("📥 Loading data...")
    df = (load_from_csv(args.file) if args.source == "csv"
          else load_from_db(args.n_ky, args.qh_ky))
    print(f"   {len(df)} ky")
    print(f"   Mega : {'✅' if 'mgn1' in df.columns else '⚠️'}")
    print(f"   JP   : {'✅' if 'jpck' in df.columns else '⚠️'}")
    print(f"   QHL  : {'✅' if 'qhl'  in df.columns else '⚠️'}")

    if args.ensemble:
        print(f"\n🚀 Training ensemble ({len(ENSEMBLE_CONFIGS)} configs × 55 so)...")
        all_results = train_ensemble(
            df, val_size=args.val_size, min_train=args.min_train,
        )
        print(f"\n🔮 Predicting (ensemble + filter={args.filter:.0%})...")
        proba   = predict_top_k_ensemble(
            all_results, top_k=args.top, filter_threshold=args.filter,
        )
        cv_prec = np.mean([all_results[0][n]["prec"] for n in ALL_NUMS])
    else:
        print("\n⚙️  Global feature engineering...")
        global_feat = make_global_features(df, lags=args.lags)
        print(f"   {global_feat.shape[1]} global features")

        print(f"\n🚀 Training 55 model (top={args.top})...")
        results = train_all(
            df, global_feat,
            lags=args.lags, val_size=args.val_size, min_train=args.min_train,
            train_window=args.train_window, decay=args.decay,
            show_importance=args.importance,
        )

        print(f"\n🔮 Predicting (filter={args.filter:.0%})...")
        proba   = predict_top_k_single(
            results, top_k=args.top, filter_threshold=args.filter,
        )
        cv_prec = np.mean([results[n]["prec"] for n in ALL_NUMS])

    show_prediction(df, proba, top_k=args.top)

    if args.save:
        save_predict(args, df, proba, cv_prec)


if __name__ == "__main__":
    main()