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
  C. Global L535: sc, sum, dd_enc, cl5_l_count, dec0, dec3 (lag 1-lags)
  D. Rolling global: sc, cl5_l_count (window 3, 5)
  E. Mega/Power: mg_dd, pw_dd, cross features
  F. JP: jpck, is_jp
  G. Sliding window + Time decay weight

Usage:
    python claude_fima_xgboost_n35.py --source db --lags 3 --top 8
    python claude_fima_xgboost_n35.py --source db --lags 3 --top 8 --importance
    python claude_fima_xgboost_n35.py --source db --lags 3 --top 8 --train_window 300 --decay 0.995
    python claude_fima_xgboost_n35.py --update_result
"""

import argparse, os, warnings
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

ALL_NUMS = list(range(1, 36))  # 1-35

# ── Version ───────────────────────────────────────────────────────────────────
VERSION = "v1.3.0"
DESCRIPTION = (
    "35 binary XGBoost (n1-n5), no-leak train/predict split, "
    "sliding window + time decay weight, Mega/Power cross features, "
    "walk-forward CV, save to l535kqpredict"
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
    df = pd.read_sql(sql, conn)
    conn.close()
    return df.sort_values("ky").reset_index(drop=True)


def load_from_csv(filepath) -> pd.DataFrame:
    df = pd.read_csv(filepath)
    return df.sort_values("ky").reset_index(drop=True)


# ─────────────────────────────────────────────
# 2. HELPERS
# ─────────────────────────────────────────────

THU_MAP = {"T2": 2, "T3": 3, "T4": 4, "T5": 5, "T6": 6, "T7": 7, "CN": 8}
DD_MAP = {"CC": 0, "CL": 1, "LC": 2, "LL": 3}


def _safe_int(s) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _encode_thu(series) -> pd.Series:
    return series.astype(str).str.strip().map(THU_MAP)


def _ngay_diff(d1: pd.Series, d2: pd.Series) -> pd.Series:
    return (
        pd.to_datetime(d1, errors="coerce") - pd.to_datetime(d2, errors="coerce")
    ).dt.days


def _encode_dd(n_first, n_last) -> pd.Series:
    def _cl(n):
        return "C" if n % 2 == 0 else "L"

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
        int(row[c])
        for c in ["n1", "n2", "n3", "n4", "n5"]
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
    """
    d = df.copy()

    # ── Encode global ──
    d["dd_enc"] = d["dd"].apply(
        lambda v: {"CC": 0, "CL": 1, "LC": 2, "LL": 3}.get(str(v).strip(), -1)
    )
    s = d["cl5"].astype(str).str.strip()
    d["cl5_l_count"] = s.apply(
        lambda v: sum(1 for c in v if c == "L") if len(v) == 5 else np.nan
    )
    d["cl5_pattern"] = s.apply(
        lambda v: (
            int("".join("0" if c == "C" else "1" for c in v), 2)
            if len(v) == 5
            else np.nan
        )
    )
    d["thu_enc"] = _encode_thu(d["thu"])
    d["dot"] = pd.to_numeric(d["dot"], errors="coerce").fillna(0).astype(int)

    if "jpck" in d.columns:
        d["jpck"] = pd.to_numeric(d["jpck"], errors="coerce").fillna(0)
    if "is_jp" in d.columns:
        d["is_jp_enc"] = d["is_jp"].astype(int)

    feat = pd.DataFrame(index=d.index)

    # A. Biet truoc (lich) — no shift
    feat["thu_enc"] = d["thu_enc"]
    feat["dot"] = d["dot"]

    # B. Global L535 ky truoc (shift 1)
    prev_cols = ["dd_enc", "sc", "sum", "cl5_l_count", "cl5_pattern", "dec0", "dec3"]
    for c in ["jpck", "is_jp_enc"]:
        if c in d.columns:
            prev_cols.append(c)
    prev_cols = [c for c in prev_cols if c in d.columns]
    for col in prev_cols:
        feat[f"{col}_prev"] = d[col].shift(1)

    # C. Lag 2 → lags
    lag_cols = ["dd_enc", "sc", "cl5_l_count", "cl5_pattern"]
    lag_cols = [c for c in lag_cols if c in d.columns]
    for lag in range(2, lags + 1):
        for col in lag_cols:
            feat[f"{col}_lag{lag}"] = d[col].shift(lag)

    # D. Rolling (window 3, 5)
    for w in [3, 5]:
        for col in ["sc", "cl5_l_count"]:
            if col in d.columns:
                base = d[col].shift(1)
                feat[f"{col}_roll{w}_mean"] = base.rolling(w).mean()
                feat[f"{col}_roll{w}_std"] = base.rolling(w).std()

    # E. Mega features
    if all(c in d.columns for c in ["mgn1", "mgn6"]):
        mn = [_safe_int(d[f"mgn{i}"]) for i in range(1, 7)]
        if "mgthu" in d.columns:
            feat["mg_thu_enc"] = _encode_thu(d["mgthu"])
        if "mgngay" in d.columns:
            feat["mg_ngay_diff"] = _ngay_diff(d["ngay"], d["mgngay"])
        feat["mg_dd"] = _encode_dd(mn[0], mn[5])

    # F. Power features
    if all(c in d.columns for c in ["pwn1", "pwn6"]):
        pn = [_safe_int(d[f"pwn{i}"]) for i in range(1, 7)]
        if "pwthu" in d.columns:
            feat["pw_thu_enc"] = _encode_thu(d["pwthu"])
        if "pwngay" in d.columns:
            feat["pw_ngay_diff"] = _ngay_diff(d["ngay"], d["pwngay"])
        feat["pw_dd"] = _encode_dd(pn[0], pn[5])

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

    return feat


def make_per_number_features(df: pd.DataFrame, num: int) -> pd.DataFrame:
    """
    Tinh features rieng cho so `num` (1-35).
    Tat ca deu shift >= 1.

    Features:
      - n_hit_prev       : so `num` co xuat hien ky truoc khong (0/1)
      - n_last_seen      : cach bao nhieu ky lan cuoi so `num` xuat hien
      - n_freq_20        : tan suat trong 20 ky gan nhat (shift 1)
      - n_freq_50        : tan suat trong 50 ky gan nhat (shift 1)
      - n_is_qhl         : so `num` co trong danh sach QHL khong
      - n_is_even        : so chan (1) hay le (0)
      - n_dec            : chuc cua so (0-3)
    """
    feat = pd.DataFrame(index=df.index)

    # Ket qua ky truoc: so `num` co xuat hien n1-n5 khong
    hit = df.apply(lambda r: int(num in get_drawn_set(r)), axis=1)

    # n_hit_prev: hit ky truoc
    feat["n_hit_prev"] = hit.shift(1)

    # n_last_seen: cach bao nhieu ky lan cuoi xuat hien
    last_seen = []
    last = np.nan
    for i, h in enumerate(hit):
        if i == 0:
            last_seen.append(np.nan)
            if h == 1:
                last = 0
        else:
            if pd.isna(last):
                last_seen.append(np.nan)
            else:
                last_seen.append(i - last)
            if h == 1:
                last = i
    feat["n_last_seen"] = pd.Series(last_seen, index=df.index).shift(1)

    # n_freq_20, n_freq_50: rolling mean (shift 1 roi rolling)
    hit_shifted = hit.shift(1)
    feat["n_freq_20"] = hit_shifted.rolling(20).mean()
    feat["n_freq_50"] = hit_shifted.rolling(50).mean()

    # n_is_qhl: so `num` co trong QHL cua ky truoc khong
    if "qhl" in df.columns:

        def _in_qhl(v):
            if pd.isna(v) or v == "":
                return 0.0
            try:
                nums = [int(x) for x in str(v).split(",") if x.strip()]
                return float(num in nums)
            except:
                return 0.0

        feat["n_is_qhl"] = df["qhl"].shift(1).apply(_in_qhl)

    # Static features (khong can shift)
    feat["n_is_even"] = float(num % 2 == 0)
    feat["n_dec"] = float(num // 10)

    return feat


def build_dataset(
    df: pd.DataFrame, num: int, global_feat: pd.DataFrame, lags: int = 3
) -> tuple:
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
    predict_row = combined.iloc[[-1]]  # ky moi nhat: dung features, bo target

    return combined_train, predict_row, feature_cols


# ─────────────────────────────────────────────
# 4. XGB BUILDER
# ─────────────────────────────────────────────


def make_xgb(params: dict = None):
    defaults = dict(
        n_estimators=400,
        max_depth=2,
        learning_rate=0.05,
        subsample=0.7,
        colsample_bytree=0.6,
        min_child_weight=5,
        gamma=1,
        reg_alpha=0.1,
        reg_lambda=2.0,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
        early_stopping_rounds=50,
    )
    if params:
        defaults.update(params)
    return XGBClassifier(**defaults)


# ─────────────────────────────────────────────
# 5. WALK-FORWARD
# ─────────────────────────────────────────────


def _walk_forward_splits(n: int, min_train: int, val_size: int):
    splits = []
    start = min_train
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
    w = decay ** (n - 1 - idx)  # row cuoi = decay^0 = 1.0
    return w / w.mean()  # normalize de tong khong doi


def train_one(
    combined: pd.DataFrame,
    feature_cols: list,
    val_size: int,
    min_train: int,
    train_window: int = 0,
    decay: float = 0.995,
    xgb_params: dict = None,
) -> tuple:
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

    splits = _walk_forward_splits(len(X), min_train, val_size)
    prec_list, rec_list = [], []

    for tr_idx, va_idx in splits:
        X_tr, X_va = X[tr_idx], X[va_idx]
        y_tr, y_va = y[tr_idx], y[va_idx]

        if y_tr.sum() == 0:
            continue

        # Sliding window trong CV: lay toi da train_window ky cuoi cua fold
        if train_window > 0 and len(tr_idx) > train_window:
            tr_idx = tr_idx[-train_window:]
            X_tr = X[tr_idx]
            y_tr = y[tr_idx]

        # Time decay weight
        sw_bal = compute_sample_weight("balanced", y_tr)
        sw_decay = _make_decay_weights(len(y_tr), decay)
        sw_tr = sw_bal * sw_decay

        mdl = make_xgb(xgb_params)
        mdl.fit(
            X_tr,
            y_tr,
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

    sw_bal_f = compute_sample_weight("balanced", y_fin)
    sw_decay_f = _make_decay_weights(len(y_fin), decay)
    sw_fin = sw_bal_f * sw_decay_f

    final_p = dict(xgb_params or {})
    final_p["early_stopping_rounds"] = None
    final = make_xgb(final_p)
    final.fit(X_fin, y_fin, sample_weight=sw_fin, verbose=False)

    mean_prec = float(np.mean(prec_list)) if prec_list else 0.0
    mean_rec = float(np.mean(rec_list)) if rec_list else 0.0
    return final, mean_prec, mean_rec


# ─────────────────────────────────────────────
# 7. TRAIN ALL 35
# ─────────────────────────────────────────────


def train_all(
    df: pd.DataFrame,
    global_feat: pd.DataFrame,
    lags: int,
    val_size: int,
    min_train: int,
    train_window: int = 0,
    decay: float = 0.995,
    xgb_params: dict = None,
    show_importance: bool = False,
) -> dict:
    """
    Train 35 model, tra ve dict: num -> {model, prec, rec, feature_cols, predict_row}.
    train_window: chi dung N ky gan nhat (0 = tat ca)
    decay       : he so giam weight theo thoi gian
    """
    results = {}
    baseline_prec = 5 * 8 / 35 / 8  # = 14.3%

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
            combined_train,
            feature_cols,
            val_size,
            min_train,
            train_window=train_window,
            decay=decay,
            xgb_params=xgb_params,
        )
        results[num] = dict(
            model=model,
            prec=prec,
            rec=rec,
            feature_cols=feature_cols,
            combined=combined_train,
            predict_row=predict_row,
        )
        flag = "✅" if prec > baseline_prec else "⚠️ "
        print(f"  {num:>4}  {prec:>8.3f}  {rec:>8.3f}  {flag}")

    precs = [results[n]["prec"] for n in ALL_NUMS]
    recs = [results[n]["rec"] for n in ALL_NUMS]
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
        mdl = r["model"]
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
# 9. PREDICT TOP-K
# ─────────────────────────────────────────────


def predict_top_k(
    df: pd.DataFrame, global_feat: pd.DataFrame, results: dict, top_k: int = 8
) -> list:
    """
    Voi moi so 1-35, lay xac suat tu model dung predict_row (ky moi nhat).
    predict_row chua duoc dung de train → no-leak thuc su.
    Tra ve list (num, proba) sap xep giam dan.
    """
    proba_list = []
    for num in ALL_NUMS:
        r = results[num]
        feat_cols = r["feature_cols"]
        model = r["model"]
        predict_row = r["predict_row"]

        last_X = predict_row[feat_cols].values
        prob = model.predict_proba(last_X)[0][1]
        proba_list.append((num, prob))

    proba_list.sort(key=lambda x: -x[1])
    return proba_list


def show_prediction(
    df: pd.DataFrame, proba_list: list, top_k: int = 8, results: dict = None
):
    # Ky cuoi trong DB = ky vua xo xong, dung features cua no de predict ky tiep
    last_ky = (df["ky"].iloc[-1] or "").strip()
    next_ky = f"(sau {last_ky})"
    top = proba_list[:top_k]
    rest = proba_list[top_k:]

    print(f"\n{'═'*56}")
    print(f"📌 Input : ky {last_ky} (features cua ky nay)")
    print(f"🎯 Du doan {top_k} so cho ky {next_ky}:")
    print(f"{'═'*56}")

    # In top k
    nums_chosen = [n for n, _ in top]
    for rank, (num, prob) in enumerate(top, 1):
        cv_prec = results[num]["prec"] if results else 0.0
        bar = "█" * int(prob * 30)
        print(f"  #{rank:>2}  So {num:>2}  P={prob:.3f}  CV_prec={cv_prec:.3f}  {bar}")

    print(f"\n  → Chon: {sorted(nums_chosen)}")

    # Thong ke tap hop
    chosen_set = set(nums_chosen)
    n_even = sum(1 for n in nums_chosen if n % 2 == 0)
    n_odd = top_k - n_even
    dec_cnt = Counter(n // 10 for n in nums_chosen)
    print(f"  → Chan/Le: {n_even}C / {n_odd}L")
    print(f"  → Chuc   : " + "  ".join(f"{d*10}x:{dec_cnt.get(d,0)}" for d in range(4)))

    # So sanh voi QHL neu co
    if results:
        qhl_hits = []
        for num in nums_chosen:
            if (
                results[num]["combined"]["n_is_qhl"].iloc[-1] == 1.0
                if "n_is_qhl" in results[num]["combined"].columns
                else False
            ):
                qhl_hits.append(num)
        if qhl_hits:
            print(f"  → So trong QHL: {sorted(qhl_hits)}")

    print(f"{'═'*56}\n")
    return nums_chosen


# ─────────────────────────────────────────────
# 10. EVALUATE HISTORY (back-test)
# ─────────────────────────────────────────────


def evaluate_history(df: pd.DataFrame, results: dict, top_k: int = 8, last_n: int = 50):
    """
    Back-test tren last_n ky cuoi: moi ky chon top_k so,
    so voi ket qua thuc te, tinh precision va hit count.
    """
    print(f"\n📊 Back-test {last_n} ky cuoi (top {top_k}):")
    print(f"  {'Ky':<8}  {'Chon':^35}  {'Trung':>5}  {'Prec':>6}")
    print(f"  {'─'*65}")

    prec_list = []
    hit_list = []

    # Lay tung ky trong khoang back-test
    n_total = len(df)
    start = max(0, n_total - last_n - 1)

    for i in range(start, n_total - 1):
        ky_str = (df["ky"].iloc[i] or "").strip()
        actual = get_drawn_set(df.iloc[i + 1])  # ket qua ky tiep theo

        # Lay xac suat tung so tai thoi diem i (row i trong combined)
        proba_i = []
        for num in ALL_NUMS:
            r = results[num]
            combined = r["combined"]
            feat_c = r["feature_cols"]
            model = r["model"]

            # Tim row trong combined tuong ung ky nay
            mask = combined["ky"].astype(str).str.strip() == ky_str
            if mask.sum() == 0:
                proba_i.append((num, 0.0))
                continue
            row_idx = combined[mask].index[0]
            X_row = combined.loc[[row_idx], feat_c].values
            prob = model.predict_proba(X_row)[0][1]
            proba_i.append((num, prob))

        proba_i.sort(key=lambda x: -x[1])
        chosen = set(n for n, _ in proba_i[:top_k])
        hits = chosen & actual
        prec = len(hits) / top_k
        prec_list.append(prec)
        hit_list.append(len(hits))

        # In moi 5 ky
        if (i - start) % 5 == 0 or i == n_total - 2:
            chosen_str = " ".join(f"{n:>2}" for n in sorted(chosen))
            print(f"  {ky_str:<8}  [{chosen_str}]" f"  {len(hits):>5}  {prec:>6.1%}")

    mean_prec = np.mean(prec_list)
    mean_hits = np.mean(hit_list)
    baseline = 5 / 35 * top_k / top_k  # = 14.3%
    lift = mean_prec - baseline
    flag = "✅" if lift > 0.02 else ("🟡" if lift > 0 else "⚠️ ")

    print(
        f"\n  Mean precision : {mean_prec:.3f}  (baseline={baseline:.3f})  "
        f"lift={lift:+.3f}  {flag}"
    )
    print(f"  Mean hits/ky   : {mean_hits:.2f} / {top_k}")


# ─────────────────────────────────────────────
# 11. OPTUNA TUNING
# ─────────────────────────────────────────────


def optuna_tune(
    df: pd.DataFrame,
    global_feat: pd.DataFrame,
    lags: int,
    val_size: int,
    min_train: int,
    n_trials: int = 30,
) -> dict:
    """
    Tune hyperparameter tren 5 so ngau nhien (nhanh hon tune ca 35).
    """
    try:
        import optuna

        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        print("  ⚠️  Optuna chua cai: pip install optuna")
        return None

    sample_nums = [7, 14, 21, 28, 35]  # 5 so dai dien

    def objective(trial):
        params = dict(
            max_depth=trial.suggest_int("max_depth", 2, 4),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            n_estimators=trial.suggest_int("n_estimators", 200, 600),
            subsample=trial.suggest_float("subsample", 0.5, 0.9),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.4, 0.9),
            min_child_weight=trial.suggest_int("min_child_weight", 3, 20),
            gamma=trial.suggest_float("gamma", 0.0, 3.0),
            reg_alpha=trial.suggest_float("reg_alpha", 0.0, 1.0),
            reg_lambda=trial.suggest_float("reg_lambda", 0.5, 4.0),
            early_stopping_rounds=40,
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
# 12. SAVE PREDICT TO DB
# ─────────────────────────────────────────────


def save_predict(
    args, df: pd.DataFrame, proba_list: list, results: dict, cv_prec_mean: float
):
    """
    Luu ket qua du doan vao table l535kqpredict.
    proba_list: list (num, prob) sap xep giam dan (tat ca 35 so).
    Chi luu top 18 (hoac it hon neu top_k < 18).
    """
    import psycopg2

    input_ky = (df["ky"].iloc[-1] or "").strip()

    # Uoc tinh predict_ky: ky tiep theo sau input_ky (khong biet chinh xac)
    # De null, user tu dien sau khi biet ky nao duoc xo tiep
    # Tuy nhien co the uoc tinh don gian = int(input_ky) + 1
    try:
        predict_ky = str(int(input_ky) + 1).zfill(len(input_ky))
    except:
        predict_ky = None

    # Chuan bi 18 so (pad bang None neu top_k < 18)
    top18 = proba_list[:18]
    row = {}
    for i in range(1, 19):
        if i <= len(top18):
            num, prob = top18[i - 1]
            row[f"n{i}"] = int(num)
            row[f"p{i}"] = float(prob)
            row[f"cvp{i}"] = float(results[num]["prec"])
        else:
            row[f"n{i}"] = None
            row[f"p{i}"] = None
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
        "run_at": datetime.now(timezone.utc),
        "version": VERSION,
        "description": DESCRIPTION,
        "source": getattr(args, "source", "db"),
        "lags": getattr(args, "lags", 3),
        "train_window": getattr(args, "train_window", 0),
        "decay": getattr(args, "decay", 0.995),
        "top_k": getattr(args, "top", 8),
        "score_mode": "pure_p",
        "cv_prec_mean": float(cv_prec_mean),
        "input_ky": input_ky,
        "predict_ky": predict_ky,
        **row,
    }

    conn = psycopg2.connect(DATABASE_URL)
    conn.autocommit = False
    cur = conn.cursor()
    try:
        cur.execute(sql, params)
        conn.commit()
        print(
            f"\n💾 Da luu du doan vao l535kqpredict"
            f"  (input_ky={input_ky}, predict_ky={predict_ky},"
            f"  version={VERSION})"
        )
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
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

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
            cur.execute(
                """
                SELECT n1,n2,n3,n4,n5,n6
                FROM public.l535kqdetail
                WHERE ky = %s
                LIMIT 1
            """,
                (predict_ky,),
            )
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
            top_k = int(row["top_k"] or 8)
            pred_nums = []
            for i in range(1, 19):
                n = row[f"n{i}"]
                if n is not None and len(pred_nums) < top_k:
                    pred_nums.append(int(n))

            # 5. Tinh hit
            hits = set(pred_nums) & actual_nums
            hit_cnt = len(hits)
            prec_val = hit_cnt / top_k if top_k > 0 else 0.0

            # 6. Update
            cur.execute(
                """
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
            """,
                (
                    actual["n1"],
                    actual["n2"],
                    actual["n3"],
                    actual["n4"],
                    actual["n5"],
                    actual_n6,
                    hit_cnt,
                    prec_val,
                    datetime.now(timezone.utc),
                    row["id"],
                ),
            )

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
    parser.add_argument(
        "--update_result",
        action="store_true",
        help="Cap nhat ket qua thuc te tu l535kqdetail",
    )
    parser.add_argument("--source", choices=["db", "csv"], default="db")
    parser.add_argument("--file", default="l535kqdetail_dat.csv")
    parser.add_argument("--n_ky", type=int, default=0)
    parser.add_argument("--qh_ky", type=int, default=20)
    parser.add_argument("--lags", type=int, default=3)
    parser.add_argument(
        "--top", type=int, default=8, help="So luong so chon (mac dinh: 8)"
    )
    parser.add_argument(
        "--val_size",
        type=int,
        default=20,
        help="So ky moi fold walk-forward (mac dinh: 20)",
    )
    parser.add_argument(
        "--min_train",
        type=int,
        default=200,
        help="So ky toi thieu de bat dau train (mac dinh: 200)",
    )
    parser.add_argument(
        "--importance",
        action="store_true",
        help="In feature importance trung binh 35 model",
    )
    parser.add_argument(
        "--backtest", type=int, default=0, help="Back-test N ky cuoi (0 = tat)"
    )
    parser.add_argument(
        "--train_window",
        type=int,
        default=0,
        help="Sliding window: chi dung N ky gan nhat (0 = tat ca)",
    )
    parser.add_argument(
        "--decay",
        type=float,
        default=0.995,
        help="Time decay weight (0.99-1.0, 1.0 = tat decay)",
    )
    parser.add_argument(
        "--save", action="store_true", help="Luu ket qua du doan vao l535kqpredict"
    )
    parser.add_argument("--tune", action="store_true")
    parser.add_argument("--tune_trials", type=int, default=30)
    args = parser.parse_args()

    # ── Mode: update result ──
    if args.update_result:
        print(f"🔄 Update ket qua thuc te tu l535kqdetail...")
        update_result()
        return

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

    has_mega = "mgn1" in df.columns
    has_power = "pwn1" in df.columns
    has_jp = "jpck" in df.columns
    has_qhl = "qhl" in df.columns
    print(f"   Mega  : {'✅' if has_mega  else '⚠️'}")
    print(f"   Power : {'✅' if has_power else '⚠️'}")
    print(f"   JP    : {'✅' if has_jp    else '⚠️'}")
    print(f"   QHL   : {'✅' if has_qhl   else '⚠️'}")

    print("\n⚙️  Global feature engineering...")
    global_feat = make_global_features(df, lags=args.lags)
    print(f"   {global_feat.shape[1]} global features")

    # Optuna tuning
    xgb_params = None
    if args.tune:
        xgb_params = optuna_tune(
            df,
            global_feat,
            lags=args.lags,
            val_size=args.val_size,
            min_train=args.min_train,
            n_trials=args.tune_trials,
        )

    print(f"\n🚀 Training 35 model (top={args.top})...")
    results = train_all(
        df,
        global_feat,
        lags=args.lags,
        val_size=args.val_size,
        min_train=args.min_train,
        train_window=args.train_window,
        decay=args.decay,
        xgb_params=xgb_params,
        show_importance=args.importance,
    )

    # Back-test
    if args.backtest > 0:
        evaluate_history(df, results, top_k=args.top, last_n=args.backtest)

    # Predict
    print(f"\n🔮 Predicting top {args.top}...")
    proba_list = predict_top_k(df, global_feat, results, top_k=args.top)
    show_prediction(df, proba_list, top_k=args.top, results=results)

    # CV prec mean
    cv_prec_mean = np.mean([results[n]["prec"] for n in ALL_NUMS])

    # Save to DB
    if args.save:
        save_predict(args, df, proba_list, results, cv_prec_mean)


if __name__ == "__main__":
    main()
