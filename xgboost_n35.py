"""
xgboost_nums.py
───────────────
Du doan 35 so cua L535 (n1-n5), chon top 8 theo xac suat.

Kien truc:
  - 35 XGBoost binary classifier song song (1 model / so)
  - Moi model predict P(so n xuat hien trong n1-n5 ky tiep theo)
  - Rank 35 xac suat, chon top 8 (Kieu A — pure top 8)

Baseline precision: chon random 8/35 → 5*8/35/5 = 22.9%
  (trung binh moi ky co 5 so, chon 8 → ky vong trung 8*5/35 = 1.14 so)
  Precision = so trung / so chon = 1.14/8 = 14.3%

Features (no-leak, shift >= 1):
  A. Lich: thu, dot (biet truoc)
  B. Per-number history: last_seen, freq_20, freq_50, is_qhl
  C. Global L535: sc, sum, dd_enc, cl5_l_count, dec0, dec3 (lag 1-lags)
  D. Rolling global: sc, cl5_l_count (window 3, 5)
  E. Mega/Power: mg_dd, pw_dd, cross features
  F. JP: jpck, is_jp

Usage:
    python xgboost_nums.py --source db --lags 3 --top 8
    python xgboost_nums.py --source db --lags 3 --top 8 --importance
    python xgboost_nums.py --source db --lags 3 --top 8 --tune --tune_trials 50
    python xgboost_nums.py --file data.csv --lags 3 --top 8
"""

import argparse, os, warnings
import numpy as np
import pandas as pd
from collections import Counter

from xgboost import XGBClassifier
from sklearn.metrics import precision_score, recall_score
from sklearn.utils.class_weight import compute_sample_weight
from scipy.stats import mode

warnings.filterwarnings("ignore")

DATABASE_URL = os.environ.get(
    "DATABASE_URL", "postgresql://postgres:bin@localhost:5432/katalott"
)

ALL_NUMS = list(range(1, 36))   # 1-35

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
        lambda v: int("".join("0" if c=="C" else "1" for c in v), 2)
        if len(v) == 5 else np.nan
    )
    d["thu_enc"] = _encode_thu(d["thu"])
    d["dot"]     = pd.to_numeric(d["dot"], errors="coerce").fillna(0).astype(int)

    if "jpck" in d.columns:
        d["jpck"] = pd.to_numeric(d["jpck"], errors="coerce").fillna(0)
    if "is_jp" in d.columns:
        d["is_jp_enc"] = d["is_jp"].astype(int)

    feat = pd.DataFrame(index=d.index)

    # A. Biet truoc (lich) — no shift
    feat["thu_enc"] = d["thu_enc"]
    feat["dot"]     = d["dot"]

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
                feat[f"{col}_roll{w}_std"]  = base.rolling(w).std()

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
    hit = df.apply(
        lambda r: int(num in get_drawn_set(r)), axis=1
    )

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
    feat["n_dec"]     = float(num // 10)

    return feat


def build_dataset(df: pd.DataFrame, num: int, global_feat: pd.DataFrame,
                  lags: int = 3) -> tuple:
    """
    Gop global_feat + per_number_feat, tao target cho so `num`.
    Tra ve (X_df, y) sau khi dropna.
    """
    per_feat = make_per_number_features(df, num)
    all_feat = pd.concat([global_feat, per_feat], axis=1)

    # Target: so `num` co xuat hien trong n1-n5 ky hien tai khong
    target = df.apply(lambda r: int(num in get_drawn_set(r)), axis=1)

    combined = pd.concat([all_feat, target.rename("y"), df["ky"]], axis=1)
    combined = combined.dropna().reset_index(drop=True)

    feature_cols = list(all_feat.columns)
    return combined, feature_cols


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

def train_one(combined: pd.DataFrame, feature_cols: list,
              val_size: int, min_train: int,
              xgb_params: dict = None) -> tuple:
    """
    Train walk-forward cho 1 so. Tra ve (final_model, mean_prec, mean_rec).
    """
    X     = combined[feature_cols].values
    y     = combined["y"].values

    splits   = _walk_forward_splits(len(X), min_train, val_size)
    prec_list, rec_list = [], []

    for tr_idx, va_idx in splits:
        X_tr, X_va = X[tr_idx], X[va_idx]
        y_tr, y_va = y[tr_idx], y[va_idx]

        if y_tr.sum() == 0:   # khong co positive trong train
            continue

        sw_tr = compute_sample_weight("balanced", y_tr)
        mdl   = make_xgb(xgb_params)
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

    # Final model tren toan bo data
    final_p = dict(xgb_params or {})
    final_p["early_stopping_rounds"] = None
    final = make_xgb(final_p)
    sw_all = compute_sample_weight("balanced", y)
    final.fit(X, y, sample_weight=sw_all, verbose=False)

    mean_prec = float(np.mean(prec_list)) if prec_list else 0.0
    mean_rec  = float(np.mean(rec_list))  if rec_list  else 0.0
    return final, mean_prec, mean_rec


# ─────────────────────────────────────────────
# 7. TRAIN ALL 35
# ─────────────────────────────────────────────

def train_all(df: pd.DataFrame, global_feat: pd.DataFrame,
              lags: int, val_size: int, min_train: int,
              xgb_params: dict = None,
              show_importance: bool = False) -> dict:
    """
    Train 35 model, tra ve dict: num -> {model, prec, rec, feature_cols}.
    """
    results = {}
    baseline_prec = 5 * 8 / 35 / 8   # = 14.3%

    print(f"\n  Baseline precision (random 8/35): {baseline_prec:.1%}")
    print(f"  {'Num':>4}  {'CV Prec':>8}  {'CV Rec':>8}  {'Flag'}")
    print(f"  {'─'*40}")

    for num in ALL_NUMS:
        combined, feature_cols = build_dataset(df, num, global_feat, lags)
        model, prec, rec = train_one(
            combined, feature_cols, val_size, min_train, xgb_params
        )
        results[num] = dict(
            model        = model,
            prec         = prec,
            rec          = rec,
            feature_cols = feature_cols,
            combined     = combined,
        )
        flag = "✅" if prec > baseline_prec else "⚠️ "
        print(f"  {num:>4}  {prec:>8.3f}  {rec:>8.3f}  {flag}")

    # Summary
    precs = [results[n]["prec"] for n in ALL_NUMS]
    recs  = [results[n]["rec"]  for n in ALL_NUMS]
    print(f"\n  Mean CV Precision : {np.mean(precs):.3f}")
    print(f"  Mean CV Recall    : {np.mean(recs):.3f}")
    n_above = sum(1 for p in precs if p > baseline_prec)
    print(f"  So beat baseline  : {n_above}/35")

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
# 9. PREDICT TOP-K
# ─────────────────────────────────────────────

def predict_top_k(df: pd.DataFrame, global_feat: pd.DataFrame,
                  results: dict, top_k: int = 8) -> list:
    """
    Voi moi so 1-35, lay xac suat tu model, rank va chon top_k.
    Tra ve list (num, proba) sap xep giam dan.
    """
    proba_list = []
    for num in ALL_NUMS:
        r         = results[num]
        combined  = r["combined"]
        feat_cols = r["feature_cols"]
        model     = r["model"]

        # Row cuoi cung trong combined la ky hien tai (ky moi nhat)
        last_X = combined[feat_cols].iloc[[-1]].values
        prob   = model.predict_proba(last_X)[0][1]  # P(xuat hien=1)
        proba_list.append((num, prob))

    proba_list.sort(key=lambda x: -x[1])
    return proba_list


def show_prediction(df: pd.DataFrame, proba_list: list,
                    top_k: int = 8, results: dict = None):
    last_ky = df["ky"].iloc[-1]
    top     = proba_list[:top_k]
    rest    = proba_list[top_k:]

    print(f"\n{'═'*56}")
    print(f"🎯 Du doan {top_k} so cho ky sau ky {last_ky}:")
    print(f"{'═'*56}")

    # In top k
    nums_chosen = [n for n, _ in top]
    for rank, (num, prob) in enumerate(top, 1):
        cv_prec = results[num]["prec"] if results else 0.0
        bar     = "█" * int(prob * 30)
        print(f"  #{rank:>2}  So {num:>2}  P={prob:.3f}  CV_prec={cv_prec:.3f}  {bar}")

    print(f"\n  → Chon: {sorted(nums_chosen)}")

    # Thong ke tap hop
    chosen_set = set(nums_chosen)
    n_even  = sum(1 for n in nums_chosen if n % 2 == 0)
    n_odd   = top_k - n_even
    dec_cnt = Counter(n // 10 for n in nums_chosen)
    print(f"  → Chan/Le: {n_even}C / {n_odd}L")
    print(f"  → Chuc   : " + "  ".join(
        f"{d*10}x:{dec_cnt.get(d,0)}" for d in range(4)
    ))

    # So sanh voi QHL neu co
    if results:
        qhl_hits = []
        for num in nums_chosen:
            if results[num]["combined"]["n_is_qhl"].iloc[-1] == 1.0 \
               if "n_is_qhl" in results[num]["combined"].columns else False:
                qhl_hits.append(num)
        if qhl_hits:
            print(f"  → So trong QHL: {sorted(qhl_hits)}")

    print(f"{'═'*56}\n")
    return nums_chosen


# ─────────────────────────────────────────────
# 10. EVALUATE HISTORY (back-test)
# ─────────────────────────────────────────────

def evaluate_history(df: pd.DataFrame, results: dict,
                     top_k: int = 8, last_n: int = 50):
    """
    Back-test tren last_n ky cuoi: moi ky chon top_k so,
    so voi ket qua thuc te, tinh precision va hit count.
    """
    print(f"\n📊 Back-test {last_n} ky cuoi (top {top_k}):")
    print(f"  {'Ky':<8}  {'Chon':^35}  {'Trung':>5}  {'Prec':>6}")
    print(f"  {'─'*65}")

    prec_list = []
    hit_list  = []

    # Lay tung ky trong khoang back-test
    n_total = len(df)
    start   = max(0, n_total - last_n - 1)

    for i in range(start, n_total - 1):
        ky_str  = (df["ky"].iloc[i] or "").strip()
        actual  = get_drawn_set(df.iloc[i + 1])   # ket qua ky tiep theo

        # Lay xac suat tung so tai thoi diem i (row i trong combined)
        proba_i = []
        for num in ALL_NUMS:
            r        = results[num]
            combined = r["combined"]
            feat_c   = r["feature_cols"]
            model    = r["model"]

            # Tim row trong combined tuong ung ky nay
            mask = combined["ky"].astype(str).str.strip() == ky_str
            if mask.sum() == 0:
                proba_i.append((num, 0.0))
                continue
            row_idx = combined[mask].index[0]
            X_row   = combined.loc[[row_idx], feat_c].values
            prob    = model.predict_proba(X_row)[0][1]
            proba_i.append((num, prob))

        proba_i.sort(key=lambda x: -x[1])
        chosen = set(n for n, _ in proba_i[:top_k])
        hits   = chosen & actual
        prec   = len(hits) / top_k
        prec_list.append(prec)
        hit_list.append(len(hits))

        # In moi 5 ky
        if (i - start) % 5 == 0 or i == n_total - 2:
            chosen_str = " ".join(f"{n:>2}" for n in sorted(chosen))
            print(
                f"  {ky_str:<8}  [{chosen_str}]"
                f"  {len(hits):>5}  {prec:>6.1%}"
            )

    mean_prec = np.mean(prec_list)
    mean_hits = np.mean(hit_list)
    baseline  = 5 / 35 * top_k / top_k   # = 14.3%
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
# 12. MAIN
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="XGBoost predict 35 so L535")
    parser.add_argument("--source",      choices=["db", "csv"], default="csv")
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
    parser.add_argument("--tune",        action="store_true")
    parser.add_argument("--tune_trials", type=int, default=30)
    args = parser.parse_args()

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
    global_feat = make_global_features(df, lags=args.lags)
    print(f"   {global_feat.shape[1]} global features")

    # Optuna tuning
    xgb_params = None
    if args.tune:
        xgb_params = optuna_tune(
            df, global_feat,
            lags=args.lags,
            val_size=args.val_size,
            min_train=args.min_train,
            n_trials=args.tune_trials,
        )

    print(f"\n🚀 Training 35 model (top={args.top})...")
    results = train_all(
        df, global_feat,
        lags=args.lags,
        val_size=args.val_size,
        min_train=args.min_train,
        xgb_params=xgb_params,
        show_importance=args.importance,
    )

    # Back-test neu co --backtest
    if args.backtest > 0:
        evaluate_history(df, results, top_k=args.top, last_n=args.backtest)

    # Predict ky tiep theo
    print(f"\n🔮 Predicting top {args.top}...")
    proba_list = predict_top_k(df, global_feat, results, top_k=args.top)
    show_prediction(df, proba_list, top_k=args.top, results=results)


if __name__ == "__main__":
    main()