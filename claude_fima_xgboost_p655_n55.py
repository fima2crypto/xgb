"""
xgboost_p655.py
───────────────
Du doan kha nang xuat hien cua 55 so cua Power P655.

2 models rieng biet (Option B - Stacked):
  - Model N16 : du doan 6 so chinh (N1-N6), output top 8
  - Model N7  : du doan so bonus (N7),       output top 3

Moi hang trong training = 1 cap (ky_i, so_k):
  - Label N16 = 1 neu so_k xuat hien trong N1-N6 cua ky i+1
  - Label N7  = 1 neu so_k chinh la N7 cua ky i+1

Features:
  - Chung (shift>=1): dec0-5, sc, sum, cl6, dd, ke, jpck, jp1_cnt, rolling, lags
  - Rieng so k     : freq_last_10/20/50, ky_vang, is_qhl, dec_of_k, is_prime, is_even
  - Mega (no shift): mg_thu_enc, mg_ngay_diff, mg_dd, mg_sc
  - Lotto (no shift, chon loc): lo_sc, lo_dd
  - Cross          : cross_mg_dd_match, cross_lo_dd_match, cross_sc_mg, cross_sc_lo

Khong data leak: tat ca features duoc tinh tu ky hien tai nhin VE QUA KHU.
Label la ket qua cua ky TIEP THEO.

Usage:
    python xgboost_p655.py --source db
    python xgboost_p655.py --source db --lags 5 --importance
    python xgboost_p655.py --source db --tune --tune_trials 50
    python xgboost_p655.py --file p655kqdetail.csv --lags 3
"""

import argparse
import os
import warnings

import numpy as np
import pandas as pd
from collections import Counter

from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from sklearn.utils.class_weight import compute_sample_weight

warnings.filterwarnings("ignore")

# ── constants ─────────────────────────────────────────────────────────────────

ALL_NUMS = list(range(1, 56))          # P655: 1..55
PRIMES   = {2,3,5,7,11,13,17,19,23,29,31,37,41,43,47,53}
THU_MAP  = {"T2":2,"T3":3,"T4":4,"T5":5,"T6":6,"T7":7,"CN":8}
DD_MAP   = {"CC":0,"CL":1,"LC":2,"LL":3}

DATABASE_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql://postgres:bin@localhost:5432/katalott"
)

# ── load ──────────────────────────────────────────────────────────────────────

def load_from_db(n_ky: int = 0, qh_ky: int = 20) -> pd.DataFrame:
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

def load_from_csv(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath)
    return df.sort_values("ky").reset_index(drop=True)

# ── helpers ───────────────────────────────────────────────────────────────────

def _safe_int(s) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")

def _encode_thu(s) -> pd.Series:
    return s.astype(str).str.strip().map(THU_MAP)

def _encode_dd(n_first: pd.Series, n_last: pd.Series) -> pd.Series:
    def _cl(n): return "C" if int(n) % 2 == 0 else "L"
    res = []
    for f, l in zip(n_first, n_last):
        if pd.isna(f) or pd.isna(l):
            res.append(np.nan)
        else:
            res.append(DD_MAP.get(_cl(f) + _cl(l), np.nan))
    return pd.Series(res, index=n_first.index)

def _ngay_diff(ref: pd.Series, other: pd.Series) -> pd.Series:
    return (pd.to_datetime(ref, errors="coerce")
            - pd.to_datetime(other, errors="coerce")).dt.days

def _get_n16(df: pd.DataFrame) -> np.ndarray:
    """Tra ve mang (n_ky, 6) so chinh."""
    cols = [f"n{i}" for i in range(1, 7)]
    return df[cols].apply(_safe_int).values

def _get_n7(df: pd.DataFrame) -> np.ndarray:
    return _safe_int(df["n7"]).values

# ── feature engineering (per-ky) ─────────────────────────────────────────────

def build_ky_features(df: pd.DataFrame, lags: int = 3) -> pd.DataFrame:
    """
    Tinh cac features CHUNG cho moi ky (1 hang = 1 ky).
    Tat ca deu duoc shift >= 1 de tranh data leak,
    ngoai tru Mega/Lotto (da la ky TRUOC roi).
    """
    d = df.copy()

    # ── Encode dd / cl6 ──
    d["dd_enc"] = d["dd"].astype(str).str.strip().map(DD_MAP)
    s6 = d["cl6"].astype(str).str.strip()
    for i in range(6):
        d[f"cl6_b{i}"] = s6.str[i].map({"C":0,"L":1}).fillna(-1).astype(int)
    d["cl6_l_count"] = sum(d[f"cl6_b{i}"] for i in range(6))
    d["cl6_pattern"] = s6.apply(
        lambda v: int("".join("0" if c=="C" else "1" for c in v), 2)
        if len(v)==6 else -1
    )
    d["cl6_changed"] = (d["cl6_pattern"] != d["cl6_pattern"].shift(1)).astype(int)

    # ── Encode thu ──
    d["thu_enc"] = _encode_thu(d["thu"])

    # ── jpck / jp_enc ──
    d["jpck"]   = _safe_int(d.get("jpck", 0)).fillna(0)
    d["jp1_enc"] = (_safe_int(d.get("jp1_cnt", 0)) > 0).astype(int)

    feat = pd.DataFrame(index=d.index)

    # A. Biet truoc (lich) — no shift
    feat["thu_enc"] = d["thu_enc"]

    # B. Ket qua ky truoc (shift 1)
    prev_cols = [
        "dd_enc", "sc", "sum", "ke",
        "cl6_b0","cl6_b1","cl6_b2","cl6_b3","cl6_b4","cl6_b5",
        "cl6_l_count","cl6_pattern","cl6_changed",
        "dec0","dec1","dec2","dec3","dec4","dec5",
        "jpck","jp1_enc",
    ]
    prev_cols = [c for c in prev_cols if c in d.columns]
    for c in prev_cols:
        feat[f"{c}_prev"] = d[c].shift(1)

    # C. Lag 2 → lags
    lag_cols = ["dd_enc","sc","cl6_l_count","cl6_pattern","sum"]
    lag_cols = [c for c in lag_cols if c in d.columns]
    for lag in range(2, lags + 1):
        for c in lag_cols:
            feat[f"{c}_lag{lag}"] = d[c].shift(lag)

    # D. Rolling (window 3, 5)
    roll_cols = ["sc","cl6_l_count","sum"]
    roll_cols = [c for c in roll_cols if c in d.columns]
    for w in [3, 5]:
        for c in roll_cols:
            base = d[c].shift(1)
            feat[f"{c}_roll{w}_mean"] = base.rolling(w).mean()
            feat[f"{c}_roll{w}_std"]  = base.rolling(w).std()

    # E. Mega features (no shift — da la ky TRUOC P655)
    if "mgn1" in d.columns:
        mg_nums = [_safe_int(d[f"mgn{i}"]) for i in range(1,7)]
        feat["mg_thu_enc"]   = _encode_thu(d["mgthu"])
        feat["mg_ngay_diff"] = _ngay_diff(d["ngay"], d["mgngay"])
        feat["mg_dd"]        = _encode_dd(mg_nums[0], mg_nums[5])
        feat["mg_sc"]        = sum(1 for n in mg_nums if (n % 2 == 0).any())  # vectorized below
        feat["mg_sc"]        = sum((n % 2 == 0).astype(int) for n in mg_nums)

    # F. Cross features (Mega only)
    if "mg_dd" in feat.columns and "dd_enc_prev" in feat.columns:
        feat["cross_mg_dd_match"] = (feat["mg_dd"] == feat["dd_enc_prev"]).astype(float)
    if "mg_sc" in feat.columns and "sc_prev" in feat.columns:
        feat["cross_sc_mg"] = feat["sc_prev"] * feat["mg_sc"]

    return feat


def build_num_features(df: pd.DataFrame, n16_arr: np.ndarray) -> dict:
    """
    Tinh cac features RIENG THEO SO K cho moi ky.
    Tra ve dict: so_k -> pd.DataFrame (index = ky index).
    Tat ca duoc tinh tu lich su KY HIEN TAI nhin ve QUA KHU (no leak).

    n16_arr: (n_ky, 6) so chinh da duoc parse.
    """
    n_ky = len(df)
    qhl_col = df["qhl"].astype(str) if "qhl" in df.columns else None

    # Pre-compute: voi moi ky i, tap so chinh = set(n16_arr[i])
    sets = [set(row[~np.isnan(row)].astype(int)) for row in n16_arr]

    num_feats = {}
    for k in ALL_NUMS:
        rows = []
        last_seen = -1   # index ky cuoi cung so k xuat hien
        freq10 = freq20 = freq50 = 0

        for i in range(n_ky):
            if k in sets[i]:
                last_seen = i

            # ky_vang: so ky chua thay so k (tinh tai thoi diem i, nhin qua khu)
            ky_vang = (i - last_seen) if last_seen >= 0 else i + 1

            # freq trong cua so truot (nhin ve qua khu, khong tinh i)
            start10 = max(0, i - 10)
            start20 = max(0, i - 20)
            start50 = max(0, i - 50)
            freq10  = sum(1 for j in range(start10, i) if k in sets[j])
            freq20  = sum(1 for j in range(start20, i) if k in sets[j])
            freq50  = sum(1 for j in range(start50, i) if k in sets[j])

            # is_qhl: so k co trong danh sach qua han cua ky i khong
            is_qhl = 0
            if qhl_col is not None:
                qhl_str = qhl_col.iloc[i]
                if qhl_str and qhl_str != "nan":
                    try:
                        qhl_nums = set(int(x) for x in qhl_str.split(",") if x.strip())
                        is_qhl = int(k in qhl_nums)
                    except Exception:
                        pass

            rows.append({
                "freq_last_10" : freq10,
                "freq_last_20" : freq20,
                "freq_last_50" : freq50,
                "ky_vang"      : ky_vang,
                "is_qhl"       : is_qhl,
                "dec_of_k"     : k // 10,
                "is_prime_k"   : int(k in PRIMES),
                "is_even_k"    : int(k % 2 == 0),
                "k"            : k,
            })

        num_feats[k] = pd.DataFrame(rows, index=df.index)

    return num_feats


# ── build stacked dataset ─────────────────────────────────────────────────────

def build_stacked(df: pd.DataFrame, ky_feat: pd.DataFrame,
                  num_feats: dict, target: str = "n16") -> tuple:
    """
    Xay dung stacked dataset:
      - Moi hang = 1 cap (ky_i, so_k)
      - Label = 1 neu so_k xuat hien o ky i+1

    target: 'n16' hoac 'n7'
    Tra ve (X_df, y, ky_series, k_series)
    """
    n16_arr = _get_n16(df)
    n7_arr  = _get_n7(df)
    n_ky    = len(df)

    all_rows   = []
    all_labels = []
    all_ky     = []
    all_k      = []

    feat_cols = list(ky_feat.columns)

    for i in range(n_ky - 1):   # i+1 la label → bo ky cuoi
        ky_row = ky_feat.iloc[i]
        if ky_row.isna().all():
            continue

        # Label cua ky i = ket qua ky i+1
        if target == "n16":
            next_set = set(n16_arr[i+1][~np.isnan(n16_arr[i+1])].astype(int))
        else:
            nxt = n7_arr[i+1]
            next_set = {int(nxt)} if not np.isnan(nxt) and nxt > 0 else set()

        for k in ALL_NUMS:
            nf  = num_feats[k].iloc[i]
            row = list(ky_row.values) + list(nf.drop("k").values)
            all_rows.append(row)
            all_labels.append(int(k in next_set))
            all_ky.append(df["ky"].iloc[i])
            all_k.append(k)

    col_names = feat_cols + [c for c in num_feats[1].columns if c != "k"]
    X_df = pd.DataFrame(all_rows, columns=col_names)
    y    = np.array(all_labels)
    return X_df, y, pd.Series(all_ky), pd.Series(all_k)


# ── walk-forward splits ───────────────────────────────────────────────────────

def _walk_forward_splits(n_ky: int, min_train: int, val_size: int):
    """
    Tao expanding window splits theo ky (khong phai theo hang stacked).
    Tra ve list (train_ky_idx, val_ky_idx).
    """
    splits = []
    start  = min_train
    while start + val_size <= n_ky:
        splits.append((np.arange(0, start), np.arange(start, start + val_size)))
        start += val_size
    return splits


# ── XGB builder ───────────────────────────────────────────────────────────────

def make_xgb(params: dict = None) -> XGBClassifier:
    defaults = dict(
        n_estimators          = 500,
        max_depth             = 3,
        learning_rate         = 0.05,
        subsample             = 0.8,
        colsample_bytree      = 0.7,
        min_child_weight      = 20,
        gamma                 = 1.0,
        reg_alpha             = 0.1,
        reg_lambda            = 2.0,
        objective             = "binary:logistic",
        eval_metric           = "auc",
        random_state          = 42,
        n_jobs                = -1,
        early_stopping_rounds = 30,
    )
    if params:
        defaults.update(params)
    return XGBClassifier(**defaults)


# ── train ─────────────────────────────────────────────────────────────────────

def train_model(df: pd.DataFrame, X_df: pd.DataFrame, y: np.ndarray,
                ky_series: pd.Series, k_series: pd.Series,
                n_ky: int, min_train: int = 200, val_size: int = 20,
                xgb_params: dict = None, label: str = "N16") -> tuple:
    """
    Walk-forward validation tren stacked data.
    Split theo ky (khong phai theo hang) de giu time-series integrity.
    """
    print(f"\n{'─'*60}")
    print(f"🚀 Training Model {label}")
    print(f"{'─'*60}")

    ky_list    = df["ky"].values          # (n_ky,)
    ky_to_idx  = {k: i for i, k in enumerate(ky_list)}

    # Map moi hang stacked → ky index
    row_ky_idx = ky_series.map(ky_to_idx).values

    splits  = _walk_forward_splits(n_ky - 1, min_train, val_size)
    pos_rate = y.mean()
    print(f"  Stacked rows : {len(X_df):,}  ({n_ky-1} ky × 55 so)")
    print(f"  Positive rate: {pos_rate:.4f}  "
          f"(ly thuyet: {'6/55=10.9%' if label=='N16' else '1/55=1.8%'})")
    print(f"  Walk-forward : min_train={min_train}, val_size={val_size}"
          f" → {len(splits)} folds")

    X_arr = X_df.values
    auc_list = []
    hit_list = []   # so hit trung binh moi fold (top8 hoac top3)
    top_n = 8 if label == "N16" else 3

    for fold, (tr_ky, va_ky) in enumerate(splits, 1):
        tr_mask = np.isin(row_ky_idx, tr_ky)
        va_mask = np.isin(row_ky_idx, va_ky)

        X_tr, y_tr = X_arr[tr_mask], y[tr_mask]
        X_va, y_va = X_arr[va_mask], y[va_mask]
        k_va       = k_series.values[va_mask]
        ky_va      = ky_series.values[va_mask]

        if len(np.unique(y_tr)) < 2 or len(X_va) == 0:
            continue

        sw = compute_sample_weight("balanced", y_tr)
        mdl = make_xgb(xgb_params)
        mdl.fit(
            X_tr, y_tr,
            sample_weight=sw,
            eval_set=[(X_va, y_va)],
            verbose=False,
        )

        proba = mdl.predict_proba(X_va)[:, 1]

        # AUC
        if len(np.unique(y_va)) == 2:
            auc = roc_auc_score(y_va, proba)
            auc_list.append(auc)
        else:
            auc = float("nan")

        # Hit rate: voi moi ky trong val, lay top_n so va dem so trung
        hits = []
        for ky_val in np.unique(ky_va):
            mask_ky = ky_va == ky_val
            proba_ky = proba[mask_ky]
            y_ky     = y_va[mask_ky]
            top_idx  = np.argsort(proba_ky)[::-1][:top_n]
            hit      = y_ky[top_idx].sum()
            hits.append(hit)
        mean_hit = np.mean(hits) if hits else 0
        hit_list.append(mean_hit)

        best_it = getattr(mdl, "best_iteration", "-")
        if fold % 5 == 0 or fold <= 3 or fold == len(splits):
            print(
                f"  Fold {fold:>3}: AUC={auc:.3f}  "
                f"hit/ky={mean_hit:.2f}/{top_n}  "
                f"best_iter={best_it}  "
                f"train_ky={len(tr_ky)}  val_ky={len(va_ky)}"
            )

    mean_auc = float(np.nanmean(auc_list)) if auc_list else 0
    mean_hit = float(np.mean(hit_list))    if hit_list else 0
    random_hit = 6/55*top_n if label=="N16" else 1/55*top_n
    lift_hit = mean_hit - random_hit

    flag = "✅ tot" if lift_hit > 0.3 else ("🟡 kha" if lift_hit > 0.1 else "⚠️ yeu")
    print(f"\n  Mean AUC       : {mean_auc:.3f}")
    print(f"  Mean hit/ky    : {mean_hit:.3f}/{top_n}  "
          f"(random={random_hit:.2f}, lift={lift_hit:+.3f})  {flag}")

    # Final model tren toan bo data
    print(f"\n  Fitting final model tren {len(X_df):,} rows...")
    final_p = dict(xgb_params or {})
    final_p["early_stopping_rounds"] = None
    final = make_xgb(final_p)
    sw_all = compute_sample_weight("balanced", y)
    final.fit(X_arr, y, sample_weight=sw_all, verbose=False)

    return final, mean_auc, mean_hit


# ── feature importance ────────────────────────────────────────────────────────

def show_importance(model: XGBClassifier, feature_cols: list,
                    label: str = "N16", top_n: int = 20):
    imp = pd.Series(model.feature_importances_, index=feature_cols)
    top = imp.sort_values(ascending=False).head(top_n)
    print(f"\n  Top {top_n} features [{label}]:")
    for f, s in top.items():
        bar = "█" * int(s * 300)
        print(f"    {f:<42} {s:.4f}  {bar}")

    groups = {
        "Mega"  : [c for c in imp.index if c.startswith("mg_")],
        "Cross" : [c for c in imp.index if c.startswith("cross_")],
        "NumK"  : ["freq_last_10","freq_last_20","freq_last_50",
                   "ky_vang","is_qhl","dec_of_k","is_prime_k","is_even_k"],
        "JP"    : [c for c in imp.index if "jp" in c],
    }
    print(f"\n  Importance theo nhom [{label}]:")
    total_grp = 0
    for grp, cols in groups.items():
        s = imp[[c for c in cols if c in imp.index]].sum()
        total_grp += s
        print(f"    {grp:<8}: {s:.4f}")
    print(f"    {'P655':<8}: {1 - total_grp:.4f}")


# ── optuna tuning ─────────────────────────────────────────────────────────────

def optuna_tune(X_df: pd.DataFrame, y: np.ndarray,
                ky_series: pd.Series, df: pd.DataFrame,
                min_train: int, val_size: int,
                n_trials: int = 50, label: str = "N16") -> dict:
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        print("  ⚠️  Optuna chua cai: pip install optuna")
        return {}

    ky_list   = df["ky"].values
    ky_to_idx = {k: i for i, k in enumerate(ky_list)}
    row_ky_idx= ky_series.map(ky_to_idx).values
    splits    = _walk_forward_splits(len(df) - 1, min_train, val_size)
    X_arr     = X_df.values

    def objective(trial):
        params = dict(
            max_depth             = trial.suggest_int("max_depth", 2, 5),
            learning_rate         = trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            n_estimators          = trial.suggest_int("n_estimators", 200, 800),
            subsample             = trial.suggest_float("subsample", 0.5, 0.9),
            colsample_bytree      = trial.suggest_float("colsample_bytree", 0.4, 0.9),
            min_child_weight      = trial.suggest_int("min_child_weight", 10, 50),
            gamma                 = trial.suggest_float("gamma", 0.5, 5.0),
            reg_alpha             = trial.suggest_float("reg_alpha", 0.0, 1.0),
            reg_lambda            = trial.suggest_float("reg_lambda", 0.5, 5.0),
            early_stopping_rounds = 30,
        )
        auc_list = []
        for tr_ky, va_ky in splits:
            tr_mask = np.isin(row_ky_idx, tr_ky)
            va_mask = np.isin(row_ky_idx, va_ky)
            X_tr, y_tr = X_arr[tr_mask], y[tr_mask]
            X_va, y_va = X_arr[va_mask], y[va_mask]
            if len(np.unique(y_tr)) < 2 or len(np.unique(y_va)) < 2:
                continue
            sw  = compute_sample_weight("balanced", y_tr)
            mdl = make_xgb(params)
            mdl.fit(X_tr, y_tr, sample_weight=sw,
                    eval_set=[(X_va, y_va)], verbose=False)
            auc_list.append(roc_auc_score(y_va, mdl.predict_proba(X_va)[:,1]))
        return np.mean(auc_list) if auc_list else 0.0

    print(f"\n🔍 Optuna tuning [{label}] ({n_trials} trials)...")
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    print(f"  Best AUC: {study.best_value:.3f}")
    for k, v in study.best_params.items():
        print(f"    {k:<25} = {v}")
    return study.best_params


# ── predict ───────────────────────────────────────────────────────────────────

def predict_next(df: pd.DataFrame, ky_feat: pd.DataFrame,
                 num_feats: dict, model_n16: XGBClassifier,
                 model_n7: XGBClassifier, top8: int = 8, top3: int = 3):
    """
    Du doan cho ky tiep theo (dua tren ky CUOI CUNG trong df).
    """
    last_ky  = df["ky"].iloc[-1]
    last_idx = len(df) - 1

    feat_cols_ky  = list(ky_feat.columns)
    feat_cols_num = [c for c in num_feats[1].columns if c != "k"]

    ky_row = ky_feat.iloc[-1]

    rows_n16 = []
    rows_n7  = []
    for k in ALL_NUMS:
        nf  = num_feats[k].iloc[last_idx]
        row = list(ky_row.values) + list(nf.drop("k").values)
        rows_n16.append(row)
        rows_n7.append(row)

    col_names = feat_cols_ky + feat_cols_num
    X_n16 = pd.DataFrame(rows_n16, columns=col_names)
    X_n7  = pd.DataFrame(rows_n7,  columns=col_names)

    prob_n16 = model_n16.predict_proba(X_n16.values)[:, 1]
    prob_n7  = model_n7.predict_proba(X_n7.values)[:,  1]

    top8_idx  = np.argsort(prob_n16)[::-1][:top8]
    top3_idx  = np.argsort(prob_n7)[::-1][:top3]

    top8_nums = [(ALL_NUMS[i], prob_n16[i]) for i in top8_idx]
    top3_nums = [(ALL_NUMS[i], prob_n7[i])  for i in top3_idx]

    print(f"\n{'═'*55}")
    print(f"🎯 Du doan cho ky sau ky [{last_ky}]")
    print(f"{'═'*55}")

    print(f"\n  📌 6 so chinh (N16) — Top {top8}:")
    for rank, (num, prob) in enumerate(top8_nums, 1):
        bar  = "█" * int(prob * 200)
        flag = "🔥" if prob >= 0.15 else ("⚡" if prob >= 0.10 else " ")
        print(f"    #{rank}  So {num:>2}  ({prob:.1%})  {flag}  {bar}")

    print(f"\n  🎰 So bonus (N7)   — Top {top3}:")
    for rank, (num, prob) in enumerate(top3_nums, 1):
        bar  = "█" * int(prob * 400)
        flag = "🔥" if prob >= 0.05 else " "
        print(f"    #{rank}  So {num:>2}  ({prob:.1%})  {flag}  {bar}")

    print(f"\n{'═'*55}\n")
    return top8_nums, top3_nums


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="XGBoost predict P655 — 55 so")
    parser.add_argument("--source",      choices=["db","csv"], default="db")
    parser.add_argument("--file",        default="p655kqdetail.csv")
    parser.add_argument("--n_ky",        type=int, default=0)
    parser.add_argument("--qh_ky",       type=int, default=20)
    parser.add_argument("--lags",        type=int, default=3)
    parser.add_argument("--val_size",    type=int, default=20)
    parser.add_argument("--min_train",   type=int, default=200)
    parser.add_argument("--importance",  action="store_true")
    parser.add_argument("--tune",        action="store_true")
    parser.add_argument("--tune_trials", type=int, default=50)
    parser.add_argument("--top8",        type=int, default=8)
    parser.add_argument("--top3",        type=int, default=3)
    args = parser.parse_args()

    # 1. Load
    print("📥 Loading data...")
    df = (load_from_csv(args.file) if args.source == "csv"
          else load_from_db(args.n_ky, args.qh_ky))
    print(f"   {len(df)} ky  |  cols: {len(df.columns)}")
    print(f"   Mega  : {'✅' if 'mgn1' in df.columns else '⚠️'}")

    print(f"   jpck  : {'✅' if 'jpck' in df.columns else '⚠️'}")

    # 2. Parse arrays
    n16_arr = _get_n16(df)
    n7_arr  = _get_n7(df)

    # 3. Feature engineering
    print("\n⚙️  Feature engineering...")
    ky_feat  = build_ky_features(df, lags=args.lags)
    print(f"   ky_feat  : {ky_feat.shape[1]} cols")

    print("   Building num_feats (55 so × n_ky) — co the mat vai giay...")
    num_feats = build_num_features(df, n16_arr)
    print(f"   num_feats: {len(num_feats[1].columns)-1} cols/so")

    # 4. Build stacked datasets
    print("\n📦 Building stacked datasets...")
    X_n16, y_n16, ky_n16, k_n16 = build_stacked(df, ky_feat, num_feats, target="n16")
    X_n7,  y_n7,  ky_n7,  k_n7  = build_stacked(df, ky_feat, num_feats, target="n7")
    feat_cols = list(X_n16.columns)
    print(f"   N16: {len(X_n16):,} rows  pos={y_n16.mean():.4f}")
    print(f"   N7 : {len(X_n7):,} rows  pos={y_n7.mean():.4f}")

    # 5. Optuna (optional)
    params_n16 = params_n7 = None
    if args.tune:
        params_n16 = optuna_tune(X_n16, y_n16, ky_n16, df,
                                 args.min_train, args.val_size,
                                 args.tune_trials, "N16")
        params_n7  = optuna_tune(X_n7,  y_n7,  ky_n7,  df,
                                 args.min_train, args.val_size,
                                 args.tune_trials, "N7")

    # 6. Train
    model_n16, auc_n16, hit_n16 = train_model(
        df, X_n16, y_n16, ky_n16, k_n16,
        n_ky=len(df), min_train=args.min_train, val_size=args.val_size,
        xgb_params=params_n16, label="N16"
    )
    model_n7, auc_n7, hit_n7 = train_model(
        df, X_n7, y_n7, ky_n7, k_n7,
        n_ky=len(df), min_train=args.min_train, val_size=args.val_size,
        xgb_params=params_n7, label="N7"
    )

    # 7. Feature importance
    if args.importance:
        show_importance(model_n16, feat_cols, "N16")
        show_importance(model_n7,  feat_cols, "N7")

    # 8. Predict
    predict_next(df, ky_feat, num_feats,
                 model_n16, model_n7,
                 top8=args.top8, top3=args.top3)

    # 9. Save models
    try:
        import joblib
        joblib.dump({
            "model_n16" : model_n16,
            "model_n7"  : model_n7,
            "feat_cols" : feat_cols,
            "auc_n16"   : auc_n16,
            "auc_n7"    : auc_n7,
            "hit_n16"   : hit_n16,
            "hit_n7"    : hit_n7,
            "lags"      : args.lags,
        }, "model_p655.pkl")
        print("💾 Luu model: model_p655.pkl")
    except ImportError:
        pass


if __name__ == "__main__":
    main()