"""
xgboost_dd.py
─────────────
Du doan cot DD: CC / CL / LC / LL (4 lop)
  DD = chan/le cua n1 (so dau) va n5 (so cuoi) cua L535
  C = Chan, L = Le

Khong data leak: tat ca features L535 deu shift >= 1.
Features Mega/Power: lay ky gan nhat TRUOC ky L535 (khong can shift).

Cai tien:
  1. Class weights    — can bang 4 lop khong deu (LL 34%, CL 19%)
  2. Early stopping   — tranh overfit tung fold
  3. Cross features   — tuong tac Mega/Power x L535
  4. jpck / is_jp     — ky jackpot co the co pattern rieng
  5. Optuna tuning    — tim hyperparameter tu dong (--tune)

Usage:
    python xgboost_dd.py --source db --lags 3 --importance
    python xgboost_dd.py --source db --lags 3 --tune --tune_trials 50
    python xgboost_dd.py --file data.csv --lags 3 --importance
"""

import argparse, os, warnings
import numpy as np
import pandas as pd
from collections import Counter

from xgboost import XGBClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight
from scipy.stats import mode

warnings.filterwarnings("ignore")

DATABASE_URL = os.environ.get(
    "DATABASE_URL", "postgresql://postgres:bin@localhost:5432/katalott"
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
# 2. HELPERS MEGA / POWER
# ─────────────────────────────────────────────

THU_MAP = {"T2": 2, "T3": 3, "T4": 4, "T5": 5, "T6": 6, "T7": 7, "CN": 8}
DD_MAP  = {"CC": 0, "CL": 1, "LC": 2, "LL": 3}

def _safe_int(series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")

def _encode_dd(n_first, n_last) -> pd.Series:
    """Tinh DD encode (CC=0, CL=1, LC=2, LL=3) tu 2 series so."""
    def _cl(n):
        return "C" if n % 2 == 0 else "L"
    result = []
    for f, l in zip(n_first, n_last):
        if pd.isna(f) or pd.isna(l):
            result.append(np.nan)
        else:
            result.append(DD_MAP.get(_cl(int(f)) + _cl(int(l)), np.nan))
    return pd.Series(result, index=n_first.index)

def _encode_thu(series) -> pd.Series:
    return series.astype(str).str.strip().map(THU_MAP)

def _ngay_diff(df_ngay: pd.Series, ref_ngay: pd.Series) -> pd.Series:
    d1 = pd.to_datetime(ref_ngay, errors="coerce")
    d2 = pd.to_datetime(df_ngay,  errors="coerce")
    return (d1 - d2).dt.days

def _compute_mega_features(df: pd.DataFrame) -> pd.DataFrame:
    feat = pd.DataFrame(index=df.index)
    if not all(c in df.columns for c in ["mgn1", "mgn6"]):
        return feat
    nums = [_safe_int(df[c]) for c in [f"mgn{i}" for i in range(1, 7)]]
    if "mgthu" in df.columns:
        feat["mg_thu_enc"]   = _encode_thu(df["mgthu"])
    if "mgngay" in df.columns and "ngay" in df.columns:
        feat["mg_ngay_diff"] = _ngay_diff(df["mgngay"], df["ngay"])
    feat["mg_dd"] = _encode_dd(nums[0], nums[5])
    return feat

def _compute_power_features(df: pd.DataFrame) -> pd.DataFrame:
    feat = pd.DataFrame(index=df.index)
    if not all(c in df.columns for c in ["pwn1", "pwn6"]):
        return feat
    nums = [_safe_int(df[c]) for c in [f"pwn{i}" for i in range(1, 7)]]
    if "pwthu" in df.columns:
        feat["pw_thu_enc"]   = _encode_thu(df["pwthu"])
    if "pwngay" in df.columns and "ngay" in df.columns:
        feat["pw_ngay_diff"] = _ngay_diff(df["pwngay"], df["ngay"])
    feat["pw_dd"] = _encode_dd(nums[0], nums[5])
    return feat


# ─────────────────────────────────────────────
# 3. FEATURES (no-leak)
# ─────────────────────────────────────────────

def make_features(df: pd.DataFrame, lags: int = 3) -> tuple:
    df = df.copy()

    # ── Encode DD ──
    df["dd_enc"] = df["dd"].apply(
        lambda v: {"CC": 0, "CL": 1, "LC": 2, "LL": 3}.get(str(v).strip(), -1)
    )

    # ── Encode CL5 bitwise ──
    s = df["cl5"].astype(str).str.strip()
    for i in range(5):
        df[f"cl5_b{i}"] = s.str[i].map({"C": 0, "L": 1}).fillna(-1).astype(int)

    # ── CL5 aggregate & pattern ──
    df["cl5_pattern"] = s.apply(
        lambda v: int("".join("0" if c == "C" else "1" for c in v), 2)
        if len(v) == 5 else -1
    )
    df["cl5_l_count"] = sum(df[f"cl5_b{i}"] for i in range(5))
    df["cl5_changed"] = (df["cl5_pattern"] != df["cl5_pattern"].shift(1)).astype(int)

    # ── Encode thu / dot ──
    df["thu_enc"] = df["thu"].apply(
        lambda v: {"T2": 2, "T3": 3, "T4": 4, "T5": 5,
                   "T6": 6, "T7": 7, "CN": 8}.get(str(v).strip(), -1)
    )
    df["dot"] = pd.to_numeric(df["dot"], errors="coerce").fillna(0).astype(int)

    # ── 4. jackpot features ──
    if "jpck" in df.columns:
        df["jpck"] = pd.to_numeric(df["jpck"], errors="coerce").fillna(0)
    if "is_jp" in df.columns:
        df["is_jp_enc"] = df["is_jp"].astype(int)

    # ================================================================
    feat = pd.DataFrame(index=df.index)

    # A. Biet truoc (lich) — no shift
    feat["thu_enc"] = df["thu_enc"]
    feat["dot"]     = df["dot"]

    # B. Ket qua ky truoc (shift 1)
    prev_cols = [
        "dd_enc", "sc", "sum",
        "cl5_b0", "cl5_b1", "cl5_b2", "cl5_b3", "cl5_b4",
        "cl5_l_count", "cl5_pattern", "cl5_changed",
        "dec0", "dec3",
    ]
    for c in ["jpck", "is_jp_enc"]:          # 4. them jackpot
        if c in df.columns:
            prev_cols.append(c)
    prev_cols = [c for c in prev_cols if c in df.columns]
    for col in prev_cols:
        feat[f"{col}_prev"] = df[col].shift(1)

    # C. Lag 2 → lags
    lag_cols = ["dd_enc", "sc", "cl5_l_count", "cl5_pattern"]
    lag_cols = [c for c in lag_cols if c in df.columns]
    for lag in range(2, lags + 1):
        for col in lag_cols:
            feat[f"{col}_lag{lag}"] = df[col].shift(lag)

    # D. Rolling (window 3, 5)
    roll_cols = ["sc", "cl5_l_count"]
    roll_cols = [c for c in roll_cols if c in df.columns]
    for w in [3, 5]:
        for col in roll_cols:
            base = df[col].shift(1)
            feat[f"{col}_roll{w}_mean"] = base.rolling(w).mean()
            feat[f"{col}_roll{w}_std"]  = base.rolling(w).std()

    # E. Mega / Power features (no shift)
    mg_feat = _compute_mega_features(df)
    pw_feat = _compute_power_features(df)
    feat = pd.concat([feat, mg_feat, pw_feat], axis=1)

    # F. 3. Cross features — tuong tac Mega/Power x L535
    if "mg_dd" in feat.columns and "dd_enc_prev" in feat.columns:
        # DD Mega co khop voi DD L535 ky truoc khong?
        feat["cross_mg_dd_match"] = (
            feat["mg_dd"] == feat["dd_enc_prev"]
        ).astype(float)

    if "pw_dd" in feat.columns and "dd_enc_prev" in feat.columns:
        # DD Power co khop voi DD L535 ky truoc khong?
        feat["cross_pw_dd_match"] = (
            feat["pw_dd"] == feat["dd_enc_prev"]
        ).astype(float)

    if "mg_ngay_diff" in feat.columns:
        # dot 2 cung ngay (diff=0) vs dot 1 khac ngay
        feat["cross_dot_mg_diff"] = feat["dot"] * feat["mg_ngay_diff"]

    if "pw_ngay_diff" in feat.columns:
        feat["cross_dot_pw_diff"] = feat["dot"] * feat["pw_ngay_diff"]

    if "mg_dd" in feat.columns and "sc_prev" in feat.columns:
        feat["cross_sc_mg_dd"] = feat["sc_prev"] * feat["mg_dd"]

    if "pw_dd" in feat.columns and "sc_prev" in feat.columns:
        feat["cross_sc_pw_dd"] = feat["sc_prev"] * feat["pw_dd"]

    # Target
    target = df["dd"].apply(lambda v: str(v).strip())
    out = (
        pd.concat([feat, target.rename("dd"), df["ky"]], axis=1)
        .dropna()
        .reset_index(drop=True)
    )
    return out, list(feat.columns)


# ─────────────────────────────────────────────
# 4. XGB BUILDER
# ─────────────────────────────────────────────

def make_xgb(n_cls, params: dict = None):
    defaults = dict(
        n_estimators        = 600,
        max_depth           = 2,
        learning_rate       = 0.03,
        subsample           = 0.7,
        colsample_bytree    = 0.6,
        min_child_weight    = 10,
        gamma               = 2,
        reg_alpha           = 0.1,
        reg_lambda          = 2.0,
        objective           = "multi:softprob",
        eval_metric         = "mlogloss",
        num_class           = n_cls,
        random_state        = 42,
        n_jobs              = -1,
        early_stopping_rounds = 40,   # 2. early stopping
    )
    if params:
        defaults.update(params)
        defaults["num_class"] = n_cls   # dam bao khong bi ghi de
    return XGBClassifier(**defaults)


# ─────────────────────────────────────────────
# 5. TRAIN
# ─────────────────────────────────────────────

def train(df, feature_cols, n_splits=5, xgb_params=None):
    X     = df[feature_cols].values
    y_raw = df["dd"].values

    le        = LabelEncoder()
    y         = le.fit_transform(y_raw)
    classes   = le.classes_
    n_classes = len(classes)

    maj_enc  = mode(y, keepdims=True).mode[0]
    baseline = np.mean(y == maj_enc)
    maj_lbl  = le.inverse_transform([maj_enc])[0]

    print(f"\n  TARGET: DD  |  {n_classes} lop: {list(classes)}")
    print(
        "  Phan phoi: "
        + "  ".join(
            f"{c}:{v}" for c, v in sorted(Counter(y_raw).items(), key=lambda x: -x[1])
        )
    )
    print(f"  Baseline (majority='{maj_lbl}'): {baseline:.3f}")

    tscv     = TimeSeriesSplit(n_splits=n_splits)
    acc_list = []

    for fold, (tr_idx, va_idx) in enumerate(tscv.split(X), 1):
        X_tr, X_va = X[tr_idx], X[va_idx]
        y_tr, y_va = y[tr_idx], y[va_idx]

        # 1. Class weights — can bang theo tan suat tung lop trong fold
        sw_tr = compute_sample_weight("balanced", y_tr)

        fold_cls = sorted(set(y_tr))
        remap    = {old: new for new, old in enumerate(fold_cls)}
        y_tr_r   = np.array([remap[v] for v in y_tr])
        mask     = np.array([v in remap for v in y_va])
        y_va_r   = np.array([remap[v] for v in y_va if v in remap])
        X_va_r   = X_va[mask]

        mdl = make_xgb(len(fold_cls), xgb_params)
        # 2. Early stopping tren validation set
        mdl.fit(
            X_tr, y_tr_r,
            sample_weight=sw_tr,
            eval_set=[(X_va_r, y_va_r)],
            verbose=False,
        )
        acc      = accuracy_score(y_va_r, mdl.predict(X_va_r))
        best_it  = getattr(mdl, "best_iteration", "-")
        acc_list.append(acc)
        print(
            f"    Fold {fold}: acc={acc:.3f}  best_iter={best_it}"
            f"  (val={len(y_va_r)}, "
            f"ky {df['ky'].iloc[va_idx[0]]}→{df['ky'].iloc[va_idx[-1]]})"
        )

    mean_acc = np.mean(acc_list)
    lift     = mean_acc - baseline
    flag     = "✅ tot" if lift > 0.05 else ("🟡 kha" if lift > 0.01 else "⚠️ yeu")
    print(
        f"\n  CV acc = {mean_acc:.3f} ± {np.std(acc_list):.3f}  "
        f"| lift = {lift:+.3f}  {flag}"
    )

    # Final model tren toan bo data (tat early stopping)
    final_p = dict(xgb_params or {})
    final_p["early_stopping_rounds"] = None
    final   = make_xgb(n_classes, final_p)
    sw_all  = compute_sample_weight("balanced", y)
    final.fit(X, y, sample_weight=sw_all, verbose=False)

    # Report 20% cuoi
    sp       = int(len(X) * 0.8)
    pred_lbl = le.inverse_transform(final.predict(X[sp:]))
    true_lbl = y_raw[sp:]
    print(f"\n  Classification report (20% cuoi = {len(true_lbl)} ky):")
    print(
        classification_report(
            true_lbl, pred_lbl,
            labels=sorted(set(true_lbl)), zero_division=0, digits=3
        )
    )
    return final, le, mean_acc


# ─────────────────────────────────────────────
# 6. OPTUNA TUNING
# ─────────────────────────────────────────────

def optuna_tune(df, feature_cols, n_splits=5, n_trials=50):
    """5. Tim hyperparameter tot nhat bang Optuna."""
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        print("  ⚠️  Optuna chua cai: pip install optuna")
        return None

    X     = df[feature_cols].values
    y_raw = df["dd"].values
    le    = LabelEncoder()
    y     = le.fit_transform(y_raw)
    tscv  = TimeSeriesSplit(n_splits=n_splits)

    def objective(trial):
        params = dict(
            max_depth        = trial.suggest_int("max_depth", 2, 4),
            learning_rate    = trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            n_estimators     = trial.suggest_int("n_estimators", 200, 800),
            subsample        = trial.suggest_float("subsample", 0.5, 0.9),
            colsample_bytree = trial.suggest_float("colsample_bytree", 0.4, 0.9),
            min_child_weight = trial.suggest_int("min_child_weight", 5, 30),
            gamma            = trial.suggest_float("gamma", 0.5, 5.0),
            reg_alpha        = trial.suggest_float("reg_alpha", 0.0, 1.0),
            reg_lambda       = trial.suggest_float("reg_lambda", 0.5, 5.0),
            early_stopping_rounds = 30,
        )
        acc_list = []
        for tr_idx, va_idx in tscv.split(X):
            X_tr, X_va = X[tr_idx], X[va_idx]
            y_tr, y_va = y[tr_idx], y[va_idx]

            sw_tr    = compute_sample_weight("balanced", y_tr)
            fold_cls = sorted(set(y_tr))
            remap    = {old: new for new, old in enumerate(fold_cls)}
            y_tr_r   = np.array([remap[v] for v in y_tr])
            mask     = np.array([v in remap for v in y_va])
            y_va_r   = np.array([remap[v] for v in y_va if v in remap])
            X_va_r   = X_va[mask]

            mdl = make_xgb(len(fold_cls), params)
            mdl.fit(
                X_tr, y_tr_r,
                sample_weight=sw_tr,
                eval_set=[(X_va_r, y_va_r)],
                verbose=False,
            )
            acc_list.append(accuracy_score(y_va_r, mdl.predict(X_va_r)))
        return np.mean(acc_list)

    print(f"\n🔍 Optuna tuning ({n_trials} trials, {n_splits} folds)...")
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best = study.best_params
    print(f"\n  Best CV acc : {study.best_value:.3f}")
    print("  Best params :")
    for k, v in best.items():
        print(f"    {k:<25} = {v}")
    return best


# ─────────────────────────────────────────────
# 7. FEATURE IMPORTANCE
# ─────────────────────────────────────────────

def show_importance(model, feature_cols, top_n=20):
    imp = pd.Series(model.feature_importances_, index=feature_cols)
    top = imp.sort_values(ascending=False).head(top_n)
    print(f"\n  Top {top_n} features quan trong:")
    for f, s in top.items():
        bar = "█" * int(s * 400)
        print(f"    {f:<42} {s:.4f}  {bar}")

    # Tong importance theo nhom
    mg_imp = imp[[c for c in imp.index if c.startswith("mg_")]].sum()
    pw_imp = imp[[c for c in imp.index if c.startswith("pw_")]].sum()
    cx_imp = imp[[c for c in imp.index if c.startswith("cross_")]].sum()
    jp_imp = imp[[c for c in imp.index if "jp" in c]].sum()
    l5_imp = 1 - mg_imp - pw_imp - cx_imp - jp_imp
    print(f"\n  Importance theo nhom:")
    print(f"    Mega    : {mg_imp:.4f}")
    print(f"    Power   : {pw_imp:.4f}")
    print(f"    Cross   : {cx_imp:.4f}")
    print(f"    JP      : {jp_imp:.4f}")
    print(f"    L535    : {l5_imp:.4f}")


# ─────────────────────────────────────────────
# 8. PREDICT
# ─────────────────────────────────────────────

def predict_next(df, feature_cols, model, le):
    last_row = df[feature_cols].iloc[[-1]]
    last_ky  = df["ky"].iloc[-1]

    proba   = model.predict_proba(last_row)[0]
    classes = le.classes_
    top3    = np.argsort(proba)[::-1][:3]

    best_val  = classes[top3[0]]
    best_prob = proba[top3[0]]
    conf      = "✅" if best_prob >= 0.40 else ("🟡" if best_prob >= 0.28 else "⚪")

    print(f"\n{'═'*52}")
    print(f"🎯 Du doan DD cho ky sau ky {last_ky}:")
    print(f"{'═'*52}")
    for i, idx in enumerate(top3, 1):
        lbl  = classes[idx]
        prob = proba[idx]
        n1   = "Chan" if lbl[0] == "C" else "Le"
        n5   = "Chan" if lbl[1] == "C" else "Le"
        mark = "◀" if i == 1 else ""
        print(f"  #{i}  {lbl}  ({prob:.1%})   n1={n1}, n5={n5}  {mark}")
    print(f"\n  Ket qua du doan: {conf} DD = [{best_val}]  ({best_prob:.1%})")
    print(f"{'═'*52}\n")


# ─────────────────────────────────────────────
# 9. MAIN
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="XGBoost predict DD")
    parser.add_argument("--source",      choices=["db", "csv"], default="csv")
    parser.add_argument("--file",        default="l535kqdetail_dat.csv")
    parser.add_argument("--n_ky",        type=int,  default=0)
    parser.add_argument("--qh_ky",       type=int,  default=20)
    parser.add_argument("--lags",        type=int,  default=3)
    parser.add_argument("--splits",      type=int,  default=5)
    parser.add_argument("--importance",  action="store_true")
    parser.add_argument("--tune",        action="store_true",
                        help="Chay Optuna hyperparameter tuning truoc khi train")
    parser.add_argument("--tune_trials", type=int, default=50,
                        help="So trials Optuna (mac dinh: 50)")
    args = parser.parse_args()

    print("📥 Loading data...")
    df = (
        load_from_csv(args.file)
        if args.source == "csv"
        else load_from_db(args.n_ky, args.qh_ky)
    )
    print(f"   {len(df)} ky")
    print(f"   DD phan phoi: {df['dd'].value_counts().to_dict()}")

    has_mega  = "mgn1" in df.columns
    has_power = "pwn1" in df.columns
    has_jp    = "jpck" in df.columns
    print(f"   Mega  : {'✅' if has_mega  else '⚠️  khong co cot mg*'}")
    print(f"   Power : {'✅' if has_power else '⚠️  khong co cot pw*'}")
    print(f"   JP    : {'✅' if has_jp    else '⚠️  khong co cot jpck'}")

    print("\n⚙️  Feature engineering...")
    df_feat, feature_cols = make_features(df, lags=args.lags)
    print(f"   {len(df_feat)} rows  |  {len(feature_cols)} features")

    mg_feats = [c for c in feature_cols if c.startswith("mg_")]
    pw_feats = [c for c in feature_cols if c.startswith("pw_")]
    cx_feats = [c for c in feature_cols if c.startswith("cross_")]
    jp_feats = [c for c in feature_cols if "jp" in c]
    if mg_feats: print(f"   Mega   ({len(mg_feats)}): {mg_feats}")
    if pw_feats: print(f"   Power  ({len(pw_feats)}): {pw_feats}")
    if cx_feats: print(f"   Cross  ({len(cx_feats)}): {cx_feats}")
    if jp_feats: print(f"   JP     ({len(jp_feats)}): {jp_feats}")

    # Optuna tuning (neu co --tune)
    xgb_params = None
    if args.tune:
        xgb_params = optuna_tune(
            df_feat, feature_cols,
            n_splits=args.splits,
            n_trials=args.tune_trials,
        )

    print("\n🚀 Training...")
    model, le, cv_acc = train(
        df_feat, feature_cols,
        n_splits=args.splits,
        xgb_params=xgb_params,
    )

    if args.importance:
        show_importance(model, feature_cols)

    predict_next(df_feat, feature_cols, model, le)

    try:
        import joblib
        joblib.dump(
            {"model": model, "le": le, "features": feature_cols,
             "cv_acc": cv_acc, "lags": args.lags},
            "model_dd.pkl"
        )
        print("💾 Luu model: model_dd.pkl")
    except ImportError:
        pass


if __name__ == "__main__":
    main()