"""
xgboost_dd_sc.py
────────────────
Du doan 2 cot:
  dd  — chan/le cua n1 va n5: CC / CL / LC / LL  (4 lop)
  sc  — so luong so chan trong n1-n5: 0-5          (6 lop)

Khong data leak: tat ca features deu shift >= 1.

Usage:
    python xgboost_dd_sc.py --file data.csv
    python xgboost_dd_sc.py --file data.csv --lags 5 --importance
"""

import argparse, warnings
import numpy as np
import pandas as pd
from collections import Counter

from xgboost import XGBClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
from scipy.stats import mode

warnings.filterwarnings("ignore")

TARGETS = ["dd", "sc"]

# ─────────────────────────────────────────────
# 1. LOAD
# ─────────────────────────────────────────────

def load(filepath) -> pd.DataFrame:
    df = pd.read_csv(filepath)
    return df.sort_values("ky").reset_index(drop=True)

# ─────────────────────────────────────────────
# 2. FEATURES (no-leak)
# ─────────────────────────────────────────────

def make_features(df: pd.DataFrame, lags: int = 5) -> tuple:
    df = df.copy()

    # Encode dd → 0-3
    df["dd_enc"] = df["dd"].apply(
        lambda v: {"CC":0,"CL":1,"LC":2,"LL":3}.get(str(v).strip(), -1))

    # Encode cl5 → so chu C (0-5)
    df["cl5_enc"] = df["cl5"].apply(
        lambda v: str(v).strip().count("C") if len(str(v).strip())==5 else -1)

    # Encode thu
    df["thu_enc"] = df["thu"].apply(
        lambda v: {"2":2,"3":3,"4":4,"5":5,"6":6,"7":7,"CN":8}.get(str(v).strip(), -1))

    df["dot"] = pd.to_numeric(df["dot"], errors="coerce").fillna(0).astype(int)

    feat = pd.DataFrame(index=df.index)

    # A. Biet truoc (lich) — khong shift
    feat["thu_enc"] = df["thu_enc"]
    feat["dot"]     = df["dot"]

    # B. Tat ca ket qua ky truoc (shift 1 — khong leak)
    prev_cols = ["dd_enc", "sc", "ke", "sum", "cl5_enc",
                 "t1", "g0", "p1", "snt_cnt",
                 "dec0", "dec1", "dec2", "dec3"]
    prev_cols = [c for c in prev_cols if c in df.columns]
    for col in prev_cols:
        feat[f"{col}_prev"] = df[col].shift(1)

    # C. Lag 2 → lags
    lag_cols = ["dd_enc", "sc", "ke", "sum", "cl5_enc", "dec0", "dec1"]
    lag_cols = [c for c in lag_cols if c in df.columns]
    for lag in range(2, lags + 1):
        for col in lag_cols:
            feat[f"{col}_lag{lag}"] = df[col].shift(lag)

    # D. Rolling mean / std (window 3, 5) — shift(1) truoc rolling
    roll_cols = ["dd_enc", "sc", "sum", "cl5_enc"]
    roll_cols = [c for c in roll_cols if c in df.columns]
    for w in [3, 5]:
        for col in roll_cols:
            base = df[col].shift(1)
            feat[f"{col}_roll{w}_mean"] = base.rolling(w).mean()
            feat[f"{col}_roll{w}_std"]  = base.rolling(w).std()

    # Gop target
    targets_df = df[["dd", "sc", "ky"]].copy()
    targets_df["dd"] = df["dd"].apply(lambda v: str(v).strip())
    targets_df["sc"] = pd.to_numeric(df["sc"], errors="coerce").astype("Int64")

    out = pd.concat([feat, targets_df], axis=1).dropna().reset_index(drop=True)
    feature_cols = list(feat.columns)
    return out, feature_cols

# ─────────────────────────────────────────────
# 3. TRAIN
# ─────────────────────────────────────────────

def make_xgb(n_cls):
    return XGBClassifier(
        n_estimators=400,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.7,
        min_child_weight=5,
        gamma=1,
        objective="multi:softprob",
        eval_metric="mlogloss",
        num_class=n_cls,
        random_state=42,
        n_jobs=-1,
    )

def train_one(df, feature_cols, target_col, n_splits=5):
    X     = df[feature_cols].values
    y_raw = df[target_col].values

    le = LabelEncoder()
    y  = le.fit_transform(y_raw)
    classes   = le.classes_
    n_classes = len(classes)

    maj_enc  = mode(y, keepdims=True).mode[0]
    baseline = np.mean(y == maj_enc)
    maj_lbl  = le.inverse_transform([maj_enc])[0]

    print(f"\n{'─'*55}")
    print(f"  TARGET: {target_col.upper()}  |  {n_classes} lop  "
          f"|  baseline={baseline:.3f} ('{maj_lbl}')")
    print(f"  Phan phoi: "
          + "  ".join(f"{c}:{v}" for c, v in
                      sorted(Counter(y_raw).items(), key=lambda x: -x[1])))

    tscv     = TimeSeriesSplit(n_splits=n_splits)
    acc_list = []

    for fold, (tr_idx, va_idx) in enumerate(tscv.split(X), 1):
        X_tr, X_va = X[tr_idx], X[va_idx]
        y_tr, y_va = y[tr_idx], y[va_idx]

        fold_cls = sorted(set(y_tr))
        remap    = {old: new for new, old in enumerate(fold_cls)}
        y_tr_r   = np.array([remap[v] for v in y_tr])
        mask     = np.array([v in remap for v in y_va])
        y_va_r   = np.array([remap[v] for v in y_va if v in remap])
        X_va_r   = X_va[mask]

        mdl = make_xgb(len(fold_cls))
        mdl.fit(X_tr, y_tr_r, eval_set=[(X_va_r, y_va_r)], verbose=False)
        acc = accuracy_score(y_va_r, mdl.predict(X_va_r))
        acc_list.append(acc)
        print(f"    Fold {fold}: acc={acc:.3f}  (val={len(y_va_r)} ky "
              f"{df['ky'].iloc[va_idx[0]]}→{df['ky'].iloc[va_idx[-1]]})")

    mean_acc = np.mean(acc_list)
    lift     = mean_acc - baseline
    flag     = "✅ tot" if lift > 0.05 else ("🟡 kha" if lift > 0.01 else "⚠️ yeu")
    print(f"\n  CV acc = {mean_acc:.3f} ± {np.std(acc_list):.3f}  "
          f"| lift = {lift:+.3f}  {flag}")

    # Final model
    final = make_xgb(n_classes)
    final.fit(X, y, verbose=False)

    # Report tren 20% cuoi
    sp       = int(len(X) * 0.8)
    pred_lbl = le.inverse_transform(final.predict(X[sp:]))
    true_lbl = y_raw[sp:]
    present  = sorted(set(true_lbl), key=str)
    print(f"\n  Classification report (20% cuoi = {len(true_lbl)} ky):")
    print(classification_report(true_lbl, pred_lbl,
                                labels=present, zero_division=0, digits=3))
    return final, le

# ─────────────────────────────────────────────
# 4. FEATURE IMPORTANCE
# ─────────────────────────────────────────────

def show_importance(model, feature_cols, target_col, top_n=12):
    imp = pd.Series(model.feature_importances_, index=feature_cols)
    top = imp.sort_values(ascending=False).head(top_n)
    print(f"  Top {top_n} features — {target_col}:")
    for f, s in top.items():
        bar = "█" * int(s * 400)
        print(f"    {f:<38} {s:.4f}  {bar}")

# ─────────────────────────────────────────────
# 5. PREDICT
# ─────────────────────────────────────────────

def predict_next(df, feature_cols, models, encoders):
    last_row = df[feature_cols].iloc[[-1]]
    last_ky  = df["ky"].iloc[-1]

    print(f"\n{'═'*55}")
    print(f"🎯 Du doan cho ky sau ky {last_ky}:")
    print(f"{'═'*55}")

    for col, model, le in zip(TARGETS, models, encoders):
        proba   = model.predict_proba(last_row)[0]
        classes = le.classes_
        top3    = np.argsort(proba)[::-1][:3]

        best_val  = classes[top3[0]]
        best_prob = proba[top3[0]]
        conf = "✅" if best_prob >= 0.40 else ("🟡" if best_prob >= 0.28 else "⚪")

        top_str = "  ".join(f"{classes[i]}:{proba[i]:.1%}" for i in top3)

        if col == "dd":
            note = (f"n1={'chan' if str(best_val)[0]=='C' else 'le'}, "
                    f"n5={'chan' if str(best_val)[1]=='C' else 'le'}")
        else:
            note = f"{best_val} so chan trong n1-n5"

        print(f"  {col.upper():<4}: {conf} {str(best_val):<4} ({best_prob:.1%})"
              f"   [{top_str}]")
        print(f"         → {note}")

    print(f"{'═'*55}\n")

# ─────────────────────────────────────────────
# 6. MAIN
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="XGBoost predict DD va SC")
    parser.add_argument("--file",       default="l535kqdetail.csv")
    parser.add_argument("--lags",       type=int, default=5)
    parser.add_argument("--splits",     type=int, default=5)
    parser.add_argument("--importance", action="store_true")
    args = parser.parse_args()

    print("📥 Loading data...")
    df = load(args.file)
    print(f"   {len(df)} ky  |  dd: {df['dd'].value_counts().to_dict()}")
    print(f"   sc phan phoi: {df['sc'].value_counts().sort_index().to_dict()}")

    print("\n⚙️  Feature engineering...")
    df_feat, feature_cols = make_features(df, lags=args.lags)
    print(f"   {len(df_feat)} rows  |  {len(feature_cols)} features")

    models, encoders = [], []
    for col in TARGETS:
        m, le = train_one(df_feat, feature_cols, col, n_splits=args.splits)
        models.append(m)
        encoders.append(le)
        if args.importance:
            show_importance(m, feature_cols, col)

    predict_next(df_feat, feature_cols, models, encoders)

    try:
        import joblib
        joblib.dump({"models": models, "encoders": encoders,
                     "features": feature_cols, "targets": TARGETS},
                    "model_dd_sc.pkl")
        print("💾 Luu model: model_dd_sc.pkl")
    except ImportError:
        pass

if __name__ == "__main__":
    main()