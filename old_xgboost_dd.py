"""
xgboost_dd.py
─────────────
Du doan cot DD: CC / CL / LC / LL (4 lop)
  DD = chan/le cua n1 (so dau) va n5 (so cuoi)
  C = Chan, L = Le

Khong data leak: tat ca features deu shift >= 1.

Usage:
    python xgboost_dd.py --file data.csv
    python xgboost_dd.py --file data.csv --lags 5 --importance
"""

import argparse, os, warnings
import numpy as np
import pandas as pd
from collections import Counter

from xgboost import XGBClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
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
    df = pd.read_sql(sql, conn)
    conn.close()
    return df.sort_values("ky").reset_index(drop=True)

def load_from_csv(filepath) -> pd.DataFrame:
    df = pd.read_csv(filepath)
    return df.sort_values("ky").reset_index(drop=True)

def load(filepath) -> pd.DataFrame:
    df = pd.read_csv(filepath)
    return df.sort_values("ky").reset_index(drop=True)


# ─────────────────────────────────────────────
# 2. FEATURES (no-leak)
# ─────────────────────────────────────────────
# claude AI makge_feature
def claude_make_features(df: pd.DataFrame, lags: int = 5) -> tuple:
    df = df.copy()

    # Encode
    df["dd_enc"] = df["dd"].apply(
        lambda v: {"CC": 0, "CL": 1, "LC": 2, "LL": 3}.get(str(v).strip(), -1)
    )
    # df["cl5_enc"] = df["cl5"].apply(
    #    lambda v: str(v).strip().count("C") if len(str(v).strip())==5 else -1)

    s = df["cl5"].astype(str).str.strip()
    # tạo 5 cột bit
    for i in range(5):
        df[f"cl5_b{i}"] = s.str[i].map({"C": 0, "L": 1}).fillna(-1).astype(int)

    df["thu_enc"] = df["thu"].apply(
        lambda v: {"2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7, "CN": 8}.get(
            str(v).strip(), -1
        )
    )
    df["dot"] = pd.to_numeric(df["dot"], errors="coerce").fillna(0).astype(int)

    feat = pd.DataFrame(index=df.index)

    # A. Biet truoc (lich)
    feat["thu_enc"] = df["thu_enc"]
    feat["dot"] = df["dot"]

    # B. Ket qua ky truoc — shift(1)
    prev_cols = [
        "dd_enc",
        "sc",
        "ke",
        "sum",
        "cl5_b0",
        "cl5_b1",
        "cl5_b2",
        "cl5_b3",
        "cl5_b4",
        "t1",
        "g0",
        "p1",
        "snt_cnt",
        "dec0",
        "dec1",
        "dec2",
        "dec3",
    ]
    prev_cols = [c for c in prev_cols if c in df.columns]
    for col in prev_cols:
        feat[f"{col}_prev"] = df[col].shift(1)

    # C. Lag 2 → lags
    lag_cols = [
        "dd_enc",
        "sc",
        "cl5_b0",
        "cl5_b1",
        "cl5_b2",
        "cl5_b3",
        "cl5_b4",
        "sum",
        "dec0",
        "dec1",
    ]
    lag_cols = [c for c in lag_cols if c in df.columns]
    for lag in range(2, lags + 1):
        for col in lag_cols:
            feat[f"{col}_lag{lag}"] = df[col].shift(lag)

    # D. Rolling mean / std (window 3, 5)
    roll_cols = [
        "dd_enc",
        "sc",
        "cl5_b0",
        "cl5_b1",
        "cl5_b2",
        "cl5_b3",
        "cl5_b4",
        "sum",
    ]
    roll_cols = [c for c in roll_cols if c in df.columns]
    for w in [3, 5]:
        for col in roll_cols:
            base = df[col].shift(1)
            feat[f"{col}_roll{w}_mean"] = base.rolling(w).mean()
            feat[f"{col}_roll{w}_std"] = base.rolling(w).std()

    # Target
    target = df["dd"].apply(lambda v: str(v).strip())

    out = (
        pd.concat([feat, target.rename("dd"), df["ky"]], axis=1)
        .dropna()
        .reset_index(drop=True)
    )
    return out, list(feat.columns)


# chatgpt make_features
def make_features(df: pd.DataFrame, lags: int = 5) -> tuple:

    df = df.copy()

    # =========================================================
    # 1. Encode DD
    # =========================================================

    df["dd_enc"] = df["dd"].apply(
        lambda v: {"CC": 0, "CL": 1, "LC": 2, "LL": 3}.get(str(v).strip(), -1)
    )

    # =========================================================
    # 2. Encode CL5 bitwise
    # =========================================================

    s = df["cl5"].astype(str).str.strip()

    for i in range(5):

        df[f"cl5_b{i}"] = s.str[i].map({"C": 0, "L": 1}).fillna(-1).astype(int)

    # =========================================================
    # 3. Global CL5 pattern
    # =========================================================

    def encode_cl5_pattern(v):

        s = str(v).strip()

        if len(s) != 5:
            return -1

        bits = "".join("0" if ch == "C" else "1" for ch in s)

        return int(bits, 2)

    df["cl5_pattern"] = df["cl5"].apply(encode_cl5_pattern)

    # =========================================================
    # 4. Aggregate parity feature
    # =========================================================

    df["cl5_l_count"] = (
        df["cl5_b0"] + df["cl5_b1"] + df["cl5_b2"] + df["cl5_b3"] + df["cl5_b4"]
    )

    # =========================================================
    # 5. Transition feature
    # =========================================================

    df["cl5_changed"] = (df["cl5_pattern"] != df["cl5_pattern"].shift(1)).astype(int)

    # =========================================================
    # 6. Encode thu
    # =========================================================

    df["thu_enc"] = df["thu"].apply(
        lambda v: {"2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7, "CN": 8}.get(
            str(v).strip(), -1
        )
    )

    df["dot"] = pd.to_numeric(df["dot"], errors="coerce").fillna(0).astype(int)

    # =========================================================
    # 7. Feature dataframe
    # =========================================================

    feat = pd.DataFrame(index=df.index)

    # =========================================================
    # A. Known before draw
    # =========================================================

    feat["thu_enc"] = df["thu_enc"]
    feat["dot"] = df["dot"]

    # =========================================================
    # B. Previous features (lag1)
    # =========================================================

    prev_cols = [
        "dd_enc",
        "sc",
        "sum",
        # bitwise
        "cl5_b0",
        "cl5_b1",
        "cl5_b2",
        "cl5_b3",
        "cl5_b4",
        # aggregate
        "cl5_l_count",
        # global state
        "cl5_pattern",
        # transition
        "cl5_changed",
        "dec0",
        "dec1",
        "dec2",
        "dec3",
    ]

    prev_cols = [c for c in prev_cols if c in df.columns]

    for col in prev_cols:

        feat[f"{col}_prev"] = df[col].shift(1)

    # =========================================================
    # C. Lag 2 → lags
    # =========================================================

    lag_cols = [
        "dd_enc",
        "sc",
        "sum",
        "cl5_b0",
        "cl5_b1",
        "cl5_b2",
        "cl5_b3",
        "cl5_b4",
        "cl5_l_count",
        "cl5_pattern",
        "dec0",
        "dec1",
    ]

    lag_cols = [c for c in lag_cols if c in df.columns]

    for lag in range(2, lags + 1):

        for col in lag_cols:

            feat[f"{col}_lag{lag}"] = df[col].shift(lag)

    # =========================================================
    # D. Rolling features
    # =========================================================

    roll_cols = ["sc", "sum", "cl5_l_count", "cl5_changed"]

    roll_cols = [c for c in roll_cols if c in df.columns]

    for w in [3, 5]:

        for col in roll_cols:

            base = df[col].shift(1)

            feat[f"{col}_roll{w}_mean"] = base.rolling(w).mean()

            feat[f"{col}_roll{w}_std"] = base.rolling(w).std()

    # =========================================================
    # Target
    # =========================================================

    target = df["dd"].apply(lambda v: str(v).strip())

    out = (
        pd.concat([feat, target.rename("dd"), df["ky"]], axis=1)
        .dropna()
        .reset_index(drop=True)
    )

    return out, list(feat.columns)


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


def train(df, feature_cols, n_splits=5):
    X = df[feature_cols].values
    y_raw = df["dd"].values

    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    classes = le.classes_
    n_classes = len(classes)

    maj_enc = mode(y, keepdims=True).mode[0]
    baseline = np.mean(y == maj_enc)
    maj_lbl = le.inverse_transform([maj_enc])[0]

    print(f"\n  TARGET: DD  |  {n_classes} lop: {list(classes)}")
    print(
        f"  Phan phoi: "
        + "  ".join(
            f"{c}:{v}" for c, v in sorted(Counter(y_raw).items(), key=lambda x: -x[1])
        )
    )
    print(f"  Baseline (majority='{maj_lbl}'): {baseline:.3f}")

    tscv = TimeSeriesSplit(n_splits=n_splits)
    acc_list = []

    for fold, (tr_idx, va_idx) in enumerate(tscv.split(X), 1):
        X_tr, X_va = X[tr_idx], X[va_idx]
        y_tr, y_va = y[tr_idx], y[va_idx]

        fold_cls = sorted(set(y_tr))
        remap = {old: new for new, old in enumerate(fold_cls)}
        y_tr_r = np.array([remap[v] for v in y_tr])
        mask = np.array([v in remap for v in y_va])
        y_va_r = np.array([remap[v] for v in y_va if v in remap])
        X_va_r = X_va[mask]

        mdl = make_xgb(len(fold_cls))
        mdl.fit(X_tr, y_tr_r, eval_set=[(X_va_r, y_va_r)], verbose=False)
        acc = accuracy_score(y_va_r, mdl.predict(X_va_r))
        acc_list.append(acc)
        print(
            f"    Fold {fold}: acc={acc:.3f}  "
            f"(val={len(y_va_r)}, "
            f"ky {df['ky'].iloc[va_idx[0]]}→{df['ky'].iloc[va_idx[-1]]})"
        )

    mean_acc = np.mean(acc_list)
    lift = mean_acc - baseline
    flag = "✅ tot" if lift > 0.05 else ("🟡 kha" if lift > 0.01 else "⚠️ yeu")
    print(
        f"\n  CV acc = {mean_acc:.3f} ± {np.std(acc_list):.3f}  "
        f"| lift = {lift:+.3f}  {flag}"
    )

    # Final model tren toan bo data
    final = make_xgb(n_classes)
    final.fit(X, y, verbose=False)

    # Report 20% cuoi
    sp = int(len(X) * 0.8)
    pred_lbl = le.inverse_transform(final.predict(X[sp:]))
    true_lbl = y_raw[sp:]
    print(f"\n  Classification report (20% cuoi = {len(true_lbl)} ky):")
    print(
        classification_report(
            true_lbl, pred_lbl, labels=sorted(set(true_lbl)), zero_division=0, digits=3
        )
    )
    return final, le


# ─────────────────────────────────────────────
# 4. FEATURE IMPORTANCE
# ─────────────────────────────────────────────


def show_importance(model, feature_cols, top_n=15):
    imp = pd.Series(model.feature_importances_, index=feature_cols)
    top = imp.sort_values(ascending=False).head(top_n)
    print(f"\n  Top {top_n} features quan trong:")
    for f, s in top.items():
        bar = "█" * int(s * 400)
        print(f"    {f:<38} {s:.4f}  {bar}")


# ─────────────────────────────────────────────
# 5. PREDICT
# ─────────────────────────────────────────────


def predict_next(df, feature_cols, model, le):
    last_row = df[feature_cols].iloc[[-1]]
    last_ky = df["ky"].iloc[-1]

    proba = model.predict_proba(last_row)[0]
    classes = le.classes_
    top3 = np.argsort(proba)[::-1][:3]

    best_val = classes[top3[0]]
    best_prob = proba[top3[0]]
    conf = "✅" if best_prob >= 0.40 else ("🟡" if best_prob >= 0.28 else "⚪")

    print(f"\n{'═'*50}")
    print(f"🎯 Du doan DD cho ky sau ky {last_ky}:")
    print(f"{'═'*50}")
    for i, idx in enumerate(top3, 1):
        lbl = classes[idx]
        prob = proba[idx]
        n1 = "Chan" if lbl[0] == "C" else "Le"
        n5 = "Chan" if lbl[1] == "C" else "Le"
        mark = "◀" if i == 1 else ""
        print(f"  #{i}  {lbl}  ({prob:.1%})   n1={n1}, n5={n5}  {mark}")
    print(f"\n  Ket qua du doan: {conf} DD = [{best_val}]  ({best_prob:.1%})")
    print(f"{'═'*50}\n")


# ─────────────────────────────────────────────
# 6. MAIN
# ─────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="XGBoost predict DD")
    parser.add_argument("--source", choices=["db", "csv"], default="csv")
    parser.add_argument("--file", default="l535kqdetail_dat.csv")
    parser.add_argument("--n_ky", type=int, default=0)
    parser.add_argument("--qh_ky", type=int, default=20)
    parser.add_argument("--lags", type=int, default=5)
    parser.add_argument("--splits", type=int, default=5)
    parser.add_argument("--importance", action="store_true")
    args = parser.parse_args()

    print("📥 Loading data...")
    #df = load(args.file)
    df = load_from_csv(args.file) if args.source=="csv" \
         else load_from_db(args.n_ky, args.qh_ky)
    print(f"   {len(df)} ky")
    print(f"   DD phan phoi: {df['dd'].value_counts().to_dict()}")

    print("\n⚙️  Feature engineering...")
    df_feat, feature_cols = make_features(df, lags=args.lags)
    print(f"   {len(df_feat)} rows  |  {len(feature_cols)} features")

    print("\n🚀 Training...")
    model, le = train(df_feat, feature_cols, n_splits=args.splits)

    if args.importance:
        show_importance(model, feature_cols)

    predict_next(df_feat, feature_cols, model, le)

    try:
        import joblib

        joblib.dump(
            {"model": model, "le": le, "features": feature_cols}, "model_dd.pkl"
        )
        print("💾 Luu model: model_dd.pkl")
    except ImportError:
        pass


if __name__ == "__main__":
    main()
