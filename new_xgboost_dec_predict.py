"""
xgboost_dec_predict.py  (v5)
────────────────────────────
Train 6 models song song, khong data leak:
  dec0, dec1, dec2, dec3  — phan phoi hang chuc
  dd                       — chan/le cua n1 va n5 (CC/CL/LC/LL)
  n6                       — so dac biet (1-12)

Usage:
    python xgboost_dec_predict.py --source csv --file data.csv
    python xgboost_dec_predict.py --source db --n_ky 0 --qh_ky 20
    python xgboost_dec_predict.py --source csv --file data.csv --importance
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
    "DATABASE_URL",
    "postgresql://postgres:bin@localhost:5432/katalott"
)

# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────

# 6 targets can du doan
TARGETS = ["dec0", "dec1", "dec2", "dec3", "dd", "n6"]

# Cot chi biet sau khi quay → phai shift(1) truoc khi dung
SHIFTABLE = [
    "dec0", "dec1", "dec2", "dec3",
    "sc", "ke", "sum",
    "t1", "g0", "p1",
    "snt_cnt",
    "dd_enc",       # encode cua dd
    "cl5_enc",      # encode cua cl5
    "dbx_enc",      # encode cua dbx (nhom n6)
    "dbdd_enc",     # chan/le cua n6
    "n6",           # so dac biet ky truoc
    "g1_cnt", "g2_cnt", "g3_cnt",
    "dd_cnt",
]

# Cot biet truoc khi quay (lich)
PRE_KNOWN = ["thu_enc", "dot"]

# ─────────────────────────────────────────────
# 1. LOAD DATA
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
    df = pd.read_sql(sql, conn); conn.close()
    return df.sort_values("ky").reset_index(drop=True)

def load_from_csv(filepath) -> pd.DataFrame:
    df = pd.read_csv(filepath)
    return df.sort_values("ky").reset_index(drop=True)

# ─────────────────────────────────────────────
# 2. ENCODE + FEATURE ENGINEERING
# ─────────────────────────────────────────────

def encode_raw(df: pd.DataFrame) -> pd.DataFrame:
    """Encode cac cot categorical thanh so nguyen."""
    df = df.copy()
    df["dd_enc"] = df["dd"].apply(
        lambda v: {"CC":0,"CL":1,"LC":2,"LL":3}.get(str(v).strip(), -1))
    df["cl5_enc"] = df["cl5"].apply(
        lambda v: str(v).strip().count("C") if len(str(v).strip())==5 else -1)
    df["dbx_enc"] = df["dbx"].apply(
        lambda v: {"X":0,"Y":1,"Z":2}.get(str(v).strip(), -1))
    df["dbdd_enc"] = df["dbdd"].apply(
        lambda v: 0 if str(v).strip()=="C" else 1)
    df["thu_enc"] = df["thu"].apply(
        lambda v: {"2":2,"3":3,"4":4,"5":5,"6":6,"7":7,"CN":8}.get(str(v).strip(),-1))
    df["dot"] = pd.to_numeric(df["dot"], errors="coerce").fillna(0).astype(int)
    # Dam bao n6 la int
    df["n6"] = pd.to_numeric(df["n6"], errors="coerce").fillna(0).astype(int)
    return df


def make_features(df: pd.DataFrame, lags: int = 5) -> tuple:
    """
    Tao feature matrix KHONG DATA LEAK.

    Nguyen tac:
      - Row i = features lay tu row i-1 tro ve (tat ca deu shift >= 1)
      - Thu / dot la thong tin lich biet truoc, KHONG can shift
      - Khong dung bat ky gia tri nao cua kY HIEN TAI

    Tra ve: (df_out, feature_cols)
      df_out gom feature_cols + targets (dec0-3, dd, n6) + ky
    """
    df = encode_raw(df)
    feat = pd.DataFrame(index=df.index)

    # A. Thong tin biet truoc (lich) — khong shift
    for col in PRE_KNOWN:
        if col in df.columns:
            feat[col] = df[col]

    # B. Tat ca ket qua ky truoc (shift 1)
    present = [c for c in SHIFTABLE if c in df.columns]
    for col in present:
        feat[f"{col}_prev"] = df[col].shift(1)

    # C. Lag 2 → lags
    lag_core = [
        "dec0","dec1","dec2","dec3",
        "sc","ke","sum",
        "t1","g0","p1","snt_cnt",
        "dd_enc","cl5_enc","dbx_enc","n6",
    ]
    lag_core = [c for c in lag_core if c in df.columns]
    for lag in range(2, lags+1):
        for col in lag_core:
            feat[f"{col}_lag{lag}"] = df[col].shift(lag)

    # D. Rolling mean / std (window 3, 5) — shift(1) truoc rolling
    roll_core = ["dec0","dec1","dec2","dec3","sc","sum","ke","n6"]
    roll_core = [c for c in roll_core if c in df.columns]
    for w in [3, 5]:
        for col in roll_core:
            base = df[col].shift(1)
            feat[f"{col}_roll{w}_mean"] = base.rolling(w).mean()
            feat[f"{col}_roll{w}_std"]  = base.rolling(w).std()

    # E. Gap features: khoang cach ky cuoi xuat hien moi n6 (1-12)
    #    Tinh tu lich su, shift de tranh leak
    for num in range(1, 13):
        feat[f"n6_gap_{num}"] = (
            (df["n6"].shift(1) != num)
            .groupby((df["n6"].shift(1) == num).cumsum())
            .cumcount()
        )

    # Gop target + ky
    targets_df = df[["dec0","dec1","dec2","dec3","n6","dd","ky"]].copy()
    # Encode dd thanh string giu nguyen (LC/CC...) lam target phan loai
    targets_df["dd"] = df["dd"].apply(lambda v: str(v).strip())

    out = pd.concat([feat, targets_df], axis=1).dropna().reset_index(drop=True)
    feature_cols = list(feat.columns)
    return out, feature_cols

# ─────────────────────────────────────────────
# 3. XGB MODEL
# ─────────────────────────────────────────────

def make_xgb(n_cls: int) -> XGBClassifier:
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

# ─────────────────────────────────────────────
# 4. TRAIN 1 TARGET
# ─────────────────────────────────────────────

def train_one(df, feature_cols, target_col, n_splits=5):
    X     = df[feature_cols].values
    y_raw = df[target_col].values

    le = LabelEncoder()
    y  = le.fit_transform(y_raw)
    classes   = le.classes_
    n_classes = len(classes)

    maj_enc = mode(y, keepdims=True).mode[0]
    baseline = np.mean(y == maj_enc)
    maj_label = le.inverse_transform([maj_enc])[0]

    print(f"\n  [{target_col}]  {n_classes} lop  "
          f"baseline={baseline:.3f} (majority='{maj_label}')")
    print(f"   Phan phoi: "
          + " ".join(f"{c}:{v}" for c,v in
                     sorted(Counter(y_raw).items(), key=lambda x:-x[1])[:8]))

    tscv     = TimeSeriesSplit(n_splits=n_splits)
    acc_list = []

    for fold, (tr_idx, va_idx) in enumerate(tscv.split(X), 1):
        X_tr, X_va = X[tr_idx], X[va_idx]
        y_tr, y_va = y[tr_idx], y[va_idx]

        # Remap neu fold thieu lop
        fold_cls = sorted(set(y_tr))
        remap    = {old:new for new,old in enumerate(fold_cls)}
        y_tr_r   = np.array([remap[v] for v in y_tr])
        mask     = np.array([v in remap for v in y_va])
        y_va_r   = np.array([remap[v] for v in y_va if v in remap])
        X_va_r   = X_va[mask]

        mdl = make_xgb(len(fold_cls))
        mdl.fit(X_tr, y_tr_r, eval_set=[(X_va_r, y_va_r)], verbose=False)
        acc = accuracy_score(y_va_r, mdl.predict(X_va_r))
        acc_list.append(acc)

    mean_acc = np.mean(acc_list)
    lift     = mean_acc - baseline
    flag     = "✅" if lift > 0.05 else ("🟡" if lift > 0.01 else "⚠️")
    print(f"   CV acc={mean_acc:.3f} ± {np.std(acc_list):.3f}  "
          f"lift={lift:+.3f} vs baseline  {flag}")

    # Final model tren toan bo data
    final = make_xgb(n_classes)
    final.fit(X, y, verbose=False)

    # Report 20% cuoi
    sp = int(len(X)*0.8)
    pred_lbl = le.inverse_transform(final.predict(X[sp:]))
    true_lbl = y_raw[sp:]
    present  = sorted(set(true_lbl), key=lambda x: str(x))
    print(classification_report(true_lbl, pred_lbl,
                                labels=present, zero_division=0, digits=3))
    return final, le

# ─────────────────────────────────────────────
# 5. FEATURE IMPORTANCE
# ─────────────────────────────────────────────

def show_importance(model, feature_cols, target_col, top_n=15):
    imp = pd.Series(model.feature_importances_, index=feature_cols)
    top = imp.sort_values(ascending=False).head(top_n)
    print(f"\n  Top {top_n} features — {target_col}:")
    for f, s in top.items():
        bar = "█" * int(s * 300)
        print(f"    {f:<38} {s:.4f}  {bar}")

# ─────────────────────────────────────────────
# 6. PREDICT KY KE TIEP
# ─────────────────────────────────────────────

def predict_next(df, feature_cols, models, encoders):
    last_row = df[feature_cols].iloc[[-1]]
    last_ky  = df["ky"].iloc[-1]

    sep = "─" * 62
    print(f"\n{sep}")
    print(f"🎯 Du doan cho ky sau ky {last_ky}:")
    print(f"{sep}")

    results = {}
    for col, model, le in zip(TARGETS, models, encoders):
        proba   = model.predict_proba(last_row)[0]
        classes = le.classes_
        top3    = np.argsort(proba)[::-1][:3]

        best_val  = classes[top3[0]]
        best_prob = proba[top3[0]]
        results[col] = best_val

        top_str = "  ".join(f"{classes[i]}:{proba[i]:.1%}" for i in top3)
        conf    = "✅" if best_prob >= 0.40 else ("🟡" if best_prob >= 0.28 else "⚪")
        print(f"  {col:<6}: {conf} {str(best_val):<4} ({best_prob:.1%})   [{top_str}]")

    # Tong hop dec
    dec_str = "".join(str(results[c]) for c in ["dec0","dec1","dec2","dec3"])
    dd_val  = results["dd"]
    n6_val  = results["n6"]

    print(f"\n  {'─'*58}")
    print(f"  dec  : [{dec_str}]  "
          f"(0-9:{results['dec0']} | 10-19:{results['dec1']} | "
          f"20-29:{results['dec2']} | 30-35:{results['dec3']})")
    print(f"  dd   : [{dd_val}]  "
          f"(n1={'chan' if str(dd_val)[0]=='C' else 'le'}, "
          f"n5={'chan' if str(dd_val)[1]=='C' else 'le'})")
    print(f"  n6   : [{n6_val}]")
    print(f"{sep}\n")
    return results

# ─────────────────────────────────────────────
# 7. MAIN
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="XGBoost v5: dec+dd+n6")
    parser.add_argument("--source",     choices=["db","csv"], default="csv")
    parser.add_argument("--file",       default="l535kqdetail.csv")
    parser.add_argument("--n_ky",       type=int, default=0)
    parser.add_argument("--qh_ky",      type=int, default=20)
    parser.add_argument("--lags",       type=int, default=5)
    parser.add_argument("--splits",     type=int, default=5)
    parser.add_argument("--importance", action="store_true")
    args = parser.parse_args()

    print("📥 Loading data...")
    df = load_from_csv(args.file) if args.source=="csv" \
         else load_from_db(args.n_ky, args.qh_ky)
    print(f"   {len(df)} ky  |  n6 range: {df['n6'].min()}-{df['n6'].max()}")

    print("\n⚙️  Feature engineering (no-leak)...")
    df_feat, feature_cols = make_features(df, lags=args.lags)
    print(f"   {len(df_feat)} rows  |  {len(feature_cols)} features")

    print(f"\n🚀 Training {len(TARGETS)} models "
          f"(TimeSeriesSplit {args.splits} folds)...")

    models, encoders = [], []
    for col in TARGETS:
        m, le = train_one(df_feat, feature_cols, col, n_splits=args.splits)
        models.append(m); encoders.append(le)
        if args.importance:
            show_importance(m, feature_cols, col)

    predict_next(df_feat, feature_cols, models, encoders)

    try:
        import joblib
        payload = {"models": models, "encoders": encoders,
                   "features": feature_cols, "targets": TARGETS}
        joblib.dump(payload, "dec_model_v5.pkl")
        print("💾 Model luu: dec_model_v5.pkl")
    except ImportError:
        pass

if __name__ == "__main__":
    main()