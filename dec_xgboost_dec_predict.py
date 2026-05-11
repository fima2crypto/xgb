"""
xgboost_dec_predict.py  (v3)
────────────────────────────
Thay vi du doan chuoi dec 4 ky tu (44 lop → overfit),
du doan TUNG COT dec0, dec1, dec2, dec3 rieng le.
Moi cot chi co 0-5 lop → du lieu du de hoc.

Usage:
    python xgboost_dec_predict.py --source csv --file data.csv
    python xgboost_dec_predict.py --source db --n_ky 0 --qh_ky 20
"""

import argparse, os, warnings
import numpy as np
import pandas as pd
from collections import Counter

from xgboost import XGBClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")

DATABASE_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql://postgres:bin@localhost:5432/katalott"
)

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
    import psycopg2
    conn = psycopg2.connect(DATABASE_URL)
    df = pd.read_sql(sql, conn); conn.close()
    return df.sort_values("ky").reset_index(drop=True)

def load_from_csv(filepath) -> pd.DataFrame:
    df = pd.read_csv(filepath)
    return df.sort_values("ky").reset_index(drop=True)

# ─────────────────────────────────────────────
# 2. FEATURE ENGINEERING
# ─────────────────────────────────────────────

def encode_dd(v):
    return {"CC":0,"CL":1,"LC":2,"LL":3}.get(str(v).strip(), -1)

def encode_cl5(v):
    s = str(v).strip()
    return s.count("C") if len(s)==5 else -1

def encode_dbx(v):
    return {"X":0,"Y":1,"Z":2}.get(str(v).strip(), -1)

def encode_thu(v):
    return {"2":2,"3":3,"4":4,"5":5,"6":6,"7":7,"CN":8}.get(str(v).strip(), -1)

def make_features(df: pd.DataFrame, lags=5) -> pd.DataFrame:
    df = df.copy()

    # Encode categorical
    df["dd_enc"]  = df["dd"].apply(encode_dd)
    df["cl5_enc"] = df["cl5"].apply(encode_cl5)
    df["dbx_enc"] = df["dbx"].apply(encode_dbx)
    df["thu_enc"] = df["thu"].apply(encode_thu)
    if "dot" in df.columns:
        df["dot"] = pd.to_numeric(df["dot"], errors="coerce").fillna(0).astype(int)

    # Lag columns
    lag_cols = ["dec0","dec1","dec2","dec3","sc","ke","sum",
                "t1","g0","p1","snt_cnt","dd_enc","cl5_enc","dbx_enc"]
    lag_cols = [c for c in lag_cols if c in df.columns]
    for lag in range(1, lags+1):
        for col in lag_cols:
            df[f"{col}_lag{lag}"] = df[col].shift(lag)

    # Rolling mean/std
    roll_cols = ["dec0","dec1","dec2","dec3","sc","sum"]
    roll_cols = [c for c in roll_cols if c in df.columns]
    for w in [3, 5]:
        for col in roll_cols:
            df[f"{col}_roll{w}_mean"] = df[col].shift(1).rolling(w).mean()
            df[f"{col}_roll{w}_std"]  = df[col].shift(1).rolling(w).std()

    # Xac dinh feature cols (loai bo non-numeric va cot leak)
    exclude = {"ky","ngay","thu","dd","cl5","dbx","dbdd","cam","snt","xx",
               "mod","level","qh_hit","qhl","jpm_info","updated_at",
               "is_jp","is_chia","is_day","dec","dec_ck","cl5_ck",
               "target","dec0","dec1","dec2","dec3"}   # dec0-3 la target, khong dung lam base feature chinh
    # Giu dec0-dec3 lam lag thoi (da duoc shift → khong leak)
    all_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [c for c in all_cols if c not in exclude]

    # Them dec0-3 base (cua KY TRUOC, qua lag1)
    # Chung da co trong lag_cols → da duoc tao thanh dec0_lag1 v.v.

    df = df[feature_cols + ["dec0","dec1","dec2","dec3","ky"]].dropna().reset_index(drop=True)
    return df, feature_cols

# ─────────────────────────────────────────────
# 3. TRAIN & EVALUATE 1 COT
# ─────────────────────────────────────────────

def make_xgb(n_cls):
    return XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="multi:softprob",
        eval_metric="mlogloss",
        num_class=n_cls,
        random_state=42,
        n_jobs=-1,
    )

def train_one_target(df, feature_cols, target_col, n_splits=5):
    """Train XGBoost cho 1 cot target (dec0/1/2/3)."""
    X = df[feature_cols].values
    y_raw = df[target_col].values.astype(int)

    le = LabelEncoder()
    y  = le.fit_transform(y_raw)
    classes   = le.classes_
    n_classes = len(classes)

    print(f"\n  [{target_col}] {n_classes} lop: {list(classes)}")
    print(f"   Phan phoi: {dict(sorted(Counter(y_raw).items()))}")

    tscv = TimeSeriesSplit(n_splits=n_splits)
    acc_list = []

    for fold, (tr_idx, va_idx) in enumerate(tscv.split(X), 1):
        X_tr, X_va = X[tr_idx], X[va_idx]
        y_tr, y_va = y[tr_idx], y[va_idx]

        # Remap neu fold train thieu lop
        fold_cls = sorted(set(y_tr))
        remap    = {old:new for new,old in enumerate(fold_cls)}
        n_f      = len(fold_cls)

        y_tr_r = np.array([remap[v] for v in y_tr])
        mask   = np.array([v in remap for v in y_va])
        y_va_r = np.array([remap[v] for v in y_va if v in remap])
        X_va_r = X_va[mask]

        mdl = make_xgb(n_f)
        mdl.fit(X_tr, y_tr_r, eval_set=[(X_va_r, y_va_r)], verbose=False)

        acc = accuracy_score(y_va_r, mdl.predict(X_va_r))
        acc_list.append(acc)

    mean_acc = np.mean(acc_list)
    print(f"   CV Accuracy: {mean_acc:.4f} ± {np.std(acc_list):.4f}")

    # Baseline: luon doan lop pho bien nhat
    from scipy.stats import mode
    majority = mode(y, keepdims=True).mode[0]
    baseline = np.mean(y == majority)
    lift = mean_acc - baseline
    print(f"   Baseline (majority): {baseline:.4f}  |  Lift: {lift:+.4f}"
          + ("  ✅" if lift > 0.02 else "  ⚠️ thap"))

    # Final model toan bo data
    final = make_xgb(n_classes)
    final.fit(X, y, verbose=False)

    # Report tren 20% cuoi
    split_at = int(len(X)*0.8)
    pred_lbl = le.inverse_transform(final.predict(X[split_at:]))
    true_lbl = y_raw[split_at:]
    present  = sorted(set(true_lbl))
    print(classification_report(true_lbl, pred_lbl,
                                labels=present, zero_division=0, digits=3))

    return final, le

# ─────────────────────────────────────────────
# 4. FEATURE IMPORTANCE
# ─────────────────────────────────────────────

def show_importance(model, feature_cols, target_col, top_n=15):
    imp = pd.Series(model.feature_importances_, index=feature_cols)
    imp = imp.sort_values(ascending=False).head(top_n)
    print(f"\n  Top {top_n} features cho {target_col}:")
    for f, s in imp.items():
        bar = "█" * int(s * 300)
        print(f"    {f:<35} {s:.4f}  {bar}")

# ─────────────────────────────────────────────
# 5. PREDICT KY KE TIEP
# ─────────────────────────────────────────────

def predict_next(df, feature_cols, models, encoders):
    last_row = df[feature_cols].iloc[[-1]]
    last_ky  = df["ky"].iloc[-1]

    print(f"\n{'─'*58}")
    print(f"🎯 Du doan cho ky sau ky {last_ky}:")
    print(f"{'─'*58}")

    results = {}
    for col, model, le in zip(["dec0","dec1","dec2","dec3"], models, encoders):
        proba   = model.predict_proba(last_row)[0]
        classes = le.classes_
        top_idx = np.argsort(proba)[::-1][:3]

        best_val  = classes[top_idx[0]]
        best_prob = proba[top_idx[0]]
        results[col] = best_val

        prob_str = "  ".join(f"{classes[i]}:{proba[i]:.1%}" for i in top_idx)
        flag = "✅" if best_prob >= 0.40 else ("🟡" if best_prob >= 0.25 else "⚪")
        print(f"  {col}: {flag} {best_val} ({best_prob:.1%})   [{prob_str}]")

    # Ghep lai
    dec_pred = "".join(str(results[c]) for c in ["dec0","dec1","dec2","dec3"])
    print(f"\n  → dec tong hop du doan: [{dec_pred}]")
    print(f"     Hang 0-9:{results['dec0']}  10-19:{results['dec1']}  "
          f"20-29:{results['dec2']}  30-35:{results['dec3']}")
    print(f"{'─'*58}")
    return results

# ─────────────────────────────────────────────
# 6. MAIN
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source",  choices=["db","csv"], default="csv")
    parser.add_argument("--file",    default="l535kqdetail.csv")
    parser.add_argument("--n_ky",    type=int, default=0)
    parser.add_argument("--qh_ky",   type=int, default=20)
    parser.add_argument("--lags",    type=int, default=5)
    parser.add_argument("--splits",  type=int, default=5)
    parser.add_argument("--importance", action="store_true",
                        help="Hien thi feature importance")
    args = parser.parse_args()

    print("📥 Loading data...")
    df = load_from_csv(args.file) if args.source=="csv" \
         else load_from_db(args.n_ky, args.qh_ky)
    print(f"   {len(df)} ky loaded")

    print("\n⚙️  Feature engineering...")
    df_feat, feature_cols = make_features(df, lags=args.lags)
    print(f"   {len(df_feat)} rows, {len(feature_cols)} features")

    print(f"\n🚀 Training 4 models (dec0/1/2/3), TimeSeriesSplit {args.splits} folds...")
    models, encoders = [], []
    for col in ["dec0","dec1","dec2","dec3"]:
        m, le = train_one_target(df_feat, feature_cols, col, n_splits=args.splits)
        models.append(m)
        encoders.append(le)
        if args.importance:
            show_importance(m, feature_cols, col)

    predict_next(df_feat, feature_cols, models, encoders)

    # Save
    try:
        import joblib
        joblib.dump({"models":models,"encoders":encoders,"features":feature_cols},
                    "dec_model_v3.pkl")
        print("\n💾 Model luu: dec_model_v3.pkl")
    except ImportError:
        pass

if __name__ == "__main__":
    main()