"""
xgboost_dec_predict.py
──────────────────────
Du doan cot `dec` ky ke tiep tu bang l535kqdetail.

`dec` la chuoi 4 ky tu, vd "1310":
  - vi tri 0: so luong n1-n5 thuoc hang 0-9
  - vi tri 1: so luong n1-n5 thuoc hang 10-19
  - vi tri 2: so luong n1-n5 thuoc hang 20-29
  - vi tri 3: so luong n1-n5 thuoc hang 30-35
→ Tong 4 chu so luon = 5 (co 5 so n1-n5)

Pipeline:
  1. Load data (PostgreSQL hoac CSV)
  2. Feature engineering (lag, rolling, encode)
  3. Train XGBoostClassifier (multi-class)
  4. Evaluate + Feature importance
  5. Predict ky ke tiep

Usage:
    python xgboost_dec_predict.py --source csv --file data.csv
    python xgboost_dec_predict.py --source db --n_ky 0
"""

import argparse
import os
import warnings
import numpy as np
import pandas as pd
from collections import Counter

# XGBoost + sklearn
from xgboost import XGBClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────

DATABASE_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql://postgres:bin@localhost:5432/katalott"
)

def load_from_db(n_ky: int = 0, qh_ky: int = 20) -> pd.DataFrame:
    """Load tu PostgreSQL, sap xep tang dan theo ky."""
    import psycopg2
    import psycopg2.extras

    limit_clause = f"LIMIT {n_ky}" if n_ky > 0 else ""
    sql = f"""
        SELECT *
        FROM public.l535kqdetail
        WHERE qh_ky = {qh_ky}
        ORDER BY ky DESC
        {limit_clause}
    """
    conn = psycopg2.connect(DATABASE_URL)
    df = pd.read_sql(sql, conn)
    conn.close()

    # Sap xep tang dan (cu → moi) de tinh lag dung
    df = df.sort_values("ky").reset_index(drop=True)
    return df


def load_from_csv(filepath: str) -> pd.DataFrame:
    """Load tu file CSV export, sap xep tang dan."""
    df = pd.read_csv(filepath)
    df = df.sort_values("ky").reset_index(drop=True)
    return df


# ─────────────────────────────────────────────
# 2. FEATURE ENGINEERING
# ─────────────────────────────────────────────

def encode_dd(dd_val) -> int:
    """Chuyen 'CC','CL','LC','LL' -> 0,1,2,3."""
    mapping = {"CC": 0, "CL": 1, "LC": 2, "LL": 3}
    return mapping.get(str(dd_val).strip(), -1)

def encode_cl5(cl5_val) -> int:
    """Chuyen 'CCLLL'... thanh so chan trong 5 vi tri (0-5)."""
    s = str(cl5_val).strip()
    return s.count("C") if len(s) == 5 else -1

def encode_dbx(dbx_val) -> int:
    mapping = {"X": 0, "Y": 1, "Z": 2}
    return mapping.get(str(dbx_val).strip(), -1)

def encode_thu(thu_val) -> int:
    """Thu trong tuan -> so nguyen (2-7, CN=8)."""
    mapping = {"2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7, "CN": 8}
    return mapping.get(str(thu_val).strip(), -1)

def make_features(df: pd.DataFrame, lags: int = 5) -> pd.DataFrame:
    """
    Tao feature matrix tu df da sort tang dan.
    Moi row = features cua ky i, target = dec cua ky i.
    Lag features lay tu ky i-1 → i-lags.
    """
    df = df.copy()

    # --- Encode categorical ---
    df["dd_enc"]  = df["dd"].apply(encode_dd)
    df["cl5_enc"] = df["cl5"].apply(encode_cl5)
    df["dbx_enc"] = df["dbx"].apply(encode_dbx)
    df["thu_enc"] = df["thu"].apply(encode_thu)

    # --- Base numeric features (cua chinh ky do) ---
    base_cols = [
        "dec0", "dec1", "dec2", "dec3",   # so luong theo hang chuc
        "sc",                               # so luong so chan
        "ke",                               # so cap ke tiep nhau
        "sum",                              # tong n1-n5
        "t1", "g0", "p1",                  # so trung vs ky truoc
        "snt_cnt",                          # so nguyen to
        "dd_enc", "cl5_enc", "dbx_enc",
        "thu_enc", "dot",
        "g1_cnt", "g2_cnt", "g3_cnt",      # giai
        "dec_ck",                           # khoang cach lap lai dec
    ]
    # Dam bao cac col ton tai
    base_cols = [c for c in base_cols if c in df.columns]

    # --- Lag features ---
    lag_cols = ["dec0", "dec1", "dec2", "dec3", "sc", "ke", "sum",
                "t1", "g0", "p1", "snt_cnt", "dd_enc", "cl5_enc"]
    lag_cols = [c for c in lag_cols if c in df.columns]

    for lag in range(1, lags + 1):
        for col in lag_cols:
            df[f"{col}_lag{lag}"] = df[col].shift(lag)

    # --- Rolling mean / std (window=3,5) ---
    roll_cols = ["dec0", "dec1", "dec2", "dec3", "sc", "sum"]
    roll_cols = [c for c in roll_cols if c in df.columns]
    for w in [3, 5]:
        for col in roll_cols:
            df[f"{col}_roll{w}_mean"] = df[col].shift(1).rolling(w).mean()
            df[f"{col}_roll{w}_std"]  = df[col].shift(1).rolling(w).std()

    # --- Dot (buoi) encode ---
    if "dot" in df.columns:
        df["dot"] = pd.to_numeric(df["dot"], errors="coerce").fillna(0).astype(int)

    # --- Target: dec cua chinh ky nay ---
    # dec co the la int (1310) hoac str ("1310") tuy nguon data
    # Chuan hoa ve string 4 ky tu, zero-pad neu can
    df["target"] = df["dec"].apply(
        lambda x: str(int(x)).zfill(4) if pd.notna(x) else None
    )

    # Drop rows co NaN (do lag / rolling)
    feature_cols = [c for c in df.columns if c not in
                    ["ky", "ngay", "thu", "target", "dec",
                     "dd", "cl5", "dbx", "dbdd", "cam",
                     "snt", "xx", "mod", "level", "cl5_ck",
                     "qh_hit", "qhl", "jpm_info",
                     "is_jp", "is_chia", "is_day", "updated_at",
                     "qh_ky", "dec0_lag0"  # tranh data leak
                     ]]
    # Loai bo cac col non-numeric con lai
    numeric_cols = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()

    df_feat = df[numeric_cols + ["target", "ky"]].dropna().reset_index(drop=True)
    return df_feat, numeric_cols


# ─────────────────────────────────────────────
# 3. TRAIN & EVALUATE
# ─────────────────────────────────────────────

def make_xgb(n_classes: int) -> XGBClassifier:
    """Tao XGBClassifier chuan, n_classes tu dong."""
    return XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="multi:softprob",
        eval_metric="mlogloss",
        num_class=n_classes,
        random_state=42,
        n_jobs=-1,
    )


def train_and_evaluate(df_feat: pd.DataFrame, feature_cols: list,
                       n_splits: int = 5):
    """
    Train XGBoostClassifier voi TimeSeriesSplit.
    
    Van de: mot so fold train khong co du tat ca cac lop (lop hiem),
    nen encode y tren TOAN BO data truoc, sau do trong moi fold chi
    dung cac lop xuat hien trong tap train → remap lai.
    
    Giai phap: encode toan bo, train final model tren toan bo,
    CV chi dung de do accuracy (khong can num_class khop fold).
    """
    X = df_feat[feature_cols].values
    y_raw = df_feat["target"].values

    # Encode tren toan bo data → sequential 0..N-1
    le = LabelEncoder()
    y_all = le.fit_transform(y_raw)
    classes    = le.classes_
    n_classes  = len(classes)

    print(f"\n📊 Cac lop 'dec' trong data: {len(classes)} lop")
    print(f"   Top 10 pho bien: "
          + ", ".join(f"{c}({v})" for c, v in
                      sorted(Counter(y_raw).items(), key=lambda x: -x[1])[:10]))
    print(f"   Phan phoi day du: {dict(Counter(y_raw))}")

    tscv     = TimeSeriesSplit(n_splits=n_splits)
    acc_list = []

    for fold, (tr_idx, va_idx) in enumerate(tscv.split(X), 1):
        X_tr, X_va = X[tr_idx], X[va_idx]
        y_tr, y_va = y_all[tr_idx], y_all[va_idx]

        # Lay cac lop co trong fold train, remap 0..k-1
        fold_classes = sorted(set(y_tr))
        remap        = {old: new for new, old in enumerate(fold_classes)}
        n_fold_cls   = len(fold_classes)

        y_tr_r = np.array([remap[v] for v in y_tr])
        # Val: neu gap lop khong co trong train → skip khi tinh acc
        va_mask  = np.array([v in remap for v in y_va])
        y_va_r   = np.array([remap[v] for v in y_va if v in remap])
        X_va_r   = X_va[va_mask]

        mdl = make_xgb(n_fold_cls)
        mdl.fit(X_tr, y_tr_r,
                eval_set=[(X_va_r, y_va_r)],
                verbose=False)

        preds = mdl.predict(X_va_r)
        acc   = accuracy_score(y_va_r, preds)
        acc_list.append(acc)

        skipped = (~va_mask).sum()
        print(f"  Fold {fold}: acc={acc:.4f}  "
              f"(train={len(y_tr)}, val={len(y_va_r)}"
              + (f", bo qua {skipped} mau lop moi" if skipped else "") +
              f", ky {df_feat['ky'].iloc[va_idx[0]]}→{df_feat['ky'].iloc[va_idx[-1]]})")

    print(f"\n✅ Mean CV Accuracy: {np.mean(acc_list):.4f} ± {np.std(acc_list):.4f}")

    # ── Final model: train tren TOAN BO data voi tat ca cac lop ──
    print(f"\n🔄 Retrain final model tren toan bo {len(X)} row, {n_classes} lop...")
    final_model = make_xgb(n_classes)
    final_model.fit(X, y_all, verbose=False)

    # Classification report tren 20% cuoi
    split_at   = int(len(X) * 0.8)
    preds_last = final_model.predict(X[split_at:])
    # Remap nguoc ve ten lop thuc
    pred_labels = le.inverse_transform(preds_last)
    true_labels = y_raw[split_at:]
    present_cls = sorted(set(true_labels))

    print("\n📋 Classification Report (20% cuoi lam val):")
    print(classification_report(true_labels, pred_labels,
                                labels=present_cls, zero_division=0))

    return final_model, le


# ─────────────────────────────────────────────
# 4. FEATURE IMPORTANCE
# ─────────────────────────────────────────────

def show_feature_importance(model, feature_cols: list, top_n: int = 20):
    imp = pd.Series(model.feature_importances_, index=feature_cols)
    imp = imp.sort_values(ascending=False).head(top_n)
    print(f"\n🏆 Top {top_n} features quan trong nhat:")
    for feat, score in imp.items():
        bar = "█" * int(score * 200)
        print(f"  {feat:<35} {score:.4f}  {bar}")


# ─────────────────────────────────────────────
# 5. PREDICT KY KE TIEP
# ─────────────────────────────────────────────

def predict_next(df_feat: pd.DataFrame, feature_cols: list,
                 model, le: LabelEncoder):
    """
    Lay row cuoi cung lam input, du doan dec ky ke tiep.
    In ra top-3 du doan kem xac suat.
    """
    last_row = df_feat[feature_cols].iloc[[-1]]
    proba = model.predict_proba(last_row)[0]
    classes = le.classes_

    top3_idx = np.argsort(proba)[::-1][:3]
    last_ky = df_feat["ky"].iloc[-1]

    print(f"\n🎯 Du doan `dec` cho ky sau ky {last_ky}:")
    print(f"{'Hang':>6}  {'dec':>8}  {'Xac suat':>10}  {'Mo ta'}")
    print("─" * 55)
    for rank, idx in enumerate(top3_idx, 1):
        dec_val = classes[idx]
        prob    = proba[idx]
        desc = (f"Hang 0-9: {dec_val[0]}, "
                f"Hang 10-19: {dec_val[1]}, "
                f"Hang 20-29: {dec_val[2]}, "
                f"Hang 30-35: {dec_val[3]}")
        print(f"  #{rank}    {dec_val:>8}   {prob:>9.2%}   {desc}")

    # Xac suat toan bo cac lop
    print(f"\n📊 Xac suat day du:")
    prob_df = pd.DataFrame({"dec": classes, "prob": proba})
    prob_df = prob_df.sort_values("prob", ascending=False)
    print(prob_df.to_string(index=False))


# ─────────────────────────────────────────────
# 6. MAIN
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="XGBoost du doan dec cho l535")
    parser.add_argument("--source", choices=["db", "csv"], default="csv",
                        help="Nguon data: 'db' hoac 'csv'")
    parser.add_argument("--file",   default="l535kqdetail.csv",
                        help="Duong dan file CSV (neu --source csv)")
    parser.add_argument("--n_ky",   type=int, default=0,
                        help="So ky moi nhat load tu DB (0=tat ca)")
    parser.add_argument("--qh_ky",  type=int, default=20,
                        help="Tham so qh_ky khi load tu DB")
    parser.add_argument("--lags",   type=int, default=5,
                        help="So lag ky (mac dinh 5)")
    parser.add_argument("--splits", type=int, default=5,
                        help="So fold TimeSeriesSplit")
    args = parser.parse_args()

    # --- Load ---
    print("📥 Loading data...")
    if args.source == "db":
        df = load_from_db(n_ky=args.n_ky, qh_ky=args.qh_ky)
    else:
        df = load_from_csv(args.file)

    print(f"   Tong so ky: {len(df)}")
    print(f"   Cot dec mau: {df['dec'].value_counts().head(5).to_dict()}")

    # --- Feature engineering ---
    print("\n⚙️  Feature engineering...")
    df_feat, feature_cols = make_features(df, lags=args.lags)
    print(f"   Rows sau drop NaN: {len(df_feat)}")
    print(f"   So features: {len(feature_cols)}")

    # --- Train ---
    print(f"\n🚀 Training XGBoost (TimeSeriesSplit, {args.splits} folds)...")
    model, le = train_and_evaluate(df_feat, feature_cols, n_splits=args.splits)

    # --- Feature importance ---
    show_feature_importance(model, feature_cols, top_n=20)

    # --- Predict next ---
    predict_next(df_feat, feature_cols, model, le)

    # --- Save model (optional) ---
    try:
        import joblib
        joblib.dump({"model": model, "le": le, "features": feature_cols},
                    "dec_model.pkl")
        print("\n💾 Model luu tai dec_model.pkl")
    except ImportError:
        pass


if __name__ == "__main__":
    main()