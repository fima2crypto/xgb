import os
import psycopg2
import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import top_k_accuracy_score
from xgboost import XGBClassifier

DATABASE_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql://postgres:bin@localhost:5432/katalott"
)

# =========================================================
# LOAD DATA
# =========================================================

def load_data():
    conn = psycopg2.connect(DATABASE_URL)

    sql = """
    SELECT *
    FROM public.l535kqdetail
    ORDER BY ky ASC
    """

    df = pd.read_sql(sql, conn)

    conn.close()

    return df


# =========================================================
# FEATURE ENGINEERING
# =========================================================

def add_features(df):

    # TARGET = dec kỳ kế tiếp
    df["target"] = df["dec"].shift(-1)

    # LAG FEATURES
    for lag in [1, 2, 3, 5]:
        df[f"dec_lag_{lag}"] = df["dec"].shift(lag)
        df[f"cl5_lag_{lag}"] = df["cl5"].shift(lag)
        df[f"sum_lag_{lag}"] = df["sum"].shift(lag)
        df[f"sc_lag_{lag}"] = df["sc"].shift(lag)

    # ROLLING FEATURES
    for win in [5, 10, 20]:
        df[f"sum_avg_{win}"] = df["sum"].rolling(win).mean()
        df[f"sc_avg_{win}"] = df["sc"].rolling(win).mean()
        df[f"ke_avg_{win}"] = df["ke"].rolling(win).mean()

    # FILL NULL
    df["thu"] = df["thu"].fillna("UNK")
    df["dot"] = df["dot"].fillna("UNK")
    df["dd"] = df["dd"].fillna("UNK")
    df["dbx"] = df["dbx"].fillna("UNK")
    df["dbdd"] = df["dbdd"].fillna("UNK")
    df["cl5"] = df["cl5"].fillna("UNK")
    df["cam"] = df["cam"].fillna("UNK")
    df["mod"] = df["mod"].fillna("UNK")

    cat_cols = [
        "thu",
        "dot",
        "dd",
        "dbx",
        "dbdd",
        "cl5",
        "cam",
        "mod",
        "dec_lag_1",
        "dec_lag_2",
        "dec_lag_3",
        "dec_lag_5",
        "cl5_lag_1",
        "cl5_lag_2",
        "cl5_lag_3",
        "cl5_lag_5",
    ]

    encoders = {}

    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le

    return df, encoders


# =========================================================
# TRAIN MODEL
# =========================================================

def train_model(df):

    feature_cols = [
        "n1", "n2", "n3", "n4", "n5", "n6",
        "sum",
        "sc",
        "ke",
        "t1",
        "g0",
        "p1",
        "snt_cnt",
        "qhc",
        "jpck",
        "jpm_cnt",
        "is_jp",
        "is_day",
        "is_chia",
        "sum_avg_5",
        "sum_avg_10",
        "sum_avg_20",
        "sc_avg_5",
        "sc_avg_10",
        "sc_avg_20",
        "ke_avg_5",
        "ke_avg_10",
        "ke_avg_20",
        "thu",
        "dot",
        "dd",
        "dbx",
        "dbdd",
        "cl5",
        "cam",
        "mod",
        "dec_lag_1",
        "dec_lag_2",
        "dec_lag_3",
        "dec_lag_5",
        "cl5_lag_1",
        "cl5_lag_2",
        "cl5_lag_3",
        "cl5_lag_5",
    ]

    model_df = df.dropna(subset=["target"]).copy()
    model_df = model_df.dropna()

    y_encoder = LabelEncoder()

    y = y_encoder.fit_transform(model_df["target"])
    X = model_df[feature_cols]

    # TIME SPLIT
    split_idx = int(len(model_df) * 0.8)

    X_train = X.iloc[:split_idx]
    X_test = X.iloc[split_idx:]

    y_train = y[:split_idx]
    y_test = y[split_idx:]

    test_df = model_df.iloc[split_idx:].copy()

    # MODEL
    model = XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="multi:softprob",
        eval_metric="mlogloss",
        random_state=42,
    )

    model.fit(X_train, y_train)

    probs = model.predict_proba(X_test)

    # TOP-2 ACCURACY
    top2_acc = top_k_accuracy_score(y_test, probs, k=2)

    print(f"TOP-2 ACCURACY = {top2_acc:.4f}")

    rows = []

    classes = y_encoder.classes_

    for i in range(len(test_df)):

        row_probs = probs[i]

        top2_idx = np.argsort(row_probs)[::-1][:2]

        top1_class = classes[top2_idx[0]]
        top2_class = classes[top2_idx[1]]

        top1_prob = float(row_probs[top2_idx[0]])
        top2_prob = float(row_probs[top2_idx[1]])

        actual_dec = test_df.iloc[i]["target"]

        is_top1_hit = actual_dec == top1_class
        is_top2_hit = actual_dec in [top1_class, top2_class]

        current_ky = test_df.iloc[i]["ky"]

        predict_for_ky = None

        if i + split_idx + 1 < len(df):
            predict_for_ky = df.iloc[i + split_idx + 1]["ky"]

        rows.append({
            "ky": current_ky,
            "predict_for_ky": predict_for_ky,
            "top1_dec": top1_class,
            "top1_prob": top1_prob,
            "top2_dec": top2_class,
            "top2_prob": top2_prob,
            "actual_dec": actual_dec,
            "is_top1_hit": is_top1_hit,
            "is_top2_hit": is_top2_hit,
        })

    result_df = pd.DataFrame(rows)

    return model, result_df, feature_cols, y_encoder


# =========================================================
# SAVE RESULT
# =========================================================

def save_predictions(result_df):

    conn = psycopg2.connect(DATABASE_URL)
    cur = conn.cursor()

    sql = """
    INSERT INTO public.l535_dec_predict (
        ky,
        predict_for_ky,
        top1_dec,
        top1_prob,
        top2_dec,
        top2_prob,
        actual_dec,
        is_top1_hit,
        is_top2_hit
    ) VALUES (
        %(ky)s,
        %(predict_for_ky)s,
        %(top1_dec)s,
        %(top1_prob)s,
        %(top2_dec)s,
        %(top2_prob)s,
        %(actual_dec)s,
        %(is_top1_hit)s,
        %(is_top2_hit)s
    )
    """

    for _, row in result_df.iterrows():
        cur.execute(sql, row.to_dict())

    conn.commit()

    cur.close()
    conn.close()


# =========================================================
# PREDICT NEXT KY
# =========================================================

def predict_next(df, model, feature_cols, y_encoder):

    latest = df.tail(1).copy()

    X_latest = latest[feature_cols]

    probs = model.predict_proba(X_latest)[0]

    top2_idx = np.argsort(probs)[::-1][:2]

    classes = y_encoder.classes_

    print("\n===== NEXT KY PREDICTION =====")

    for rank, idx in enumerate(top2_idx, start=1):
        print(
            f"TOP {rank}: dec={classes[idx]} prob={probs[idx]:.6f}"
        )


# =========================================================
# MAIN
# =========================================================

def main():

    print("Loading data...")

    df = load_data()

    print(f"Rows = {len(df)}")

    print("Building features...")

    df, encoders = add_features(df)

    print("Training model...")

    model, result_df, feature_cols, y_encoder = train_model(df)

    print(result_df.tail())

    print("Saving predictions...")

    save_predictions(result_df)

    predict_next(df, model, feature_cols, y_encoder)

    print("Done.")


if __name__ == "__main__":
    main()
