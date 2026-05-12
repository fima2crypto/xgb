# gpt_fima_xgboost_l535_n35.py

"""
gpt_fima_xgboost_l535_n35.py
Improved XGBoost pipeline for L535 prediction
"""

import argparse
import os
import warnings
from collections import Counter

import numpy as np
import pandas as pd

from sklearn.metrics import precision_score
from sklearn.utils.class_weight import compute_sample_weight

from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

DATABASE_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql://postgres:bin@localhost:5432/katalott"
)

ALL_NUMS = list(range(1, 36))

THU_MAP = {
    "T2": 2,
    "T3": 3,
    "T4": 4,
    "T5": 5,
    "T6": 6,
    "T7": 7,
    "CN": 8,
}


# ─────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────

def load_from_db(n_ky=0, qh_ky=20):
    import psycopg2

    limit = f"LIMIT {n_ky}" if n_ky > 0 else ""

    sql = f"""
        SELECT *
        FROM public.l535kqdetail
        WHERE qh_ky = {qh_ky}
        ORDER BY ky DESC {limit}
    """

    conn = psycopg2.connect(DATABASE_URL)
    df = pd.read_sql(sql, conn)
    conn.close()

    return df.sort_values("ky").reset_index(drop=True)


def load_from_csv(filepath):
    df = pd.read_csv(filepath)
    return df.sort_values("ky").reset_index(drop=True)


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def _encode_thu(series):
    return series.astype(str).str.strip().map(THU_MAP)


def get_drawn_set(row):
    return {
        int(row[c])
        for c in ["n1", "n2", "n3", "n4", "n5"]
        if pd.notna(row[c])
    }


# ─────────────────────────────────────────────
# GLOBAL FEATURES
# ─────────────────────────────────────────────

def make_global_features(df, lags=3):
    d = df.copy()

    feat = pd.DataFrame(index=d.index)

    feat["thu_enc"] = _encode_thu(d["thu"])
    feat["dot"] = pd.to_numeric(d["dot"], errors="coerce").fillna(0)

    if "sc" in d.columns:
        feat["sc_prev"] = d["sc"].shift(1)

        for lag in range(2, lags + 1):
            feat[f"sc_lag{lag}"] = d["sc"].shift(lag)

        for w in [3, 5, 10]:
            feat[f"sc_roll{w}_mean"] = (
                d["sc"].shift(1).rolling(w).mean()
            )

            feat[f"sc_roll{w}_std"] = (
                d["sc"].shift(1).rolling(w).std()
            )

    return feat


# ─────────────────────────────────────────────
# PER NUMBER FEATURES
# ─────────────────────────────────────────────

def make_per_number_features(df, num):
    feat = pd.DataFrame(index=df.index)

    hit = df.apply(
        lambda r: int(num in get_drawn_set(r)),
        axis=1,
    )

    # previous hit
    feat["n_hit_prev"] = hit.shift(1)

    # last seen distance
    gaps = []
    last_idx = None

    for i, h in enumerate(hit):
        if last_idx is None:
            gaps.append(np.nan)
        else:
            gaps.append(i - last_idx)

        if h == 1:
            last_idx = i

    feat["n_last_seen"] = pd.Series(gaps).shift(1)

    # frequencies
    h_shift = hit.shift(1)

    feat["n_freq_5"] = h_shift.rolling(5).mean()
    feat["n_freq_20"] = h_shift.rolling(20).mean()
    feat["n_freq_50"] = h_shift.rolling(50).mean()

    # momentum
    feat["n_momentum"] = (
        feat["n_freq_5"] - feat["n_freq_50"]
    )

    # hot score
    feat["n_hot_score"] = (
        feat["n_freq_5"] /
        (feat["n_freq_50"] + 1e-6)
    )

    # static features
    feat["n_is_even"] = float(num % 2 == 0)
    feat["n_dec"] = float(num // 10)

    return feat


# ─────────────────────────────────────────────
# BUILD DATASET
# ─────────────────────────────────────────────

def build_dataset(df, num, global_feat):
    per_feat = make_per_number_features(df, num)

    all_feat = pd.concat([
        global_feat,
        per_feat,
    ], axis=1)

    target = df.apply(
        lambda r: int(num in get_drawn_set(r)),
        axis=1,
    )

    combined = pd.concat([
        all_feat,
        target.rename("y"),
        df[["ky"]],
    ], axis=1)

    combined = combined.dropna().reset_index(drop=True)

    feature_cols = list(all_feat.columns)

    return combined, feature_cols


# ─────────────────────────────────────────────
# MODEL
# ─────────────────────────────────────────────

def make_xgb(scale_pos_weight=1.0):
    return XGBClassifier(
        n_estimators=300,
        max_depth=2,
        learning_rate=0.03,
        subsample=0.7,
        colsample_bytree=0.7,
        min_child_weight=8,
        gamma=1.0,
        reg_alpha=0.2,
        reg_lambda=3.0,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
        scale_pos_weight=scale_pos_weight,
    )


# ─────────────────────────────────────────────
# TRAIN ONE
# ─────────────────────────────────────────────

def train_one(combined, feature_cols, val_size=50):
    X = combined[feature_cols].values
    y = combined["y"].values

    split = len(X) - val_size

    if split <= 100:
        return None, 0.0

    X_train = X[:split]
    y_train = y[:split]

    X_val = X[split:]
    y_val = y[split:]

    pos = y_train.sum()
    neg = len(y_train) - pos

    scale_pos_weight = max(1.0, neg / max(pos, 1))

    model = make_xgb(scale_pos_weight)

    sample_weight = compute_sample_weight(
        class_weight="balanced",
        y=y_train,
    )

    model.fit(
        X_train,
        y_train,
        sample_weight=sample_weight,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    pred = model.predict(X_val)

    precision = precision_score(
        y_val,
        pred,
        zero_division=0,
    )

    return model, precision


# ─────────────────────────────────────────────
# TRAIN ALL
# ─────────────────────────────────────────────

def train_all(df, global_feat):
    results = {}

    print("\nTraining 35 models...\n")

    for num in ALL_NUMS:
        combined, feature_cols = build_dataset(
            df,
            num,
            global_feat,
        )

        model, precision = train_one(
            combined,
            feature_cols,
            val_size=50,
        )

        results[num] = {
            "model": model,
            "precision": precision,
            "combined": combined,
            "feature_cols": feature_cols,
        }

        print(f"{num:>2} | precision={precision:.3f}")

    return results


# ─────────────────────────────────────────────
# PREDICT NEXT DRAW
# ─────────────────────────────────────────────

def predict_next(df, global_feat, results, top_k=8):
    probs = []

    for num in ALL_NUMS:
        r = results[num]

        combined = r["combined"]
        feature_cols = r["feature_cols"]
        model = r["model"]

        X_last = combined[feature_cols].iloc[[-1]].values

        prob = model.predict_proba(X_last)[0][1]

        probs.append((num, prob))

    probs.sort(key=lambda x: -x[1])

    return probs[:top_k]


# ─────────────────────────────────────────────
# STRICT WALK-FORWARD BACKTEST
# ─────────────────────────────────────────────

def strict_backtest(df, top_k=8, backtest_size=50):
    print(f"\nStrict walk-forward backtest ({backtest_size} draws)\n")

    start_idx = max(200, len(df) - backtest_size - 1)

    precisions = []

    for end_idx in range(start_idx, len(df) - 1):
        train_df = df.iloc[:end_idx].copy()

        next_row = df.iloc[end_idx]

        global_feat = make_global_features(train_df)

        results = train_all(
            train_df,
            global_feat,
        )

        top = predict_next(
            train_df,
            global_feat,
            results,
            top_k=top_k,
        )

        chosen = set(n for n, _ in top)

        actual = get_drawn_set(next_row)

        hits = chosen & actual

        precision = len(hits) / top_k

        precisions.append(precision)

        print(
            f"KY={next_row['ky']} | "
            f"hits={len(hits)} | "
            f"precision={precision:.2%}"
        )

    print("\n────────────────────────────")
    print(f"Mean precision: {np.mean(precisions):.3f}")
    print("────────────────────────────")


# ─────────────────────────────────────────────
# SHOW RESULT
# ─────────────────────────────────────────────

def show_prediction(top):
    print("\n════════════════════════════════════")
    print("TOP PREDICTED NUMBERS")
    print("════════════════════════════════════")

    nums = []

    for rank, (num, prob) in enumerate(top, 1):
        nums.append(num)

        bar = "█" * int(prob * 30)

        print(
            f"#{rank:>2} "
            f"num={num:>2} "
            f"prob={prob:.3f} "
            f"{bar}"
        )

    print("\nSelected:")
    print(sorted(nums))

    even = sum(1 for n in nums if n % 2 == 0)

    print(f"Even/Odd: {even}/{len(nums)-even}")

    dec = Counter(n // 10 for n in nums)

    print("Decades:", dict(dec))

    print("════════════════════════════════════\n")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--source",
        choices=["csv", "db"],
        default="csv",
    )

    parser.add_argument(
        "--file",
        default="l535kqdetail_dat.csv",
    )

    parser.add_argument(
        "--top",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--backtest",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--n_ky",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--qh_ky",
        type=int,
        default=20,
    )

    args = parser.parse_args()

    print("Loading data...")

    if args.source == "csv":
        df = load_from_csv(args.file)
    else:
        df = load_from_db(
            args.n_ky,
            args.qh_ky,
        )

    print(f"Rows: {len(df)}")

    global_feat = make_global_features(df)

    results = train_all(df, global_feat)

    top = predict_next(
        df,
        global_feat,
        results,
        top_k=args.top,
    )

    show_prediction(top)

    if args.backtest > 0:
        strict_backtest(
            df,
            top_k=args.top,
            backtest_size=args.backtest,
        )


if __name__ == "__main__":
    main()
