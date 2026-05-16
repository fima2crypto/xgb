
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from xgboost import XGBClassifier
from sklearn.metrics import precision_score

from sqlalchemy import create_engine

DATABASE_URL = "postgresql://postgres:bin@localhost:5432/katalott"

ALL_NUMS = list(range(1, 46))

ROLL_WINDOWS = [5, 10, 20, 50]


# ============================================================
# DB
# ============================================================

engine = create_engine(DATABASE_URL)


# ============================================================
# LOAD DATA
# ============================================================

def load_data():
    sql = """
    SELECT *
    FROM public.m645kqdetail
    ORDER BY ngay ASC, ky ASC
    """

    df = pd.read_sql(sql, engine)
    # =========================================================
    # DROP RAW STRING COLUMNS
    # =========================================================

    drop_cols = [
        "dd",
        "cam",
        "snt",
        "xx",
        "mod",
        "dec",
        "cl6",
        "qh_hit",
        "qhl",
        "jpm_info",
        "pwthu",
        "lothu",
    ]

    df = df.drop(columns=drop_cols, errors="ignore")

    return df


# ============================================================
# HELPERS
# ============================================================

def get_drawn_set(row):
    return {
        int(row["n1"]),
        int(row["n2"]),
        int(row["n3"]),
        int(row["n4"]),
        int(row["n5"]),
        int(row["n6"]),
    }


# ============================================================
# POWER FEATURES
# ============================================================

def build_power_features(df):

    # availability
    df["has_power_data"] = df["pwky"].notna().astype(int)

    # parity
    pw_cols = ["pwn1", "pwn2", "pwn3", "pwn4", "pwn5", "pwn6"]

    df["pw_sc"] = df[pw_cols].apply(
        lambda r: sum((x % 2 == 0) for x in r if pd.notna(x)),
        axis=1,
    )

    # sum
    df["pw_sum"] = df[pw_cols].sum(axis=1, min_count=1)

    # overlap with previous Mega
    overlaps = []

    prev_sets = []
    for _, row in df.iterrows():
        prev_sets.append(get_drawn_set(row))

    for i, row in df.iterrows():

        if i == 0:
            overlaps.append(np.nan)
            continue

        prev_mega = prev_sets[i - 1]

        pw_nums = {
            row["pwn1"],
            row["pwn2"],
            row["pwn3"],
            row["pwn4"],
            row["pwn5"],
            row["pwn6"],
        }

        pw_nums = {x for x in pw_nums if pd.notna(x)}

        overlaps.append(len(prev_mega & pw_nums))

    df["pw_overlap_prev"] = overlaps

    return df


# ============================================================
# LOTTO FEATURES
# ============================================================

def build_lotto_features(df):

    df["has_lotto_data"] = df["loky"].notna().astype(int)

    lo_cols = ["lon1", "lon2", "lon3", "lon4", "lon5"]

    df["lo_sum"] = df[lo_cols].sum(axis=1, min_count=1)

    df["lo_sc"] = df[lo_cols].apply(
        lambda r: sum((x % 2 == 0) for x in r if pd.notna(x)),
        axis=1,
    )

    return df


# ============================================================
# GLOBAL FEATURES
# ============================================================

def build_global_features(df):

    # repeat counts
    repeat_1 = []
    repeat_2 = []
    repeat_3 = []

    all_sets = [get_drawn_set(r) for _, r in df.iterrows()]

    for i in range(len(df)):

        curr = all_sets[i]

        if i >= 1:
            r1 = len(curr & all_sets[i - 1])
        else:
            r1 = np.nan

        if i >= 2:
            prev2 = all_sets[i - 1] | all_sets[i - 2]
            r2 = len(curr & prev2)
        else:
            r2 = np.nan

        if i >= 3:
            prev3 = (
                all_sets[i - 1]
                | all_sets[i - 2]
                | all_sets[i - 3]
            )
            r3 = len(curr & prev3)
        else:
            r3 = np.nan

        repeat_1.append(r1)
        repeat_2.append(r2)
        repeat_3.append(r3)

    df["repeat_1"] = repeat_1
    df["repeat_2"] = repeat_2
    df["repeat_3"] = repeat_3

    # rolling statistics
    for w in ROLL_WINDOWS:

        df[f"sum_roll_mean_{w}"] = (
            df["sum"]
            .rolling(w)
            .mean()
        )

        df[f"sum_roll_std_{w}"] = (
            df["sum"]
            .rolling(w)
            .std()
        )

        df[f"sc_roll_mean_{w}"] = (
            df["sc"]
            .rolling(w)
            .mean()
        )

    return df


# ============================================================
# PER NUMBER FEATURES
# ============================================================

def build_per_number_features(df, num):

    hit = []

    all_sets = [get_drawn_set(r) for _, r in df.iterrows()]

    for s in all_sets:
        hit.append(1 if num in s else 0)

    df[f"n{num}_hit"] = hit

    # frequency windows
    for w in [5, 10, 20, 50]:

        df[f"n{num}_freq_{w}"] = (
            pd.Series(hit)
            .shift(1)
            .rolling(w)
            .sum()
        )

    # last seen
    last_seen = []

    gap = 999

    for h in hit:

        if h == 1:
            gap = 0
        else:
            gap += 1

        last_seen.append(gap)

    df[f"n{num}_last_seen"] = pd.Series(last_seen).shift(1)

    return df


# ============================================================
# PAIR GRAPH
# ============================================================

def build_pair_features(df, num, window=50):

    pair_strength = []

    all_sets = [get_drawn_set(r) for _, r in df.iterrows()]

    for i in range(len(df)):

        if i < window:
            pair_strength.append(np.nan)
            continue

        recent = all_sets[i - window:i]

        cnt = 0

        for s in recent:
            if num in s:
                cnt += len(s)

        pair_strength.append(cnt)

    df[f"n{num}_pair_strength"] = pair_strength

    return df


# ============================================================
# FEATURE COLUMNS
# ============================================================
def get_feature_columns(df, num):

    exclude = {
        "ky",
        "thu",
        "ngay",
        "updated_at",
        "pwky",
        "loky",
    }

    target_cols = {
        f"n{i}_hit"
        for i in ALL_NUMS
    }

    cols = []

    for c in df.columns:

        if c in exclude:
            continue

        if c in target_cols:
            continue

        # =====================================================
        # ONLY NUMERIC / BOOL
        # =====================================================

        if not (
            pd.api.types.is_numeric_dtype(df[c])
            or pd.api.types.is_bool_dtype(df[c])
        ):
            continue

        cols.append(c)

    return cols


# ============================================================
# TRAIN SINGLE MODEL
# ============================================================

def train_single_model(df, num):

    target_col = f"n{num}_hit"

    feature_cols = get_feature_columns(df, num)

    train_df = df.dropna(subset=[target_col]).copy()

    train_df = train_df.iloc[50:].copy()

    X = train_df[feature_cols]
    y = train_df[target_col]

    split = int(len(train_df) * 0.85)

    X_train = X.iloc[:split]
    y_train = y.iloc[:split]

    X_valid = X.iloc[split:]
    y_valid = y.iloc[split:]

    model = XGBClassifier(
        n_estimators=400,
        max_depth=3,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.7,
        min_child_weight=5,
        gamma=1,
        objective="binary:logistic",
        eval_metric="logloss",
        scale_pos_weight=7,
        random_state=42,
        n_jobs=-1,
    )

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_valid, y_valid)],
        verbose=False,
    )

    pred = model.predict_proba(X_valid)[:, 1]

    pred_bin = (pred >= 0.20).astype(int)

    precision = precision_score(y_valid, pred_bin)

    return {
        "model": model,
        "precision": precision,
        "features": feature_cols,
    }


# ============================================================
# PREDICT NEXT DRAW
# ============================================================

def predict_next(df, models):

    latest = df.iloc[-1:].copy()

    results = []

    for num in ALL_NUMS:

        m = models[num]

        X = latest[m["features"]]

        prob = m["model"].predict_proba(X)[0][1]

        results.append({
            "num": num,
            "prob": prob,
        })

    pred_df = pd.DataFrame(results)

    pred_df = pred_df.sort_values(
        "prob",
        ascending=False,
    )

    return pred_df


# ============================================================
# MAIN
# ============================================================

def main():

    print("Loading data...")

    df = load_data()

    print("Building Power features...")
    df = build_power_features(df)

    print("Building Lotto features...")
    df = build_lotto_features(df)

    print("Building global features...")
    df = build_global_features(df)

    models = {}

    for num in ALL_NUMS:

        print(f"Building features for {num}...")

        df = build_per_number_features(df, num)

        df = build_pair_features(df, num)

        print(f"Training model {num}...")

        result = train_single_model(df, num)

        models[num] = result

        print(
            f"[{num:02d}] precision="
            f"{result['precision']:.4f}"
        )

    pred_df = predict_next(df, models)

    print("\nTOP PREDICTIONS")
    print(pred_df.head(12))


if __name__ == "__main__":
    main()