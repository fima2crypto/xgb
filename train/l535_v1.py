# ==========================================================
# final_lotto535_xgboost_v3_pro.py
# USING core.database.get_db()
# Lotto 5/35 + Bonus 1/12
# ==========================================================

import pandas as pd
import numpy as np

from xgboost import XGBClassifier

from core.database import get_engine

# ==========================================================
# CONFIG
# ==========================================================
TABLE_NAME = "l535kq"

WINDOW = 30

TOTAL_MAIN = 35
TOTAL_BONUS = 12

TOP_MAIN = 10
TOP_BONUS = 3

BACKTEST_LAST = 100

# ensemble weight
W_XGB = 0.55
W_FREQ = 0.25
W_DELAY = 0.20

# ==========================================================
# LOAD DATA
# ==========================================================

engine = get_engine()

sql = f"""
SELECT *
FROM {TABLE_NAME}
ORDER BY ky ASC
"""

print(sql)
df = pd.read_sql(sql, engine)


print("Rows loaded:", len(df))

# ==========================================================
# CLEAN
# ======rm .git/index====================================================
for c in ["ky", "n1", "n2", "n3", "n4", "n5", "n6"]:
    df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)

# bonus valid 1-12
df = df[
    (df["n6"] >= 1) &
    (df["n6"] <= TOTAL_BONUS)
].reset_index(drop=True)

# ==========================================================
# CREATE MAIN TARGET
# ==========================================================
for i in range(1, TOTAL_MAIN + 1):
    df[f"num_{i:02d}"] = 0

for idx, row in df.iterrows():

    nums = {
        row["n1"],
        row["n2"],
        row["n3"],
        row["n4"],
        row["n5"]
    }

    for n in nums:

        if 1 <= n <= TOTAL_MAIN:
            df.loc[idx, f"num_{n:02d}"] = 1

main_cols = [
    f"num_{i:02d}"
    for i in range(1, TOTAL_MAIN + 1)
]

# ==========================================================
# FEATURE ENGINEERING
# ==========================================================
def build_feature(hist):

    feat = {}

    # ------------------------------------------------------
    # MAIN FEATURES
    # ------------------------------------------------------
    for n in range(1, TOTAL_MAIN + 1):

        col = f"num_{n:02d}"

        # frequency
        feat[f"freq_{n:02d}"] = int(hist[col].sum())

        # recent 10
        feat[f"freq10_{n:02d}"] = int(
            hist.tail(10)[col].sum()
        )

        # overdue
        arr = hist[col].values[::-1]

        delay = WINDOW

        for i, v in enumerate(arr):

            if v == 1:
                delay = i
                break

        feat[f"delay_{n:02d}"] = delay

    # ------------------------------------------------------
    # BONUS FEATURES
    # ------------------------------------------------------
    for b in range(1, TOTAL_BONUS + 1):

        feat[f"bonus_freq_{b:02d}"] = int(
            (hist["n6"] == b).sum()
        )

    # ------------------------------------------------------
    # SUM FEATURES
    # ------------------------------------------------------
    sums = hist[
        ["n1", "n2", "n3", "n4", "n5"]
    ].sum(axis=1)

    feat["sum_mean"] = float(sums.mean())
    feat["sum_std"] = float(sums.std())

    # ------------------------------------------------------
    # ODD / EVEN
    # ------------------------------------------------------
    odd_counts = []

    for _, row in hist.iterrows():

        nums = [
            row["n1"],
            row["n2"],
            row["n3"],
            row["n4"],
            row["n5"],
        ]

        odd = sum(x % 2 == 1 for x in nums)

        odd_counts.append(odd)

    feat["odd_mean"] = float(np.mean(odd_counts))

    return feat

# ==========================================================
# BUILD DATASET
# ==========================================================
rows_main = []
rows_bonus = []

for i in range(WINDOW, len(df) - 1):

    hist = df.iloc[i-WINDOW:i]

    feat = build_feature(hist)

    # target main
    y_main = df.iloc[i + 1][main_cols].values.tolist()

    # target bonus
    y_bonus = int(df.iloc[i + 1]["n6"]) - 1

    rows_main.append((feat, y_main))
    rows_bonus.append((feat, y_bonus))

X = pd.DataFrame([
    r[0]
    for r in rows_main
])

Y_MAIN = np.array([
    r[1]
    for r in rows_main
])

Y_BONUS = np.array([
    r[1]
    for r in rows_bonus
])

print("Train rows:", len(X))

# ==========================================================
# TRAIN MAIN MODELS
# ==========================================================
main_models = {}

for idx, col in enumerate(main_cols):

    y = Y_MAIN[:, idx]

    if len(np.unique(y)) < 2:
        continue

    model = XGBClassifier(
        n_estimators=260,
        max_depth=5,
        learning_rate=0.03,
        subsample=0.85,
        colsample_bytree=0.85,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1
    )

    model.fit(X, y)

    main_models[col] = model

print("Main trained:", len(main_models))

# ==========================================================
# TRAIN BONUS MODEL
# ==========================================================
bonus_model = XGBClassifier(
    n_estimators=280,
    max_depth=5,
    learning_rate=0.03,
    subsample=0.85,
    colsample_bytree=0.85,
    objective="multi:softprob",
    num_class=TOTAL_BONUS,
    eval_metric="mlogloss",
    random_state=42,
    n_jobs=-1
)

bonus_model.fit(X, Y_BONUS)

print("Bonus trained")

# ==========================================================
# ENSEMBLE SCORE
# ==========================================================
def ensemble_score(X_pred, hist):

    result = []

    for n in range(1, TOTAL_MAIN + 1):

        col = f"num_{n:02d}"

        # XGB SCORE
        if col in main_models:

            px = main_models[col] \
                .predict_proba(X_pred)[0][1]

        else:
            px = 0

        # FREQ SCORE
        pf = hist[col].sum() / WINDOW

        # DELAY SCORE
        arr = hist[col].values[::-1]

        delay = WINDOW

        for i, v in enumerate(arr):

            if v == 1:
                delay = i
                break

        pdelay = delay / WINDOW

        # FINAL
        score = (
            (px * W_XGB) +
            (pf * W_FREQ) +
            (pdelay * W_DELAY)
        )

        result.append((n, score))

    return sorted(
        result,
        key=lambda x: x[1],
        reverse=True
    )

# ==========================================================
# BACKTEST
# ==========================================================
print("\nBACKTEST LAST", BACKTEST_LAST)
print("-" * 60)

tested = 0
hit_total = 0
bonus_hit = 0

start_idx = max(
    WINDOW,
    len(df) - BACKTEST_LAST
)

for i in range(start_idx, len(df) - 1):

    hist = df.iloc[i-WINDOW:i]

    feat = build_feature(hist)

    X_pred = pd.DataFrame([feat])

    ranks = ensemble_score(X_pred, hist)

    pred_main = set([
        x[0]
        for x in ranks[:5]
    ])

    real_main = {
        int(df.iloc[i + 1]["n1"]),
        int(df.iloc[i + 1]["n2"]),
        int(df.iloc[i + 1]["n3"]),
        int(df.iloc[i + 1]["n4"]),
        int(df.iloc[i + 1]["n5"]),
    }

    hit_total += len(
        pred_main & real_main
    )

    # BONUS
    bp = bonus_model.predict_proba(X_pred)[0]

    pred_bonus = int(np.argmax(bp)) + 1

    real_bonus = int(df.iloc[i + 1]["n6"])

    if pred_bonus == real_bonus:
        bonus_hit += 1

    tested += 1

print("Rounds:", tested)
print(
    "Avg main hit:",
    round(hit_total / tested, 3)
)

print(
    "Bonus hit rate:",
    round(bonus_hit / tested, 3)
)

# ==========================================================
# PREDICT NEXT ROUND
# ==========================================================
hist = df.iloc[-WINDOW:]

feat = build_feature(hist)

X_pred = pd.DataFrame([feat])

ranks = ensemble_score(X_pred, hist)

# BONUS
bp = bonus_model.predict_proba(X_pred)[0]

bonus_probs = []

for i in range(TOTAL_BONUS):

    bonus_probs.append(
        (i + 1, bp[i])
    )

bonus_probs = sorted(
    bonus_probs,
    key=lambda x: x[1],
    reverse=True
)

# ==========================================================
# FINAL RESULT
# ==========================================================
print("\nTOP MAIN")
print("-" * 60)

for n, s in ranks[:TOP_MAIN]:

    print(f"{n:02d} => {s:.4f}")

pick = sorted([
    x[0]
    for x in ranks[:5]
])

print("\nGOI Y BO MAIN:")

print(
    " ".join(
        f"{x:02d}"
        for x in pick
    )
)

print("\nTOP BONUS")
print("-" * 60)

for n, p in bonus_probs[:TOP_BONUS]:

    print(f"{n:02d} => {p:.4f}")

print("=" * 60)