# ==========================================================
# compare_predict_result.py
#
# Compare:
# predict from previous round
# VS
# actual current round result
#
# DEFAULT: compare last 12 rounds
# ==========================================================

import json
import pandas as pd

from core.database import get_db

# ==========================================================
# CONFIG
# ==========================================================
PREDICT_TABLE = "l535kqtrain"
RESULT_TABLE = "l535kq"

COMPARE_LAST = 12

# ==========================================================
# LOAD DATA
# ==========================================================
conn = get_db()

# predict table
predict_sql = f"""
SELECT *
FROM {PREDICT_TABLE}
ORDER BY id DESC
LIMIT {COMPARE_LAST}
"""

predict_df = pd.read_sql(predict_sql, conn)

# result table
result_sql = f"""
SELECT *
FROM {RESULT_TABLE}
ORDER BY ky ASC
"""

result_df = pd.read_sql(result_sql, conn)

conn.close()

# ==========================================================
# RESULT MAP
# ==========================================================
result_map = {}

for _, row in result_df.iterrows():

    ky = int(row["ky"])

    result_map[ky] = {
        "main": {
            int(row["n1"]),
            int(row["n2"]),
            int(row["n3"]),
            int(row["n4"]),
            int(row["n5"]),
        },
        "bonus": int(row["n6"])
    }

# ==========================================================
# COMPARE
# ==========================================================
print("\nCOMPARE PREDICT VS REAL")
print("=" * 80)

summary_hit = 0
summary_bonus = 0
summary_round = 0

predict_df = predict_df.sort_values("id")

for _, row in predict_df.iterrows():

    ky_ref = int(row["ky_ref"])

    next_ky = ky_ref + 1

    # ----------------------------------------------
    # real result
    # ----------------------------------------------
    if next_ky not in result_map:

        print(f"\nKY {next_ky} -> NO RESULT YET")

        continue

    real_main = result_map[next_ky]["main"]
    real_bonus = result_map[next_ky]["bonus"]

    # ----------------------------------------------
    # load predict
    # ----------------------------------------------
    final_sets = row["final_sets"]

    if isinstance(final_sets, str):
        final_sets = json.loads(final_sets)

    best_set = final_sets[0]

    pred_main = set(best_set["main"])
    pred_bonus = int(best_set["bonus"])

    # ----------------------------------------------
    # compare
    # ----------------------------------------------
    hit_main = len(pred_main & real_main)

    hit_bonus = pred_bonus == real_bonus

    summary_hit += hit_main

    if hit_bonus:
        summary_bonus += 1

    summary_round += 1

    # ----------------------------------------------
    # print
    # ----------------------------------------------
    pred_main_str = " ".join(
        f"{x:02d}"
        for x in sorted(pred_main)
    )

    real_main_str = " ".join(
        f"{x:02d}"
        for x in sorted(real_main)
    )

    print(f"\nKY REF: {ky_ref}")
    print(f"TARGET KY: {next_ky}")

    print(f"PREDICT : {pred_main_str} | BONUS {pred_bonus:02d}")
    print(f"REAL    : {real_main_str} | BONUS {real_bonus:02d}")

    print(
        f"HIT MAIN={hit_main} | "
        f"HIT BONUS={hit_bonus}"
    )

# ==========================================================
# SUMMARY
# ==========================================================
print("\n" + "=" * 80)

if summary_round > 0:

    print("SUMMARY")

    print(f"Rounds compared : {summary_round}")

    print(
        f"Avg main hit    : "
        f"{round(summary_hit / summary_round, 3)}"
    )

    print(
        f"Bonus hit rate  : "
        f"{round(summary_bonus / summary_round, 3)}"
    )

else:

    print("NO ROUND TO COMPARE")

print("=" * 80)