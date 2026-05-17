import warnings
warnings.filterwarnings("ignore")

import os
import numpy as np
import pandas as pd

from collections import Counter

from xgboost import XGBClassifier

from sklearn.metrics import (
    precision_score,
    recall_score,
    average_precision_score,
    brier_score_loss,
)

from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator

from sklearn.utils.class_weight import compute_sample_weight

import psycopg2


# ============================================================
# CONFIG
# ============================================================

DATABASE_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql://postgres:bin@localhost:5432/katalott"
)

ALL_NUMS = list(range(1, 46))

VERSION = "v2.1.0"

ROLL_WINDOWS = [5, 10, 20]

ENSEMBLE_CONFIGS = [
    {
        "lags": 3,
        "decay": 0.995,
        "train_window": 0,
        "label": "M1",
    },
    {
        "lags": 5,
        "decay": 0.990,
        "train_window": 300,
        "label": "M2",
    },
    {
        "lags": 7,
        "decay": 0.985,
        "train_window": 200,
        "label": "M3",
    },
]


# ============================================================
# LOAD DATA
# ============================================================

def load_from_db(qh_ky=20):

    sql = f"""
        SELECT *
        FROM public.m645kqdetail
        WHERE qh_ky = {qh_ky}
        ORDER BY ky ASC
    """

    conn = psycopg2.connect(DATABASE_URL)

    df = pd.read_sql(sql, conn)

    conn.close()

    return df.reset_index(drop=True)


# ============================================================
# HELPERS
# ============================================================

def get_drawn_set(row):

    return {
        int(row[c])
        for c in ["n1", "n2", "n3", "n4", "n5", "n6"]
        if pd.notna(row[c])
    }


THU_MAP = {
    "T2": 2,
    "T3": 3,
    "T4": 4,
    "T5": 5,
    "T6": 6,
    "T7": 7,
    "CN": 8,
}


# ============================================================
# GLOBAL FEATURES
# ============================================================

def make_global_features(df):

    d = df.copy()

    feat = pd.DataFrame(index=d.index)

    feat["thu_enc"] = (
        d["thu"]
        .astype(str)
        .map(THU_MAP)
    )

    # --------------------------------------------------------
    # POWER FEATURES
    # --------------------------------------------------------

    pw_cols = [f"pwn{i}" for i in range(1, 7)]

    if all(c in d.columns for c in pw_cols):

        feat["pw_sum"] = (
            d[pw_cols]
            .shift(1)
            .sum(axis=1)
        )

        feat["pw_even"] = (
            d[pw_cols]
            .shift(1)
            .apply(
                lambda r: sum(
                    (x % 2 == 0)
                    for x in r
                    if pd.notna(x)
                ),
                axis=1,
            )
        )

    # --------------------------------------------------------
    # REPEAT FEATURES
    # --------------------------------------------------------

    all_sets = d.apply(get_drawn_set, axis=1).tolist()

    r1 = []
    r2 = []
    r3 = []

    for i in range(len(d)):

        curr = all_sets[i]

        if i >= 1:
            v1 = len(curr & all_sets[i - 1])
        else:
            v1 = np.nan

        if i >= 2:
            prev2 = all_sets[i - 1] | all_sets[i - 2]
            v2 = len(curr & prev2)
        else:
            v2 = np.nan

        if i >= 3:
            prev3 = (
                all_sets[i - 1]
                | all_sets[i - 2]
                | all_sets[i - 3]
            )
            v3 = len(curr & prev3)
        else:
            v3 = np.nan

        r1.append(v1)
        r2.append(v2)
        r3.append(v3)

    feat["repeat_1_prev"] = pd.Series(r1).shift(1)
    feat["repeat_2_prev"] = pd.Series(r2).shift(1)
    feat["repeat_3_prev"] = pd.Series(r3).shift(1)

    # --------------------------------------------------------
    # ROLLING
    # --------------------------------------------------------

    for w in ROLL_WINDOWS:

        base_sum = d["sum"].shift(1)
        base_sc = d["sc"].shift(1)

        feat[f"sum_roll_mean_{w}"] = (
            base_sum.rolling(w).mean()
        )

        feat[f"sum_roll_std_{w}"] = (
            base_sum.rolling(w).std()
        )

        feat[f"sc_roll_mean_{w}"] = (
            base_sc.rolling(w).mean()
        )

    return feat


# ============================================================
# GRAPH FEATURE
# ============================================================

def make_graph_feature(df, num, window=50):

    all_sets = df.apply(get_drawn_set, axis=1).tolist()

    vals = []

    for i in range(len(df)):

        if i < window:
            vals.append(np.nan)
            continue

        recent = all_sets[i - window:i]

        pair_counter = Counter()

        for s in recent:

            if num in s:

                for other in s:

                    if other != num:
                        pair_counter[other] += 1

        if len(pair_counter) == 0:
            vals.append(0)
        else:
            vals.append(
                np.mean(list(pair_counter.values()))
            )

    return pd.Series(vals, index=df.index)


# ============================================================
# PER NUMBER FEATURES
# ============================================================

def make_per_number_features(df, num):

    feat = pd.DataFrame(index=df.index)

    hit = df.apply(
        lambda r: int(num in get_drawn_set(r)),
        axis=1,
    )

    feat["n_hit_prev"] = hit.shift(1)

    # --------------------------------------------------------
    # LAST SEEN
    # --------------------------------------------------------

    gaps = []

    last = None

    for i, h in enumerate(hit):

        if last is None:
            gaps.append(np.nan)
        else:
            gaps.append(i - last)

        if h == 1:
            last = i

    feat["n_last_seen"] = (
        pd.Series(gaps).shift(1)
    )

    # --------------------------------------------------------
    # FREQUENCY
    # --------------------------------------------------------

    for w in [10, 20, 50]:

        feat[f"n_freq_{w}"] = (
            hit.shift(1)
            .rolling(w)
            .mean()
        )

    # --------------------------------------------------------
    # GAP STATS
    # --------------------------------------------------------

    feat["n_gap_mean_20"] = (
        feat["n_last_seen"]
        .rolling(20)
        .mean()
    )

    feat["n_gap_std_20"] = (
        feat["n_last_seen"]
        .rolling(20)
        .std()
    )

    # --------------------------------------------------------
    # GRAPH
    # --------------------------------------------------------

    feat["n_graph_strength"] = (
        make_graph_feature(df, num)
    )

    # --------------------------------------------------------
    # BAYES
    # --------------------------------------------------------

    alpha = 6
    beta = 39

    feat["n_bayes_freq_20"] = (
        (
            feat["n_freq_20"] * 20 + alpha
        )
        /
        (20 + alpha + beta)
    )

    feat["n_even"] = float(num % 2 == 0)

    feat["n_decade"] = float(num // 10)

    return feat


# ============================================================
# DATASET
# ============================================================

def build_dataset(df, num, global_feat):

    per_feat = make_per_number_features(df, num)

    feat = pd.concat(
        [global_feat, per_feat],
        axis=1,
    )

    feat = feat.fillna(-1)

    y = df.apply(
        lambda r: int(num in get_drawn_set(r)),
        axis=1,
    )

    combined = pd.concat(
        [
            feat,
            y.rename("y"),
        ],
        axis=1,
    )

    combined = combined.reset_index(drop=True)

    train_df = combined.iloc[:-1].copy()

    pred_row = combined.iloc[[-1]].copy()

    feature_cols = [c for c in feat.columns]

    return train_df, pred_row, feature_cols


# ============================================================
# WALK FORWARD
# ============================================================

def walk_forward_splits(
    n,
    min_train=200,
    val_size=30,
):

    splits = []

    start = min_train

    while start + val_size <= n:

        tr = np.arange(0, start)

        va = np.arange(start, start + val_size)

        splits.append((tr, va))

        start += val_size

    return splits


# ============================================================
# DECAY
# ============================================================

def make_decay_weights(n, decay=0.995):

    idx = np.arange(n)

    w = decay ** (n - 1 - idx)

    return w / w.mean()


# ============================================================
# POS WEIGHT
# ============================================================

def calc_scale_pos_weight(y):

    pos = max(1, int(np.sum(y == 1)))

    neg = max(1, int(np.sum(y == 0)))

    return neg / pos


# ============================================================
# MODEL
# ============================================================

def make_xgb(
    scale_pos_weight=6,
    use_early_stopping=False,
):

    params = dict(

        n_estimators=120,
        max_depth=2,
        learning_rate=0.03,

        subsample=0.65,
        colsample_bytree=0.45,

        min_child_weight=8,
        gamma=2.0,

        reg_alpha=0.5,
        reg_lambda=3.0,

        objective="binary:logistic",
        eval_metric="logloss",

        scale_pos_weight=scale_pos_weight,

        random_state=42,
        n_jobs=-1,
    )

    if use_early_stopping:
        params["early_stopping_rounds"] = 20

    return XGBClassifier(**params)


# ============================================================
# TRAIN ONE
# ============================================================

def train_one(
    combined,
    feature_cols,
    val_size=30,
    min_train=200,
    train_window=0,
    decay=0.995,
):

    X = combined[feature_cols].values

    y = combined["y"].values

    splits = walk_forward_splits(
        len(X),
        min_train=min_train,
        val_size=val_size,
    )

    precs = []
    recs = []
    praucs = []
    briers = []

    # ========================================================
    # WALK FORWARD CV
    # ========================================================

    for tr_idx, va_idx in splits:

        y_tr_check = y[tr_idx]
        y_va_check = y[va_idx]

        if len(np.unique(y_tr_check)) < 2:
            continue

        if len(np.unique(y_va_check)) < 2:
            continue

        if train_window > 0:
            tr_idx = tr_idx[-train_window:]

        X_tr = X[tr_idx]
        y_tr = y[tr_idx]

        X_va = X[va_idx]
        y_va = y[va_idx]

        spw = calc_scale_pos_weight(y_tr)

        sw_bal = compute_sample_weight(
            class_weight="balanced",
            y=y_tr,
        )

        sw_decay = make_decay_weights(
            len(y_tr),
            decay=decay,
        )

        sw = sw_bal * sw_decay

        mdl = make_xgb(
            spw,
            use_early_stopping=True,
        )

        mdl.fit(
            X_tr,
            y_tr,
            sample_weight=sw,
            eval_set=[(X_va, y_va)],
            verbose=False,
        )

        # ====================================================
        # CALIBRATION
        # ====================================================

        try:

            cal = CalibratedClassifierCV(
                FrozenEstimator(mdl),
                method="sigmoid",
                cv=2,
            )

            cal.fit(X_va, y_va)

            pred_prob = (
                cal.predict_proba(X_va)[:, 1]
            )

        except Exception:

            pred_prob = (
                mdl.predict_proba(X_va)[:, 1]
            )

            cal = mdl

        pred_bin = (
            pred_prob >= 0.5
        ).astype(int)

        precs.append(
            precision_score(
                y_va,
                pred_bin,
                zero_division=0,
            )
        )

        recs.append(
            recall_score(
                y_va,
                pred_bin,
                zero_division=0,
            )
        )

        praucs.append(
            average_precision_score(
                y_va,
                pred_prob,
            )
        )

        briers.append(
            brier_score_loss(
                y_va,
                pred_prob,
            )
        )

    # ========================================================
    # FINAL FULL RETRAIN
    # ========================================================

    spw = calc_scale_pos_weight(y)

    sw_bal = compute_sample_weight(
        class_weight="balanced",
        y=y,
    )

    sw_decay = make_decay_weights(
        len(y),
        decay=decay,
    )

    sw = sw_bal * sw_decay

    final_xgb = make_xgb(
        spw,
        use_early_stopping=False,
    )

    final_xgb.fit(
        X,
        y,
        sample_weight=sw,
        verbose=False,
    )

    # ========================================================
    # FINAL CALIBRATION
    # ========================================================

    try:

        final_model = CalibratedClassifierCV(
            FrozenEstimator(final_xgb),
            method="sigmoid",
            cv=3,
        )

        final_model.fit(X, y)

    except Exception:

        print(
            "WARNING: calibration failed -> raw XGB"
        )

        final_model = final_xgb

    return {
        "model": final_model,
        "precision": float(np.mean(precs)),
        "recall": float(np.mean(recs)),
        "prauc": float(np.mean(praucs)),
        "brier": float(np.mean(briers)),
    }


# ============================================================
# TRAIN ALL
# ============================================================

def train_all(
    df,
    global_feat,
    val_size=30,
    min_train=200,
    train_window=0,
    decay=0.995,
):

    results = {}

    for num in ALL_NUMS:

        train_df, pred_row, feature_cols = (
            build_dataset(
                df,
                num,
                global_feat,
            )
        )

        result = train_one(
            train_df,
            feature_cols,
            val_size=val_size,
            min_train=min_train,
            train_window=train_window,
            decay=decay,
        )

        result["feature_cols"] = feature_cols
        result["predict_row"] = pred_row

        results[num] = result

        print(
            f"[{num:02d}] "
            f"prec={result['precision']:.4f} "
            f"prauc={result['prauc']:.4f}"
        )

    return results


# ============================================================
# ENSEMBLE
# ============================================================

def train_ensemble(df):

    all_results = []

    for cfg in ENSEMBLE_CONFIGS:

        print(f"\n===== {cfg['label']} =====")

        gf = make_global_features(df)

        r = train_all(
            df,
            gf,
            train_window=cfg["train_window"],
            decay=cfg["decay"],
        )

        all_results.append(r)

    return all_results


# ============================================================
# PREDICT
# ============================================================

def predict_ensemble(
    all_results,
    top_k=8,
):

    rows = []

    for num in ALL_NUMS:

        probs = []

        for r in all_results:

            rr = r[num]

            X = rr["predict_row"][
                rr["feature_cols"]
            ].values

            prob = rr["model"].predict_proba(X)[0][1]

            probs.append(prob)

        rows.append({
            "num": num,
            "prob": np.mean(probs),
        })

    out = pd.DataFrame(rows)

    out = out.sort_values(
        "prob",
        ascending=False,
    )

    return out.head(top_k)


# ============================================================
# RANDOM BASELINE
# ============================================================

def simulate_random_baseline(
    draws=10000,
    top_k=8,
):

    vals = []

    nums = list(range(1, 46))

    for _ in range(draws):

        pred = set(
            np.random.choice(
                nums,
                top_k,
                replace=False,
            )
        )

        actual = set(
            np.random.choice(
                nums,
                6,
                replace=False,
            )
        )

        vals.append(
            len(pred & actual) / top_k
        )

    return {
        "mean": np.mean(vals),
        "std": np.std(vals),
        "p95": np.quantile(vals, 0.95),
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print(f"Version: {VERSION}")

    print("Loading data...")

    df = load_from_db()

    print(f"Loaded {len(df)} rows")

    baseline = simulate_random_baseline()

    print("\nRandom baseline")
    print(baseline)

    print("\nTraining ensemble...")

    all_results = train_ensemble(df)

    print("\nPredicting next draw...")

    pred = predict_ensemble(
        all_results,
        top_k=8,
    )

    print("\nTOP PREDICTIONS")
    print(pred)

    print("\nChosen numbers:")
    print(sorted(pred["num"].tolist()))


if __name__ == "__main__":
    main()