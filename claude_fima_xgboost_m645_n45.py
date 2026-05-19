"""
claude_fima_triboost_m645_n45.py
────────────────────────────────
Nâng cấp M645: Ensemble XGBoost + LightGBM + CatBoost
Cùng kiến trúc walk-forward, sliding window, time decay với bản gốc.

Thay đổi chính:
  - Thêm make_lgb() / make_cat() builders
  - train_one() nhận learner="xgb"|"lgb"|"cat"
  - ENSEMBLE_CONFIGS mở rộng: mỗi config chạy 3 learners
  - blend_scores(): rank-average hoặc weighted-by-cv-prec
  - score_mode lưu "tri_boost" vào m645kqpredict

Usage:
    python claude_fima_xgboost_m645_n45.py --source db --top 8 --ensemble --save
    python claude_fima_xgboost_m645_n45.py --source db --top 10 --ensemble --blend weighted --save
    python claude_fima_xgboost_m645_n45.py --update_result
"""

import argparse, os, warnings
import numpy as np
import pandas as pd
from collections import Counter
from datetime import datetime, timezone

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import precision_score, recall_score
from sklearn.utils.class_weight import compute_sample_weight

warnings.filterwarnings("ignore")

DATABASE_URL = os.environ.get(
    "DATABASE_URL", "postgresql://postgres:bin@localhost:5432/katalott"
)

ALL_NUMS = list(range(1, 46))

VERSION = "v2.0.0"
DESCRIPTION = (
    "TriBoost M645: XGBoost+LightGBM+CatBoost, 3 configs × 3 learners × 45 binary, "
    "walk-forward CV precision, rank-avg/weighted blend, save to m645kqpredict"
)


# ─────────────────────────────────────────────
# 1. LOAD  (giữ nguyên)
# ─────────────────────────────────────────────


def load_from_db(n_ky=0, qh_ky=20) -> pd.DataFrame:
    import psycopg2

    limit = f"LIMIT {n_ky}" if n_ky > 0 else ""
    sql = f"""
        SELECT * FROM public.m645kqdetail
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


# ─────────────────────────────────────────────
# 2. HELPERS  (giữ nguyên)
# ─────────────────────────────────────────────

THU_MAP = {"T2": 2, "T3": 3, "T4": 4, "T5": 5, "T6": 6, "T7": 7, "CN": 8}
DD_MAP = {"CC": 0, "CL": 1, "LC": 2, "LL": 3}


def _safe_int(s):
    return pd.to_numeric(s, errors="coerce")


def _encode_thu(s):
    return s.astype(str).str.strip().map(THU_MAP)


def _encode_dd(n_first, n_last):
    def _cl(n):
        return "C" if n % 2 == 0 else "L"

    result = []
    for f, l in zip(n_first, n_last):
        if pd.isna(f) or pd.isna(l):
            result.append(np.nan)
        else:
            result.append(DD_MAP.get(_cl(int(f)) + _cl(int(l)), np.nan))
    return pd.Series(result, index=n_first.index)


def _ngay_diff(d1, d2):
    return (
        pd.to_datetime(d1, errors="coerce") - pd.to_datetime(d2, errors="coerce")
    ).dt.days


def get_drawn_set(row):
    return {
        int(row[c])
        for c in ["n1", "n2", "n3", "n4", "n5", "n6"]
        if pd.notna(row[c]) and row[c] != 0
    }


# ─────────────────────────────────────────────
# 3–5. FEATURE ENGINEERING  (giữ nguyên)
# ─────────────────────────────────────────────


def make_global_features(df, lags=3):
    d = df.copy()
    d["dd_enc"] = d["dd"].apply(
        lambda v: {"CC": 0, "CL": 1, "LC": 2, "LL": 3}.get(str(v).strip(), -1)
    )
    s = d["cl6"].astype(str).str.strip()
    for i in range(6):
        d[f"cl6_b{i}"] = s.str[i].map({"C": 0, "L": 1}).fillna(-1).astype(int)
    d["cl6_l_count"] = sum(d[f"cl6_b{i}"] for i in range(6))
    d["cl6_pattern"] = s.apply(
        lambda v: (
            int("".join("0" if c == "C" else "1" for c in v), 2)
            if len(v) == 6
            else np.nan
        )
    )
    d["cl6_changed"] = (d["cl6_pattern"] != d["cl6_pattern"].shift(1)).astype(float)
    d["thu_enc"] = _encode_thu(d["thu"])
    if "jpck" in d.columns:
        d["jpck"] = _safe_int(d["jpck"]).fillna(0)
    if "is_jp" in d.columns:
        d["is_jp_enc"] = d["is_jp"].astype(int)
    if all(c in d.columns for c in ["pwn1", "pwn6"]):
        pn = [_safe_int(d[f"pwn{i}"]) for i in range(1, 7)]
        if "pwthu" in d.columns:
            d["pw_thu_enc"] = _encode_thu(d["pwthu"])
        if "pwngay" in d.columns:
            d["pw_ngay_diff"] = _ngay_diff(d["ngay"], d["pwngay"])
        d["pw_dd"] = _encode_dd(pn[0], pn[5])
        d["pw_sc"] = sum((n % 2 == 0).astype(float) for n in pn)

    feat = pd.DataFrame(index=d.index)
    feat["thu_enc"] = d["thu_enc"]
    prev_cols = [
        "dd_enc",
        "sc",
        "sum",
        "ke",
        "cl6_b0",
        "cl6_b1",
        "cl6_b2",
        "cl6_b3",
        "cl6_b4",
        "cl6_b5",
        "cl6_l_count",
        "cl6_pattern",
        "cl6_changed",
        "dec0",
        "dec1",
        "dec2",
        "dec3",
        "dec4",
    ]
    for c in ["jpck", "is_jp_enc"]:
        if c in d.columns:
            prev_cols.append(c)
    for col in [c for c in prev_cols if c in d.columns]:
        feat[f"{col}_prev"] = d[col].shift(1)
    lag_cols = [
        c for c in ["dd_enc", "sc", "cl6_l_count", "cl6_pattern"] if c in d.columns
    ]
    for lag in range(2, lags + 1):
        for col in lag_cols:
            feat[f"{col}_lag{lag}"] = d[col].shift(lag)
    for w in [3, 5]:
        for col in ["sc", "cl6_l_count"]:
            if col in d.columns:
                base = d[col].shift(1)
                feat[f"{col}_roll{w}_mean"] = base.rolling(w).mean()
                feat[f"{col}_roll{w}_std"] = base.rolling(w).std()
    for c in ["pw_thu_enc", "pw_ngay_diff", "pw_dd", "pw_sc"]:
        if c in d.columns:
            feat[c] = d[c]
    if "pw_dd" in feat.columns and "dd_enc_prev" in feat.columns:
        feat["cross_pw_dd_match"] = (feat["pw_dd"] == feat["dd_enc_prev"]).astype(float)
    if "pw_sc" in feat.columns and "sc_prev" in feat.columns:
        feat["cross_sc_pw"] = feat["sc_prev"] * feat["pw_sc"]
    return feat


def make_per_number_features(df, num):
    feat = pd.DataFrame(index=df.index)
    hit = df.apply(lambda r: int(num in get_drawn_set(r)), axis=1)
    feat["n_hit_prev"] = hit.shift(1)
    last_seen, last = [], np.nan
    for i, h in enumerate(hit):
        last_seen.append(np.nan if i == 0 else (np.nan if pd.isna(last) else i - last))
        if h == 1:
            last = i
    feat["n_last_seen"] = pd.Series(last_seen, index=df.index).shift(1)
    streak, cur = [], 0
    for i, h in enumerate(hit):
        streak.append(np.nan if i == 0 else float(cur))
        cur = 0 if h == 1 else cur + 1
    feat["n_streak"] = pd.Series(streak, index=df.index).shift(1)
    hs = hit.shift(1)
    feat["n_freq_10"] = hs.rolling(10).mean()
    feat["n_freq_20"] = hs.rolling(20).mean()
    feat["n_freq_50"] = hs.rolling(50).mean()
    if "qhl" in df.columns:

        def _in_qhl(v):
            if pd.isna(v) or v == "":
                return 0.0
            try:
                return float(num in [int(x) for x in str(v).split(",") if x.strip()])
            except:
                return 0.0

        feat["n_is_qhl"] = df["qhl"].shift(1).apply(_in_qhl)
    prev_nums = df.apply(get_drawn_set, axis=1).shift(1)
    pair_vals = []
    for i in range(len(df)):
        if i < 31:
            pair_vals.append(np.nan)
            continue
        pn = prev_nums.iloc[i]
        if not pn:
            pair_vals.append(0.0)
            continue
        count = sum(
            1
            for j in range(i - 30, i)
            if num in get_drawn_set(df.iloc[j])
            and len(get_drawn_set(df.iloc[j]) & pn) > 0
        )
        pair_vals.append(count / 30.0)
    feat["n_pair_freq"] = pd.Series(pair_vals, index=df.index)
    feat["n_is_even"] = float(num % 2 == 0)
    feat["n_dec"] = float(num // 10)
    return feat


def build_dataset(df, num, global_feat):
    per_feat = make_per_number_features(df, num)
    all_feat = pd.concat([global_feat, per_feat], axis=1).fillna(-1)
    target = df.apply(lambda r: int(num in get_drawn_set(r)), axis=1)
    combined = (
        pd.concat([all_feat, target.rename("y"), df["ky"]], axis=1)
        .dropna(subset=["n_freq_50"])
        .reset_index(drop=True)
    )
    feature_cols = list(all_feat.columns)
    combined_train = combined.iloc[:-1].reset_index(drop=True)
    predict_row = combined.iloc[[-1]]
    return combined_train, predict_row, feature_cols


# ─────────────────────────────────────────────
# 6. MODEL BUILDERS  ← THÊM LGB + CAT
# ─────────────────────────────────────────────


def make_xgb(params=None):
    defaults = dict(
        n_estimators=300,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.7,
        min_child_weight=3,
        gamma=0.1,
        reg_alpha=0.05,
        reg_lambda=1.0,
        scale_pos_weight=6,  # ~6/45
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
        early_stopping_rounds=30,
    )
    if params:
        defaults.update(params)
    return XGBClassifier(**defaults)


def make_lgb(params=None):
    """
    LightGBM binary classifier.
    - num_leaves=31 (conservative, tránh overfit)
    - min_child_samples=20 (tương đương min_child_weight XGB)
    - is_unbalance=True thay vì scale_pos_weight (LGB native)
    - feature_fraction = colsample_bytree tương đương
    - early_stopping: dùng callbacks
    """
    defaults = dict(
        n_estimators=400,
        num_leaves=31,
        max_depth=-1,
        learning_rate=0.05,
        subsample=0.8,  # bagging_fraction
        subsample_freq=1,  # bật bagging
        colsample_bytree=0.7,  # feature_fraction
        min_child_samples=20,
        reg_alpha=0.05,
        reg_lambda=1.0,
        is_unbalance=True,  # tự xử lý imbalance
        objective="binary",
        metric="binary_logloss",
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )
    if params:
        defaults.update(params)
    return LGBMClassifier(**defaults)


def make_cat(params=None):
    """
    CatBoost binary classifier.
    - depth=4 (cân bằng XGB depth=3)
    - auto_class_weights='Balanced' (native imbalance handling)
    - od_type='Iter' + od_wait=30 (early stopping tương đương)
    - CatBoost dùng ordered boosting → ít overfit hơn trên tập nhỏ
    """
    defaults = dict(
        iterations=300,
        depth=4,
        learning_rate=0.05,
        l2_leaf_reg=3.0,
        subsample=0.8,
        colsample_bylevel=0.7,
        auto_class_weights="Balanced",
        loss_function="Logloss",
        eval_metric="Precision",
        random_seed=42,
        od_type="Iter",
        od_wait=30,
        verbose=0,
        thread_count=-1,
    )
    if params:
        defaults.update(params)
    return CatBoostClassifier(**defaults)


LEARNER_BUILDERS = {
    "xgb": make_xgb,
    "lgb": make_lgb,
    "cat": make_cat,
}


# ─────────────────────────────────────────────
# 7. WALK-FORWARD  (giữ nguyên)
# ─────────────────────────────────────────────


def _walk_forward_splits(n, min_train, val_size):
    splits, start = [], min_train
    while start + val_size <= n:
        splits.append((np.arange(0, start), np.arange(start, start + val_size)))
        start += val_size
    return splits


def _make_decay_weights(n, decay=0.995):
    idx = np.arange(n)
    w = decay ** (n - 1 - idx)
    return w / w.mean()


# ─────────────────────────────────────────────
# 8. TRAIN ONE NUMBER  ← hỗ trợ 3 learners
# ─────────────────────────────────────────────


def train_one(
    combined,
    feature_cols,
    val_size,
    min_train,
    train_window=0,
    decay=0.995,
    learner="xgb",
    model_params=None,
):
    """
    Walk-forward train cho 1 số, 1 learner.
    Trả về (model, mean_prec, mean_rec).
    """
    X = combined[feature_cols].values
    y = combined["y"].values
    builder = LEARNER_BUILDERS[learner]

    splits = _walk_forward_splits(len(X), min_train, val_size)
    prec_list, rec_list = [], []

    for tr_idx, va_idx in splits:
        X_tr, X_va = X[tr_idx], X[va_idx]
        y_tr, y_va = y[tr_idx], y[va_idx]
        if y_tr.sum() == 0:
            continue

        if train_window > 0 and len(tr_idx) > train_window:
            tr_idx = tr_idx[-train_window:]
            X_tr, y_tr = X[tr_idx], y[tr_idx]

        sw_bal = compute_sample_weight("balanced", y_tr)
        sw_decay = _make_decay_weights(len(y_tr), decay)
        sw_tr = sw_bal * sw_decay

        mdl = builder(model_params)

        # Fit theo từng learner (early stopping API khác nhau)
        if learner == "xgb":
            mdl.fit(
                X_tr, y_tr, sample_weight=sw_tr, eval_set=[(X_va, y_va)], verbose=False
            )
        elif learner == "lgb":
            from lightgbm import early_stopping, log_evaluation

            mdl.fit(
                X_tr,
                y_tr,
                sample_weight=sw_tr,
                eval_set=[(X_va, y_va)],
                callbacks=[early_stopping(30, verbose=False), log_evaluation(-1)],
            )
        elif learner == "cat":
            from catboost import Pool

            mdl.fit(
                Pool(X_tr, y_tr, weight=sw_tr), eval_set=Pool(X_va, y_va), verbose=False
            )

        pred = mdl.predict(X_va)
        prec_list.append(precision_score(y_va, pred, zero_division=0))
        rec_list.append(recall_score(y_va, pred, zero_division=0))

    # Final model (no early stopping, full data / sliding window)
    if train_window > 0 and len(X) > train_window:
        X_fin, y_fin = X[-train_window:], y[-train_window:]
    else:
        X_fin, y_fin = X, y

    sw_fin = compute_sample_weight("balanced", y_fin) * _make_decay_weights(
        len(y_fin), decay
    )

    fin_params = dict(model_params or {})
    if learner == "xgb":
        fin_params["early_stopping_rounds"] = None
    elif learner == "lgb":
        fin_params.pop("early_stopping_rounds", None)
    elif learner == "cat":
        fin_params["od_type"] = "Iter"
        fin_params["od_wait"] = 9999

    final = builder(fin_params)
    if learner in ("xgb", "lgb"):
        ##      final.fit(X_fin, y_fin, sample_weight=sw_fin, verbose=False)
        final.fit(X_fin, y_fin, sample_weight=sw_fin)
    else:  # cat
        from catboost import Pool

        final.fit(Pool(X_fin, y_fin, weight=sw_fin), verbose=False)

    mean_prec = float(np.mean(prec_list)) if prec_list else 0.0
    mean_rec = float(np.mean(rec_list)) if rec_list else 0.0
    return final, mean_prec, mean_rec


# ─────────────────────────────────────────────
# 9. TRAIN ALL 45 (cho 1 learner)
# ─────────────────────────────────────────────


def train_all(
    df,
    global_feat,
    lags,
    val_size,
    min_train,
    train_window=0,
    decay=0.995,
    learner="xgb",
    model_params=None,
):
    baseline = 6 * 8 / 45 / 8
    results = {}
    print(f"\n  Learner: {learner.upper()}  |  Baseline prec: {baseline:.1%}")
    print(f"  {'Num':>4}  {'CV Prec':>8}  {'CV Rec':>8}  {'Flag'}")
    print(f"  {'─' * 40}")

    for num in ALL_NUMS:
        combined_train, predict_row, feature_cols = build_dataset(df, num, global_feat)
        model, prec, rec = train_one(
            combined_train,
            feature_cols,
            val_size,
            min_train,
            train_window,
            decay,
            learner,
            model_params,
        )
        results[num] = dict(
            model=model,
            prec=prec,
            rec=rec,
            feature_cols=feature_cols,
            predict_row=predict_row,
        )
        flag = "✅" if prec > baseline else "⚠️ "
        print(f"  {num:>4}  {prec:>8.3f}  {rec:>8.3f}  {flag}")

    precs = [results[n]["prec"] for n in ALL_NUMS]
    recs = [results[n]["rec"] for n in ALL_NUMS]
    print(
        f"\n  Mean CV Prec: {np.mean(precs):.3f}  |  Mean CV Rec: {np.mean(recs):.3f}"
    )
    print(f"  Beat baseline: {sum(1 for p in precs if p > baseline)}/45")
    return results


# ─────────────────────────────────────────────
# 10. ENSEMBLE CONFIGS  ← mở rộng 3 learners
# ─────────────────────────────────────────────

ENSEMBLE_CONFIGS = [
    {"lags": 3, "decay": 0.995, "train_window": 0, "label": "C1(lag3,d0.995,all)"},
    {"lags": 5, "decay": 0.990, "train_window": 0, "label": "C2(lag5,d0.990,all)"},
    {"lags": 3, "decay": 0.995, "train_window": 400, "label": "C3(lag3,d0.995,w400)"},
]
LEARNERS = ["xgb", "lgb", "cat"]


def train_ensemble_tri(df, val_size, min_train):
    """
    Chạy 3 configs × 3 learners = 9 runs × 45 models = 405 models tổng.
    Trả về list[dict] mỗi phần tử có: results, learner, cfg_label.
    """
    all_runs = []
    for cfg in ENSEMBLE_CONFIGS:
        gf = make_global_features(df, lags=cfg["lags"])
        for lrn in LEARNERS:
            print(f"\n  ══ {cfg['label']} | {lrn.upper()} ══")
            r = train_all(
                df,
                gf,
                lags=cfg["lags"],
                val_size=val_size,
                min_train=min_train,
                train_window=cfg["train_window"],
                decay=cfg["decay"],
                learner=lrn,
            )
            all_runs.append({"results": r, "learner": lrn, "label": cfg["label"]})
    return all_runs


# ─────────────────────────────────────────────
# 11. BLEND  ← HAI CHIẾN LƯỢC
# ─────────────────────────────────────────────


def blend_scores(all_runs, top_k=8, filter_threshold=0.12, strategy="rank_avg"):
    """
    strategy:
      "rank_avg"  — trung bình rank xác suất qua tất cả runs (rank thấp = tốt)
      "weighted"  — trung bình xác suất, trọng số = CV_prec của run đó
    """
    # Tính prob + cv_prec cho từng (num, run)
    run_probs = []  # list[dict num→(prob, cv_prec)]
    for run in all_runs:
        row = {}
        for num in ALL_NUMS:
            res = run["results"][num]
            last = res["predict_row"][res["feature_cols"]].values
            prob = res["model"].predict_proba(last)[0][1]
            row[num] = (prob, res["prec"])
        run_probs.append(row)

    if strategy == "rank_avg":
        # rank trong từng run (1 = prob cao nhất)
        rank_sum = {n: 0.0 for n in ALL_NUMS}
        for row in run_probs:
            sorted_nums = sorted(ALL_NUMS, key=lambda n: -row[n][0])
            for r, n in enumerate(sorted_nums, 1):
                rank_sum[n] += r
        n_runs = len(run_probs)
        # cv_prec = trung bình qua tất cả runs
        cv_avg = {n: np.mean([row[n][1] for row in run_probs]) for n in ALL_NUMS}
        # score = âm rank_avg (thấp = tốt → sort ngược lại)
        scores = [(n, -rank_sum[n] / n_runs, cv_avg[n]) for n in ALL_NUMS]

    elif strategy == "weighted":
        agg = {}
        for num in ALL_NUMS:
            probs = [row[num][0] for row in run_probs]
            cv_ps = [row[num][1] for row in run_probs]
            w = np.array(cv_ps)
            w_sum = w.sum()
            if w_sum > 0:
                wprob = float(np.dot(probs, w) / w_sum)
            else:
                wprob = float(np.mean(probs))
            agg[num] = (wprob, float(np.mean(cv_ps)))
        scores = [(n, agg[n][0], agg[n][1]) for n in ALL_NUMS]

    # Filter
    if filter_threshold > 0:
        before = len(scores)
        scores = [(n, s, c) for n, s, c in scores if c >= filter_threshold]
        removed = before - len(scores)
        if removed:
            print(f"\n  🔍 Filter {filter_threshold:.0%}: loại {removed} số")

    scores.sort(key=lambda x: -x[1])
    return scores


# ─────────────────────────────────────────────
# 12. SHOW PREDICTION  (giữ nguyên)
# ─────────────────────────────────────────────


def show_prediction(df, proba_list, top_k=8):
    last_ky = (df["ky"].iloc[-1] or "").strip()
    top = proba_list[:top_k]
    print(f"\n{'═' * 58}")
    print(f"📌 Input : kỳ {last_ky}")
    print(f"🎯 Dự đoán {top_k} số cho kỳ tiếp theo:")
    print(f"{'═' * 58}")
    nums_chosen = []
    for rank, (num, score, cvp) in enumerate(top, 1):
        bar = "█" * int(abs(score) * 20 if score < 0 else score * 30)
        print(f"  #{rank:>2}  Số {num:>2}  score={score:.4f}  CV_prec={cvp:.3f}  {bar}")
        nums_chosen.append(num)
    print(f"\n  → Chọn: {sorted(nums_chosen)}")
    n_even = sum(1 for n in nums_chosen if n % 2 == 0)
    dec_cnt = Counter(n // 10 for n in nums_chosen)
    print(f"  → Chẵn/Lẻ : {n_even}C / {top_k - n_even}L")
    print(
        f"  → Chục    : "
        + "  ".join(
            f"{d * 10}x:{dec_cnt.get(d, 0)}" for d in range(5) if dec_cnt.get(d, 0) > 0
        )
    )
    print(f"{'═' * 58}\n")
    return nums_chosen


# ─────────────────────────────────────────────
# 13. SAVE PREDICT  (cập nhật score_mode)
# ─────────────────────────────────────────────


def save_predict(args, df, proba_list, cv_prec_mean):
    import psycopg2

    input_ky = (df["ky"].iloc[-1] or "").strip()
    try:
        predict_ky = str(int(input_ky) + 1).zfill(len(input_ky))
    except:
        predict_ky = None

    score_mode = f"tri_{getattr(args, 'blend', 'rank_avg')}"

    row = {}
    for i in range(1, 19):
        if i <= len(proba_list):
            num, score, cvp = proba_list[i - 1]
            row[f"n{i}"] = int(num)
            row[f"p{i}"] = float(abs(score))  # rank_avg scores âm → lấy abs
            row[f"cvp{i}"] = float(cvp)
        else:
            row[f"n{i}"] = row[f"p{i}"] = row[f"cvp{i}"] = None

    sql = """
    INSERT INTO public.m645kqpredict (
        run_at, version, description,
        source, lags, train_window, decay, top_k, score_mode,
        cv_prec_mean, input_ky, predict_ky,
        n1,p1,cvp1, n2,p2,cvp2, n3,p3,cvp3, n4,p4,cvp4,
        n5,p5,cvp5, n6,p6,cvp6, n7,p7,cvp7, n8,p8,cvp8,
        n9,p9,cvp9, n10,p10,cvp10, n11,p11,cvp11, n12,p12,cvp12,
        n13,p13,cvp13, n14,p14,cvp14, n15,p15,cvp15, n16,p16,cvp16,
        n17,p17,cvp17, n18,p18,cvp18
    ) VALUES (
        %(run_at)s, %(version)s, %(description)s,
        %(source)s, %(lags)s, %(train_window)s, %(decay)s,
        %(top_k)s, %(score_mode)s, %(cv_prec_mean)s,
        %(input_ky)s, %(predict_ky)s,
        %(n1)s,%(p1)s,%(cvp1)s, %(n2)s,%(p2)s,%(cvp2)s,
        %(n3)s,%(p3)s,%(cvp3)s, %(n4)s,%(p4)s,%(cvp4)s,
        %(n5)s,%(p5)s,%(cvp5)s, %(n6)s,%(p6)s,%(cvp6)s,
        %(n7)s,%(p7)s,%(cvp7)s, %(n8)s,%(p8)s,%(cvp8)s,
        %(n9)s,%(p9)s,%(cvp9)s, %(n10)s,%(p10)s,%(cvp10)s,
        %(n11)s,%(p11)s,%(cvp11)s, %(n12)s,%(p12)s,%(cvp12)s,
        %(n13)s,%(p13)s,%(cvp13)s, %(n14)s,%(p14)s,%(cvp14)s,
        %(n15)s,%(p15)s,%(cvp15)s, %(n16)s,%(p16)s,%(cvp16)s,
        %(n17)s,%(p17)s,%(cvp17)s, %(n18)s,%(p18)s,%(cvp18)s
    )
    ON CONFLICT (predict_ky, version, score_mode) DO UPDATE SET
        run_at=EXCLUDED.run_at, description=EXCLUDED.description,
        cv_prec_mean=EXCLUDED.cv_prec_mean, input_ky=EXCLUDED.input_ky,
        n1=EXCLUDED.n1,p1=EXCLUDED.p1,cvp1=EXCLUDED.cvp1,
        n2=EXCLUDED.n2,p2=EXCLUDED.p2,cvp2=EXCLUDED.cvp2,
        n3=EXCLUDED.n3,p3=EXCLUDED.p3,cvp3=EXCLUDED.cvp3,
        n4=EXCLUDED.n4,p4=EXCLUDED.p4,cvp4=EXCLUDED.cvp4,
        n5=EXCLUDED.n5,p5=EXCLUDED.p5,cvp5=EXCLUDED.cvp5,
        n6=EXCLUDED.n6,p6=EXCLUDED.p6,cvp6=EXCLUDED.cvp6,
        n7=EXCLUDED.n7,p7=EXCLUDED.p7,cvp7=EXCLUDED.cvp7,
        n8=EXCLUDED.n8,p8=EXCLUDED.p8,cvp8=EXCLUDED.cvp8,
        n9=EXCLUDED.n9,p9=EXCLUDED.p9,cvp9=EXCLUDED.cvp9,
        n10=EXCLUDED.n10,p10=EXCLUDED.p10,cvp10=EXCLUDED.cvp10,
        n11=EXCLUDED.n11,p11=EXCLUDED.p11,cvp11=EXCLUDED.cvp11,
        n12=EXCLUDED.n12,p12=EXCLUDED.p12,cvp12=EXCLUDED.cvp12,
        n13=EXCLUDED.n13,p13=EXCLUDED.p13,cvp13=EXCLUDED.cvp13,
        n14=EXCLUDED.n14,p14=EXCLUDED.p14,cvp14=EXCLUDED.cvp14,
        n15=EXCLUDED.n15,p15=EXCLUDED.p15,cvp15=EXCLUDED.cvp15,
        n16=EXCLUDED.n16,p16=EXCLUDED.p16,cvp16=EXCLUDED.cvp16,
        n17=EXCLUDED.n17,p17=EXCLUDED.p17,cvp17=EXCLUDED.cvp17,
        n18=EXCLUDED.n18,p18=EXCLUDED.p18,cvp18=EXCLUDED.cvp18,
        updated_at=now();
    """
    params = {
        "run_at": datetime.now(timezone.utc),
        "version": VERSION,
        "description": DESCRIPTION,
        "source": getattr(args, "source", "db"),
        "lags": getattr(args, "lags", 3),
        "train_window": getattr(args, "train_window", 0),
        "decay": getattr(args, "decay", 0.995),
        "top_k": getattr(args, "top", 8),
        "score_mode": score_mode,
        "cv_prec_mean": float(cv_prec_mean),
        "input_ky": input_ky,
        "predict_ky": predict_ky,
        **row,
    }
    conn = psycopg2.connect(DATABASE_URL)
    conn.autocommit = False
    cur = conn.cursor()
    try:
        cur.execute(sql, params)
        conn.commit()
        print(
            f"\n💾 Đã lưu → m645kqpredict  (predict_ky={predict_ky}, mode={score_mode})"
        )
    except Exception as e:
        conn.rollback()
        print(f"\n⚠️  Lưu thất bại: {e}")
    finally:
        cur.close()
        conn.close()


# ─────────────────────────────────────────────
# 14. UPDATE RESULT  (giữ nguyên)
# ─────────────────────────────────────────────


def update_result():
    import psycopg2, psycopg2.extras

    conn = psycopg2.connect(DATABASE_URL)
    conn.autocommit = False
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    try:
        cur.execute("""
            SELECT id, predict_ky, top_k, n1,n2,n3,n4,n5,n6,n7,n8
            FROM public.m645kqpredict
            WHERE actual_n1 IS NULL AND predict_ky IS NOT NULL
            ORDER BY predict_ky
        """)
        pending = cur.fetchall()
        print(f"📋 Tìm thấy {len(pending)} kỳ chưa có kết quả")
        updated = 0
        for row in pending:
            predict_ky = (row["predict_ky"] or "").strip()
            cur.execute(
                "SELECT n1,n2,n3,n4,n5,n6 FROM public.m645kqdetail WHERE ky=%s LIMIT 1",
                (predict_ky,),
            )
            actual = cur.fetchone()
            if actual is None:
                print(f"  kỳ {predict_ky}: chưa có kết quả, bỏ qua")
                continue
            actual_nums = {
                int(actual[f"n{i}"])
                for i in range(1, 7)
                if actual[f"n{i}"] is not None and actual[f"n{i}"] != 0
            }
            top_k = int(row["top_k"] or 8)
            pred_nums = [
                int(row[f"n{i}"])
                for i in range(1, 19)
                if row.get(f"n{i}") is not None and len([x for x in [] if True]) < top_k
            ]
            pred_nums = pred_nums[:top_k]
            hits = set(pred_nums) & actual_nums
            hit_cnt = len(hits)
            prec_val = hit_cnt / top_k if top_k > 0 else 0.0
            cur.execute(
                """
                UPDATE public.m645kqpredict SET
                    actual_n1=%(an1)s,actual_n2=%(an2)s,actual_n3=%(an3)s,
                    actual_n4=%(an4)s,actual_n5=%(an5)s,actual_n6=%(an6)s,
                    hit_cnt=%(hit_cnt)s,precision_val=%(prec)s,updated_at=%(upd)s
                WHERE id=%(id)s
            """,
                {
                    "an1": actual["n1"],
                    "an2": actual["n2"],
                    "an3": actual["n3"],
                    "an4": actual["n4"],
                    "an5": actual["n5"],
                    "an6": actual["n6"],
                    "hit_cnt": hit_cnt,
                    "prec": prec_val,
                    "upd": datetime.now(timezone.utc),
                    "id": row["id"],
                },
            )
            flag = "✅" if prec_val > 0.133 else ("🟡" if prec_val > 0 else "⚠️ ")
            print(
                f"  kỳ {predict_ky}: dự={sorted(pred_nums)} thực={sorted(actual_nums)} "
                f"hit={hit_cnt}/{top_k} prec={prec_val:.1%} {flag}"
            )
            updated += 1
        conn.commit()
        print(f"\n✅ Đã update {updated}/{len(pending)} kỳ")
    except Exception as e:
        conn.rollback()
        print(f"[ERROR] {e}")
        raise
    finally:
        cur.close()
        conn.close()


# ─────────────────────────────────────────────
# 15. MAIN
# ─────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="TriBoost M645: XGB+LGB+CAT")
    parser.add_argument("--update_result", action="store_true")
    parser.add_argument("--source", choices=["db", "csv"], default="db")
    parser.add_argument("--file", default="m645kqdetail.csv")
    parser.add_argument("--n_ky", type=int, default=0)
    parser.add_argument("--qh_ky", type=int, default=20)
    parser.add_argument("--lags", type=int, default=3)
    parser.add_argument("--top", type=int, default=8)
    parser.add_argument("--val_size", type=int, default=30)
    parser.add_argument("--min_train", type=int, default=200)
    parser.add_argument("--train_window", type=int, default=0)
    parser.add_argument("--decay", type=float, default=0.995)
    parser.add_argument("--filter", type=float, default=0.12)
    parser.add_argument(
        "--blend",
        choices=["rank_avg", "weighted"],
        default="rank_avg",
        help="Chiến lược blend: rank_avg (mặc định) hoặc weighted by CV_prec",
    )
    parser.add_argument("--save", action="store_true")
    args = parser.parse_args()

    if args.update_result:
        print("🔄 Update kết quả thực tế từ m645kqdetail...")
        update_result()
        return

    print(f"📌 Version : {VERSION}")
    print(f"📝 Desc    : {DESCRIPTION[:80]}...")
    print("📥 Loading data...")
    df = (
        load_from_csv(args.file)
        if args.source == "csv"
        else load_from_db(args.n_ky, args.qh_ky)
    )
    print(f"   {len(df)} kỳ")

    print(
        f"\n🚀 Training TriBoost ensemble "
        f"({len(ENSEMBLE_CONFIGS)} configs × {len(LEARNERS)} learners × 45 số)..."
    )
    all_runs = train_ensemble_tri(df, val_size=args.val_size, min_train=args.min_train)

    print(f"\n🔮 Blending ({args.blend}) + filter={args.filter:.0%}...")
    proba_list = blend_scores(
        all_runs, top_k=args.top, filter_threshold=args.filter, strategy=args.blend
    )
    cv_prec_mean = np.mean([all_runs[0]["results"][n]["prec"] for n in ALL_NUMS])

    show_prediction(df, proba_list, top_k=args.top)

    if args.save:
        save_predict(args, df, proba_list, cv_prec_mean)


if __name__ == "__main__":
    main()
