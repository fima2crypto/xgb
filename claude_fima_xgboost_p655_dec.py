"""
claude_fima_xgboost_p655_dec.py
────────────────────────────────
Predict phan phoi hang chuc cua P655:
  - 6 XGBoost regression model cho dec0..dec5
  - dec0: so luong so trong 1-9   (max 6)
  - dec1: so luong so trong 10-19 (max 6)
  - dec2: so luong so trong 20-29 (max 6)
  - dec3: so luong so trong 30-39 (max 6)
  - dec4: so luong so trong 40-49 (max 6)
  - dec5: so luong so trong 50-55 (max 6)
  - Walk-forward CV voi MAE metric
  - Ensemble 3 configs (M1/M2/M3)
  - Output: count du doan moi hang chuc (tham khao phan phoi)

Usage:
    python claude_fima_xgboost_p655_dec.py --source db
    python claude_fima_xgboost_p655_dec.py --source db --ensemble
    python claude_fima_xgboost_p655_dec.py --source csv --file p655kqdetail.csv
"""

import argparse, os, warnings
import numpy as np
import pandas as pd
from datetime import datetime, timezone

from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error

warnings.filterwarnings("ignore")

DATABASE_URL = os.environ.get(
    "DATABASE_URL", "postgresql://postgres:bin@localhost:5432/katalott"
)

# dec range: (dec_index, start, end inclusive)
DEC_RANGES = [
    (0,  1,  9),
    (1, 10, 19),
    (2, 20, 29),
    (3, 30, 39),
    (4, 40, 49),
    (5, 50, 55),
]

VERSION     = "v1.0.0"
DESCRIPTION = "6 XGBoost regression (dec0-dec5 count), walk-forward CV MAE, ensemble 3 configs"


# ─────────────────────────────────────────────
# 1. LOAD
# ─────────────────────────────────────────────

def load_from_db(n_ky=0, qh_ky=20) -> pd.DataFrame:
    import psycopg2
    limit = f"LIMIT {n_ky}" if n_ky > 0 else ""
    sql = f"""
        SELECT * FROM public.p655kqdetail
        WHERE qh_ky = {qh_ky}
        ORDER BY ky DESC {limit}
    """
    conn = psycopg2.connect(DATABASE_URL)
    df   = pd.read_sql(sql, conn)
    conn.close()
    return df.sort_values("ky").reset_index(drop=True)

def load_from_csv(filepath) -> pd.DataFrame:
    df = pd.read_csv(filepath)
    return df.sort_values("ky").reset_index(drop=True)


# ─────────────────────────────────────────────
# 2. HELPERS
# ─────────────────────────────────────────────

THU_MAP = {"T2":2,"T3":3,"T4":4,"T5":5,"T6":6,"T7":7,"CN":8}
DD_MAP  = {"CC":0,"CL":1,"LC":2,"LL":3}

def _safe_int(s) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")

def _encode_thu(series) -> pd.Series:
    return series.astype(str).str.strip().map(THU_MAP)

def _encode_dd(n_first, n_last) -> pd.Series:
    def _cl(n): return "C" if n % 2 == 0 else "L"
    result = []
    for f, l in zip(n_first, n_last):
        if pd.isna(f) or pd.isna(l):
            result.append(np.nan)
        else:
            result.append(DD_MAP.get(_cl(int(f)) + _cl(int(l)), np.nan))
    return pd.Series(result, index=n_first.index)

def _ngay_diff(d1: pd.Series, d2: pd.Series) -> pd.Series:
    return (pd.to_datetime(d1, errors="coerce") -
            pd.to_datetime(d2, errors="coerce")).dt.days

def get_drawn_set(row) -> set:
    return {
        int(row[c]) for c in ["n1","n2","n3","n4","n5","n6"]
        if pd.notna(row[c]) and row[c] != 0
    }

def get_dec_count(row, dec_start: int, dec_end: int) -> int:
    """Dem so luong so trong khoang [dec_start, dec_end]."""
    nums = get_drawn_set(row)
    return sum(1 for n in nums if dec_start <= n <= dec_end)


# ─────────────────────────────────────────────
# 3. GLOBAL FEATURES
# ─────────────────────────────────────────────

def make_global_features(df: pd.DataFrame, lags: int = 3) -> pd.DataFrame:
    d = df.copy()

    # DD encode
    d["dd_enc"] = d["dd"].apply(
        lambda v: {"CC":0,"CL":1,"LC":2,"LL":3}.get(str(v).strip(), -1)
    )

    # CL6
    s = d["cl6"].astype(str).str.strip()
    for i in range(6):
        d[f"cl6_b{i}"] = s.str[i].map({"C":0,"L":1}).fillna(-1).astype(int)
    d["cl6_pattern"] = s.apply(
        lambda v: int("".join("0" if c=="C" else "1" for c in v), 2)
        if len(v)==6 else np.nan
    )
    d["cl6_changed"] = (d["cl6_pattern"] != d["cl6_pattern"].shift(1)).astype(float)

    # Thu
    d["thu_enc"] = _encode_thu(d["thu"])

    # JP
    if "jpck" in d.columns:
        d["jpck"] = _safe_int(d["jpck"]).fillna(0)
    if "jp1_cnt" in d.columns:
        d["jp1_enc"] = (_safe_int(d["jp1_cnt"]) > 0).astype(int)

    # Mega
    if all(c in d.columns for c in ["mgn1","mgn6"]):
        mn = [_safe_int(d[f"mgn{i}"]) for i in range(1,7)]
        if "mgthu" in d.columns:
            d["mg_thu_enc"]   = _encode_thu(d["mgthu"])
        if "mgngay" in d.columns and "ngay" in d.columns:
            d["mg_ngay_diff"] = _ngay_diff(d["ngay"], d["mgngay"])
        d["mg_dd"] = _encode_dd(mn[0], mn[5])
        d["mg_sc"] = sum((n % 2 == 0).astype(float) for n in mn)

    feat = pd.DataFrame(index=d.index)

    # A. Biet truoc
    feat["thu_enc"] = d["thu_enc"]

    # B. Ky truoc (shift 1)
    prev_cols = [
        "dd_enc","sc","sum","ke",
        "cl6_b0","cl6_b1","cl6_b2","cl6_b3","cl6_b4","cl6_b5",
        "cl6_pattern","cl6_changed",
        "dec0","dec1","dec2","dec3","dec4","dec5",
    ]
    for c in ["jpck","jp1_enc"]:
        if c in d.columns: prev_cols.append(c)
    prev_cols = [c for c in prev_cols if c in d.columns]
    for col in prev_cols:
        feat[f"{col}_prev"] = d[col].shift(1)

    # C. Lag 2 → lags
    lag_cols = ["dd_enc","sc","cl6_pattern"]
    lag_cols = [c for c in lag_cols if c in d.columns]
    for lag in range(2, lags+1):
        for col in lag_cols:
            feat[f"{col}_lag{lag}"] = d[col].shift(lag)

    # D. Rolling sc
    for w in [3, 5]:
        base = d["sc"].shift(1)
        feat[f"sc_roll{w}_mean"] = base.rolling(w).mean()
        feat[f"sc_roll{w}_std"]  = base.rolling(w).std()

    # E. Mega
    for c in ["mg_thu_enc","mg_ngay_diff","mg_dd","mg_sc"]:
        if c in d.columns:
            feat[c] = d[c]

    # F. Cross Mega x P655
    if "mg_dd" in feat.columns and "dd_enc_prev" in feat.columns:
        feat["cross_mg_dd_match"] = (
            feat["mg_dd"] == feat["dd_enc_prev"]
        ).astype(float)
    if "mg_sc" in feat.columns and "sc_prev" in feat.columns:
        feat["cross_sc_mg"] = feat["sc_prev"] * feat["mg_sc"]

    return feat


# ─────────────────────────────────────────────
# 4. PER-DEC FEATURES
# ─────────────────────────────────────────────

def make_per_dec_features(df: pd.DataFrame,
                           dec_idx: int,
                           dec_start: int,
                           dec_end: int) -> pd.DataFrame:
    """Features rieng cho hang chuc dec_idx."""
    feat = pd.DataFrame(index=df.index)

    count = df.apply(lambda r: get_dec_count(r, dec_start, dec_end), axis=1)

    # count_prev: so luong so cua chuc nay ky truoc
    feat["dec_count_prev"] = count.shift(1)

    # dec_hit_prev: chuc nay co xuat hien ky truoc khong (binary)
    feat["dec_hit_prev"] = (count > 0).astype(float).shift(1)

    # dec_last_seen: bao nhieu ky truoc lan xuat hien gan nhat
    hit = (count > 0).astype(int)
    last_seen = []
    last = np.nan
    for i, h in enumerate(hit):
        if i == 0:
            last_seen.append(np.nan)
            if h == 1: last = 0
        else:
            last_seen.append(np.nan if pd.isna(last) else i - last)
            if h == 1: last = i
    feat["dec_last_seen"] = pd.Series(last_seen, index=df.index).shift(1)

    # dec_streak: dang nghi bao nhieu ky lien tiep
    streak = []
    cur = 0
    for i, h in enumerate(hit):
        if i == 0:
            streak.append(np.nan)
        else:
            streak.append(float(cur))
        cur = 0 if h == 1 else cur + 1
    feat["dec_streak"] = pd.Series(streak, index=df.index).shift(1)

    # dec_freq_10/20/50: tan suat chuc nay xuat hien
    hit_s = hit.shift(1)
    feat["dec_freq_10"] = hit_s.rolling(10).mean()
    feat["dec_freq_20"] = hit_s.rolling(20).mean()
    feat["dec_freq_50"] = hit_s.rolling(50).mean()

    # dec_mean_count_10/20: trung binh so luong so moi ky (rolling)
    count_s = count.shift(1)
    feat["dec_mean_count_10"] = count_s.rolling(10).mean()
    feat["dec_mean_count_20"] = count_s.rolling(20).mean()

    # Static
    feat["dec_size"] = float(dec_end - dec_start + 1)  # so luong so trong chuc
    feat["dec_idx"]  = float(dec_idx)

    return feat


# ─────────────────────────────────────────────
# 5. BUILD DATASET
# ─────────────────────────────────────────────

def build_dataset(df: pd.DataFrame,
                  dec_idx: int, dec_start: int, dec_end: int,
                  global_feat: pd.DataFrame) -> tuple:
    """No-leak: combined_train = ky 0→N-2, predict_row = ky N-1."""
    per_feat = make_per_dec_features(df, dec_idx, dec_start, dec_end)
    all_feat = pd.concat([global_feat, per_feat], axis=1).fillna(-1)

    target = df.apply(lambda r: get_dec_count(r, dec_start, dec_end), axis=1)

    combined = pd.concat(
        [all_feat, target.rename("y"), df["ky"]], axis=1
    ).dropna(subset=["dec_freq_50"]).reset_index(drop=True)

    feature_cols   = list(all_feat.columns)
    combined_train = combined.iloc[:-1].reset_index(drop=True)
    predict_row    = combined.iloc[[-1]]

    return combined_train, predict_row, feature_cols


# ─────────────────────────────────────────────
# 6. XGB REGRESSOR
# ─────────────────────────────────────────────

def make_xgb_reg(params: dict = None):
    defaults = dict(
        n_estimators          = 300,
        max_depth             = 3,
        learning_rate         = 0.05,
        subsample             = 0.8,
        colsample_bytree      = 0.7,
        min_child_weight      = 3,
        gamma                 = 0.1,
        reg_alpha             = 0.05,
        reg_lambda            = 1.0,
        objective             = "reg:squarederror",
        eval_metric           = "mae",
        random_state          = 42,
        n_jobs                = -1,
        early_stopping_rounds = 30,
    )
    if params:
        defaults.update(params)
    return XGBRegressor(**defaults)


# ─────────────────────────────────────────────
# 7. WALK-FORWARD
# ─────────────────────────────────────────────

def _walk_forward_splits(n: int, min_train: int, val_size: int):
    splits = []
    start  = min_train
    while start + val_size <= n:
        splits.append((np.arange(0, start), np.arange(start, start+val_size)))
        start += val_size
    return splits

def _make_decay_weights(n: int, decay: float = 0.995) -> np.ndarray:
    idx = np.arange(n)
    w   = decay ** (n - 1 - idx)
    return w / w.mean()


# ─────────────────────────────────────────────
# 8. TRAIN ONE DEC
# ─────────────────────────────────────────────

def train_one(combined: pd.DataFrame, feature_cols: list,
              val_size: int, min_train: int,
              train_window: int = 0, decay: float = 0.995,
              xgb_params: dict = None) -> tuple:
    X = combined[feature_cols].values
    y = combined["y"].values

    splits   = _walk_forward_splits(len(X), min_train, val_size)
    mae_list = []

    for tr_idx, va_idx in splits:
        X_tr, X_va = X[tr_idx], X[va_idx]
        y_tr, y_va = y[tr_idx], y[va_idx]

        if train_window > 0 and len(tr_idx) > train_window:
            tr_idx = tr_idx[-train_window:]
            X_tr   = X[tr_idx]
            y_tr   = y[tr_idx]

        sw = _make_decay_weights(len(y_tr), decay)

        mdl = make_xgb_reg(xgb_params)
        mdl.fit(X_tr, y_tr, sample_weight=sw,
                eval_set=[(X_va, y_va)], verbose=False)
        pred = mdl.predict(X_va)
        mae_list.append(mean_absolute_error(y_va, pred))

    # Final model
    if train_window > 0 and len(X) > train_window:
        X_fin, y_fin = X[-train_window:], y[-train_window:]
    else:
        X_fin, y_fin = X, y

    sw_fin  = _make_decay_weights(len(y_fin), decay)
    final_p = dict(xgb_params or {})
    final_p["early_stopping_rounds"] = None
    final   = make_xgb_reg(final_p)
    final.fit(X_fin, y_fin, sample_weight=sw_fin, verbose=False)

    mean_mae = float(np.mean(mae_list)) if mae_list else 0.0
    return final, mean_mae


# ─────────────────────────────────────────────
# 9. TRAIN ALL (6 dec)
# ─────────────────────────────────────────────

def train_all(df: pd.DataFrame, global_feat: pd.DataFrame,
              val_size: int, min_train: int,
              train_window: int = 0, decay: float = 0.995,
              xgb_params: dict = None) -> dict:
    """Train 6 regression model cho dec0..dec5."""
    results  = {}

    # Baseline MAE: neu luon predict mean count
    all_counts = []
    for dec_idx, ds, de in DEC_RANGES:
        c = df.apply(lambda r: get_dec_count(r, ds, de), axis=1)
        all_counts.append(c.mean())
    print(f"\n  Mean count per dec (baseline): "
          + "  ".join(f"dec{i}={v:.2f}" for i,v in enumerate(all_counts)))

    print(f"\n  {'Dec':>5}  {'Range':>8}  {'CV MAE':>8}  {'Mean Y':>8}")
    print(f"  {'─'*42}")

    for dec_idx, ds, de in DEC_RANGES:
        combined_train, predict_row, feature_cols = build_dataset(
            df, dec_idx, ds, de, global_feat
        )
        model, mae = train_one(
            combined_train, feature_cols,
            val_size, min_train, train_window, decay, xgb_params
        )
        mean_y = combined_train["y"].mean()
        results[dec_idx] = dict(
            model=model, mae=mae, mean_y=mean_y,
            feature_cols=feature_cols,
            predict_row=predict_row,
            dec_start=ds, dec_end=de,
        )
        print(f"  dec{dec_idx}  {ds:>3}-{de:<3}   {mae:>8.4f}  {mean_y:>8.4f}")

    print(f"\n  Mean CV MAE: {np.mean([results[i]['mae'] for i in range(6)]):.4f}")
    return results


# ─────────────────────────────────────────────
# 10. ENSEMBLE
# ─────────────────────────────────────────────

ENSEMBLE_CONFIGS = [
    {"lags":3, "decay":0.995, "train_window":0,   "label":"M1(lag3,d0.995,all)"},
    {"lags":5, "decay":0.990, "train_window":0,   "label":"M2(lag5,d0.990,all)"},
    {"lags":3, "decay":0.995, "train_window":500, "label":"M3(lag3,d0.995,w500)"},
]

def train_ensemble(df: pd.DataFrame, val_size: int, min_train: int,
                   xgb_params: dict = None) -> list:
    all_results = []
    for cfg in ENSEMBLE_CONFIGS:
        print(f"\n  ── Ensemble {cfg['label']} ──")
        gf = make_global_features(df, lags=cfg["lags"])
        r  = train_all(
            df, gf,
            val_size=val_size, min_train=min_train,
            train_window=cfg["train_window"], decay=cfg["decay"],
            xgb_params=xgb_params,
        )
        all_results.append(r)
    return all_results


def predict_ensemble(all_results: list) -> list:
    """Average predicted count qua 3 config cho tung dec."""
    preds = []
    for dec_idx, ds, de in DEC_RANGES:
        vals = []
        for r in all_results:
            res    = r[dec_idx]
            last_X = res["predict_row"][res["feature_cols"]].values
            val    = float(res["model"].predict(last_X)[0])
            vals.append(max(0.0, val))   # clip âm
        preds.append((dec_idx, ds, de, float(np.mean(vals))))
    return preds


def predict_single(results: dict) -> list:
    preds = []
    for dec_idx, ds, de in DEC_RANGES:
        res    = results[dec_idx]
        last_X = res["predict_row"][res["feature_cols"]].values
        val    = float(max(0.0, res["model"].predict(last_X)[0]))
        preds.append((dec_idx, ds, de, val))
    return preds


# ─────────────────────────────────────────────
# 11. SHOW PREDICTION
# ─────────────────────────────────────────────

def show_prediction(df: pd.DataFrame, preds: list):
    last_ky  = (df["ky"].iloc[-1] or "").strip()
    total    = sum(p[3] for p in preds)

    print(f"\n{'═'*58}")
    print(f"📌 Input : ky {last_ky}")
    print(f"🎯 Du doan phan phoi hang chuc cho ky (sau {last_ky}):")
    print(f"{'═'*58}")
    print(f"\n  {'Dec':>5}  {'Range':>8}  {'Count':>7}  {'Bar'}")
    print(f"  {'─'*50}")

    for dec_idx, ds, de, count in preds:
        bar   = "█" * int(count * 10)
        frac  = count / total * 100 if total > 0 else 0
        print(f"  dec{dec_idx}  {ds:>3}-{de:<3}  {count:>6.3f}  {bar}  ({frac:.1f}%)")

    print(f"\n  Tong predicted count : {total:.3f}  (thuc te luon = 6)")

    print(f"{'═'*58}\n")


# ─────────────────────────────────────────────
# 12. MAIN
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="XGBoost regression predict dec count P655"
    )
    parser.add_argument("--source",       choices=["db","csv"], default="db")
    parser.add_argument("--file",         default="p655kqdetail.csv")
    parser.add_argument("--n_ky",         type=int,   default=0)
    parser.add_argument("--qh_ky",        type=int,   default=20)
    parser.add_argument("--lags",         type=int,   default=3)
    parser.add_argument("--val_size",     type=int,   default=50)
    parser.add_argument("--min_train",    type=int,   default=300)
    parser.add_argument("--train_window", type=int,   default=0)
    parser.add_argument("--decay",        type=float, default=0.995)
    parser.add_argument("--ensemble",     action="store_true")
    args = parser.parse_args()

    print(f"📌 Version : {VERSION}")
    print(f"📝 Desc    : {DESCRIPTION}")
    print("📥 Loading data...")
    df = (load_from_csv(args.file) if args.source == "csv"
          else load_from_db(args.n_ky, args.qh_ky))
    print(f"   {len(df)} ky")
    print(f"   Mega : {'✅' if 'mgn1' in df.columns else '⚠️'}")
    print(f"   JP   : {'✅' if 'jpck' in df.columns else '⚠️'}")

    if args.ensemble:
        print(f"\n🚀 Training ensemble ({len(ENSEMBLE_CONFIGS)} configs × 6 dec)...")
        all_results = train_ensemble(
            df, val_size=args.val_size, min_train=args.min_train,
        )
        print(f"\n🔮 Predicting (ensemble)...")
        preds = predict_ensemble(all_results)
    else:
        print("\n⚙️  Global feature engineering...")
        global_feat = make_global_features(df, lags=args.lags)
        print(f"   {global_feat.shape[1]} global features")

        print(f"\n🚀 Training 6 regression model...")
        results = train_all(
            df, global_feat,
            val_size=args.val_size, min_train=args.min_train,
            train_window=args.train_window, decay=args.decay,
        )
        print(f"\n🔮 Predicting...")
        preds = predict_single(results)

    show_prediction(df, preds)


if __name__ == "__main__":
    main()