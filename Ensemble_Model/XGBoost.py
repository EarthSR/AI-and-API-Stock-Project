# XGB_PriceMeta.py  (Meta-regressor สำหรับ "ราคา" + คำนวนทิศทางจากราคา + รายงานแยกหุ้น)
# -*- coding: utf-8 -*-
"""
สิ่งที่ปรับปรุงจากเวอร์ชันก่อนหน้า (backward-compatible):
1) Feature Engineering เพิ่มเติมแบบไม่เกิด leakage:
   - prev_ret_1d (log-return ของวันก่อนหน้า), vol_ema (EMA ของ |prev_ret_1d| ต่อ Ticker)
   - sign_agree (LSTM/GRU เห็นทิศทางตรงกันไหม), mag_mean, ret_pred_ratio, rel_conf_* (อัตราส่วนความมั่นใจเทียบ EMA error)
2) Isotonic Calibration ของ P(up) จาก predicted return บน validation (หรือ CV) → ได้คอลัมน์ Meta_Prob_Up
3) Neutral Zone จาก conformal residual (gamma = max(|q_lo|,|q_hi|)) → คอลัมน์ Meta_Pred_Dir_Th (tri-state: {-1,0,1})
4) รายงานสรุปเพิ่ม AUC/Brier (ถ้ามี val) และคอลัมน์ *_META_TH ใน overall summaries
5) ความทนทาน: winsorize ฟีเจอร์แบบ return เพิ่ม, ใส่ n_jobs=-1 ให้ XGBRegressor
6) ใหม่: ไฟล์ "Simple metrics" 3 ไฟล์ พร้อมหัวคอลัมน์มาตรฐาน
   - meta_price_simple_overall.csv
   - meta_price_simple_val_by_ticker.csv
   - meta_price_simple_all_by_ticker.csv
"""

import os, joblib, warnings
import numpy as np
import pandas as pd
import xgboost as xgb
from xgboost import XGBRegressor
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, f1_score, precision_score, recall_score, matthews_corrcoef,
    roc_auc_score, brier_score_loss
)
from sklearn.isotonic import IsotonicRegression

warnings.filterwarnings("ignore", category=FutureWarning)

# =======================
# CONFIG
# =======================
LSTM_DIR = r'../LSTM_model/logs/'
GRU_DIR  = r'../GRU_model/logs/'
DAILY_FILE = 'daily_all_predictions.csv'   # ใช้ไฟล์นี้ทั้ง LSTM/GRU

# Out paths
OUT_MODEL_PATH   = 'xgb_price.json'
OUT_MODEL_META   = 'xgb_price.meta.joblib'
OUT_PRED_CSV     = 'meta_price_predictions.csv'
OUT_VAL_SUMMARY  = 'meta_price_val_summary.csv'  # (ยังคงไว้ เผื่อขยายในอนาคต)
OUT_VAL_BY_TKR   = 'meta_price_val_metrics_per_ticker.csv'
OUT_FEAT_GAIN    = 'xgb_price_feature_gain.csv'
OUT_DIR_BY_TKR_ALL = 'meta_price_dir_metrics_per_ticker_all.csv'         # All rows
OUT_DIR_SUMMARY_OVERALL = 'meta_price_dir_summary_overall.csv'           # สรุปรวม

# NEW simple metric snapshots
OUT_SIMPLE_OVERALL         = 'meta_price_simple_overall.csv'
OUT_SIMPLE_VAL_BY_TICKER   = 'meta_price_simple_val_by_ticker.csv'
OUT_SIMPLE_ALL_BY_TICKER   = 'meta_price_simple_all_by_ticker.csv'

# Validation split (จะ auto-adjust ถ้าข้อมูลน้อย)
VAL_RATIO_DEFAULT    = 0.12
EMBARGO_DAYS_DEFAULT = 5

# Training knobs
USE_TIME_DECAY = True
TIME_DECAY_STRENGTH = 2.0
WINSOR_K = 4.0
EWM_ALPHA = 0.20

MIN_TREES = 50
STEP      = 10
N_ESTIMATORS = 800
RANDOM_STATE = 42
OBJECTIVE = 'reg:squarederror'  # 'reg:squarederror' หรือ 'reg:absoluteerror'

# เกณฑ์ขั้นต่ำใน time-split
MIN_TRAIN_WANTED = 50
MIN_VAL_WANTED   = 15

# =======================
# Helpers
# =======================
REQ_MIN_COLS = {'Ticker','Date','Chunk_Index','Step','Predicted_Price','Actual_Price'}

np.seterr(all='ignore')
EPS = 1e-12

def rmse_np(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

def load_daily(model_dir, daily_file):
    path = os.path.join(model_dir, daily_file)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Cannot find {daily_file} in {model_dir}")
    df = pd.read_csv(path, parse_dates=['Date'])
    df.columns = df.columns.str.strip()
    if not REQ_MIN_COLS.issubset(df.columns):
        missing = sorted(list(REQ_MIN_COLS - set(df.columns)))
        raise RuntimeError(f"{daily_file} in {model_dir} missing columns: {missing}")
    out = (pd.DataFrame({
        'Ticker': df['Ticker'].astype(str),
        'Date':   pd.to_datetime(df['Date']),
        'chunk':  df['Chunk_Index'].astype(int),
        'step':   df['Step'].astype(int),
        'pred_price': df['Predicted_Price'].astype(float),
        'actual_price': df['Actual_Price'].astype(float),
    })
    .sort_values(['Ticker','Date','step'])
    .reset_index(drop=True))
    # เก็บ step สุดท้ายของวันต่อ (Ticker,Date)
    out = out.groupby(['Ticker','Date'], as_index=False).tail(1).reset_index(drop=True)
    return out

def winsorize_by_ticker(feat_df, tickers, cols, k=WINSOR_K):
    feat_df = feat_df.copy()
    for c in cols:
        g = feat_df.groupby(tickers)[c]
        mu = g.transform('mean')
        sd = g.transform('std').replace(0, 1e-6)
        feat_df[c] = np.clip(feat_df[c], mu - k*sd, mu + k*sd)
    return feat_df

def _coalesce(a, b):
    return np.where(np.isfinite(a), a, b)

def _time_decay_weights(dates: pd.Series):
    t = dates.values.astype('datetime64[ns]').astype('int64')
    t = (t - t.min()) / max(1, (t.max() - t.min()))
    return np.exp(t * TIME_DECAY_STRENGTH)

def _map_gain_to_names(booster, feat_cols):
    gain = booster.get_score(importance_type='gain')  # keys: f0,f1,...
    fmap = {f"f{i}": name for i, name in enumerate(feat_cols)}
    mapped = {fmap.get(k, k): v for k, v in gain.items()}
    return mapped

def _price_from_ret(prev, ret):
    return prev * np.exp(ret)

def dir_metrics_basic(y_true, y_pred):
    return {
        'ACC':  accuracy_score(y_true, y_pred),
        'F1':   f1_score(y_true, y_pred, zero_division=0),
        'PREC': precision_score(y_true, y_pred, zero_division=0),
        'REC':  recall_score(y_true, y_pred, zero_division=0),
        'MCC':  matthews_corrcoef(y_true, y_pred) if len(np.unique(y_pred))>1 else 0.0
    }

def dir_report(y_true, y_pred, ignore_val=-1):
    """คำนวณเมตริก + confusion counts โดย 'กรอง' ค่าที่ y_pred==ignore_val ออก"""
    mask = (y_pred != ignore_val) & np.isfinite(y_true)
    n = int(mask.sum())
    if n == 0:
        return dict(Support=0, TP=0, FP=0, TN=0, FN=0,
                    ACC=np.nan, F1=np.nan, PREC=np.nan, REC=np.nan, MCC=np.nan,
                    Pred_Pos_Count=0, Pred_Pos_Rate=np.nan)
    yt = y_true[mask].astype(int)
    yp = y_pred[mask].astype(int)
    tp = int(np.sum((yt==1)&(yp==1)))
    tn = int(np.sum((yt==0)&(yp==0)))
    fp = int(np.sum((yt==0)&(yp==1)))
    fn = int(np.sum((yt==1)&(yp==0)))
    acc  = (tp+tn)/n if n>0 else np.nan
    f1   = f1_score(yt, yp, zero_division=0)
    prec = precision_score(yt, yp, zero_division=0)
    rec  = recall_score(yt, yp, zero_division=0)
    mcc  = matthews_corrcoef(yt, yp) if len(np.unique(yp))>1 else 0.0
    pred_pos = int(np.sum(yp==1))
    return dict(
        Support=n, TP=tp, FP=fp, TN=tn, FN=fn,
        ACC=float(acc), F1=float(f1), PREC=float(prec), REC=float(rec), MCC=float(mcc),
        Pred_Pos_Count=pred_pos, Pred_Pos_Rate=float(pred_pos/n if n>0 else np.nan)
    )

# =======================
# Load & merge
# =======================
lstm = load_daily(LSTM_DIR, DAILY_FILE)
gru  = load_daily(GRU_DIR,  DAILY_FILE)

df = (pd.merge(lstm, gru, on=['Ticker','Date'], how='outer', suffixes=('_lstm','_gru'))
        .sort_values(['Ticker','Date'])
        .reset_index(drop=True))

# เลือก actual & prev_actual
df['actual_price'] = _coalesce(df.get('actual_price_lstm').values, df.get('actual_price_gru').values)
df = df.sort_values(['Ticker','Date']).reset_index(drop=True)
df['prev_actual'] = df.groupby('Ticker')['actual_price'].shift(1)

# กรองแถวที่มี prev_actual และ actual
mask_ok = np.isfinite(df['actual_price'].values) & np.isfinite(df['prev_actual'].values)
df = df.loc[mask_ok].reset_index(drop=True)

# =======================
# Features
# =======================
pred_lstm = df.get('pred_price_lstm').values if 'pred_price_lstm' in df else np.full(len(df), np.nan)
pred_gru  = df.get('pred_price_gru' ).values if 'pred_price_gru'  in df else np.full(len(df), np.nan)

# returns เทียบ prev_actual
ret_pred_lstm = np.where(np.isfinite(pred_lstm), np.log((pred_lstm+EPS)/df['prev_actual'].values), np.nan)
ret_pred_gru  = np.where(np.isfinite(pred_gru ), np.log((pred_gru +EPS)/df['prev_actual'].values), np.nan)

ret_pred_mean = np.where(
    np.isfinite(ret_pred_lstm) & np.isfinite(ret_pred_gru),
    0.5*(ret_pred_lstm + ret_pred_gru),
    _coalesce(ret_pred_lstm, ret_pred_gru)
)
ret_pred_diff = np.where(
    np.isfinite(ret_pred_lstm) & np.isfinite(ret_pred_gru),
    ret_pred_lstm - ret_pred_gru,
    0.0
)

# ฟีเจอร์คุณภาพฐาน (EMA ของ |pred - prev_actual| ต่อ ticker)
for src, arr in [('lstm', pred_lstm), ('gru', pred_gru)]:
    diff = np.abs(arr - df['prev_actual'].values)
    s = pd.Series(diff, index=df.index, name='diff')

    def _ema_fill(series: pd.Series):
        g = series.to_numpy()
        finite = np.isfinite(g)
        if not finite.any():
            base = np.zeros_like(g, dtype=float)
        else:
            med = np.median(g[finite])
            base = np.where(finite, g, med)
        return pd.Series(base).ewm(alpha=EWM_ALPHA, adjust=False).mean()

    df[f'ema_mae_{src}'] = s.groupby(df['Ticker']).transform(_ema_fill)

# target: actual return (y)
df['ret_actual'] = np.log((df['actual_price']+EPS) / (df['prev_actual']+EPS))

# === NEW: non-leaky lag features ===
# prev_ret_1d = ret_actual ของวันก่อนหน้า (safe: shift 1)
df['prev_ret_1d'] = df.groupby('Ticker')['ret_actual'].shift(1)
# vol_ema = EMA ของ |prev_ret_1d|
_df_abs_prev = df['prev_ret_1d'].abs().rename('abs_prev')
df['vol_ema'] = _df_abs_prev.groupby(df['Ticker']).transform(
    lambda s: s.fillna(s.median()).ewm(alpha=EWM_ALPHA, adjust=False).mean()
)

# ความเห็นตรงกันของ LSTM/GRU และความแรงของสัญญาณ
sgn_l = np.sign(ret_pred_lstm)
sgn_g = np.sign(ret_pred_gru)
sign_agree = (sgn_l == sgn_g).astype(float)
mag_mean = np.abs(ret_pred_mean)
ret_pred_ratio = np.where(
    np.isfinite(ret_pred_lstm) & np.isfinite(ret_pred_gru),
    ret_pred_diff / (np.abs(ret_pred_lstm) + np.abs(ret_pred_gru) + EPS),
    0.0
)
# แปลง EMA error (หน่วยราคา) → หน่วย return โดยหารด้วย prev_actual
ema_mae_lstm_ret = df['ema_mae_lstm'] / (df['prev_actual'] + EPS)
ema_mae_gru_ret  = df['ema_mae_gru'] / (df['prev_actual'] + EPS)
rel_conf_lstm = np.abs(ret_pred_lstm) / (ema_mae_lstm_ret + EPS)
rel_conf_gru  = np.abs(ret_pred_gru ) / (ema_mae_gru_ret  + EPS)

# time features
df['dow'] = pd.to_datetime(df['Date']).dt.weekday.astype(int)
df['dom'] = pd.to_datetime(df['Date']).dt.day.astype(int)

# รวมฟีเจอร์
feat_cols = [
    # base returns
    'ret_pred_lstm','ret_pred_gru','ret_pred_mean','ret_pred_diff',
    # NEW confidence & structure
    'prev_ret_1d','vol_ema','sign_agree','mag_mean','ret_pred_ratio','rel_conf_lstm','rel_conf_gru',
    # time
    'dow','dom'
]

# สร้างตารางฟีเจอร์
_feat = pd.DataFrame({
    'ret_pred_lstm': ret_pred_lstm,
    'ret_pred_gru':  ret_pred_gru,
    'ret_pred_mean': ret_pred_mean,
    'ret_pred_diff': ret_pred_diff,
    'prev_ret_1d':   df['prev_ret_1d'].values,
    'vol_ema':       df['vol_ema'].values,
    'sign_agree':    sign_agree,
    'mag_mean':      mag_mean,
    'ret_pred_ratio':ret_pred_ratio,
    'rel_conf_lstm': rel_conf_lstm,
    'rel_conf_gru':  rel_conf_gru,
    'dow':           df['dow'].values,
    'dom':           df['dom'].values,
})

# winsorize returns-like features by ticker
_feat = winsorize_by_ticker(
    _feat.assign(Ticker=df['Ticker']),
    tickers='Ticker',
    cols=['ret_pred_lstm','ret_pred_gru','ret_pred_mean','ret_pred_diff','prev_ret_1d'],
    k=WINSOR_K
).drop(columns=['Ticker'])

feat = _feat.copy()
y = df['ret_actual'].values

# =======================
# Flexible split (time-split หรือ CV fallback)
# =======================
dates_sorted = np.sort(df['Date'].unique())
split_mode = 'time_split'
val_mask = None

if len(dates_sorted) < 6:
    print("[WARN] จำนวนวันน้อยมาก → CV mode")
    split_mode = 'cv_only'

if split_mode == 'time_split':
    tried = []
    found = False
    for vr in [VAL_RATIO_DEFAULT, 0.10, 0.08, 0.06, 0.05]:
        for emb in [EMBARGO_DAYS_DEFAULT, max(0, EMBARGO_DAYS_DEFAULT-3), 0]:
            cut_idx = int(max(1, len(dates_sorted)*(1.0-vr))) - 1
            cut_idx = max(0, min(cut_idx, len(dates_sorted)-1))
            cut = dates_sorted[cut_idx]
            cut_embargo = cut - np.timedelta64(emb, 'D')
            train_idx = (df['Date'] <= cut_embargo)
            val_idx   = (df['Date'] >  cut)
            tried.append((vr, emb, int(train_idx.sum()), int(val_idx.sum())))
            if train_idx.sum() >= MIN_TRAIN_WANTED and val_idx.sum() >= MIN_VAL_WANTED:
                found = True
                pick_vr, pick_emb = vr, emb
                break
        if found: break
    if not found:
        best = max(tried, key=lambda t: (t[2]+t[3], t[2], t[3])) if tried else None
        if best and best[2] >= 10 and best[3] >= 5:
            pick_vr, pick_emb = best[0], best[1]
            cut = dates_sorted[int(max(1, len(dates_sorted)*(1.0-pick_vr)))-1]
            cut_embargo = cut - np.timedelta64(pick_emb, 'D')
            train_idx = (df['Date'] <= cut_embargo)
            val_idx   = (df['Date'] >  cut)
            print(f"[WARN] ใช้ time-split ที่ดีที่สุด: train={int(train_idx.sum())}, val={int(val_idx.sum())} (vr={pick_vr}, embargo={pick_emb})")
        else:
            print("[WARN] time-split ไม่ถึงเกณฑ์ → CV mode")
            split_mode = 'cv_only'
    else:
        print(f"[INFO] time-split: vr={pick_vr}, embargo={pick_emb}; train={int(train_idx.sum())}, val={int(val_idx.sum())}")

# =======================
# Train + best_k scan
# =======================
xgb_reg = XGBRegressor(
    n_estimators=N_ESTIMATORS,
    learning_rate=0.05,
    max_depth=4,
    subsample=0.9,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    reg_alpha=0.0,
    objective=OBJECTIVE,
    tree_method='hist',
    random_state=RANDOM_STATE,
    n_jobs=-1
)

def _fit_with_decay(X, y, dates):
    if USE_TIME_DECAY:
        w = _time_decay_weights(dates)
        xgb_reg.fit(X, y, sample_weight=w)
    else:
        xgb_reg.fit(X, y)

def _scan_best_k(booster, X_val, y_val):
    dval = xgb.DMatrix(X_val, label=y_val)
    best_k, best_rmse = MIN_TREES, 1e18
    n_estim = int(xgb_reg.get_params().get('n_estimators', N_ESTIMATORS))
    for k in range(MIN_TREES, n_estim+1, STEP):
        preds_val_ret = booster.predict(dval, iteration_range=(0, k))
        rmse = rmse_np(y_val, preds_val_ret)
        if rmse < best_rmse:
            best_rmse, best_k = rmse, k
    return best_k, best_rmse

residuals_for_conformal = []
iso_cal = None  # Isotonic calibrator (P(up) | pred_ret)
auc_val = np.nan
brier_val = np.nan

if split_mode == 'time_split':
    X_train, y_train = feat.loc[train_idx].values, y[train_idx]
    X_val,   y_val   = feat.loc[val_idx].values,   y[val_idx]
    print(f"[META-PRICE] Train size: {X_train.shape}, Val size: {X_val.shape}")

    _fit_with_decay(X_train, y_train, df.loc[train_idx, 'Date'])
    booster = xgb_reg.get_booster()

    final_best_k, best_rmse = _scan_best_k(booster, X_val, y_val)
    dval = xgb.DMatrix(X_val, label=y_val)
    val_preds_ret = booster.predict(dval, iteration_range=(0, final_best_k))
    residuals_for_conformal.extend(list(y_val - val_preds_ret))

    # === NEW: calibrate P(up) ด้วย isotonic ===
    actual_dir_val = (df.loc[val_idx, 'actual_price'].values > df.loc[val_idx, 'prev_actual'].values).astype(int)
    try:
        iso_cal = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds='clip')
        iso_cal.fit(val_preds_ret, actual_dir_val)
        prob_val = iso_cal.predict(val_preds_ret)
        if len(np.unique(actual_dir_val)) > 1:
            auc_val = float(roc_auc_score(actual_dir_val, prob_val))
        brier_val = float(brier_score_loss(actual_dir_val, prob_val))
    except Exception as e:
        print(f"[WARN] Isotonic calibration failed: {e}")
        iso_cal = None

    val_mask = val_idx.values if isinstance(val_idx, pd.Series) else val_idx

else:
    # CV แบบกลุ่มวัน (เรียงตามเวลา)
    dates = dates_sorted
    n_dates = len(dates)
    n_folds = min(5, max(2, n_dates // 4))
    fold_sizes = [n_dates // n_folds + (1 if i < n_dates % n_folds else 0) for i in range(n_folds)]
    bounds = np.cumsum([0] + fold_sizes)

    best_k_list, rmse_list = [], []
    all_val_idx = np.zeros(len(df), dtype=bool)

    val_pred_buf = []  # เก็บ (pred_ret, actual_dir) จากทุก fold เพื่อใช้ calibrate

    for i in range(n_folds):
        val_start, val_end = bounds[i], bounds[i+1]
        val_days = dates[val_start:val_end]
        if len(val_days) == 0:
            continue
        train_days = dates[:val_start]
        if len(train_days) < 2:
            continue

        tr_idx = df['Date'].isin(train_days)
        vl_idx = df['Date'].isin(val_days)

        X_tr, y_tr = feat.loc[tr_idx].values, y[tr_idx]
        X_vl, y_vl = feat.loc[vl_idx].values, y[vl_idx]

        _fit_with_decay(X_tr, y_tr, df.loc[tr_idx, 'Date'])
        booster = xgb_reg.get_booster()

        k_i, rmse_i = _scan_best_k(booster, X_vl, y_vl)
        best_k_list.append(k_i)
        rmse_list.append(rmse_i)

        dvl = xgb.DMatrix(X_vl, label=y_vl)
        preds_vl = booster.predict(dvl, iteration_range=(0, k_i))
        residuals_for_conformal.extend(list(y_vl - preds_vl))
        all_val_idx = all_val_idx | vl_idx.values

        # buffer สำหรับ calibration
        actual_dir_vl = (df.loc[vl_idx, 'actual_price'].values > df.loc[vl_idx, 'prev_actual'].values).astype(int)
        val_pred_buf.append((preds_vl, actual_dir_vl))

    if not best_k_list:
        final_best_k = min(max(MIN_TREES, int(N_ESTIMATORS * 0.35)), N_ESTIMATORS)
        print(f"[WARN] CV fold เล็กเกินไป → small mode best_k={final_best_k}")
        _fit_with_decay(feat.values, y, df['Date'])
        booster = xgb_reg.get_booster()
        dtr = xgb.DMatrix(feat.values, label=y)
        preds_tr = booster.predict(dtr, iteration_range=(0, final_best_k))
        residuals_for_conformal.extend(list(y - preds_tr))
        val_mask = np.zeros(len(df), dtype=bool)
    else:
        final_best_k = int(np.median(best_k_list))
        print(f"[INFO] CV folds={len(best_k_list)}, median best_k={final_best_k}, mean RMSE={np.mean(rmse_list):.6f}")
        _fit_with_decay(feat.values, y, df['Date'])
        booster = xgb_reg.get_booster()
        val_mask = all_val_idx

        # isotonic จากทุก fold (stacked) ถ้า class มีทั้ง 0/1
        try:
            if val_pred_buf:
                preds_all = np.concatenate([p for (p, a) in val_pred_buf])
                actual_all = np.concatenate([a for (p, a) in val_pred_buf])
                if len(np.unique(actual_all)) > 1:
                    iso_cal = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds='clip')
                    iso_cal.fit(preds_all, actual_all)
                    prob_all = iso_cal.predict(preds_all)
                    auc_val = float(roc_auc_score(actual_all, prob_all))
                    brier_val = float(brier_score_loss(actual_all, prob_all))
        except Exception as e:
            print(f"[WARN] Isotonic calibration (CV) failed: {e}")
            iso_cal = None

# =======================
# Validation metrics (ราคา)
# =======================
if val_mask.sum() > 0:
    dval_full = xgb.DMatrix(feat.loc[val_mask].values, label=y[val_mask])
    val_preds_ret = booster.predict(dval_full, iteration_range=(0, final_best_k))
    val_preds_price = _price_from_ret(df.loc[val_mask, 'prev_actual'].values, val_preds_ret)
    val_actual_price = df.loc[val_mask, 'actual_price'].values
    val_mae  = mean_absolute_error(val_actual_price, val_preds_price)
    val_rmse = rmse_np(val_actual_price, val_preds_price)
    try:
        val_r2   = r2_score(val_actual_price, val_preds_price)
    except Exception:
        val_r2 = np.nan
else:
    print("[WARN] ไม่มี validation แยกชัดเจน → รายงานราคาเฉพาะ summary รวม และใช้ conformal จาก residuals ที่มี")
    val_mae = val_rmse = val_r2 = np.nan

print("\n===== META-PRICE VALIDATION (best trees) =====")
print(f"Best #Trees : {final_best_k}")
print(f"MAE         : {val_mae:.6f}" if np.isfinite(val_mae) else "MAE         : N/A")
print(f"RMSE        : {val_rmse:.6f}" if np.isfinite(val_rmse) else "RMSE        : N/A")
print(f"R2          : {val_r2:.6f}"   if np.isfinite(val_r2)   else "R2          : N/A")
if np.isfinite(auc_val) or np.isfinite(brier_val):
    print(f"AUC(VAL)    : {auc_val:.6f}" if np.isfinite(auc_val) else "AUC(VAL)    : N/A")
    print(f"Brier(VAL)  : {brier_val:.6f}" if np.isfinite(brier_val) else "Brier(VAL)  : N/A")

# =======================
# Conformal intervals (return residuals)
# =======================
residuals_for_conformal = np.array(residuals_for_conformal)
if residuals_for_conformal.size >= 5:
    q_lo, q_hi = np.quantile(residuals_for_conformal, [0.05, 0.95])
else:
    q_lo, q_hi = -0.02, 0.02
print(f"Conformal residual quantiles (return): lo={q_lo:.6f}, hi={q_hi:.6f}")

# Neutral zone margin (บนสเกล return) สำหรับจัด -1
gamma_margin = float(max(abs(q_lo), abs(q_hi)))

# Save model & meta
meta_blob = {
    'best_k': int(final_best_k),
    'q_lo': float(q_lo),
    'q_hi': float(q_hi),
    'gamma_margin': gamma_margin,
    'feat_cols': feat_cols,
    'objective': OBJECTIVE
}
if iso_cal is not None:
    meta_blob['iso_cal'] = iso_cal  # picklable
    meta_blob['auc_val'] = float(auc_val) if np.isfinite(auc_val) else None
    meta_blob['brier_val'] = float(brier_val) if np.isfinite(brier_val) else None

xgb_reg.save_model(OUT_MODEL_PATH)
joblib.dump(meta_blob, OUT_MODEL_META)
print(f"Saved model: {OUT_MODEL_PATH} / meta: {OUT_MODEL_META}")

# =======================
# Predict all + Directions
# =======================
dall = xgb.DMatrix(feat.values)
all_ret = booster.predict(dall, iteration_range=(0, final_best_k))
meta_price = df['prev_actual'].values * np.exp(all_ret)

# 90% interval on price
price_lo = df['prev_actual'].values * np.exp(all_ret + q_lo)
price_hi = df['prev_actual'].values * np.exp(all_ret + q_hi)

# Directions: เทียบ prev_actual
prev_act = df['prev_actual'].values
actual_dir = (df['actual_price'].values > prev_act).astype(int)

# หากราคาทำนายของ LSTM/GRU เป็น NaN → ตั้ง dir เป็น -1 เพื่อกันไม่ให้ปนในเมตริก
lstm_dir = np.where(np.isfinite(pred_lstm), (pred_lstm > prev_act).astype(int), -1)
gru_dir  = np.where(np.isfinite(pred_gru ), (pred_gru  > prev_act).astype(int), -1)
meta_dir = (meta_price > prev_act).astype(int)

# === NEW: calibrated prob + neutral zone ===
if iso_cal is not None:
    meta_prob_up = iso_cal.predict(all_ret)
else:
    meta_prob_up = np.full_like(all_ret, np.nan, dtype=float)

# Neutral zone บนสเกล return
meta_dir_th = np.where(np.abs(all_ret) < gamma_margin, -1, (all_ret > 0).astype(int))

# correctness flags เพื่อเขียนลง CSV (แถวไหนที่ LSTM/GRU ไม่มีพยากรณ์จะให้เป็น NaN)
meta_dir_correct = (meta_dir == actual_dir).astype(int)
lstm_dir_correct = np.where(lstm_dir!=-1, (lstm_dir==actual_dir).astype(int), np.nan)
gru_dir_correct  = np.where(gru_dir !=-1, (gru_dir ==actual_dir).astype(int), np.nan)

out = pd.DataFrame({
    'Ticker': df['Ticker'],
    'Date':   df['Date'],
    'Prev_Actual': prev_act,

    'Meta_Predicted_Price': meta_price,
    'Meta_Price_Lo95': price_lo,
    'Meta_Price_Hi95': price_hi,
    'Meta_Pred_Dir': meta_dir,
    'Meta_Pred_Dir_Th': meta_dir_th,                # NEW tri-state (-1,0,1)
    'Meta_Prob_Up': meta_prob_up,                   # NEW calibrated probability (ถ้ามี)

    'LSTM_Predicted_Price': pred_lstm,
    'GRU_Predicted_Price':  pred_gru,
    'LSTM_Pred_Dir': lstm_dir,
    'GRU_Pred_Dir':  gru_dir,

    'Actual_Price': df['actual_price'].values,
    'Actual_Dir':   actual_dir,

    # ธงบอก “ถูกทิศไหม”
    'Meta_Dir_Correct': meta_dir_correct,
    'LSTM_Dir_Correct': lstm_dir_correct,
    'GRU_Dir_Correct':  gru_dir_correct,
}).sort_values(['Ticker','Date'])

out.to_csv(OUT_PRED_CSV, index=False)
print(f"Saved predictions: {OUT_PRED_CSV}")

# =======================
# Direction metrics — Validation only (per-ticker)  [META only, slim columns]
# =======================
if val_mask.sum() > 0:
    vt = pd.DataFrame({
        'Ticker': df.loc[val_mask, 'Ticker'].values,
        'y_true_dir': actual_dir[val_mask],
        'meta_dir':   meta_dir[val_mask],
        'y_true_price': df.loc[val_mask, 'actual_price'].values,
        'y_pred_price': _price_from_ret(
            df.loc[val_mask, 'prev_actual'].values,
            booster.predict(xgb.DMatrix(feat.loc[val_mask].values), iteration_range=(0, final_best_k))
        ),
    })

    rows = []
    for tkr, g in vt.groupby('Ticker'):
        # price metrics ต่อ ticker (VAL)
        try:
            price_mae  = mean_absolute_error(g['y_true_price'], g['y_pred_price'])
            price_rmse = rmse_np(g['y_true_price'], g['y_pred_price'])
            price_r2   = r2_score(g['y_true_price'], g['y_pred_price']) if len(np.unique(g['y_true_price'])) > 1 else np.nan
        except Exception:
            price_mae = price_rmse = price_r2 = np.nan

        # META direction metrics
        rep = dir_report(g['y_true_dir'].to_numpy(), g['meta_dir'].to_numpy(), ignore_val=-1)

        rows.append({
            'Ticker':    tkr,
            'mae':       float(price_mae)  if np.isfinite(price_mae)  else np.nan,
            'rmse':      float(price_rmse) if np.isfinite(price_rmse) else np.nan,
            'r2':        float(price_r2)   if np.isfinite(price_r2)   else np.nan,
            'acc':       float(rep['ACC'])  if np.isfinite(rep['ACC'])  else np.nan,
            'f1':        float(rep['F1'])   if np.isfinite(rep['F1'])   else np.nan,
            'precision': float(rep['PREC']) if np.isfinite(rep['PREC']) else np.nan,
            'recall':    float(rep['REC'])  if np.isfinite(rep['REC'])  else np.nan,
        })

    pd.DataFrame(rows).sort_values(['Ticker']).to_csv(OUT_VAL_BY_TKR, index=False)
    print(f"Saved validation per-ticker (META only, slim): {OUT_VAL_BY_TKR}")
else:
    # ไม่มี validation แยก — เขียนไฟล์หัวคอลัมน์ว่างไว้กันพังพายป์ไลน์
    cols = ['Ticker','mae','rmse','r2','acc','f1','precision','recall']
    pd.DataFrame(columns=cols).to_csv(OUT_VAL_BY_TKR, index=False)
    print(f"[WARN] No validation split; wrote empty {OUT_VAL_BY_TKR} with headers.")


# =======================
# Direction metrics — All rows (per-ticker)
# =======================
base_df_all = pd.DataFrame({
    'Ticker': df['Ticker'],
    'y_true_dir': actual_dir,
    'meta_dir':   meta_dir,
    'meta_dir_th':meta_dir_th,
    'lstm_dir':   lstm_dir,
    'gru_dir':    gru_dir
})

full_rows = []
for tkr, g in base_df_all.groupby('Ticker'):
    rep_meta = dir_report(g['y_true_dir'].to_numpy(), g['meta_dir'].to_numpy(), ignore_val=-1)
    rep_meta_th = dir_report(g['y_true_dir'].to_numpy(), g['meta_dir_th'].to_numpy(), ignore_val=-1)
    rep_lstm = dir_report(g['y_true_dir'].to_numpy(), g['lstm_dir'].to_numpy(), ignore_val=-1)
    rep_gru  = dir_report(g['y_true_dir'].to_numpy(), g['gru_dir'].to_numpy(),  ignore_val=-1)

    full_rows.append({
        'Ticker': tkr,

        'Support_META': rep_meta['Support'], 'TP_META': rep_meta['TP'], 'FP_META': rep_meta['FP'],
        'TN_META': rep_meta['TN'], 'FN_META': rep_meta['FN'],
        'DirACC_META': rep_meta['ACC'], 'DirF1_META': rep_meta['F1'],
        'DirPREC_META': rep_meta['PREC'], 'DirREC_META': rep_meta['REC'], 'DirMCC_META': rep_meta['MCC'],

        # NEW: META_TH
        'Support_META_TH': rep_meta_th['Support'], 'TP_META_TH': rep_meta_th['TP'], 'FP_META_TH': rep_meta_th['FP'],
        'TN_META_TH': rep_meta_th['TN'], 'FN_META_TH': rep_meta_th['FN'],
        'DirACC_META_TH': rep_meta_th['ACC'], 'DirF1_META_TH': rep_meta_th['F1'],
        'DirPREC_META_TH': rep_meta_th['PREC'], 'DirREC_META_TH': rep_meta_th['REC'], 'DirMCC_META_TH': rep_meta_th['MCC'],

        'Support_LSTM': rep_lstm['Support'], 'TP_LSTM': rep_lstm['TP'], 'FP_LSTM': rep_lstm['FP'],
        'TN_LSTM': rep_lstm['TN'], 'FN_LSTM': rep_lstm['FN'],
        'DirACC_LSTM': rep_lstm['ACC'], 'DirF1_LSTM': rep_lstm['F1'],
        'DirPREC_LSTM': rep_lstm['PREC'], 'DirREC_LSTM': rep_lstm['REC'], 'DirMCC_LSTM': rep_lstm['MCC'],

        'Support_GRU': rep_gru['Support'], 'TP_GRU': rep_gru['TP'], 'FP_GRU': rep_gru['FP'],
        'TN_GRU': rep_gru['TN'], 'FN_GRU': rep_gru['FN'],
        'DirACC_GRU': rep_gru['ACC'], 'DirF1_GRU': rep_gru['F1'],
        'DirPREC_GRU': rep_gru['PREC'], 'DirREC_GRU': rep_gru['REC'], 'DirMCC_GRU': rep_gru['MCC'],
    })

pd.DataFrame(full_rows).sort_values(['Ticker']).to_csv(OUT_DIR_BY_TKR_ALL, index=False)
print(f"Saved per-ticker (ALL rows) dir metrics: {OUT_DIR_BY_TKR_ALL}")

# =======================
# Overall summaries (Val-only & All-rows)
# =======================
overall_rows = []

# Val-only (ถ้ามี)
if val_mask.sum() > 0:
    rep_meta_v   = dir_report(actual_dir[val_mask], meta_dir[val_mask], ignore_val=-1)
    rep_meta_th_v= dir_report(actual_dir[val_mask], meta_dir_th[val_mask], ignore_val=-1)
    rep_lstm_v   = dir_report(actual_dir[val_mask], lstm_dir[val_mask], ignore_val=-1)
    rep_gru_v    = dir_report(actual_dir[val_mask], gru_dir[val_mask],  ignore_val=-1)
    row_val = {
        'Scope': 'VAL_ONLY',
        # META
        'DirACC_META': rep_meta_v['ACC'], 'DirF1_META': rep_meta_v['F1'],
        'DirPREC_META': rep_meta_v['PREC'], 'DirREC_META': rep_meta_v['REC'], 'DirMCC_META': rep_meta_v['MCC'],
        'Support_META': rep_meta_v['Support'],
        # META_TH
        'DirACC_META_TH': rep_meta_th_v['ACC'], 'DirF1_META_TH': rep_meta_th_v['F1'],
        'DirPREC_META_TH': rep_meta_th_v['PREC'], 'DirREC_META_TH': rep_meta_th_v['REC'], 'DirMCC_META_TH': rep_meta_th_v['MCC'],
        'Support_META_TH': rep_meta_th_v['Support'],
        # LSTM
        'DirACC_LSTM': rep_lstm_v['ACC'], 'DirF1_LSTM': rep_lstm_v['F1'],
        'DirPREC_LSTM': rep_lstm_v['PREC'], 'DirREC_LSTM': rep_lstm_v['REC'], 'DirMCC_LSTM': rep_lstm_v['MCC'],
        'Support_LSTM': rep_lstm_v['Support'],
        # GRU
        'DirACC_GRU': rep_gru_v['ACC'], 'DirF1_GRU': rep_gru_v['F1'],
        'DirPREC_GRU': rep_gru_v['PREC'], 'DirREC_GRU': rep_gru_v['REC'], 'DirMCC_GRU': rep_gru_v['MCC'],
        'Support_GRU': rep_gru_v['Support'],
        # Price
        'MAE': float(val_mae) if np.isfinite(val_mae) else np.nan,
        'RMSE': float(val_rmse) if np.isfinite(val_rmse) else np.nan,
        'R2': float(val_r2) if np.isfinite(val_r2) else np.nan,
        'BestTrees': int(final_best_k),
    }
    if np.isfinite(auc_val) or np.isfinite(brier_val):
        row_val['AUC_META'] = float(auc_val) if np.isfinite(auc_val) else np.nan
        row_val['Brier_META'] = float(brier_val) if np.isfinite(brier_val) else np.nan
    overall_rows.append(row_val)

# All-rows
rep_meta_a    = dir_report(actual_dir, meta_dir, ignore_val=-1)
rep_meta_th_a = dir_report(actual_dir, meta_dir_th, ignore_val=-1)
rep_lstm_a    = dir_report(actual_dir, lstm_dir, ignore_val=-1)
rep_gru_a     = dir_report(actual_dir, gru_dir,  ignore_val=-1)
row_all = {
    'Scope': 'ALL_ROWS',
    # META
    'DirACC_META': rep_meta_a['ACC'], 'DirF1_META': rep_meta_a['F1'],
    'DirPREC_META': rep_meta_a['PREC'], 'DirREC_META': rep_meta_a['REC'], 'DirMCC_META': rep_meta_a['MCC'],
    'Support_META': rep_meta_a['Support'],
    # META_TH
    'DirACC_META_TH': rep_meta_th_a['ACC'], 'DirF1_META_TH': rep_meta_th_a['F1'],
    'DirPREC_META_TH': rep_meta_th_a['PREC'], 'DirREC_META_TH': rep_meta_th_a['REC'], 'DirMCC_META_TH': rep_meta_th_a['MCC'],
    'Support_META_TH': rep_meta_th_a['Support'],
    # LSTM
    'DirACC_LSTM': rep_lstm_a['ACC'], 'DirF1_LSTM': rep_lstm_a['F1'],
    'DirPREC_LSTM': rep_lstm_a['PREC'], 'DirREC_LSTM': rep_lstm_a['REC'], 'DirMCC_LSTM': rep_lstm_a['MCC'],
    'Support_LSTM': rep_lstm_a['Support'],
    # GRU
    'DirACC_GRU': rep_gru_a['ACC'], 'DirF1_GRU': rep_gru_a['F1'],
    'DirPREC_GRU': rep_gru_a['PREC'], 'DirREC_GRU': rep_gru_a['REC'], 'DirMCC_GRU': rep_gru_a['MCC'],
    'Support_GRU': rep_gru_a['Support'],
    'BestTrees': int(final_best_k),
}
overall_rows.append(row_all)

pd.DataFrame(overall_rows).to_csv(OUT_DIR_SUMMARY_OVERALL, index=False)
print(f"Saved overall dir summaries: {OUT_DIR_SUMMARY_OVERALL}")

# =======================
# Feature importance (gain)
# =======================
gain_named = _map_gain_to_names(xgb_reg.get_booster(), feat_cols)
pd.Series(gain_named).sort_values(ascending=False).to_csv(OUT_FEAT_GAIN, header=False)
print(f"Saved feature importance (gain): {OUT_FEAT_GAIN}")

# =======================
# NEW: Simple, standardized metric snapshots
# =======================
def _pick_dir_metrics(rep):
    """แปลง dir_report() → เมตริกมาตรฐาน"""
    return dict(
        acc = float(rep['ACC']) if np.isfinite(rep['ACC']) else np.nan,
        f1  = float(rep['F1'])  if np.isfinite(rep['F1'])  else np.nan,
        precision = float(rep['PREC']) if np.isfinite(rep['PREC']) else np.nan,
        recall    = float(rep['REC'])  if np.isfinite(rep['REC'])  else np.nan,
    )

# -------- Overall (2 scopes: VAL_ONLY ถ้ามี + ALL_ROWS) --------
overall_std_rows = []

def _append_overall(scope, price_mae, price_rmse, price_r2,
                    rep_meta, rep_meta_th, rep_lstm, rep_gru):
    for model_name, rep in [
        ('META',    rep_meta),
        ('META_TH', rep_meta_th),
        ('LSTM',    rep_lstm),
        ('GRU',     rep_gru),
    ]:
        d = _pick_dir_metrics(rep)
        overall_std_rows.append({
            'scope': scope,
            'model': model_name,
            'mae': float(price_mae) if np.isfinite(price_mae) else np.nan,
            'rmse': float(price_rmse) if np.isfinite(price_rmse) else np.nan,
            'r2':   float(price_r2) if np.isfinite(price_r2) else np.nan,
            'acc': d['acc'], 'f1': d['f1'],
            'precision': d['precision'], 'recall': d['recall'],
        })

# VAL_ONLY
if val_mask.sum() > 0:
    rep_meta_v    = dir_report(actual_dir[val_mask], meta_dir[val_mask],    ignore_val=-1)
    rep_meta_th_v = dir_report(actual_dir[val_mask], meta_dir_th[val_mask], ignore_val=-1)
    rep_lstm_v    = dir_report(actual_dir[val_mask], lstm_dir[val_mask],    ignore_val=-1)
    rep_gru_v     = dir_report(actual_dir[val_mask], gru_dir[val_mask],     ignore_val=-1)
    _append_overall('VAL_ONLY', val_mae, val_rmse, val_r2,
                    rep_meta_v, rep_meta_th_v, rep_lstm_v, rep_gru_v)

# ALL_ROWS (ราคา: ไม่สรุปใน scope นี้ → NaN)
rep_meta_a     = dir_report(actual_dir, meta_dir,    ignore_val=-1)
rep_meta_th_a  = dir_report(actual_dir, meta_dir_th, ignore_val=-1)
rep_lstm_a     = dir_report(actual_dir, lstm_dir,    ignore_val=-1)
rep_gru_a      = dir_report(actual_dir, gru_dir,     ignore_val=-1)

_append_overall('ALL_ROWS', np.nan, np.nan, np.nan,
                rep_meta_a, rep_meta_th_a, rep_lstm_a, rep_gru_a)

pd.DataFrame(overall_std_rows).to_csv(OUT_SIMPLE_OVERALL, index=False)
print(f"Saved standardized overall metrics: {OUT_SIMPLE_OVERALL}")

# -------- Per-ticker (VAL_ONLY) --------
if val_mask.sum() > 0:
    vt = pd.DataFrame({
        'Ticker': df.loc[val_mask, 'Ticker'].values,
        'y_true_dir': actual_dir[val_mask],
        'meta_dir':   meta_dir[val_mask],
        'meta_dir_th':meta_dir_th[val_mask],
        'lstm_dir':   lstm_dir[val_mask],
        'gru_dir':    gru_dir[val_mask],
        'y_true_price': df.loc[val_mask, 'actual_price'].values,
        'y_pred_price': _price_from_ret(
            df.loc[val_mask, 'prev_actual'].values,
            booster.predict(xgb.DMatrix(feat.loc[val_mask].values), iteration_range=(0, final_best_k))
        ),
    })

    rows_val_by_tk = []
    for tkr, g in vt.groupby('Ticker'):
        # price metrics ต่อ ticker (VAL)
        try:
            p_mae  = mean_absolute_error(g['y_true_price'], g['y_pred_price'])
            p_rmse = rmse_np(g['y_true_price'], g['y_pred_price'])
            p_r2   = r2_score(g['y_true_price'], g['y_pred_price']) if len(np.unique(g['y_true_price']))>1 else np.nan
        except Exception:
            p_mae = p_rmse = p_r2 = np.nan

        for model_name, pred_col in [
            ('META',    'meta_dir'),
            ('META_TH', 'meta_dir_th'),
            ('LSTM',    'lstm_dir'),
            ('GRU',     'gru_dir'),
        ]:
            rep = dir_report(g['y_true_dir'].to_numpy(), g[pred_col].to_numpy(), ignore_val=-1)
            d = _pick_dir_metrics(rep)
            rows_val_by_tk.append({
                'scope': 'VAL_ONLY',
                'ticker': tkr,
                'model': model_name,
                'mae': float(p_mae) if np.isfinite(p_mae) else np.nan,
                'rmse': float(p_rmse) if np.isfinite(p_rmse) else np.nan,
                'r2':   float(p_r2) if np.isfinite(p_r2) else np.nan,
                'acc': d['acc'], 'f1': d['f1'],
                'precision': d['precision'], 'recall': d['recall'],
            })
    pd.DataFrame(rows_val_by_tk).sort_values(['ticker','model']).to_csv(OUT_SIMPLE_VAL_BY_TICKER, index=False)
    print(f"Saved standardized per-ticker (VAL_ONLY): {OUT_SIMPLE_VAL_BY_TICKER}")

# -------- Per-ticker (ALL_ROWS) --------
rows_all_by_tk = []
base_df_all = pd.DataFrame({
    'Ticker': df['Ticker'],
    'y_true_dir': actual_dir,
    'meta_dir':   meta_dir,
    'meta_dir_th':meta_dir_th,
    'lstm_dir':   lstm_dir,
    'gru_dir':    gru_dir,
})
for tkr, g in base_df_all.groupby('Ticker'):
    for model_name, pred_col in [
        ('META',    'meta_dir'),
        ('META_TH', 'meta_dir_th'),
        ('LSTM',    'lstm_dir'),
        ('GRU',     'gru_dir'),
    ]:
        rep = dir_report(g['y_true_dir'].to_numpy(), g[pred_col].to_numpy(), ignore_val=-1)
        d = _pick_dir_metrics(rep)
        rows_all_by_tk.append({
            'scope': 'ALL_ROWS',
            'ticker': tkr,
            'model': model_name,
            'mae': np.nan,   # ราคาไม่สรุปใน snapshot นี้สำหรับ ALL_ROWS ต่อ ticker
            'rmse': np.nan,
            'r2':   np.nan,
            'acc': d['acc'], 'f1': d['f1'],
            'precision': d['precision'], 'recall': d['recall'],
        })
pd.DataFrame(rows_all_by_tk).sort_values(['ticker','model']).to_csv(OUT_SIMPLE_ALL_BY_TICKER, index=False)
print(f"Saved standardized per-ticker (ALL_ROWS): {OUT_SIMPLE_ALL_BY_TICKER}")
