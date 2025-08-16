# stack_xgb_meta.py
# -*- coding: utf-8 -*-
import os, json, joblib
import numpy as np
import pandas as pd
from datetime import timedelta
from xgboost import XGBClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (
    average_precision_score, accuracy_score, f1_score, precision_score,
    recall_score, matthews_corrcoef, balanced_accuracy_score
)

# ====== ปรับ path ให้ตรงโฟลเดอร์โมเดลของคุณ ======
LSTM_DIR = r'../LSTM_model'
GRU_DIR  = r'../GRU_model'

# จะพยายามโหลดไฟล์นี้ก่อน (บางคอลัมน์ไม่ครบ) ถ้าไม่พอจะ fallback ไป predictions_chunk_walkforward.csv
LSTM_PREF_FILE = 'all_predictions_per_day_multi_task.csv'
GRU_PREF_FILE  = 'all_predictions_per_day_multi_task.csv'

# fallback ไฟล์ (มี Last_Close และฟีเจอร์มากกว่า)
LSTM_FALLBACK_FILE = 'predictions_chunk_walkforward.csv'
GRU_FALLBACK_FILE  = 'predictions_chunk_walkforward.csv'

# thresholds ต่อหุ้นที่สคริปต์ LSTM/GRU เซฟไว้ (ถ้ามี)
LSTM_THR_JSON = 'dir_thresholds_per_ticker.json'
GRU_THR_JSON  = 'dir_thresholds_per_ticker.json'

# ====== Output ======
OUT_MODEL_PATH   = 'xgb_meta.json'
OUT_CALIB_PATH   = 'meta_isotonic.joblib'
OUT_PRED_CSV     = 'meta_predictions.csv'
OUT_VAL_SUMMARY  = 'meta_val_summary.csv'         # รวมทุกหุ้นในวาลิเดชัน
OUT_VAL_BY_TKR   = 'meta_val_metrics_per_ticker.csv'

# ====== split/time config ======
VAL_RATIO    = 0.12      # ให้สอดคล้องกับ LSTM/GRU
EMBARGO_DAYS = 5

# ====== ฟังก์ชันเลือก threshold ภายใต้ข้อจำกัด ======
def pick_thr_rate_constrained(probs, y_true, grid,
                              min_rec=0.0, min_prec=0.55,
                              base_rate=0.50, rate_tol=0.05,
                              metric='mcc'):
    y_true = np.asarray(y_true).astype(int)
    best_t, best_score = None, -1e9
    for t in grid:
        yhat = (probs >= t).astype(int)
        rec  = recall_score(y_true, yhat, zero_division=0)
        prec = precision_score(y_true, yhat, zero_division=0)
        posr = yhat.mean()
        if rec < min_rec or prec < min_prec:          # ข้อจำกัดขั้นต่ำ
            continue
        if base_rate is not None and abs(posr - base_rate) > rate_tol:
            continue
        score = matthews_corrcoef(y_true, yhat) if metric=='mcc' else balanced_accuracy_score(y_true, yhat)
        if np.isnan(score):
            continue
        if score > best_score:
            best_score, best_t = score, float(t)
    if best_t is not None:
        return best_t
    # fallback: เอาคะแนนดีที่สุด แล้วเลือกที่ base-rate ใกล้ base_rate
    cand = []
    for t in grid:
        yhat = (probs >= t).astype(int)
        score = matthews_corrcoef(y_true, yhat)
        if np.isnan(score):
            continue
        target_rate = base_rate if base_rate is not None else y_true.mean()
        cand.append((float(t), score, abs(yhat.mean() - target_rate)))
    if not cand:
        return float(np.median(grid))
    cand.sort(key=lambda z: (-z[1], z[2]))
    return cand[0][0]

def safe_log_ratio(num, den):
    num = np.asarray(num, float); den = np.asarray(den, float)
    with np.errstate(divide='ignore', invalid='ignore'):
        r = np.log(np.clip(num / den, 1e-12, None))
    r[~np.isfinite(r)] = 0.0
    return r

def load_thresholds_json(model_dir, thr_json_name):
    path = os.path.join(model_dir, thr_json_name)
    if not os.path.exists(path):
        return {}
    with open(path, 'r', encoding='utf-8') as f:
        d = json.load(f)
    # keys อาจเป็น str(ticker_id); เก็บกลับเป็น dict[str]->float
    return {str(k): float(v) for k, v in d.items()}

def standardize_predictions(model_dir, pref_file, fallback_file, thr_json_name):
    """
    คืน DataFrame ที่ normalize ชื่อคอลัมน์ให้เป็นชุดเดียวกัน:
    ['Ticker','Date','p','thr','pred_price','last_close','pred_dir','actual_dir','ticker_id']
    """
    # 1) เลือกไฟล์ที่จะโหลด
    pref_path  = os.path.join(model_dir, pref_file)
    fall_path  = os.path.join(model_dir, fallback_file)
    if os.path.exists(pref_path):
        df = pd.read_csv(pref_path, parse_dates=['Date'])
    elif os.path.exists(fall_path):
        df = pd.read_csv(fall_path, parse_dates=['Date'])
    else:
        raise FileNotFoundError(f"Cannot find prediction CSV in {model_dir}")

    cols = df.columns

    # 2) เลือกคอลัมน์ probability ที่เหมาะสม
    p_col = None
    for c in ['Dir_Prob', 'Dir_Prob_Cal', 'Dir_Prob_Raw']:
        if c in cols:
            p_col = c; break
    if p_col is None:
        raise RuntimeError(f"No probability column found in {model_dir}")

    # 3) last_close
    if 'Last_Close' in cols:
        last_close = df['Last_Close'].astype(float).values
    else:
        # พยายามคำนวณจากราคา ถ้าไม่มี ให้ 0 (แต่จะกระทบฟีเจอร์ ret_pred)
        if 'Actual_Price' in cols and 'Price_Change_Actual' in cols:
            last_close = (df['Actual_Price'] - df['Price_Change_Actual']).astype(float).values
        else:
            last_close = np.zeros(len(df), float)

    # 4) pred/actual price
    pred_price = df['Predicted_Price'].astype(float).values if 'Predicted_Price' in cols else np.zeros(len(df))
    # actual_dir: ถ้ามีใช้เลย; ไม่มีก็คำนวณจาก Actual_Price vs Last_Close
    if 'Actual_Dir' in cols:
        actual_dir = df['Actual_Dir'].astype(int).values
    else:
        if 'Actual_Price' in cols and np.any(last_close > 0):
            actual_dir = (df['Actual_Price'].astype(float).values > last_close).astype(int)
        else:
            raise RuntimeError(f"Actual_Dir not found and cannot be derived in {model_dir}")

    # 5) predicted dir (ถ้ามีใช้เลย; ไม่มีก็สร้างจาก threshold)
    if 'Predicted_Dir' in cols:
        pred_dir = df['Predicted_Dir'].astype(int).values
    else:
        pred_dir = None  # จะคำนวณทีหลังหากต้องการ

    # 6) thresholds ต่อหุ้น
    thr_map = load_thresholds_json(model_dir, thr_json_name)  # dict[str(ticker_id)]=thr
    # หา ticker_id ถ้ามี
    if 'Ticker_ID' in cols:
        t_ids = df['Ticker_ID'].astype(int).astype(str).values
    else:
        # ถ้าไม่มี Ticker_ID ก็ใส่ None ไว้ก่อน
        t_ids = np.array([None]*len(df), dtype=object)

    # ถ้ามีคอลัมน์ Thr_Used ก็ใช้มันเป็นหลัก
    if 'Thr_Used' in cols:
        thr_used = df['Thr_Used'].astype(float).values
    else:
        # พยายาม map จากไฟล์ thresholds ต่อหุ้น
        thr_used = np.array([
            float(thr_map.get(str(tid), 0.5)) if tid is not None else 0.5
            for tid in t_ids
        ], dtype=float)

    # ถ้ายังไม่มี pred_dir ให้สร้างจาก p_col + thr_used
    if pred_dir is None:
        pred_dir = (df[p_col].astype(float).values >= thr_used).astype(int)

    out = pd.DataFrame({
        'Ticker': df['Ticker'].astype(str),
        'Date':   pd.to_datetime(df['Date']),
        'p':      df[p_col].astype(float).values,
        'thr':    thr_used,
        'pred_price': pred_price,
        'last_close': last_close,
        'pred_dir': pred_dir,
        'actual_dir': actual_dir
    })
    # เก็บ ticker_id ถ้ามี (ใช้ตอน merge threshold/วิเคราะห์ต่อ)
    if 'Ticker_ID' in cols:
        out['ticker_id'] = df['Ticker_ID'].astype(int).values
    else:
        out['ticker_id'] = -1
    return out

# ====== โหลดและมาตรฐานข้อมูลสองโมเดล ======
lstm_df = standardize_predictions(LSTM_DIR, LSTM_PREF_FILE, LSTM_FALLBACK_FILE, LSTM_THR_JSON)
gru_df  = standardize_predictions(GRU_DIR,  GRU_PREF_FILE,  GRU_FALLBACK_FILE,  GRU_THR_JSON)

# merge ด้วย (Ticker, Date)
df = pd.merge(
    lstm_df.add_suffix('_lstm').rename(columns={'Ticker_lstm':'Ticker', 'Date_lstm':'Date'}),
    gru_df.add_suffix('_gru').rename(columns={'Ticker_gru':'Ticker',  'Date_gru':'Date'}),
    on=['Ticker','Date'],
    how='inner'
)

# target (ควรตรงกันทั้งสองฝั่ง)
y = df['actual_dir_lstm'].astype(int).values
if 'actual_dir_gru' in df.columns:
    assert (y == df['actual_dir_gru'].astype(int).values).all(), "Actual_Dir mismatch between LSTM and GRU!"

# ====== meta features ======
feat = pd.DataFrame(index=df.index)
feat['p_lstm'] = df['p_lstm'].astype(float)
feat['p_gru']  = df['p_gru'].astype(float)
feat['m_lstm'] = df['p_lstm'] - df['thr_lstm']
feat['m_gru']  = df['p_gru']  - df['thr_gru']
feat['agree']  = (df['pred_dir_lstm'] == df['pred_dir_gru']).astype(int)
feat['p_mean'] = (feat['p_lstm'] + feat['p_gru'])/2.0
feat['p_diff'] = feat['p_lstm'] - feat['p_gru']
feat['p_prod'] = feat['p_lstm'] * feat['p_gru']

feat['ret_pred_lstm'] = safe_log_ratio(df['pred_price_lstm'], df['last_close_lstm'])
feat['ret_pred_gru']  = safe_log_ratio(df['pred_price_gru'],  df['last_close_gru'])
feat['ret_pred_mean'] = (feat['ret_pred_lstm'] + feat['ret_pred_gru'])/2.0
feat['ret_pred_diff'] = feat['ret_pred_lstm'] - feat['ret_pred_gru']

feat['dow'] = pd.to_datetime(df['Date']).dt.weekday.astype(int)
feat['dom'] = pd.to_datetime(df['Date']).dt.day.astype(int)

# ====== split แบบ time-based + embargo ======
dates_sorted = np.sort(df['Date'].unique())
cut = dates_sorted[int(len(dates_sorted)*(1.0-VAL_RATIO))]
cut_embargo = cut - np.timedelta64(EMBARGO_DAYS, 'D')

train_idx = df['Date'] <= cut_embargo
val_idx   = df['Date'] >  cut

X_train, y_train = feat[train_idx].values, y[train_idx]
X_val,   y_val   = feat[val_idx].values,   y[val_idx]

print(f"[META] Train size: {X_train.shape}, Val size: {X_val.shape}")

# ====== เทรน XGB ======
xgb = XGBClassifier(
    n_estimators=600,
    learning_rate=0.05,
    max_depth=4,
    subsample=0.9,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    reg_alpha=0.0,
    objective='binary:logistic',
    eval_metric='aucpr',
    tree_method='hist'
)
xgb.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    verbose=False,
    early_stopping_rounds=60
)

# ====== คาลิเบรต & เลือก threshold ======
val_probs_raw = xgb.predict_proba(X_val)[:,1]
iso = IsotonicRegression(out_of_bounds='clip').fit(val_probs_raw, y_val)
val_probs_cal = iso.transform(val_probs_raw)

grid = np.linspace(0.2, 0.8, 121)
best_thr = pick_thr_rate_constrained(
    val_probs_cal, y_val, grid,
    min_rec=0.0, min_prec=0.55,
    base_rate=0.50, rate_tol=0.05, metric='mcc'
)
yhat_val = (val_probs_cal >= best_thr).astype(int)

ap  = average_precision_score(y_val, val_probs_cal)
acc = accuracy_score(y_val, yhat_val)
f1  = f1_score(y_val, yhat_val)
prec= precision_score(y_val, yhat_val, zero_division=0)
rec = recall_score(y_val, yhat_val, zero_division=0)
mcc = matthews_corrcoef(y_val, yhat_val)

print("\n===== META VALIDATION (overall) =====")
print(f"AUC-PR : {ap:.4f}")
print(f"ACC    : {acc:.4f}")
print(f"F1     : {f1:.4f}")
print(f"Prec   : {prec:.4f}")
print(f"Recall : {rec:.4f}")
print(f"MCC    : {mcc:.4f}")
print(f"BestThr: {best_thr:.3f}")

# ====== เซฟ model + calibrator ======
xgb.save_model(OUT_MODEL_PATH)
joblib.dump({'iso': iso, 'thr': float(best_thr)}, OUT_CALIB_PATH)
print(f"\nSaved: {OUT_MODEL_PATH}, {OUT_CALIB_PATH}")

# ====== ทำพยากรณ์ทั้งชุดแล้วเซฟ ======
all_probs_raw = xgb.predict_proba(feat.values)[:,1]
all_probs_cal = iso.transform(all_probs_raw)
meta_pred_dir = (all_probs_cal >= best_thr).astype(int)

out = pd.DataFrame({
    'Ticker': df['Ticker'],
    'Date':   df['Date'],
    'Meta_Prob': all_probs_cal,
    'Meta_Pred_Dir': meta_pred_dir,
    'Actual_Dir': y
}).sort_values(['Ticker','Date'])
out.to_csv(OUT_PRED_CSV, index=False)
print(f"Saved predictions: {OUT_PRED_CSV}")

# ====== (เสริม) สรุปเมตริกวาลิเดชันเป็นตารางอ่านง่าย + per-ticker ======
val_tab = pd.DataFrame({
    'Ticker': df.loc[val_idx, 'Ticker'].values,
    'Date':   df.loc[val_idx, 'Date'].values,
    'y_true': y_val,
    'y_prob': val_probs_cal,
    'y_hat' : yhat_val
})

# รวมภาพรวม (1 แถว)
overall_row = pd.DataFrame([{
    'scope':'overall', 'AUC_PR':ap, 'ACC':acc, 'F1':f1, 'PREC':prec, 'REC':rec, 'MCC':mcc, 'BestThr':best_thr
}])
overall_row.to_csv(OUT_VAL_SUMMARY, index=False)

# per-ticker (เขียนไฟล์ CSV ไม่พิมพ์ยาวใน console)
rows = []
for tkr, g in val_tab.groupby('Ticker'):
    yt = g['y_true'].values; yp = g['y_prob'].values; yh = g['y_hat'].values
    rows.append({
        'Ticker': tkr,
        'Support': len(g),
        'ACC': accuracy_score(yt,yh),
        'F1': f1_score(yt,yh, zero_division=0),
        'PREC': precision_score(yt,yh, zero_division=0),
        'REC': recall_score(yt,yh, zero_division=0),
        'MCC': matthews_corrcoef(yt,yh) if len(np.unique(yh))>1 else 0.0,
        'AUC_PR': average_precision_score(yt, yp)
    })
pd.DataFrame(rows).sort_values('MCC', ascending=False).to_csv(OUT_VAL_BY_TKR, index=False)
print(f"Saved validation summaries: {OUT_VAL_SUMMARY}, {OUT_VAL_BY_TKR}")
