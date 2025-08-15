# stack_xgb_meta.py
import os, joblib
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (average_precision_score, accuracy_score, f1_score,
                             precision_score, recall_score, matthews_corrcoef)
from datetime import timedelta

# ======= ปรับ path ตรงนี้ให้ชี้ไปยัง CSV ของแต่ละโมเดล =======
LSTM_CSV = r'../LSTM_model/all_predictions_per_day_multi_task.csv'
GRU_CSV  = r'../GRU_model/all_predictions_per_day_multi_task.csv'

OUT_MODEL_PATH   = 'xgb_meta.json'
OUT_CALIB_PATH   = 'meta_isotonic.joblib'
OUT_PRED_CSV     = 'meta_predictions.csv'
VAL_RATIO        = 0.12      # ให้สอดคล้องกับสคริปต์หลัก
EMBARGO_DAYS     = 5         # กันการรั่วไหลขอบเขตเวลาเล็กน้อย

# ======= helper: เลือก threshold แบบคุม precision + base-rate =======
from sklearn.metrics import balanced_accuracy_score
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
        if rec < min_rec or prec < min_prec:
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
        cand.append((float(t), score, abs(yhat.mean() - (base_rate if base_rate is not None else y_true.mean()))))
    if not cand:
        return float(np.median(grid))
    cand.sort(key=lambda z: (-z[1], z[2]))
    return cand[0][0]

# ======= โหลดผลจาก 2 โมเดล =======
need_cols = ['Ticker','Date','Dir_Prob','Thr_Used','Predicted_Price','Last_Close','Predicted_Dir','Actual_Dir']
lstm = pd.read_csv(LSTM_CSV, parse_dates=['Date'])
gru  = pd.read_csv(GRU_CSV , parse_dates=['Date'])

for c in need_cols:
    if c not in lstm.columns or c not in gru.columns:
        raise ValueError(f"Column '{c}' not found in both CSVs. Found LSTM:{c in lstm.columns}, GRU:{c in gru.columns}")

# ตั้งชื่อคอลัมน์ให้แยกกันชัดเจนแล้ว merge ตาม (Ticker, Date)
lstm_ = lstm[need_cols].copy().rename(columns={k:f"{k}_lstm" for k in need_cols if k not in ['Ticker','Date']})
gru_  = gru [need_cols].copy().rename(columns={k:f"{k}_gru"  for k in need_cols if k not in ['Ticker','Date']})

df = pd.merge(lstm_, gru_, on=['Ticker','Date'], how='inner')

# เป้าหมาย y (ใช้จากฝั่ง LSTM; ควรเท่ากับฝั่ง GRU)
if 'Actual_Dir_lstm' not in df.columns: 
    raise RuntimeError("Missing Actual_Dir_lstm after merge.")
if 'Actual_Dir_gru' in df.columns:
    assert (df['Actual_Dir_lstm'].values == df['Actual_Dir_gru'].values).all(), "Actual_Dir mismatch."
y = df['Actual_Dir_lstm'].astype(int).values

# ======= สร้าง meta-features =======
def safe_log(x):
    x = np.asarray(x, dtype=float)
    with np.errstate(divide='ignore', invalid='ignore'):
        r = np.log(np.clip(x, 1e-12, None))
    r[~np.isfinite(r)] = 0.0
    return r

feat = pd.DataFrame(index=df.index)

# ความน่าจะเป็น/มาร์จินจากแต่ละโมเดล
feat['p_lstm']   = df['Dir_Prob_lstm'].astype(float)
feat['p_gru']    = df['Dir_Prob_gru'].astype(float)
feat['m_lstm']   = df['Dir_Prob_lstm'] - df['Thr_Used_lstm']
feat['m_gru']    = df['Dir_Prob_gru']  - df['Thr_Used_gru']
feat['agree']    = (df['Predicted_Dir_lstm'] == df['Predicted_Dir_gru']).astype(int)
feat['p_mean']   = (feat['p_lstm'] + feat['p_gru'])/2.0
feat['p_diff']   = feat['p_lstm'] - feat['p_gru']
feat['p_prod']   = feat['p_lstm'] * feat['p_gru']

# ผลตอบแทนคาดการณ์จากราคา (log-return)
feat['ret_pred_lstm'] = safe_log(df['Predicted_Price_lstm'] / df['Last_Close_lstm'])
feat['ret_pred_gru']  = safe_log(df['Predicted_Price_gru']  / df['Last_Close_gru'])
feat['ret_pred_mean'] = (feat['ret_pred_lstm'] + feat['ret_pred_gru'])/2.0
feat['ret_pred_diff'] = feat['ret_pred_lstm'] - feat['ret_pred_gru']

# เวลาพื้นฐาน
feat['dow'] = df['Date'].dt.weekday.astype(int)
feat['dom'] = df['Date'].dt.day.astype(int)

# ======= split แบบ time-based + embargo =======
dates_sorted = np.sort(df['Date'].unique())
cut = dates_sorted[int(len(dates_sorted)*(1.0-VAL_RATIO))]
cut_embargo = cut - np.timedelta64(EMBARGO_DAYS, 'D')

train_idx = df['Date'] <= cut_embargo
val_idx   = df['Date'] >  cut

X_train, y_train = feat[train_idx].values, y[train_idx]
X_val,   y_val   = feat[val_idx].values,   y[val_idx]

print(f"Train size: {X_train.shape}, Val size: {X_val.shape}")

# ======= เทรน XGB (โฟกัสทิศทาง) =======
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
    tree_method='hist'  # เปลือง RAM น้อย
)
xgb.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    verbose=False,
    early_stopping_rounds=60
)

# ======= คาลิเบรต & เลือก threshold บนชุดวาลิเดชัน =======
val_probs_raw = xgb.predict_proba(X_val)[:,1]
iso = IsotonicRegression(out_of_bounds='clip').fit(val_probs_raw, y_val)
val_probs_cal = iso.transform(val_probs_raw)

grid = np.linspace(0.2, 0.8, 121)
best_thr = pick_thr_rate_constrained(val_probs_cal, y_val, grid,
                                     min_rec=0.0, min_prec=0.55,
                                     base_rate=0.50, rate_tol=0.05, metric='mcc')
print(f"[META] Best global threshold: {best_thr:.3f}")

# ======= สรุปเมตริกบนวาลิเดชัน =======
from sklearn.metrics import precision_recall_curve
ap = average_precision_score(y_val, val_probs_cal)
yhat_val = (val_probs_cal >= best_thr).astype(int)
print(f"[META] AUC-PR={ap:.4f} | Acc={accuracy_score(y_val,yhat_val):.4f} | "
      f"F1={f1_score(y_val,yhat_val):.4f} | "
      f"P={precision_score(y_val,yhat_val,zero_division=0):.4f} | "
      f"R={recall_score(y_val,yhat_val,zero_division=0):.4f} | "
      f"MCC={matthews_corrcoef(y_val,yhat_val):.4f}")

# ======= เซฟโมเดล/คาลิเบรเตอร์ =======
xgb.save_model(OUT_MODEL_PATH)
joblib.dump({'iso': iso, 'thr': float(best_thr)}, OUT_CALIB_PATH)
print(f"Saved: {OUT_MODEL_PATH}, {OUT_CALIB_PATH}")

# ======= ทำพยากรณ์ meta ทั้งหมด (เพื่อดูภาพรวม/จัดเก็บ) =======
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
