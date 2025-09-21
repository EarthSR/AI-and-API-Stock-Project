import os, random, json, joblib, warnings, sys, time, math, csv, gc, traceback
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore", category=FutureWarning)

# ======= CPU only (ปรับตามเครื่องได้) =======
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# ======= Seeds =======
SEED = 42
random.seed(SEED); np.random.seed(SEED)

# ===================== FULL CONFIG BLOCK (paths + targeted FN relief) =====================
import os, time

# ---------- Log & output paths ----------
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))

# โฟลเดอร์หลัก
LOG_DIR   = os.path.join(BASE_DIR, "logs")
DAILY_DIR = os.path.join(LOG_DIR, "daily_csv")
MODEL_DIR = os.path.join(LOG_DIR, "models")  # เก็บโมเดล/คาลิเบรต/อาร์ติแฟกต์
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(DAILY_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# Run id / logs
RUN_ID           = time.strftime("%Y%m%d-%H%M%S")
DIAG_LOG_PATH    = os.path.join(LOG_DIR, "diagnostics.log")
PROGRESS_PATH    = os.path.join(LOG_DIR, "progress.jsonl")
PRED_LOG_PATH    = os.path.join(LOG_DIR, f"predictions_{RUN_ID}.jsonl")
PRED_LATEST_PATH = os.path.join(LOG_DIR, "predictions_latest.jsonl")

# ผลลัพธ์สรุป
STREAM_PRED_PATH    = os.path.join(LOG_DIR, "predictions_chunk_walkforward.csv")
STREAM_CHUNK_PATH   = os.path.join(LOG_DIR, "chunk_metrics.csv")
STREAM_OVERALL_PATH = os.path.join(LOG_DIR, "overall_metrics_per_ticker.csv")
DAILY_ALL_PATH      = os.path.join(LOG_DIR, "daily_all_predictions.csv")

# ---------- Model artifact paths ----------
BEST_MODEL_PATH_STATIC = os.path.join(MODEL_DIR, "best_model_static.keras")
BEST_MODEL_PATH_ONLINE = os.path.join(MODEL_DIR, "best_model_online_last.keras")
BEST_MODEL_PATH        = BEST_MODEL_PATH_STATIC  # alias เดิม

ISO_CAL_PATH           = os.path.join(MODEL_DIR, "iso_calibrators_per_ticker.pkl")
META_LR_PATH           = os.path.join(MODEL_DIR, "meta_lr_per_ticker.pkl")
DIR_THR_PATH           = os.path.join(MODEL_DIR, "dir_thresholds_per_ticker.json")
VAL_PREV_MAP_PATH      = os.path.join(MODEL_DIR, "val_prev_map.json")
DIR_WEIGHT_OVR_PATH    = os.path.join(MODEL_DIR, "dir_weight_ovr.json")
DIR_TEMPERATURE_PATH   = os.path.join(MODEL_DIR, "dir_temperature_per_ticker.json")
SERVING_ARTIFACTS_PATH = os.path.join(MODEL_DIR, "serving_artifacts.pkl")
PRODUCTION_CONFIG_PATH = os.path.join(MODEL_DIR, "production_model_config.json")

# ---------- Runtime switches ----------
# Daily:       SKIP_TRAIN=True,  SKIP_CALIBRATION=True,  STRICT_LOAD=True
# Weekly recal:SKIP_TRAIN=True,  SKIP_CALIBRATION=False, STRICT_LOAD=False
SKIP_TRAIN=True
SKIP_CALIBRATION=True
STRICT_LOAD=True
USE_WFV_MODEL_CLONE    = True
PERSIST_ONLINE_UPDATES = False

# ---------- Model hyperparams ----------
BEST_PARAMS = {
    'chunk_size': 100,
    'embedding_dim': 24,
    'LSTM_units_1': 48,
    'LSTM_units_2': 24,
    'dropout_rate': 0.15,
    'dense_units': 66,
    'learning_rate': 1.6e-4,
    'retrain_frequency': 10,
    'seq_length': 10
}

# ---------- Performance & losses ----------
MC_DIR_SAMPLES   = 8
DIR_LABEL_SMOOTH = 0.00
DIR_LOSS_WEIGHT  = 0.00

# ---------- Online learning gates ----------
CONF_GATE, UNC_MAX, MARGIN = True, 0.10, 0.05
ALLOW_PRICE_ONLINE = True
Z_GATE_ONLINE = 1.05

# ---------- Calibration objective ----------
THR_OBJECTIVE_US = 'acc'
THR_OBJECTIVE_TH = 'acc'
FBETA_TH         = 1.20

# ---------- Thresholds & search window ----------
THRESH_MIN    = 0.50
MIN_RECALL    = 0.50
MIN_PRECISION = 0.00
THR_CLIP_LOW, THR_CLIP_HIGH = 0.44, 0.86          # คลาย clamp ล่าง global
ANCHOR_RADIUS = 0.18
THR_CLIP_LOW_TH = 0.448                         # คลาย floor ฝั่ง TH เพิ่ม (จาก 0.465)

# ---------- Smoothing ----------
USE_EMA_PROB     = True
ALPHA_EMA_LOWVOL = 0.68
ALPHA_EMA_HIVOL  = 0.62
SIGMA_VOL_SPLIT  = 0.013

MAJORITY_K       = 9
HYSTERESIS_BAND  = 0.0060
Z_STRONG_CUT     = 0.90

# ---------- Consensus (reserve; price-only ไม่ใช้ p_dir) ----------
LAMBDA_AGREE   = 0.15
DISAGREE_DELTA = 0.020
AGREE_BAND     = 0.015
Z_HIGH         = 0.90
UNC_SOFT       = 0.14

# ---------- PSC (Prior Shift Correction) ----------
USE_PSC          = True
PRIOR_EMA_ALPHA  = 0.07
TARGET_EMA_ALPHA = 0.15
PRIOR_MIN_N      = 40
ACT_PREV_MIN_N   = 12
PSC_LOGIT_CAP    = 0.20

# ---------- Trend prior ----------
USE_TREND_PRIOR   = True
TREND_WIN         = 7
TREND_KAPPA       = 2.0
TREND_W_LOWVOL_TH = 0.08
TREND_W_HIVOL_TH  = 0.12
TREND_W_LOWVOL_US = 0.04
TREND_W_HIVOL_US  = 0.07
TREND_W_OVR = {'GOOGL':0.04,'NVDA':0.05,'AAPL':0.05,'MSFT':0.05, 'ADVANC':0.14}  # boost โมเมนตัม ADVANC

# ---------- Threshold adapt from PSC ----------
USE_THR_ADAPT_FROM_PSC = False
THR_ADAPT_GAIN = 0.03
THR_ADAPT_CLIP = 0.015

# ---------- Per-ticker threshold offsets (persistent) ----------
THR_DELTA_OVR = {
    # US – คงเดิม (มี fine-tune ผ่าน precision_tune)
    'NVDA': -0.006, 'GOOGL': -0.006, 'AVGO': -0.006, 'AAPL': -0.006, 'MSFT': -0.006,
    'TSM':  -0.006, 'AMD':  -0.006, 'META': -0.010, 'AMZN': -0.010,
    'TSLA': +0.006,

    # TH – ผ่อน ADVANC เพิ่มเติมเพื่อลด FN
    'INSET': -0.004, 'JAS': 0.000, 'ADVANC': -0.010, 'DITTO': 0.000, 'DIF': 0.000,
    'TRUE':  0.000, 'HUMAN': 0.000, 'INET': 0.000, 'JMART': 0.000,
}

# ---------- Market-level base delta ----------
# ผ่อน TH ลงเพื่อยก PosRate/Recall (final)
TH_MARKET_DELTA = {'TH': -0.012, 'US': 0.000, 'OTHER': 0.000}

# ---------- Minimum recall overrides ----------
MIN_RECALL_OVR = {
    'ADVANC':0.50,'DITTO':0.50,'HUMAN':0.50,'INET':0.50,'JAS':0.50,
    'DIF':0.50,'TRUE':0.50,'INSET':0.50,'JMART':0.50
}
MIN_RECALL_TH_MARKET_BONUS = 0.00

# ---------- EMA override (TH ให้นิ่งขึ้น) ----------
ALPHA_EMA_OVR = {
    'ADVANC':0.62,'DITTO':0.62,'HUMAN':0.62,'INET':0.62,'JAS':0.62,
    'DIF':0.60,'TRUE':0.60,'INSET':0.62,'JMART':0.60
}

# ---------- (Base) Precision tune (รายตัว) ----------
# ==== precision_tune (FINAL, consolidated – no .update calls) ====
# ==== precision_tune (AGGRESSIVE FINAL – single block, no .update) ====
BASE_PRECISION_TUNE = {
    # ---------- US : เร่ง Recall แบบคุม FP ----------
    'AAPL':  {'thr_bump': -0.100, 'ema_alpha': 0.46, 'majk': 1, 'hys': 0.0032,
              'z_gate': 0.95, 'unc_plus': -0.012},
    'AVGO':  {'thr_bump': -0.100, 'ema_alpha': 0.50, 'majk': 1, 'hys': 0.0034},
    'AMZN':  {'thr_bump': -0.085, 'ema_alpha': 0.50, 'majk': 2, 'hys': 0.0040},
    'GOOGL': {'thr_bump': -0.046, 'ema_alpha': 0.52, 'majk': 4, 'hys': 0.0046},
    'META':  {'thr_bump': -0.044, 'ema_alpha': 0.50, 'majk': 2, 'hys': 0.0038},
    'TSLA':  {'thr_bump': -0.040, 'ema_alpha': 0.50, 'majk': 2, 'hys': 0.0040},
    'TSM':   {'thr_bump': -0.028, 'ema_alpha': 0.50, 'majk': 3, 'hys': 0.0042},
    'MSFT':  {'thr_bump': -0.006},
    'AMD':   {'thr_bump': -0.004},
    'NVDA':  {'thr_bump': -0.004},

    # ---------- TH : ปลดล็อก Recall รอบสุดท้าย (ยังคุม FP) ----------
    # เป้า: ADVANC R ~0.18–0.22 / PosRate ~12–16%
    'ADVANC':{'thr_bump': -0.640, 'ema_alpha': 0.16, 'majk': 1, 'hys': 0.0010,
              'z_gate': 0.66, 'unc_plus': -0.30},

    # เป้า: TRUE รักษา P สูง ดัน R ไป ~0.22–0.26 / PosRate ~11–13%
    'TRUE':  {'thr_bump': -0.300, 'ema_alpha': 0.42, 'majk': 1, 'hys': 0.0028,
              'z_gate': 0.78, 'unc_plus': -0.10},

    # ยิงเยอะ → เบรก FP ต่อเนื่อง
    'INSET': {'thr_bump': +0.012, 'ema_alpha': 0.58, 'majk': 9, 'hys': 0.0065},

    # ดันที่ยังอั้น (คุมเสถียรภาพระดับกลาง)
    'JAS':   {'thr_bump': -0.022, 'ema_alpha': 0.54, 'majk': 5, 'hys': 0.0048},
    'JMART': {'thr_bump': -0.100, 'ema_alpha': 0.50, 'majk': 2, 'hys': 0.0038},
    'INET':  {'thr_bump': -0.028, 'ema_alpha': 0.52, 'majk': 4, 'hys': 0.0046},
    'HUMAN': {'thr_bump': -0.022, 'ema_alpha': 0.52, 'majk': 4, 'hys': 0.0046},
    'DITTO': {'thr_bump': -0.060, 'ema_alpha': 0.53, 'majk': 4, 'hys': 0.0043},
    'DIF':   {'thr_bump': -0.060, 'ema_alpha': 0.52, 'majk': 4, 'hys': 0.0042},
}

# ---------- Eval ----------
EVAL_RETHRESH_BALANCED = False
INDIFF_BAND_FOR_EVAL   = 0.0

# ---------- Price target ----------
PRICE_TARGET_MODE = 'logret'
EPS_RET = 0.0011

# ---------- Adapt thresholds online ----------
ADAPT_WIN = 40
ADAPT_CHECK_EVERY = 18
ADAPT_STEP = 0.03
ADAPT_CAP  = 0.18
ADAPT_MARGIN = 1

# ---------- High-vol threshold shifts ----------
HIVOL_THR_SHIFT_US = -0.022
HIVOL_THR_SHIFT_TH = -0.017

# ---------- RAM-lite WFV toggles ----------
MEMORY_LIGHT_WFV            = True
MC_DIR_SAMPLES_WFV          = 3
MC_TRIGGER_BAND             = 0.12
ONLINE_UPDATE_EVERY         = 16
ONLINE_UPDATE_MAX_PER_CHUNK = 48

# ---------- Diagnostics thresholds ----------
MEM_LOW_MB  = 800.0
MEM_CRIT_MB = 400.0

# ---------- Market policies ----------
APPLY_PSC_MARKET          = {'US': True, 'TH': True, 'OTHER': True}
ALLOW_PRICE_ONLINE_MARKET = {'US': True, 'TH': False, 'OTHER': False}

WDIR_CAP_TH = 0.55
WDIR_CAP_US = 0.50
T_DIR_BASE_TH = 0.98
T_DIR_BASE_US = 1.15

# ---------- Hybrid online policy ----------
ONLINE_WEIGHT_UPDATE = {'US': True, 'TH': False, 'OTHER': False}
Z_GATE_ONLINE_US = 1.05
UNC_MAX_US       = 0.10
ONLINE_UPDATE_EVERY_US = 24
ONLINE_UPDATE_MAX_PER_CHUNK_US = 12

# ---------- Auto precision_tune from OVERALL CSV (manual override > auto) ----------
def build_precision_tune_from_overall(path: str,
                                      low_acc: float = 0.70,
                                      fp_bump: float = +0.016,
                                      fn_bump: float = -0.012):
    import os as _os, pandas as _pd, numpy as _np
    if not _os.path.exists(path):
        return None
    df = _pd.read_csv(path)
    df.columns = [c.strip().replace(" ", "_") for c in df.columns]
    required = {"Ticker","Direction_Accuracy","Direction_Precision","Direction_Recall"}
    if not required.issubset(set(df.columns)):
        return None

    def thr_bump_rule(p, r, acc):
        if _pd.isna(p) or _pd.isna(r): return 0.0
        gap = p - r
        if gap > 0.05:   # FN-heavy
            return fn_bump
        if gap < -0.05:  # FP-heavy
            return fp_bump
        if acc < low_acc and p < 0.60:
            return +0.010
        return 0.0

    rec = {}
    for _, row in df.iterrows():
        t   = str(row["Ticker"])
        acc = float(row["Direction_Accuracy"])
        p   = float(row["Direction_Precision"]) if _pd.notna(row["Direction_Precision"]) else _np.nan
        r   = float(row["Direction_Recall"])    if _pd.notna(row["Direction_Recall"])    else _np.nan
        bump = round(thr_bump_rule(p, r, acc), 3)
        ema_alpha = 0.54 if acc < low_acc else 0.50
        hys      = 0.0060 if acc < low_acc else 0.0052
        majk     = int(9 if acc < low_acc else 7)
        unc_plus = 0.010 if acc < low_acc else 0.008
        z_gate   = 0.98 if acc < low_acc else 0.95
        rec[t] = {"thr_bump": bump, "ema_alpha": ema_alpha, "hys": hys, "majk": majk,
                  "unc_plus": unc_plus, "z_gate": z_gate}
    return rec

_auto_ptune = build_precision_tune_from_overall(STREAM_OVERALL_PATH)
# IMPORTANT: ให้ manual (BASE_PRECISION_TUNE) ชนะ auto
PRECISION_TUNE = dict(_auto_ptune) if _auto_ptune else {}
PRECISION_TUNE.update(BASE_PRECISION_TUNE)
# ===================== END CONFIG BLOCK =====================

# ===================== CONSTRAINED THRESHOLD TUNER (drop-in) =====================
# จุดประสงค์: หาค่า threshold ต่อ Ticker โดยบังคับให้ recall และ pred_pos_rate ไม่ต่ำผิดปกติ
# ใช้กับผลทำนายที่มีต่อแถว: Ticker, y_true(dir 0/1), prob_up/meta(0..1)
# จะเซฟไฟล์ผลลัพธ์เป็น JSON ที่ DIR_THR_PATH (merge ทับเฉพาะ tickers ที่จูนได้)

THR_CLIP_LOW   = globals().get('THR_CLIP_LOW', 0.50)
THR_CLIP_HIGH  = globals().get('THR_CLIP_HIGH', 0.86)

_DEFAULT_CONSTRAINTS = dict(
    min_recall=0.30,
    pos_rate_lo=0.25,
    pos_rate_hi=0.45
)

# ฟิลด์ candidate สำหรับ label / prob / ticker (เพิ่มชื่อคอลัมน์ที่เราเขียนใน DAILY_ALL)
_LABEL_CANDS = ['Actual_Dir','y_dir','y_true','label','Direction','dir','target_dir','DirLabel','Actual_Direction']
_PROB_CANDS  = ['Prob_meta_adj','p_meta','p_iso','p_up','p_dir','prob_up','p_long','p_buy','p1','p_pos','p_up_ema','prob','p_use']
_TICK_CANDS  = ['Ticker','ticker','Symbol','symbol','asset','name']

def _first_col(df: pd.DataFrame, names):
    for c in names:
        if c in df.columns:
            return c
    return None

def _coerce_label_to01(s: pd.Series) -> pd.Series:
    x = s.copy()
    if set(pd.unique(x.dropna())) <= set([-1,0,1]):
        x = (x.astype(float) > 0).astype(int)
    else:
        x = (x.astype(float) >= 0.5).astype(int) if x.dropna().between(0,1).all() else x.astype(int)
    return (x > 0).astype(int)

def _coerce_prob_01(s: pd.Series) -> pd.Series:
    x = pd.to_numeric(s, errors='coerce')
    if x.min() < 0 and x.max() > 1 and (x.abs().max() > 5):
        x = 1.0 / (1.0 + np.exp(-x))
    return x.clip(0.0, 1.0)

def _load_predictions() -> tuple[pd.DataFrame, str]:
    for path in [DAILY_ALL_PATH, STREAM_PRED_PATH, PRED_LATEST_PATH]:
        if not os.path.exists(path): 
            continue
        try:
            if path.endswith(".jsonl"):
                rows = []
                with open(path, "r", encoding="utf-8") as f:
                    for line in f:
                        try:
                            rows.append(json.loads(line))
                        except Exception:
                            continue
                df = pd.DataFrame(rows)
            else:
                df = pd.read_csv(path)
        except Exception:
            continue

        if df is None or df.empty: 
            continue

        tcol = _first_col(df, _TICK_CANDS)
        ycol = _first_col(df, _LABEL_CANDS)
        pcol = _first_col(df, _PROB_CANDS)
        if tcol and ycol and pcol:
            df = df[[tcol,ycol,pcol]].rename(columns={tcol:'Ticker', ycol:'y', pcol:'p'})
            df['y'] = _coerce_label_to01(df['y'])
            df['p'] = _coerce_prob_01(df['p'])
            df = df.dropna(subset=['Ticker','y','p'])
            if not df.empty:
                return df, path
    raise FileNotFoundError("ไม่พบไฟล์ผลทำนายที่รองรับ (ลองเช็ค DAILY_ALL_PATH/STREAM_PRED_PATH/PRED_LATEST_PATH).")

def _metrics_at_threshold(y: np.ndarray, p: np.ndarray, thr: float) -> dict:
    pred = (p >= thr).astype(int)
    tp = int(((pred==1)&(y==1)).sum())
    fp = int(((pred==1)&(y==0)).sum())
    tn = int(((pred==0)&(y==0)).sum())
    fn = int(((pred==0)&(y==1)).sum())
    pos = pred.mean() if len(pred) else 0.0

    prec = tp/(tp+fp) if (tp+fp)>0 else 0.0
    rec  = tp/(tp+fn) if (tp+fn)>0 else 0.0
    f1   = (2*prec*rec)/(prec+rec) if (prec+rec)>0 else 0.0
    acc  = (tp+tn)/len(y) if len(y)>0 else 0.0
    tpr  = rec
    tnr  = tn/(tn+fp) if (tn+fp)>0 else 0.0
    bal_acc = 0.5*(tpr+tnr)
    return dict(tp=tp,fp=fp,tn=tn,fn=fn,prec=prec,rec=rec,f1=f1,acc=acc,bal_acc=bal_acc,pos_rate=pos)

def _search_best_threshold(y: np.ndarray,
                           p: np.ndarray,
                           thr_low: float,
                           thr_high: float,
                           step: float,
                           constraints: dict) -> tuple[float, dict]:
    """
    ค้นหาแบบหลายสเตจ:
      S1: recall >= min_recall และ pos_rate อยู่ใน [pos_rate_lo, pos_rate_hi] → maximize bal_acc
      S2: pos_rate เป็น [0.15, 0.55], recall >= min(min_recall, 0.25) → maximize bal_acc
      S3: ตัด pos-rate ออก เหลือ recall >= 0.20 → maximize bal_acc
      S4: เงื่อนไขเบาสุด pos_rate >= 0.05 → maximize F1
      S5: fallback = 0.50
    """
    grid = np.arange(thr_low, thr_high + 1e-9, step)
    cand = []
    for thr in grid:
        m = _metrics_at_threshold(y, p, thr)
        cand.append((thr, m))
    def pick(cond, key='bal_acc'):
        filt = [(thr,m) for (thr,m) in cand if cond(m)]
        if not filt: 
            return None
        best = sorted(filt, key=lambda tm: (tm[1].get(key,0.0), -abs(0.5-tm[0])), reverse=True)[0]
        return best

    s1 = pick(lambda m: (m['rec']>=constraints['min_recall']) and 
                        (constraints['pos_rate_lo']<=m['pos_rate']<=constraints['pos_rate_hi']), 'bal_acc')
    if s1: return s1

    s2 = pick(lambda m: (m['rec']>=min(constraints['min_recall'],0.25)) and 
                        (0.15<=m['pos_rate']<=0.55), 'bal_acc')
    if s2: return s2

    s3 = pick(lambda m: m['rec']>=0.20, 'bal_acc')
    if s3: return s3

    s4 = pick(lambda m: m['pos_rate']>=0.05, 'f1')
    if s4: return s4

    m50 = _metrics_at_threshold(y, p, 0.50)
    return 0.50, m50

def tune_dir_thresholds(step: float = 0.01,
                        constraints: dict = None,
                        extra_loosen: bool = True,
                        save_path: str = None) -> pd.DataFrame:
    """
    คืน DataFrame สรุปผลต่อ Ticker และบันทึกไฟล์ JSON thresholds
    parameters:
      - step: ระยะก้าว threshold
      - constraints: dict(min_recall, pos_rate_lo, pos_rate_hi)
      - extra_loosen: ถ้า pos_rate ทั้งเดือดหรือหด จะขยายกรอบ auto
      - save_path: path ไฟล์ json (default = DIR_THR_PATH)
    """
    df, used_path = _load_predictions()
    if constraints is None:
        constraints = _DEFAULT_CONSTRAINTS.copy()

    # ปรับ constraints อัตโนมัติเล็กน้อยตามสภาวะ
    if extra_loosen:
        pr_all = (df.groupby('Ticker')['p']
                    .apply(lambda s: (s>=0.5).mean()))
        if pr_all.mean() < 0.20:
            constraints['pos_rate_lo'] = max(0.18, constraints['pos_rate_lo']-0.05)
            constraints['pos_rate_hi'] = min(0.55, constraints['pos_rate_hi']+0.05)

    if save_path is None: 
        save_path = DIR_THR_PATH

    old = {}
    if os.path.exists(save_path):
        try:
            with open(save_path, "r", encoding="utf-8") as f:
                old = json.load(f)
        except Exception:
            old = {}

    rows = []
    new_thr = {}

    for ticker, g in df.groupby('Ticker'):
        y = g['y'].astype(int).values
        p = g['p'].astype(float).values

        if (g['y'].sum()==0) or (g['y'].sum()==len(g)):
            best_thr, best_m = 0.50, _metrics_at_threshold(y, p, 0.50)
        else:
            best_thr, best_m = _search_best_threshold(
                y, p,
                thr_low = THR_CLIP_LOW,
                thr_high= THR_CLIP_HIGH,
                step = step,
                constraints = constraints
            )

        new_thr[ticker] = round(float(best_thr), 3)
        rows.append({
            'Ticker': ticker,
            'BestThr': round(float(best_thr), 3),
            'PosRate@Best': round(best_m['pos_rate'], 3),
            'Recall@Best':  round(best_m['rec'], 3),
            'Prec@Best':    round(best_m['prec'], 3),
            'F1@Best':      round(best_m['f1'], 3),
            'Acc@Best':     round(best_m['acc'], 3),
            'BalAcc@Best':  round(best_m['bal_acc'], 3),
            'TP': best_m['tp'], 'FP': best_m['fp'], 'TN': best_m['tn'], 'FN': best_m['fn'],
        })

    merged = dict(old)
    merged.update(new_thr)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)

    report = pd.DataFrame(rows).sort_values(['BalAcc@Best','F1@Best','Acc@Best'], ascending=False)
    print(f"[threshold_tuner] loaded preds from: {used_path}")
    print(f"[threshold_tuner] thresholds saved to: {save_path}")
    return report
# ===================== END THRESHOLD TUNER BLOCK =====================


# ===================== TF/Keras =====================
import tensorflow as tf
tf.random.set_seed(SEED)
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, Embedding, Bidirectional, concatenate
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger, Callback
from tensorflow.keras.optimizers.schedules import CosineDecay
try:
    from tensorflow.keras.optimizers import AdamW
except Exception:
    from tensorflow_addons.optimizers import AdamW

from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.metrics import (accuracy_score, f1_score, precision_score, recall_score,
                             matthews_corrcoef, balanced_accuracy_score, fbeta_score)
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from collections import deque

try:
    from keras.saving import register_keras_serializable
except Exception:
    from tensorflow.keras.utils import register_keras_serializable

try:
    import psutil
except Exception:
    psutil = None

# ==== Safety/Utils ====
import numpy as _np, math as _math

def sigmoid_np(x): return 1.0 / (1.0 + _np.exp(-x))
def logit(p, eps=1e-6):
    pp = _np.clip(p, eps, 1.0 - eps)
    return _np.log(pp / (1.0 - pp))

# ===================== Diagnostics helpers =====================
def diag_log(msg: str):
    ts = time.strftime('%Y-%m-%d %H:%M:%S')
    line = f"[{ts}] {msg}"
    print(line)
    try:
        with open(DIAG_LOG_PATH, 'a', encoding='utf-8') as f:
            f.write(line + "\n")
    except Exception:
        pass

def get_free_ram_mb() -> float:
    try:
        if psutil is None:
            return float('inf')
        return float(psutil.virtual_memory().available) / 1e6
    except Exception:
        return float('inf')

def save_progress(**kwargs):
    rec = dict(ts=time.time(), **kwargs)
    try:
        with open(PROGRESS_PATH, 'a', encoding='utf-8') as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except Exception:
        pass

# ===================== Loss/Utils =====================
@register_keras_serializable(package="custom")
def gaussian_nll(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    mu_s, log_sigma_s = tf.split(y_pred, 2, axis=-1)
    sigma_s = tf.nn.softplus(log_sigma_s) + 1e-6
    z = (y_true - mu_s) / sigma_s
    return tf.reduce_mean(0.5*tf.math.log(2.0*np.pi) + tf.math.log(sigma_s) + 0.5*tf.square(z))

@register_keras_serializable(package="custom")
def mae_on_mu(y_true, y_pred):
    mu_s, _ = tf.split(y_pred, 2, axis=-1)
    return tf.reduce_mean(tf.abs(tf.cast(y_true, tf.float32) - mu_s))

def sanitize(arr):
    arr = np.asarray(arr, dtype=np.float32); mask = np.isfinite(arr)
    if not mask.all():
        med = np.nanmedian(arr[mask]); arr[~mask] = med
    arr[np.isnan(arr)] = np.nanmedian(arr[np.isfinite(arr)])
    return arr.astype(np.float32, copy=False)

def softplus_np(x): return np.log1p(np.exp(x))
def norm_cdf(x): return 0.5*(1.0 + math.erf(x / math.sqrt(2.0)))

def mu_sigma_to_raw(mu_s, log_sigma_s, ps):
    sigma_s = softplus_np(log_sigma_s) + 1e-6
    scale  = getattr(ps, 'scale_',  np.array([1.0], dtype=np.float32))[0]
    center = getattr(ps, 'center_', np.array([0.0], dtype=np.float32))[0]
    mu_raw = float(mu_s) * scale + center
    sigma_raw = float(sigma_s) * scale
    return float(mu_raw), float(sigma_raw)

def mu_sigma_to_pup(mu_raw: float, sigma_raw: float, eps=EPS_RET):
    if sigma_raw <= 1e-9:
        return (1.0 if (mu_raw - eps) > 0 else 0.0), 0.0
    zz = (mu_raw - eps) / sigma_raw
    return norm_cdf(zz), zz

# ===================== Data =====================
DATA_PATH = '../Preproces/data/Stock/merged_stock_sentiment_financial.csv'
df = pd.read_csv(DATA_PATH).sort_values(['Ticker','Date']).reset_index(drop=True)

# Sentiment fallback
if 'Sentiment' in df.columns:
    df['Sentiment'] = df['Sentiment'].map({'Positive':1,'Negative':-1,'Neutral':0}).fillna(0).astype(np.int8)
else:
    df['Sentiment'] = 0

df['Change']    = df['Close'] - df['Open']
df['Change (%)']= df.groupby('Ticker')['Close'].pct_change()*100.0
upper = df['Change (%)'].quantile(0.99); lower = df['Change (%)'].quantile(0.01)
df['Change (%)']= df['Change (%)'].clip(lower, upper)

# ===== Technicals =====
import ta
def add_ta(g):
    g=g.copy()
    g['EMA_12']=g['Close'].ewm(span=12, adjust=False).mean()
    g['EMA_26']=g['Close'].ewm(span=26, adjust=False).mean()
    g['EMA_10']=g['Close'].ewm(span=10, adjust=False).mean()
    g['EMA_20']=g['Close'].ewm(span=20, adjust=False).mean()
    g['SMA_50']=g['Close'].rolling(50, min_periods=1).mean()
    g['SMA_200']=g['Close'].rolling(200, min_periods=1).mean()
    try: g['RSI']=ta.momentum.RSIIndicator(close=g['Close'], window=14).rsi()
    except: g['RSI']=np.nan
    g['RSI']=g['RSI'].fillna(g['RSI'].rolling(5, min_periods=1).mean()).fillna(50.0)
    g['MACD']=g['EMA_12']-g['EMA_26']
    g['MACD_Signal']=g['MACD'].rolling(9, min_periods=1).mean()
    try:
        bb=ta.volatility.BollingerBands(close=g['Close'], window=20, window_dev=2)
        g['Bollinger_High']=bb.bollinger_hband(); g['Bollinger_Low']=bb.bollinger_lband()
    except:
        g['Bollinger_High']=g['Close'].rolling(20, min_periods=1).max()
        g['Bollinger_Low']=g['Close'].rolling(20, min_periods=1).min()
    try:
        atr=ta.volatility.AverageTrueRange(high=g['High'], low=g['Low'], close=g['Close'], window=14)
        g['ATR']=atr.average_true_rate() if hasattr(atr,'average_true_rate') else atr.average_true_range()
    except:
        g['ATR']=(g['High']-g['Low']).rolling(14, min_periods=1).mean()
    try:
        kc=ta.volatility.KeltnerChannel(high=g['High'], low=g['Low'], close=g['Close'], window=20, window_atr=10)
        g['Keltner_High']=kc.keltner_channel_hband()
        g['Keltner_Low']=kc.keltner_channel_lband()
        g['Keltner_Middle']=kc.keltner_channel_mband()
    except:
        rng=(g['High']-g['Low']).rolling(20, min_periods=1).mean()
        mid=g['Close'].rolling(20, min_periods=1).mean()
        g['Keltner_High']=mid+rng; g['Keltner_Low']=mid-rng; g['Keltner_Middle']=mid
    g['High_Low_Diff']=g['High']-g['Low']
    g['High_Low_EMA']=g['High_Low_Diff'].ewm(span=10, adjust=False).mean()
    g['Chaikin_Vol']=g['High_Low_EMA'].pct_change(10)*100.0
    g['Donchian_High']=g['High'].rolling(20, min_periods=1).max()
    g['Donchian_Low']=g['Low'].rolling(20, min_periods=1).min()
    try: g['PSAR']=ta.trend.PSARIndicator(high=g['High'], low=g['Low'], close=g['Close'], step=0.02, max_step=0.2).psar()
    except: g['PSAR']=(g['High']+g['Low'])/2.0
    return g

# ป้องกัน pandas ลบคอลัมน์ Ticker
_tickers_backup = df['Ticker'].values
try:
    df = df.groupby('Ticker', group_keys=False).apply(add_ta, include_groups=True)
except TypeError:
    df = df.groupby('Ticker', group_keys=False).apply(add_ta)
    if 'Ticker' not in df.columns and len(df) == len(_tickers_backup):
        df.insert(0, 'Ticker', _tickers_backup)
if 'Ticker' not in df.columns:
    raise RuntimeError("หลัง apply(add_ta) ไม่พบคอลัมน์ 'Ticker'")

us_stock  = ['AAPL','NVDA','MSFT','AMZN','GOOGL','META','TSLA','AVGO','TSM','AMD']
thai_stock= ['ADVANC','TRUE','DITTO','DIF','INSET','JMART','INET','JAS','HUMAN']
df['Market_ID']=np.where(df['Ticker'].isin(us_stock),'US',np.where(df['Ticker'].isin(thai_stock),'TH','OTHER')).astype(str)

financial_columns=['Total Revenue','QoQ Growth (%)','Earnings Per Share (EPS)','ROE (%)',
                   'Net Profit Margin (%)','Debt to Equity','P/E Ratio','P/BV Ratio','Dividend Yield (%)']
for c in financial_columns:
    if c not in df.columns: df[c]=np.nan
df[financial_columns]=df[financial_columns].replace(0,np.nan)
df[financial_columns]=df.groupby('Ticker')[financial_columns].ffill()

feature_columns=[
    'Open','High','Low','Close','Volume','Change (%)','Sentiment',
    'positive_news','negative_news','neutral_news',
    'Total Revenue','QoQ Growth (%)','Earnings Per Share (EPS)','ROE (%)',
    'ATR','Keltner_High','Keltner_Low','Keltner_Middle','Chaikin_Vol',
    'Donchian_High','Donchian_Low','PSAR',
    'Net Profit Margin (%)','Debt to Equity','P/E Ratio','P/BV Ratio','Dividend Yield (%)',
    'RSI','EMA_10','EMA_20','MACD','MACD_Signal',
    'Bollinger_High','Bollinger_Low','SMA_50','SMA_200'
]
for c in feature_columns:
    if c not in df.columns: df[c]=0.0
df[feature_columns]=(df.groupby('Ticker')[feature_columns]
                       .apply(lambda g: g.fillna(method='ffill'))
                       .reset_index(level=0, drop=True))
df[feature_columns]=df[feature_columns].fillna(0.0)

df['TargetPrice']=df.groupby('Ticker')['Close'].shift(-1)
df=df.dropna(subset=['TargetPrice']).reset_index(drop=True)
df['DirLabel']=(df['TargetPrice']>df['Close']).astype(np.int8)

# ===================== Encoders & Split =====================
from sklearn.preprocessing import LabelEncoder as _LE
market_encoder=_LE(); df['Market_ID_enc']=market_encoder.fit_transform(df['Market_ID'])
num_markets=len(market_encoder.classes_); joblib.dump(market_encoder, os.path.join(MODEL_DIR,'market_encoder.pkl'))
ticker_encoder=_LE(); df['Ticker_ID']=ticker_encoder.fit_transform(df['Ticker'])
num_tickers=len(ticker_encoder.classes_); joblib.dump(ticker_encoder, os.path.join(MODEL_DIR,'ticker_encoder.pkl'))

sorted_dates=df['Date'].unique()
train_cutoff=sorted_dates[int(len(sorted_dates)*6/7)]
train_df=df[df['Date']<=train_cutoff].copy()
test_df =df[df['Date'] >train_cutoff].copy()
train_df.to_csv(os.path.join(LOG_DIR,'train_df.csv'), index=False); test_df.to_csv(os.path.join(LOG_DIR,'test_df.csv'), index=False)

# ===================== Scaling =====================
SEQ_LEN=int(BEST_PARAMS['seq_length'])
train_df['PriceTargetRaw']=np.log(train_df['TargetPrice']/train_df['Close']).astype(np.float32)
test_df ['PriceTargetRaw']=np.log(test_df ['TargetPrice']/test_df ['Close']).astype(np.float32)

train_features=train_df[feature_columns].values.astype(np.float32)
test_features =test_df [feature_columns].values.astype(np.float32)
train_price_t=train_df['PriceTargetRaw'].values.reshape(-1,1).astype(np.float32)
test_price_t =test_df ['PriceTargetRaw'].values.reshape(-1,1).astype(np.float32)
train_dir_lbl=train_df['DirLabel'].values.astype(np.int8)
test_dir_lbl =test_df ['DirLabel'].values.astype(np.int8)
train_ticker_id=train_df['Ticker_ID'].values
test_ticker_id =test_df ['Ticker_ID'].values

train_features=sanitize(train_features); test_features=sanitize(test_features)
train_price_t=sanitize(train_price_t); test_price_t=sanitize(test_price_t)

train_features_scaled=np.zeros_like(train_features,dtype=np.float32)
test_features_scaled =np.zeros_like(test_features ,dtype=np.float32)
train_price_scaled   =np.zeros_like(train_price_t ,dtype=np.float32)
test_price_scaled    =np.zeros_like(test_price_t  ,dtype=np.float32)

ticker_scalers={}; id2ticker={}
for t_id in np.unique(train_ticker_id):
    gmask=(train_ticker_id==t_id)
    X_part=train_features[gmask]; y_part=train_price_t[gmask]
    fs=RobustScaler(); ps=RobustScaler()
    Xs=fs.fit_transform(X_part).astype(np.float32)
    ys=ps.fit_transform(y_part).astype(np.float32)
    train_features_scaled[gmask]=Xs; train_price_scaled[gmask]=ys
    tname=train_df.loc[gmask,'Ticker'].iloc[0]
    id2ticker[t_id]=tname; ticker_scalers[t_id]={'feature_scaler':fs,'price_scaler':ps,'ticker':tname}
    del X_part,y_part,Xs,ys; gc.collect()

for t_id in np.unique(test_ticker_id):
    if t_id not in ticker_scalers: continue
    gmask=(test_ticker_id==t_id)
    fs=ticker_scalers[t_id]['feature_scaler']; ps=ticker_scalers[t_id]['price_scaler']
    test_features_scaled[gmask]=fs.transform(test_features[gmask]).astype(np.float32)
    test_price_scaled[gmask]=ps.transform(test_price_t[gmask]).astype(np.float32)

joblib.dump(ticker_scalers, os.path.join(MODEL_DIR,'ticker_scalers.pkl')); joblib.dump(feature_columns, os.path.join(MODEL_DIR,'feature_columns.pkl'))

# ===================== Sequences =====================
def create_sequences_for_ticker(features, ticker_ids, market_ids, targets_price, dir_labels, seq_length=SEQ_LEN):
    Xf,Xt,Xm,Yp,Yd=[],[],[],[],[]
    for i in range(len(features)-seq_length):
        Xf.append(features[i:i+seq_length])
        Xt.append(ticker_ids[i:i+seq_length])
        Xm.append(market_ids[i:i+seq_length])
        Yp.append(targets_price[i+seq_length])
        Yd.append(dir_labels[i+seq_length])
    return (np.array(Xf,np.float32),
            np.array(Xt,np.int32),
            np.array(Xm,np.int32),
            np.array(Yp,np.float32),
            np.array(Yd,np.int8))

def build_dataset_sequences(base_df, features_scaled, price_scaled, dir_labels, seq_length=SEQ_LEN):
    Xf_list,Xt_list,Xm_list,Yp_list,Yd_list=[],[],[],[],[]
    for t_id in range(num_tickers):
        idx = base_df.index[base_df['Ticker_ID']==t_id].tolist()
        if len(idx)<=seq_length: continue
        mask=np.isin(base_df.index, idx)
        f=features_scaled[mask]; p=price_scaled[mask]; d=dir_labels[mask]
        t=base_df.loc[mask,'Ticker_ID'].values.astype(np.int32)
        m=base_df.loc[mask,'Market_ID_enc'].values.astype(np.int32)
        Xf,Xt,Xm,Yp,Yd=create_sequences_for_ticker(f,t,m,p,d,seq_length)
        if len(Xf): Xf_list.append(Xf); Xt_list.append(Xt); Xm_list.append(Xm); Yp_list.append(Yp); Yd_list.append(Yd)
        del f,p,d,t,m,Xf,Xt,Xm,Yp,Yd; gc.collect()
    if len(Xf_list)==0:
        zf=np.zeros((0,seq_length,len(feature_columns)),np.float32)
        zi=np.zeros((0,seq_length),np.int32)
        zp=np.zeros((0,1),np.float32)
        zd=np.zeros((0,),np.int8)
        return zf,zi,zi,zp,zd
    Xf=np.concatenate(Xf_list,0); Xt=np.concatenate(Xt_list,0); Xm=np.concatenate(Xm_list,0)
    Yp=np.concatenate(Yp_list,0); Yd=np.concatenate(Yd_list,0)
    del Xf_list,Xt_list,Xm_list,Yp_list,Yd_list; gc.collect()
    return Xf,Xt,Xm,Yp,Yd

X_price_train,X_ticker_train,X_market_train,y_price_train,y_dir_train=build_dataset_sequences(
    train_df,train_features_scaled,train_price_scaled,train_dir_lbl,SEQ_LEN)
X_price_test,X_ticker_test,X_market_test,y_price_test,y_dir_test=build_dataset_sequences(
    test_df,test_features_scaled,test_price_scaled,test_dir_lbl,SEQ_LEN)

num_feature=len(feature_columns)

# ===================== Model (Price-only) =====================
features_input=Input(shape=(SEQ_LEN,num_feature),name='features_input')
ticker_input  =Input(shape=(SEQ_LEN,),name='ticker_input')
market_input  =Input(shape=(SEQ_LEN,),name='market_input')

embedding_dim_ticker=int(BEST_PARAMS['embedding_dim'])
embedding_dim_market=8

tick_emb=Embedding(num_tickers,embedding_dim_ticker,
                   embeddings_regularizer=tf.keras.regularizers.l2(1e-7),
                   name="ticker_embedding")(ticker_input)
tick_emb=Dense(16,activation="relu")(tick_emb)

mkt_emb=Embedding(num_markets,embedding_dim_market,
                  embeddings_regularizer=tf.keras.regularizers.l2(1e-7),
                  name="market_embedding")(market_input)
mkt_emb=Dense(8,activation="relu")(mkt_emb)

merged=concatenate([features_input,tick_emb,mkt_emb],axis=-1)
x=Bidirectional(LSTM(int(BEST_PARAMS['LSTM_units_1']),return_sequences=True,
                     dropout=float(BEST_PARAMS['dropout_rate'])))(merged)
x=Dropout(float(BEST_PARAMS['dropout_rate']))(x)
x=Bidirectional(LSTM(int(BEST_PARAMS['LSTM_units_2']),return_sequences=False,
                     dropout=float(BEST_PARAMS['dropout_rate'])))(x)
x=Dropout(float(BEST_PARAMS['dropout_rate']))(x)
shared=Dense(int(BEST_PARAMS['dense_units']),activation="relu",
             kernel_regularizer=tf.keras.regularizers.l2(1e-6))(x)

price_head=Dense(32,activation="relu",kernel_regularizer=tf.keras.regularizers.l2(1e-6))(shared)
price_head=Dropout(0.22)(price_head)
price_params=Dense(2,name="price_params")(price_head)  # [μ, log σ]

model=Model(inputs=[features_input,ticker_input,market_input],outputs=[price_params])

BATCH_SIZE=33; VAL_SPLIT=0.12; EPOCHS=200
steps_per_epoch=max(1,int(len(X_price_train)*(1-VAL_SPLIT))//BATCH_SIZE)
decay_steps=max(1,int(steps_per_epoch*EPOCHS*1.2))
lr_schedule=CosineDecay(initial_learning_rate=float(BEST_PARAMS['learning_rate']),
                        decay_steps=decay_steps,alpha=1e-5)
optimizer=AdamW(learning_rate=lr_schedule,weight_decay=1.2e-5,clipnorm=1.0)

model.compile(
    optimizer=optimizer,
    loss={"price_params":gaussian_nll},
    loss_weights={"price_params":1.0},
    metrics={"price_params":[mae_on_mu]}
)

class StabilityCallback(Callback):
    def __init__(self): super().__init__(); self.best=float('inf')
    def on_epoch_end(self, epoch, logs=None):
        val_mae=logs.get('val_price_params_mae_on_mu',0.0)
        if val_mae<self.best:
            self.best=val_mae; print(f"  🎯 New best val MAE(μ): {val_mae:.4f}")

early_stopping=EarlyStopping(monitor="val_loss",patience=25,restore_best_weights=True,
                             verbose=1,min_delta=1e-4,start_from_epoch=12)

# ===================== Train or Load =====================
custom_objs = {"gaussian_nll": gaussian_nll, "mae_on_mu": mae_on_mu}

def _load_model(path):
    return tf.keras.models.load_model(path, custom_objects=custom_objs, safe_mode=False)

def _load_best_model_or_fail():
    try:
        m = _load_model(BEST_MODEL_PATH_STATIC if os.path.exists(BEST_MODEL_PATH_STATIC) else BEST_MODEL_PATH)
        print(f"✅ Loaded existing model from {BEST_MODEL_PATH_STATIC if os.path.exists(BEST_MODEL_PATH_STATIC) else BEST_MODEL_PATH}")
        return m
    except Exception as e:
        msg = f"❌ ไม่พบ/โหลดโมเดลไม่ได้: {e}"
        if STRICT_LOAD:
            raise FileNotFoundError(msg)
        print("⚠️", msg, " → จะใช้โมเดลที่เพิ่ง build แทน")
        return model

if SKIP_TRAIN:
    best_model = _load_best_model_or_fail()
else:
    checkpoint=ModelCheckpoint(BEST_MODEL_PATH_STATIC,monitor="val_loss",save_best_only=True,mode="min",verbose=1)
    csv_logger=CSVLogger(os.path.join(LOG_DIR,'training_log.csv'))
    callbacks=[early_stopping,checkpoint,csv_logger,StabilityCallback()]
    history=model.fit(
        [X_price_train,X_ticker_train,X_market_train],
        {"price_params":y_price_train},
        epochs=EPOCHS,batch_size=BATCH_SIZE,verbose=1,shuffle=False,
        validation_split=VAL_SPLIT,callbacks=callbacks
    )
    pd.DataFrame(history.history).to_csv(os.path.join(LOG_DIR,'training_history.csv'),index=False)
    try:
        best_model=_load_model(BEST_MODEL_PATH_STATIC)
        print("✅ Loaded best model.")
    except Exception as e:
        print("⚠️ Could not load best model:",e); best_model=model

# ===================== Calibration (Val) =====================
def _load_calibrators_from_disk():
    def _maybe(path, loader):
        return loader(path) if os.path.exists(path) else None
    # ใช้ PATHS จาก CONFIG ก่อน ถ้าไม่มี fallback เป็นชื่อไฟล์เดิมใน cwd
    iso   = _maybe(ISO_CAL_PATH, joblib.load) or _maybe('iso_calibrators_per_ticker.pkl', joblib.load) or {}
    meta  = _maybe(META_LR_PATH, joblib.load) or _maybe('meta_lr_per_ticker.pkl', joblib.load) or {}
    try:
        with open(DIR_THR_PATH, 'r', encoding='utf-8') as f:
            thr = json.load(f)
    except Exception:
        try:
            with open('dir_thresholds_per_ticker.json', 'r', encoding='utf-8') as f:
                thr = json.load(f)
        except Exception:
            thr = {}
    try:
        with open(VAL_PREV_MAP_PATH, 'r', encoding='utf-8') as f:
            vprev = json.load(f)
    except Exception:
        try:
            with open('val_prev_map.json', 'r', encoding='utf-8') as f:
                vprev = json.load(f)
        except Exception:
            vprev = {}
    try:
        with open(DIR_WEIGHT_OVR_PATH, 'r', encoding='utf-8') as f:
            wdir = json.load(f)
    except Exception:
        try:
            with open('dir_weight_ovr.json', 'r', encoding='utf-8') as f:
                wdir = json.load(f)
        except Exception:
            wdir = {}
    try:
        with open(DIR_TEMPERATURE_PATH, 'r', encoding='utf-8') as f:
            tdir = json.load(f)
    except Exception:
        try:
            with open('dir_temperature_per_ticker.json', 'r', encoding='utf-8') as f:
                tdir = json.load(f)
        except Exception:
            tdir = {}
    return iso, meta, thr, vprev, wdir, tdir

# ฟังก์ชันสร้าง X_meta ให้เข้ากับจำนวนฟีเจอร์ของ LR (รองรับทั้ง 5 และ 7)
def build_xmeta_for_lr(p_iso, zz, sigma_raw, mu_raw, p_unc, pdir_T=None, p_unc_dir=None, lr=None):
    want = getattr(lr, 'n_features_in_', 5) if lr is not None else 5
    if want >= 7:
        if pdir_T is None: pdir_T = p_iso
        if p_unc_dir is None: p_unc_dir = p_unc
        return np.array([[p_iso, zz, sigma_raw, mu_raw, pdir_T, p_unc, p_unc_dir]], np.float32)
    else:
        return np.array([[p_iso, zz, sigma_raw, mu_raw, p_unc]], np.float32)

if SKIP_CALIBRATION:
    iso_cals, meta_lrs, thresholds, val_prev_map, dir_weight_ovr, dir_temperature_ovr = _load_calibrators_from_disk()
    iso_cals = iso_cals or {}
    meta_lrs = meta_lrs or {}
    thresholds = thresholds or {}
    val_prev_map = {int(k): float(v) for k, v in (val_prev_map or {}).items()}
    dir_weight_ovr = {int(k): float(v) for k, v in (dir_weight_ovr or {}).items()}
    dir_temperature_ovr = {int(k): float(v) for k, v in (dir_temperature_ovr or {}).items()}
    if STRICT_LOAD:
        missing = []
        for name, obj in [('iso_cals',iso_cals),('meta_lrs',meta_lrs),('thresholds',thresholds),('val_prev_map',val_prev_map)]:
            if not obj: missing.append(name)
        if missing:
            raise FileNotFoundError(f"SKIP_CALIBRATION=True แต่ไม่พบ artifacts: {missing}. โปรดรันคาลิเบรตสักครั้งก่อน")
    print("✅ Loaded calibrators from disk.")
else:
    # สร้างชุด validation
    n_total = len(X_price_train)
    n_val = int(np.ceil(n_total * VAL_SPLIT))
    val_slice = slice(n_total - n_val, n_total)
    Xf_val, Xt_val, Xm_val = X_price_train[val_slice], X_ticker_train[val_slice], X_market_train[val_slice]
    y_dir_true_val = y_dir_train[val_slice].reshape(-1).astype(np.int8)

    # predict price params
    pred_val = best_model.predict([Xf_val, Xt_val, Xm_val], verbose=0)
    y_price_val = np.asarray(pred_val)
    if y_price_val.ndim == 1:
        if y_price_val.size % 2 != 0:
            raise ValueError(f"Expected even length for [mu, log_sigma], got shape {y_price_val.shape}")
        y_price_val = y_price_val.reshape(-1, 2)

    tkr_val_last = Xt_val[:, -1]
    mkt_val_last = Xm_val[:, -1]
    Nval = len(Xf_val)

    p_up_raw = np.zeros((Nval,), np.float32)
    z_val     = np.zeros((Nval,), np.float32)
    sg_val    = np.zeros((Nval,), np.float32)
    mu_val    = np.zeros((Nval,), np.float32)
    p_unc_val = np.zeros((Nval,), np.float32)

    def best_threshold_constrained(y_true, prob, metric='acc', beta=1.7,
                                   min_recall=0.60, min_precision=0.0, anchor=None, radius=0.30):
        ths = np.linspace(max(0.05, (anchor or 0.5)-radius), min(0.95, (anchor or 0.5)+radius), 81) if anchor is not None else np.linspace(0.10,0.90,81)
        best_score, best_th = -1.0, 0.5
        for th in ths:
            yhat = (prob >= th).astype(int)
            rec  = recall_score(y_true, yhat, zero_division=0)
            prec = precision_score(y_true, yhat, zero_division=0)
            bal  = balanced_accuracy_score(y_true, yhat)
            f1   = f1_score(y_true, yhat, zero_division=0)
            fbeta = fbeta_score(y_true, yhat, beta=beta, zero_division=0)
            acc  = accuracy_score(y_true, yhat)
            if   metric == 'acc':   val = acc
            elif metric == 'fbeta': val = fbeta
            elif metric == 'f1':    val = f1
            else:                   val = bal
            if (rec >= min_recall) and (prec >= min_precision) and (val > best_score):
                best_th, best_score = th, val
        if best_score < 0:
            for th in ths:
                yhat = (prob >= th).astype(int)
                f1   = f1_score(y_true, yhat, zero_division=0)
                fbeta = fbeta_score(y_true, yhat, beta=beta, zero_division=0)
                bal  = balanced_accuracy_score(y_true, yhat)
                acc  = accuracy_score(y_true, yhat)
                if   metric == 'acc':   val = acc
                elif metric == 'fbeta': val = fbeta
                elif metric == 'f1':    val = f1
                else:                   val = bal
                if val > best_score:
                    best_score, best_th = val, th
        return best_th, best_score

    # คำนวณ base + MC uncertainty บน val (หัวเดียว)
    for i in range(Nval):
        t_id = int(tkr_val_last[i])
        ps = ticker_scalers[t_id]['price_scaler']

        mu_s, log_sigma_s = float(y_price_val[i, 0]), float(y_price_val[i, 1])
        mu_raw, sigma_raw = mu_sigma_to_raw(mu_s, log_sigma_s, ps)
        mu_val[i] = mu_raw
        sg_val[i] = sigma_raw

        p_up, zz = mu_sigma_to_pup(mu_raw, sigma_raw)
        p_up_raw[i] = p_up
        z_val[i] = zz

        Xf_i = Xf_val[i:i+1]; Xt_i = Xt_val[i:i+1]; Xm_i = Xm_val[i:i+1]
        if MC_DIR_SAMPLES_WFV > 0:
            pups = []
            for _ in range(MC_DIR_SAMPLES_WFV):
                y_price2 = best_model([Xf_i, Xt_i, Xm_i], training=True)
                y_price2 = np.asarray(y_price2).reshape(-1,2)
                mu_s2, log_sigma_s2 = float(y_price2[0,0]), float(y_price2[0,1])
                sigma_s2 = max(softplus_np(log_sigma_s2) + 1e-6, 1e-6)
                scale  = getattr(ps, "scale_",  np.array([1.0], dtype=np.float32))[0]
                center = getattr(ps, "center_", np.array([0.0], dtype=np.float32))[0]
                mu_raw2 = mu_s2 * scale + center
                p_up2 = norm_cdf((mu_raw2 - EPS_RET) / (sigma_s2 * scale))
                pups.append(p_up2)
            p_unc_val[i] = float(np.std(np.asarray(pups, np.float32), ddof=0))
        else:
            p_unc_val[i] = 0.0

    # ===== Per-ticker: calibrators + thresholds =====
    iso_cals = {}
    meta_lrs = {}
    thresholds = {}
    val_prev_map = {}
    dir_weight_ovr = {}
    dir_temperature_ovr = {}

    temp_grid = np.arange(1.0, 1.01, 0.1)   # ไม่ใช้หัว dir โดยตรง → fix 1.0
    w_grid    = np.arange(0.0, 0.01, 1.0)   # ไม่ blend อะไร → 0.0

    for t in np.unique(tkr_val_last):
        idx = (tkr_val_last == t)
        if idx.sum() < 30 or len(np.unique(y_dir_true_val[idx])) < 2:
            iso_cals[int(t)] = None
            meta_lrs[int(t)] = None
            thresholds[str(int(t))] = 0.5
            val_prev_map[int(t)] = 0.5
            dir_weight_ovr[int(t)] = 0.0
            dir_temperature_ovr[int(t)] = 1.0
            continue

        y_meta_true = y_dir_true_val[idx]
        z_arr   = z_val[idx]
        sg_arr  = sg_val[idx]
        mu_arr  = mu_val[idx]
        pup_arr = np.clip(p_up_raw[idx], 0.02, 0.98)
        punc_arr = p_unc_val[idx]

        tkr_name = id2ticker.get(int(t), str(int(t)))
        mkt_id   = int(mkt_val_last[idx][0]) if idx.sum() > 0 else 0
        mkt_name = market_encoder.inverse_transform([mkt_id])[0] if mkt_id in range(len(market_encoder.classes_)) else 'OTHER'

        min_rec = MIN_RECALL_OVR.get(tkr_name, MIN_RECALL)
        if mkt_name == 'TH':
            min_rec     = min(0.95, max(min_rec, MIN_RECALL + MIN_RECALL_TH_MARKET_BONUS))
            metric_name = THR_OBJECTIVE_TH
            beta_use    = FBETA_TH
        else:
            metric_name = THR_OBJECTIVE_US
            beta_use    = 1.0

        best_acc = -1.0
        best_iso = None
        best_lr  = None
        best_th  = 0.5

        for T in temp_grid:
            pdir_T = pup_arr  # ใช้เท่ากับ p_up (หัวเดียว)
            for _w in w_grid:
                p_mix0 = pup_arr

                # Isotonic calibration
                try:
                    iso = IsotonicRegression(out_of_bounds='clip').fit(p_mix0, y_meta_true)
                    p_iso = iso.transform(p_mix0)
                except Exception:
                    iso = None
                    p_iso = p_mix0

                # Meta logistic (5 features)
                X_meta = np.column_stack([p_iso, z_arr, sg_arr, mu_arr, punc_arr]).astype(np.float32)
                try:
                    lr = LogisticRegression(solver='liblinear', class_weight='balanced', max_iter=700, C=0.7, penalty='l2')
                    lr.fit(X_meta, y_meta_true)
                    p_meta = lr.predict_proba(X_meta)[:, 1]
                except Exception:
                    lr = None
                    p_meta = p_iso

                prevalence = float(np.mean(y_meta_true))
                anchor = float(np.quantile(p_meta, 1.0 - prevalence))
                th, _ = best_threshold_constrained(
                    y_meta_true, p_meta, metric=metric_name, beta=beta_use,
                    min_recall=min_rec, min_precision=MIN_PRECISION,
                    anchor=anchor, radius=ANCHOR_RADIUS
                )
                yhat = (p_meta >= th).astype(int)
                acc = float(accuracy_score(y_meta_true, yhat))
                if acc > best_acc:
                    best_acc = acc
                    best_iso = iso
                    best_lr  = lr
                    best_th  = float(np.clip(max(THRESH_MIN, th), THR_CLIP_LOW, THR_CLIP_HIGH))

        prevalence = float(np.mean(y_meta_true))
        val_prev_map[int(t)] = prevalence
        iso_cals[int(t)] = best_iso
        meta_lrs[int(t)] = best_lr
        thresholds[str(int(t))] = best_th
        dir_weight_ovr[int(t)] = 0.0
        dir_temperature_ovr[int(t)] = 1.0

    # dump calibrators (ใช้ PATHS)
    joblib.dump(iso_cals, ISO_CAL_PATH)
    joblib.dump(meta_lrs, META_LR_PATH)
    with open(DIR_THR_PATH, 'w', encoding='utf-8') as f:
        json.dump(thresholds, f, indent=2, ensure_ascii=False)
    with open(VAL_PREV_MAP_PATH, 'w', encoding='utf-8') as f:
        json.dump({str(k): float(v) for k, v in val_prev_map.items()}, f, indent=2, ensure_ascii=False)
    with open(DIR_WEIGHT_OVR_PATH, 'w', encoding='utf-8') as f:
        json.dump({str(k): float(v) for k, v in dir_weight_ovr.items()}, f, indent=2, ensure_ascii=False)
    with open(DIR_TEMPERATURE_PATH, 'w', encoding='utf-8') as f:
        json.dump({str(k): float(v) for k, v in dir_temperature_ovr.items()}, f, indent=2, ensure_ascii=False)

# ===================== Trend prior helpers =====================
def trend_prior_from_hist(hist_close):
    if len(hist_close)<2: return 0.5
    logret=np.diff(np.log(hist_close))[-TREND_WIN:]
    if len(logret)==0: return 0.5
    z=np.mean(logret)/(np.std(logret)+1e-8)
    return norm_cdf(TREND_KAPPA*z)

def trend_weight_for(ticker_id:int, sigma_raw:float, market_id_last:int)->float:
    tkr = ticker_scalers[ticker_id]['ticker']
    if tkr in TREND_W_OVR:
        return TREND_W_OVR[tkr]
    mkt = market_encoder.inverse_transform([market_id_last])[-1]
    if mkt=='US':
        return TREND_W_LOWVOL_US if sigma_raw<SIGMA_VOL_SPLIT else TREND_W_HIVOL_US
    else:
        return TREND_W_LOWVOL_TH if sigma_raw<SIGMA_VOL_SPLIT else TREND_W_HIVOL_TH

# ===================== Walk-Forward Validation (streaming) =====================
def walk_forward_validation_prob_batch(
    model, df, feature_columns, ticker_scalers, ticker_encoder, market_encoder,
    seq_length=int(BEST_PARAMS['seq_length']), retrain_frequency=int(BEST_PARAMS['retrain_frequency']),
    chunk_size=int(BEST_PARAMS['chunk_size']), online_learning=True, use_mc_dropout=True,
    iso_cals=None, meta_lrs=None, thresholds=None,
    conf_gate=CONF_GATE, unc_max=UNC_MAX, margin=MARGIN, allow_price_online=ALLOW_PRICE_ONLINE,
    verbose=True, verbose_every=200, ticker_limit=None,
    stream_preds_path=STREAM_PRED_PATH, stream_chunk_path=STREAM_CHUNK_PATH, stream_overall_path=STREAM_OVERALL_PATH
):
    def _normalize_price_out(out):
        y = out.numpy() if hasattr(out, "numpy") else np.asarray(out)
        if y.ndim == 1:
            if y.size != 2:
                raise ValueError(f"Price head must output 2 values [mu, log_sigma], got shape {y.shape}")
            y = y.reshape(1, 2)
        elif y.ndim == 2 and y.shape[1] == 2:
            pass
        else:
            if y.size % 2 == 0:
                y = y.reshape(-1, 2)
            else:
                raise ValueError(f"Unexpected price head output shape: {y.shape}")
        return y

    # เขียนหัวไฟล์
    with open(stream_preds_path,'w',newline='',encoding='utf-8') as fp:
        csv.writer(fp).writerow(['Ticker','Date','Chunk_Index','Step','Predicted_Price','Actual_Price',
                                 'Predicted_Dir','Actual_Dir','Prob_meta_adj','Prob_unc','Thr_Used','Last_Close',
                                 'Price_Δ_Actual','Price_Δ_Pred'])
    with open(stream_chunk_path,'w',newline='',encoding='utf-8') as fc:
        csv.writer(fc).writerow(['Ticker','Chunk_Index','Chunk_Start_Date','Chunk_End_Date','Predictions_Count',
                                 'MAE','RMSE','R2_Score','Direction_Accuracy','Direction_F1','Direction_MCC'])

    daily_all_fh = open(DAILY_ALL_PATH, 'w', newline='', encoding='utf-8')
    daily_all_wt = csv.writer(daily_all_fh)
    daily_all_wt.writerow([
        'Run_ID','Ticker','Date','Chunk_Index','Step',
        'Predicted_Price','Actual_Price','Predicted_Dir','Actual_Dir',
        'Prob_meta_adj','Prob_unc','Thr_Used','Last_Close',
        'Price_Δ_Actual','Price_Δ_Pred',
        'mu_raw','sigma_raw','zscore',
        'p_up','p_iso','p_meta','wdir_used','t_dir_used','market'
    ])

    pred_jsonl   = open(PRED_LOG_PATH, 'a', encoding='utf-8')
    pred_latest  = open(PRED_LATEST_PATH, 'w', encoding='utf-8')

    overall_accum={}
    tickers=df['Ticker'].unique()
    if ticker_limit is not None: tickers=tickers[:int(ticker_limit)]
    print(f"▶️ WFV start: tickers={len(tickers)}, chunk={chunk_size}, seq={seq_length}, MC={MC_DIR_SAMPLES_WFV if MEMORY_LIGHT_WFV else 0}")
    diag_log(f"WFV config: MEMORY_LIGHT_WFV={MEMORY_LIGHT_WFV}, MC_DIR_SAMPLES_WFV={MC_DIR_SAMPLES_WFV}, ONLINE_UPDATE_EVERY={ONLINE_UPDATE_EVERY}, ONLINE_UPDATE_MAX_PER_CHUNK={ONLINE_UPDATE_MAX_PER_CHUNK}")

    adapt_state = {}
    prior_state = {}

    for t_idx,ticker in enumerate(tickers, start=1):
        try:
            g=df[df['Ticker']==ticker].sort_values('Date').reset_index(drop=True)
            total_days=len(g)
            # === แก้เฉพาะจุด ===
            if total_days <= seq_length:
                print(f"⚠️ Skip {ticker}: rows={total_days} too small")
                diag_log(f"Skip {ticker} rows={total_days} too small for chunk_size={chunk_size}+seq={seq_length}")
                continue
            # ใช้ ceil และจะทำ overlap ตอนตัดชังก์
            num_chunks = math.ceil(total_days / chunk_size)
            # === จบจุดที่แก้ ===
            print(f"\n🧩 [{t_idx}/{len(tickers)}] {ticker} rows={total_days} chunks={num_chunks}")
            diag_log(f"Ticker {ticker} rows={total_days} chunks={num_chunks}")
        except Exception as e:
            diag_log(f"ERROR preparing ticker {ticker}: {e}\n{traceback.format_exc()}")
            continue

        for cidx in range(num_chunks):
            try:
                # === แก้เฉพาะจุด (overlap ด้วย lookback = seq_length) ===
                s = 0 if cidx == 0 else max(0, cidx*chunk_size - seq_length)
                e = min((cidx+1)*chunk_size, total_days)
                chunk=g.iloc[s:e].reset_index(drop=True)
                step_total=len(chunk)-seq_length
                if step_total <= 0:
                    print(f"  ⚠️ chunk {cidx+1} too small after overlap: size={len(chunk)}")
                    diag_log(f"{ticker} chunk {cidx+1} too small after overlap: size={len(chunk)}")
                    continue
                # === จบจุดที่แก้ ===

                print(f"  📦 Chunk {cidx+1}/{num_chunks} {chunk['Date'].min()}→{chunk['Date'].max()} steps={step_total}")
                diag_log(f"{ticker} chunk {cidx+1}/{num_chunks} steps={step_total}")
                save_progress(ticker=ticker, chunk_idx=cidx+1, step=0, note="chunk_start")

                free_mb_chunk = get_free_ram_mb()
                if free_mb_chunk < MEM_CRIT_MB:
                    diag_log(f"MEM CRITICAL at chunk start {ticker} chunk={cidx+1}: free={free_mb_chunk:.1f}MB -> disable MC+online")
                    local_mc_samples = 0; local_allow_online = False
                elif free_mb_chunk < MEM_LOW_MB:
                    diag_log(f"MEM LOW at chunk start {ticker} chunk={cidx+1}: free={free_mb_chunk:.1f}MB -> MC=1")
                    local_mc_samples = 1; local_allow_online = True
                else:
                    local_mc_samples = MC_DIR_SAMPLES_WFV if MEMORY_LIGHT_WFV else 0
                    local_allow_online = True

                sum_abs_err=sum_sq_err=sum_y=sum_y2=0.0
                tp=fp=tn=fn=0
                online_updates=0

                with open(stream_preds_path,'a',newline='',encoding='utf-8') as fpred:
                    writer_pred=csv.writer(fpred)
                    ema_state=None; prev_dir=None
                    tid=int(chunk['Ticker_ID'].iloc[0]); tkr_name = ticker_scalers[tid]['ticker']
                    tune = PRECISION_TUNE.get(tkr_name, {})
                    eff_maj_k = int(tune.get('majk', MAJORITY_K))
                    roll=deque(maxlen=eff_maj_k)
                    pi_val=float(val_prev_map.get(tid,0.5)) if 'val_prev_map' in globals() else 0.5
                    if tid not in prior_state:prior_state[tid] = {'pi_pred_ema': pi_val, 'pi_target_ema': pi_val, 'n': 0, 'na': 0}
                    if tid not in adapt_state: adapt_state[tid]={'log':deque(maxlen=ADAPT_WIN),'adj':0.0,'ctr':0}
                    eff_hys = float(tune.get('hys', HYSTERESIS_BAND))
                    eff_unc_max = max(0.0, float(UNC_MAX - float(tune.get('unc_plus', 0.0))))
                    eff_z_gate = float(tune.get('z_gate', Z_GATE_ONLINE))

                    for i in range(step_total):
                        save_progress(ticker=ticker, chunk_idx=cidx+1, step=i+1, note="step")
                        if (i % 25) == 0:
                            free_mb_mid = get_free_ram_mb()
                            if free_mb_mid < MEM_CRIT_MB:
                                local_mc_samples = 0; local_allow_online = False
                            elif free_mb_mid < MEM_LOW_MB and local_mc_samples > 1:
                                local_mc_samples = 1

                        try:
                            hist=chunk.iloc[i:i+seq_length]; targ=chunk.iloc[i+seq_length]
                            t_id_last=int(hist['Ticker_ID'].iloc[-1]); mk_id_last=int(hist['Market_ID_enc'].iloc[-1])
                            if t_id_last not in ticker_scalers: continue
                            fs=ticker_scalers[t_id_last]['feature_scaler']; ps=ticker_scalers[t_id_last]['price_scaler']

                            Xf=fs.transform(hist[feature_columns].values.astype(np.float32)).reshape(1,seq_length,-1)
                            Xt=hist['Ticker_ID'].values.astype(np.int32).reshape(1,seq_length)
                            Xm=hist['Market_ID_enc'].values.astype(np.int32).reshape(1,seq_length)

                            # ---- PRICE-ONLY forward ----
                            y_price = _normalize_price_out(model([Xf, Xt, Xm], training=False))
                            mu_s,log_sigma_s=float(y_price[0,0]),float(y_price[0,1])

                            scale  = getattr(ps, 'scale_',  np.array([1.0], dtype=np.float32))[0]
                            center = getattr(ps, 'center_', np.array([0.0], dtype=np.float32))[0]
                            sigma_s = max(np.log1p(np.exp(log_sigma_s)) + 1e-6,1e-6)
                            mu_raw = mu_s * scale + center
                            sigma_raw = sigma_s * scale

                            last_close=float(hist['Close'].iloc[-1])
                            price_pred=float(last_close*math.exp(mu_raw))

                            # P(UP) + z
                            if sigma_raw<=1e-9:
                                p_up=1.0 if (mu_raw-EPS_RET)>0.0 else 0.0; zz=0.0
                            else:
                                zz=(mu_raw-EPS_RET)/sigma_raw; p_up=norm_cdf(zz)

                            # MC uncertainty (เฉพาะเมื่อจำเป็น)
                            p_unc=0.0
                            need_mc = (abs(p_up-0.5) <= MC_TRIGGER_BAND)
                            if use_mc_dropout and local_mc_samples>0 and need_mc:
                                pups=[]
                                for _ in range(local_mc_samples):
                                    y_price2 = _normalize_price_out(model([Xf, Xt, Xm], training=True))
                                    mu_s2,log_sigma_s2=float(y_price2[0,0]),float(y_price2[0,1])
                                    sigma_s2=max(np.log1p(np.exp(log_sigma_s2))+1e-6,1e-6)
                                    mu_raw2 = mu_s2*scale + center
                                    pups.append(norm_cdf((mu_raw2 - EPS_RET)/(sigma_s2*scale)))
                                p_unc=float(np.std(np.asarray(pups,np.float32),ddof=0))

                            # Calibrations
                            iso=iso_cals.get(t_id_last,None) if iso_cals else None
                            p_iso=float(iso.transform([p_up])[0]) if iso is not None else float(p_up)
                            lr =meta_lrs.get(t_id_last,None) if meta_lrs else None
                            if lr is not None:
                                x_meta=build_xmeta_for_lr(p_iso, zz, sigma_raw, mu_raw, p_unc, pdir_T=p_up, p_unc_dir=p_unc, lr=lr)
                                p_meta=float(lr.predict_proba(x_meta)[0,1])
                            else:
                                p_meta=p_iso

                            # ===== PSC / smoothing / threshold =====
                            st = prior_state.get(t_id_last, {'pi_pred_ema': p_meta, 'pi_target_ema': 0.5, 'n': 0, 'na': 0})
                            pi_pred_ema = float(st['pi_pred_ema'])
                            pi_target_ema = float(st['pi_target_ema'])
                            n_seen = int(st['n'])
                            n_actual = int(st.get('na', 0))
                            pi_train = float(val_prev_map.get(t_id_last, 0.5)) if 'val_prev_map' in globals() else 0.5
                            mkt_name = market_encoder.inverse_transform([mk_id_last])[0]
                            if USE_PSC and APPLY_PSC_MARKET.get(mkt_name, True) and n_seen >= PRIOR_MIN_N and n_actual >= ACT_PREV_MIN_N:
                                def _logit(x, eps=1e-6): x=float(np.clip(x,eps,1-eps)); return math.log(x/(1-x))
                                delta = _logit(pi_target_ema) - _logit(pi_train)
                                delta = np.clip(delta, -PSC_LOGIT_CAP, PSC_LOGIT_CAP)
                                p_meta = 1.0 / (1.0 + math.exp(-(math.log(p_meta/(1-p_meta)) + delta)))
                            st['pi_pred_ema'] = (1 - PRIOR_EMA_ALPHA) * pi_pred_ema + PRIOR_EMA_ALPHA * p_meta
                            st['n'] = n_seen + 1
                            prior_state[t_id_last] = st

                            if USE_TREND_PRIOR:
                                closes=hist['Close'].values.astype(np.float32)
                                logret=np.diff(np.log(closes))[-TREND_WIN:]
                                p_trend=0.5 if len(logret)==0 else norm_cdf(TREND_KAPPA*(np.mean(logret)/(np.std(logret)+1e-8)))
                                w=trend_weight_for(t_id_last, sigma_raw, mk_id_last)
                                p_meta=(1-w)*p_meta + w*p_trend

                            base_alpha = ALPHA_EMA_LOWVOL if sigma_raw<SIGMA_VOL_SPLIT else ALPHA_EMA_HIVOL
                            alpha_base = float(ALPHA_EMA_OVR.get(tkr_name, base_alpha))
                            alpha = float(tune.get('ema_alpha', alpha_base))
                            ema_state=p_meta if ema_state is None else (alpha*ema_state + (1-alpha)*p_meta)
                            p_use=float(np.clip(ema_state,1e-4,1-1e-4))

                            thr_base=float(thresholds.get(str(t_id_last), thresholds.get(t_id_last, 0.5))) if thresholds else 0.5
                            if tkr_name in THR_DELTA_OVR: thr_base=float(np.clip(thr_base+THR_DELTA_OVR[tkr_name], THR_CLIP_LOW, THR_CLIP_HIGH))
                            thr_base=float(np.clip(thr_base + TH_MARKET_DELTA.get(mkt_name, 0.0), THR_CLIP_LOW, THR_CLIP_HIGH))
                            thr_base=float(np.clip(thr_base + float(tune.get('thr_bump', 0.0)), THR_CLIP_LOW, THR_CLIP_HIGH))
                            if t_id_last not in adapt_state: adapt_state[t_id_last]={'adj':0.0,'ctr':0,'log':deque(maxlen=ADAPT_WIN)}
                            thr_adj=float(adapt_state[t_id_last]['adj'])
                            hivol_shift = HIVOL_THR_SHIFT_TH if mkt_name=='TH' else HIVOL_THR_SHIFT_US
                            thr_eff = thr_base + thr_adj + (hivol_shift if sigma_raw>=SIGMA_VOL_SPLIT else 0.0)
                            if USE_THR_ADAPT_FROM_PSC:
                                pi_train=float(val_prev_map.get(t_id_last,0.5)) if 'val_prev_map' in globals() else 0.5
                                delta_pi = float(prior_state[t_id_last]['pi_pred_ema'] - pi_train)
                                thr_eff -= np.clip(THR_ADAPT_GAIN * delta_pi, -THR_ADAPT_CLIP, THR_ADAPT_CLIP)
                            thr_low = THR_CLIP_LOW_TH if mkt_name=='TH' else THR_CLIP_LOW
                            thr_eff = float(np.clip(thr_eff, thr_low, THR_CLIP_HIGH))


                            pred_dir_inst=int(p_use>=thr_eff)
                            if abs(zz)>=Z_STRONG_CUT and (p_unc<=eff_unc_max):
                                pred_dir_inst = 1 if zz>0 else 0

                            pred_after_hys=pred_dir_inst
                            if (prev_dir is not None) and (abs(p_use-thr_eff)<eff_hys):
                                pred_after_hys=prev_dir
                            roll.append(pred_after_hys)
                            pred_dir = 1 if (sum(roll) >= len(roll)-sum(roll)) else 0
                            prev_dir=pred_dir

                            actual_price=float(targ['Close'])
                            actual_dir=int(actual_price>last_close)

                            st = prior_state[t_id_last]
                            st['pi_target_ema'] = (1 - TARGET_EMA_ALPHA) * float(st['pi_target_ema']) + TARGET_EMA_ALPHA * float(actual_dir)
                            st['na'] = int(st.get('na', 0)) + 1
                            prior_state[t_id_last] = st

                            writer_pred.writerow([ticker, targ['Date'], cidx+1, i+1, price_pred, actual_price,
                                                  pred_dir, actual_dir, p_use, p_unc, thr_eff, last_close,
                                                  actual_price-last_close, price_pred-last_close])

                            rec = {
                                "run_id": RUN_ID,
                                "ticker": ticker,
                                "date": str(targ['Date']),
                                "chunk_index": int(cidx+1),
                                "step": int(i+1),
                                "market": mkt_name,
                                "last_close": float(last_close),
                                "price_pred": float(price_pred),
                                "price_actual": float(actual_price),
                                "price_delta_pred": float(price_pred-last_close),
                                "price_delta_actual": float(actual_price-last_close),
                                "mu_raw": float(mu_raw),
                                "sigma_raw": float(sigma_raw),
                                "zscore": float(zz),
                                "p_up": float(p_up),
                                "p_iso": float(p_iso),
                                "p_meta": float(p_meta),
                                "p_use": float(p_use),
                                "p_unc": float(p_unc),
                                "thr_eff": float(thr_eff),
                                "pred_dir": int(pred_dir),
                                "actual_dir": int(actual_dir),
                                "wdir_used": 0.0, "t_dir_used": 1.0
                            }
                            line = json.dumps(rec, ensure_ascii=False)
                            pred_jsonl.write(line + "\n")
                            pred_latest.write(line + "\n")

                            daily_all_wt.writerow([
                                RUN_ID, ticker, str(targ['Date']), int(cidx+1), int(i+1),
                                float(price_pred), float(actual_price),
                                int(pred_dir), int(actual_dir), float(p_use), float(p_unc),
                                float(thr_eff), float(last_close),
                                float(actual_price-last_close), float(price_pred-last_close),
                                float(mu_raw), float(sigma_raw), float(zz),
                                float(p_up), float(p_iso), float(p_meta),
                                0.0, 1.0, mkt_name
                            ])

                            err = actual_price - price_pred
                            sum_abs_err += abs(err); sum_sq_err += err*err
                            sum_y += actual_price; sum_y2 += actual_price*actual_price
                            if pred_dir==1 and actual_dir==1: tp+=1
                            elif pred_dir==1 and actual_dir==0: fp+=1
                            elif pred_dir==0 and actual_dir==0: tn+=1
                            elif pred_dir==0 and actual_dir==1: fn+=1

                            if online_learning and allow_price_online and ALLOW_PRICE_ONLINE_MARKET.get(mkt_name, True):
                                do_update = ((i+1) % ONLINE_UPDATE_EVERY == 0) and (online_updates < ONLINE_UPDATE_MAX_PER_CHUNK) and local_allow_online
                                if mkt_name=='US':
                                    do_update = ((i+1) % ONLINE_UPDATE_EVERY_US == 0) and (online_updates < ONLINE_UPDATE_MAX_PER_CHUNK_US) and local_allow_online
                                if do_update:
                                    ok=True
                                    if conf_gate:
                                        conf=abs(p_use-thr_eff)
                                        ok=(conf>=MARGIN) and (p_unc<=eff_unc_max) and (abs(zz)>= (Z_GATE_ONLINE_US if mkt_name=='US' else eff_z_gate))
                                    if ok:
                                        true_logret=float(np.log(actual_price/last_close))
                                        true_logret=float(np.clip(true_logret,-0.25,0.25))
                                        y_price_true = ps.transform(np.array([[true_logret]],np.float32))
                                        model.train_on_batch([Xf, Xt, Xm], {'price_params': y_price_true})
                                        online_updates += 1

                            if (i+1) % 25 == 0:
                                gc.collect()
                            del Xf, Xt, Xm, y_price

                        except Exception as e:
                            diag_log(f"ERROR step {ticker} chunk={cidx+1} step={i+1}: {e}\n{traceback.format_exc()}")
                            gc.collect()
                            continue

                # ===== chunk metrics =====
                n = step_total
                if n>0:
                    mae = sum_abs_err/n; rmse=math.sqrt(sum_sq_err/n)
                    y_mean = sum_y/n; ss_tot = sum_y2 - n*(y_mean**2)
                    r2 = 1.0 - (sum_sq_err/ss_tot) if ss_tot>1e-9 else 0.0
                    tot = tp+fp+tn+fn
                    if tot>0:
                        acc = (tp+tn)/tot
                        prec = tp/(tp+fp) if (tp+fp)>0 else 0.0
                        rec  = tp/(tp+fn) if (tp+fn)>0 else 0.0
                        f1 = (2*prec*rec)/(prec+rec) if (prec+rec)>0 else 0.0
                        mcc = ((tp*tn - fp*fn)/ math.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))) if ((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))>0 else 0.0
                    else:
                        acc=prec=rec=f1=mcc=0.0

                    with open(stream_chunk_path,'a',newline='',encoding='utf-8') as fc:
                        csv.writer(fc).writerow([ticker,cidx+1,str(chunk['Date'].min()),str(chunk['Date'].max()),
                                                 n,mae,rmse,r2,acc,f1,mcc])

                    acc_tkr=overall_accum.get(ticker)
                    if acc_tkr is None:
                        acc_tkr={'count':0,'sum_abs_err':0.0,'sum_sq_err':0.0,'sum_y':0.0,'sum_y2':0.0,'tp':0,'fp':0,'tn':0,'fn':0}
                    acc_tkr['count']+=int(n)
                    acc_tkr['sum_abs_err']+=float(sum_abs_err)
                    acc_tkr['sum_sq_err'] +=float(sum_sq_err)
                    acc_tkr['sum_y']+=float(sum_y); acc_tkr['sum_y2']+=float(sum_y2)
                    acc_tkr['tp']+=tp; acc_tkr['fp']+=fp; acc_tkr['tn']+=tn; acc_tkr['fn']+=fn
                    overall_accum[ticker]=acc_tkr

                del chunk
                gc.collect()
                save_progress(ticker=ticker, chunk_idx=cidx+1, step=step_total, note="chunk_done")

            except Exception as e:
                diag_log(f"ERROR chunk loop {ticker} chunk={cidx+1}: {e}\n{traceback.format_exc()}")
                save_progress(ticker=ticker, chunk_idx=cidx+1, step=None, note=f"chunk_error: {e}")
                gc.collect()
                continue

    # ===== overall writer =====
    def _safe_div(a, b): return float(a) / float(b) if (b is not None and b != 0) else 0.0
    def _clf_metrics(tp, fp, tn, fn):
        tot = tp + fp + tn + fn
        acc = _safe_div(tp + tn, tot)
        prec = _safe_div(tp, tp + fp)
        rec  = _safe_div(tp, tp + fn)
        f1   = _safe_div(2 * prec * rec, (prec + rec)) if (prec + rec) > 0 else 0.0
        return acc, f1, prec, rec, tot

    header = [
        'Ticker','Total_Predictions','MAE','RMSE','R2_Score',
        'Direction_Accuracy','Direction_F1_Score','Direction_Precision','Direction_Recall',
        'TP','FP','TN','FN','Pred_Pos_Count','Pred_Pos_Rate'
    ]
    with open(stream_overall_path,'w',newline='',encoding='utf-8') as fo:
        w=csv.writer(fo); w.writerow(header)
        for tkr in sorted(overall_accum.keys()):
            acc_tkr=overall_accum.get(tkr, {})
            n   = int(acc_tkr.get('count', 0))
            sae = float(acc_tkr.get('sum_abs_err', 0.0))
            sse = float(acc_tkr.get('sum_sq_err',  0.0))
            sy  = float(acc_tkr.get('sum_y',       0.0))
            sy2 = float(acc_tkr.get('sum_y2',      0.0))
            tp  = int(acc_tkr.get('tp', 0))
            fp  = int(acc_tkr.get('fp', 0))
            tn  = int(acc_tkr.get('tn', 0))
            fn  = int(acc_tkr.get('fn', 0))

            if n <= 0:
                w.writerow([tkr, 0, float('nan'), float('nan'), float('nan'),
                            float('nan'), float('nan'), float('nan'), float('nan'),
                            0,0,0,0,0,0.0])
                continue

            mae  = sae / n
            rmse = math.sqrt(sse / n)
            y_mean = sy / n
            ss_tot = sy2 - n * (y_mean ** 2)
            r2 = 1.0 - (sse / ss_tot) if ss_tot > 1e-9 else 0.0

            acc, f1, prec, rec, _ = _clf_metrics(tp, fp, tn, fn)
            pred_pos = tp + fp
            pred_pos_rate = _safe_div(pred_pos, n)

            w.writerow([tkr, n, mae, rmse, r2, acc, f1, prec, rec,
                        tp, fp, tn, fn, pred_pos, pred_pos_rate])

    # ปิดไฟล์
    try:
        pred_jsonl.close()
        pred_latest.close()
        daily_all_fh.close()
    except Exception:
        pass

    diag_log("=== WFV DONE ===")
    print(f"\n🏁 WFV done. Outputs:\n - {stream_preds_path}\n - {stream_chunk_path}\n - {stream_overall_path}\n - {DAILY_ALL_PATH}\n - {PRED_LOG_PATH}")
    return None, overall_accum

# ===================== Run WFV =====================
# ใช้สำเนาโมเดลเพื่อความปลอดภัย
wfv_model = best_model
if USE_WFV_MODEL_CLONE:
    wfv_model = tf.keras.models.clone_model(best_model)
    wfv_model.build(best_model.input_shape)
    wfv_model.set_weights(best_model.get_weights())
    wfv_model.compile(optimizer=optimizer,
                      loss={"price_params": gaussian_nll},
                      metrics={"price_params":[mae_on_mu]})

try:
    _, results_per_ticker = walk_forward_validation_prob_batch(
        model=wfv_model, df=test_df, feature_columns=feature_columns,
        ticker_scalers=ticker_scalers, ticker_encoder=ticker_encoder, market_encoder=market_encoder,
        seq_length=int(BEST_PARAMS['seq_length']), retrain_frequency=int(BEST_PARAMS['retrain_frequency']),
        chunk_size=int(BEST_PARAMS['chunk_size']), online_learning=True, use_mc_dropout=True,
        iso_cals=globals().get('iso_cals', {}), meta_lrs=globals().get('meta_lrs', {}), thresholds=globals().get('thresholds', {}),
        conf_gate=CONF_GATE, unc_max=UNC_MAX, margin=MARGIN, allow_price_online=ALLOW_PRICE_ONLINE,
        verbose=True, verbose_every=200, ticker_limit=None
    )
except Exception as e:
    diag_log(f"FATAL in WFV: {e}\n{traceback.format_exc()}")
    raise

# ---------- BestThr report from produced predictions ----------
def _best_thr_report(daily_path, thr_min=0.50, thr_max=0.86, thr_step=0.01):
    import pandas as _pd, numpy as _np
    if not os.path.exists(daily_path):
        raise FileNotFoundError(f"not found: {daily_path}")
    df = _pd.read_csv(daily_path)
    if df.empty:
        raise RuntimeError("daily_all is empty")
    rows=[]
    ths = _np.arange(thr_min, thr_max + 1e-9, thr_step)
    for tkr, g in df.groupby('Ticker'):
        y = g['Actual_Dir'].astype(int).to_numpy()
        p = g['Prob_meta_adj'].astype(float).to_numpy()
        best = (-1.0, 0.50, (0,0,0,0,0,0,0,0,0,0))  # acc, thr, metrics
        tot = len(y)
        for th in ths:
            yhat = (p >= th).astype(int)
            tp = int(((yhat==1)&(y==1)).sum()); fp=int(((yhat==1)&(y==0)).sum())
            tn = int(((yhat==0)&(y==0)).sum()); fn=int(((yhat==0)&(y==1)).sum())
            acc = (tp+tn)/tot if tot else 0.0
            rec = tp/(tp+fn) if (tp+fn)>0 else 0.0
            prec= tp/(tp+fp) if (tp+fp)>0 else 0.0
            f1  = (2*prec*rec)/(prec+rec) if (prec+rec)>0 else 0.0
            tpr = rec
            tnr = tn/(tn+fp) if (tn+fp)>0 else 0.0
            bal = 0.5*(tpr+tnr)
            pos = (tp+fp)/tot if tot else 0.0
            if acc > best[0]:
                best = (acc, th, (pos, rec, prec, f1, acc, bal, tp, fp, tn, fn))
        acc, th, (pos, rec, prec, f1, acc, bal, tp, fp, tn, fn) = best
        rows.append([tkr, round(th,2), round(pos,3), round(rec,3), round(prec,3),
                     round(f1,3), round(acc,3), round(bal,3), tp, fp, tn, fn])
    return _pd.DataFrame(rows, columns=['Ticker','BestThr','PosRate@Best','Recall@Best','Prec@Best','F1@Best',
                                        'Acc@Best','BalAcc@Best','TP','FP','TN','FN'])

try:
    bestthr_path = os.path.join(LOG_DIR, f"threshold_tune_report_{RUN_ID}.csv")
    df_best = _best_thr_report(DAILY_ALL_PATH, thr_min=THR_CLIP_LOW, thr_max=THR_CLIP_HIGH, thr_step=0.01)
    df_best.to_csv(bestthr_path, index=False)
    print(f"📝 BestThr report -> {bestthr_path}")
except Exception as e:
    print("⚠️ BestThr report skipped:", e)

# (ออปชัน) apply BestThr เป็น thresholds สำหรับรอบถัดไป
APPLY_BEST_THR_NEXT_RUN = False
if APPLY_BEST_THR_NEXT_RUN:
    try:
        import pandas as _pd
        df_thr = df_best if 'df_best' in globals() else _pd.read_csv(bestthr_path)
        # โหลด thresholds ปัจจุบัน (id-based)
        thr_map = {}
        if os.path.exists(DIR_THR_PATH):
            thr_map = json.load(open(DIR_THR_PATH,'r',encoding='utf-8'))
        # อัปเดตค่าตาม ticker-id
        for _, r in df_thr.iterrows():
            tname = str(r['Ticker'])
            if tname in list(ticker_encoder.classes_):
                tid = int(ticker_encoder.transform([tname])[0])
                thr_map[str(tid)] = float(r['BestThr'])
        with open(DIR_THR_PATH,'w',encoding='utf-8') as f:
            json.dump(thr_map, f, indent=2, ensure_ascii=False)
        print(f"✅ Updated thresholds for next run -> {DIR_THR_PATH}")
    except Exception as e:
        print("⚠️ Apply BestThr skipped:", e)

# บันทึกโมเดลออนไลน์ (ถ้าต้องการ)
if PERSIST_ONLINE_UPDATES:
    try:
        wfv_model.save(BEST_MODEL_PATH_ONLINE)
        print(f"💾 Saved online-updated model to {BEST_MODEL_PATH_ONLINE}")
    except Exception as e:
        print("⚠️ Could not save online-updated model:", e)

# ===================== Save artifacts =====================
production_config={
    'model_config': BEST_PARAMS,
    'inference_config': {
        'eps_ret': EPS_RET,
        'use_ema_prob': USE_EMA_PROB,
        'alpha_ema_lowvol': ALPHA_EMA_LOWVOL,
        'alpha_ema_hivol': ALPHA_EMA_HIVOL,
        'alpha_ema_overrides': ALPHA_EMA_OVR,
        'sigma_vol_split': SIGMA_VOL_SPLIT,
        'threshold_min': THRESH_MIN,
        'min_recall': MIN_RECALL,
        'thr_objective_us': THR_OBJECTIVE_US,
        'thr_objective_th': THR_OBJECTIVE_TH,
        'fbeta_th': FBETA_TH,
        'majority_k': MAJORITY_K,
        'hysteresis_band': HYSTERESIS_BAND,
        'z_strong_cut': Z_STRONG_CUT,
        'prior_shift_correction': USE_PSC,
        'prior_ema_alpha': PRIOR_EMA_ALPHA,
        'target_ema_alpha': TARGET_EMA_ALPHA,
        'trend_prior': USE_TREND_PRIOR,
        'trend_kappa': TREND_KAPPA,
        'trend_w_lowvol_th': TREND_W_LOWVOL_TH,
        'trend_w_hivol_th': TREND_W_HIVOL_TH,
        'trend_w_lowvol_us': TREND_W_LOWVOL_US,
        'trend_w_hivol_us': TREND_W_HIVOL_US,
        'trend_w_ovr': TREND_W_OVR,
        'thr_adapt_from_psc': USE_THR_ADAPT_FROM_PSC,
        'thr_adapt_gain': THR_ADAPT_GAIN,
        'thr_adapt_clip': THR_ADAPT_CLIP,
        'psc_logit_cap': PSC_LOGIT_CAP,
        'thr_delta_ovr': THR_DELTA_OVR,
        'thr_market_delta': TH_MARKET_DELTA,
        'min_recall_overrides': MIN_RECALL_OVR,
        'precision_tune': PRECISION_TUNE,
        'hivol_thr_shift_us': HIVOL_THR_SHIFT_US,
        'hivol_thr_shift_th': HIVOL_THR_SHIFT_TH,
        'market_policy': {
            'apply_psc': APPLY_PSC_MARKET,
            'allow_online': ALLOW_PRICE_ONLINE_MARKET,
            'wdir_cap_th': WDIR_CAP_TH,
            'wdir_cap_us': WDIR_CAP_US,
            't_dir_base_th': T_DIR_BASE_TH,
            't_dir_base_us': T_DIR_BASE_US
        }
    },
    'paths': {
        'log_dir': LOG_DIR,
        'daily_all': DAILY_ALL_PATH,
        'stream_pred': STREAM_PRED_PATH,
        'stream_chunk': STREAM_CHUNK_PATH,
        'stream_overall': STREAM_OVERALL_PATH,
        'best_model_static': BEST_MODEL_PATH_STATIC,
        'best_model_online': BEST_MODEL_PATH_ONLINE,
        'iso_calibrators': ISO_CAL_PATH,
        'meta_lr': META_LR_PATH,
        'dir_thresholds': DIR_THR_PATH,
        'val_prev_map': VAL_PREV_MAP_PATH
    }
}
with open(PRODUCTION_CONFIG_PATH,'w', encoding='utf-8') as f: json.dump(production_config,f,indent=2, ensure_ascii=False)

# เซฟโมเดลหลัก (เผื่อไม่มีไฟล์ static)
try:
    best_model.save(BEST_MODEL_PATH_STATIC)
except Exception as e:
    print("⚠️ Could not resave static model:", e)

artifacts={
    'ticker_scalers': ticker_scalers,
    'ticker_encoder': ticker_encoder,
    'market_encoder': market_encoder,
    'feature_columns': feature_columns,
    'iso_cals': {int(k):v for k,v in globals().get('iso_cals', {}).items()},
    'meta_lrs': {int(k):v for k,v in globals().get('meta_lrs', {}).items()},
    'thresholds': globals().get('thresholds', {}),
    'val_prev_map': globals().get('val_prev_map', {}),
    'mc_dir_samples': MC_DIR_SAMPLES_WFV
}
joblib.dump(artifacts, SERVING_ARTIFACTS_PATH)
print("✅ Saved all artifacts.")
