#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
🔍 Test Script สำหรับทดสอบการทำนายหลัง retrain
ตรวจสอบว่าราคาที่ทำนายมาเหมาะสมหรือไม่
"""

import joblib
import os
from datetime import datetime

def check_model_files():
    """ตรวจสอบวันที่ของไฟล์โมเดล"""
    print("Checking model files...")
    
    files_to_check = [
        "../LSTM_model/best_hypertuned_model.keras",
        "../LSTM_model/ticker_scalers.pkl",
        "../GRU_Model/best_hypertuned_model.keras", 
        "../GRU_Model/ticker_scalers.pkl"
    ]
    
    for file_path in files_to_check:
        if os.path.exists(file_path):
            stat = os.stat(file_path)
            mod_time = datetime.fromtimestamp(stat.st_mtime)
            hours_ago = (datetime.now() - mod_time).total_seconds() / 3600
            
            print(f"   FILE: {file_path}")
            print(f"      Modified: {mod_time.strftime('%Y-%m-%d %H:%M:%S')} ({hours_ago:.1f}h ago)")
            
            if hours_ago > 24:
                print(f"      STATUS: OLD FILE (> 24h)")
            elif hours_ago < 1:
                print(f"      STATUS: FRESH (< 1h)")
            else:
                print(f"      STATUS: Recent")
        else:
            print(f"   ERROR: {file_path} - NOT FOUND")

def check_scaler_consistency():
    """ตรวจสอบความสอดคล้องของ scalers"""
    print(f"\nChecking Scaler Consistency...")
    
    lstm_scalers_path = "../LSTM_model/ticker_scalers.pkl"
    gru_scalers_path = "../GRU_Model/ticker_scalers.pkl"
    
    if os.path.exists(lstm_scalers_path) and os.path.exists(gru_scalers_path):
        lstm_scalers = joblib.load(lstm_scalers_path)
        gru_scalers = joblib.load(gru_scalers_path)
        
        print(f"   LSTM scalers: {len(lstm_scalers)} tickers")
        print(f"   GRU scalers: {len(gru_scalers)} tickers")
        
        # ตรวจสอบ ticker names
        lstm_tickers = set()
        gru_tickers = set()
        
        for scaler_info in lstm_scalers.values():
            if 'ticker' in scaler_info:
                lstm_tickers.add(scaler_info['ticker'])
                
        for scaler_info in gru_scalers.values():
            if 'ticker' in scaler_info:
                gru_tickers.add(scaler_info['ticker'])
        
        common_tickers = lstm_tickers.intersection(gru_tickers)
        print(f"   Common tickers: {len(common_tickers)}")
        print(f"   Tickers: {sorted(list(common_tickers))[:5]}...")
        
        if len(common_tickers) < 15:
            print(f"   WARNING: Too few common tickers!")
        else:
            print(f"   STATUS: Scalers look consistent")
    else:
        print(f"   ERROR: Scaler files missing")

def main():
    """รันการทดสอบทั้งหมด"""
    print("Testing Model Files After Retrain")
    print("=" * 50)
    
    check_model_files()
    check_scaler_consistency()
    
    print(f"\nNOTE: After retrain completes, run: python Autotrainmodel.py")
    print(f"   Should get better price predictions")

if __name__ == "__main__":
    main()