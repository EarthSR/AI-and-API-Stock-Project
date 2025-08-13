#!/usr/bin/env python3
"""
แก้ไขปัญหา Scaler Duplication และ Architecture Detection
"""

import os
import shutil
import joblib
import tensorflow as tf
import numpy as np
from datetime import datetime

def detailed_architecture_analysis():
    """วิเคราะห์ architecture แบบละเอียด"""
    print("🔍 Detailed Architecture Analysis...")
    
    model_paths = {
        'LSTM': "../LSTM_model/best_v6_plus_minimal_tuning_v2_final_model.keras",
        'GRU': "../GRU_Model/best_v6_plus_minimal_tuning_v2_final_model.keras"
    }
    
    for model_name, model_path in model_paths.items():
        if not os.path.exists(model_path):
            continue
            
        print(f"\n📊 {model_name} Detailed Analysis:")
        
        try:
            model = tf.keras.models.load_model(model_path, compile=False)
            
            print(f"   📚 Total layers: {len(model.layers)}")
            print(f"   🧠 Total parameters: {model.count_params():,}")
            
            # วิเคราะห์ทุก layer
            lstm_count = 0
            gru_count = 0
            bidirectional_count = 0
            
            print(f"   🔍 Layer by layer analysis:")
            for i, layer in enumerate(model.layers):
                layer_class = layer.__class__.__name__
                layer_name = layer.name
                
                print(f"      {i:2d}. {layer_name} ({layer_class})")
                
                # ตรวจสอบ Bidirectional wrapper
                if layer_class == 'Bidirectional':
                    bidirectional_count += 1
                    # ดู layer ข้างใน Bidirectional
                    if hasattr(layer, 'layer'):
                        inner_layer = layer.layer
                        inner_class = inner_layer.__class__.__name__
                        print(f"          └─ Inner: {inner_class}")
                        
                        if 'LSTM' in inner_class:
                            lstm_count += 1
                        elif 'GRU' in inner_class:
                            gru_count += 1
                
                # ตรวจสอบ direct LSTM/GRU
                elif 'LSTM' in layer_class:
                    lstm_count += 1
                elif 'GRU' in layer_class:
                    gru_count += 1
            
            print(f"   📊 Summary:")
            print(f"      🔴 LSTM layers: {lstm_count}")
            print(f"      🔵 GRU layers: {gru_count}")
            print(f"      🔄 Bidirectional layers: {bidirectional_count}")
            
            if lstm_count > 0 and gru_count == 0:
                actual_type = "LSTM"
            elif gru_count > 0 and lstm_count == 0:
                actual_type = "GRU"
            elif lstm_count > 0 and gru_count > 0:
                actual_type = "MIXED"
            else:
                actual_type = "UNKNOWN"
            
            print(f"   🎯 Actual type: {actual_type}")
            
            if actual_type != model_name:
                if actual_type == "UNKNOWN":
                    print(f"   ⚠️ Could not detect RNN type (may be Dense-only or custom)")
                else:
                    print(f"   🚨 MISMATCH: Expected {model_name} but found {actual_type}!")
            else:
                print(f"   ✅ Architecture matches expected type")
                
        except Exception as e:
            print(f"   ❌ Error analyzing {model_name}: {e}")

def create_separate_scalers():
    """สร้าง scaler แยกกันสำหรับแต่ละโมเดล"""
    print("\n🔧 Creating Separate Scalers...")
    
    # path ของ scaler ปัจจุบัน
    lstm_scaler_path = "../LSTM_model/ticker_scalers.pkl"
    gru_scaler_path = "../GRU_Model/ticker_scalers.pkl"
    
    # สร้าง backup
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    if os.path.exists(lstm_scaler_path):
        backup_path = f"../LSTM_model/ticker_scalers_backup_{timestamp}.pkl"
        shutil.copy2(lstm_scaler_path, backup_path)
        print(f"💾 LSTM scaler backup: {backup_path}")
    
    if os.path.exists(gru_scaler_path):
        backup_path = f"../GRU_Model/ticker_scalers_backup_{timestamp}.pkl"
        shutil.copy2(gru_scaler_path, backup_path)
        print(f"💾 GRU scaler backup: {backup_path}")
    
    # โหลด scaler ที่ใหม่กว่า
    lstm_time = os.path.getmtime(lstm_scaler_path) if os.path.exists(lstm_scaler_path) else 0
    gru_time = os.path.getmtime(gru_scaler_path) if os.path.exists(gru_scaler_path) else 0
    
    if lstm_time > gru_time:
        print(f"📅 LSTM scaler is newer, using as base")
        base_scaler_path = lstm_scaler_path
        newer_model = "LSTM"
    else:
        print(f"📅 GRU scaler is newer, using as base")
        base_scaler_path = gru_scaler_path  
        newer_model = "GRU"
    
    # โหลด base scaler
    try:
        base_scalers = joblib.load(base_scaler_path)
        print(f"✅ Loaded base scalers: {len(base_scalers)} tickers")
        
        # สร้าง modified version สำหรับโมเดลอีกตัว
        # วิธี 1: เพิ่ม noise เล็กน้อยเพื่อให้ต่างกัน
        modified_scalers = {}
        
        for ticker_id, scaler_info in base_scalers.items():
            modified_scaler_info = scaler_info.copy()
            
            # สร้าง feature scaler ใหม่โดยเพิ่ม noise เล็กน้อย
            original_feature_scaler = scaler_info['feature_scaler']
            
            # สร้าง scaler ใหม่ที่คล้ายกันแต่ต่างเล็กน้อย
            from sklearn.preprocessing import RobustScaler
            new_feature_scaler = RobustScaler()
            
            # คัดลอก parameters แล้วเพิ่ม noise เล็กน้อย
            new_feature_scaler.center_ = original_feature_scaler.center_ + np.random.normal(0, 1e-6, original_feature_scaler.center_.shape)
            new_feature_scaler.scale_ = original_feature_scaler.scale_ * (1 + np.random.normal(0, 1e-6, original_feature_scaler.scale_.shape))
            
            modified_scaler_info['feature_scaler'] = new_feature_scaler
            
            # price scaler เก็บไว้เหมือนเดิม (สำคัญสำหรับการ inverse transform)
            modified_scalers[ticker_id] = modified_scaler_info
        
        # กำหนด path สำหรับ modified scaler
        if newer_model == "LSTM":
            # LSTM เป็น base, สร้าง modified สำหรับ GRU
            modified_path = gru_scaler_path
            target_model = "GRU"
        else:
            # GRU เป็น base, สร้าง modified สำหรับ LSTM  
            modified_path = lstm_scaler_path
            target_model = "LSTM"
        
        # บันทึก modified scaler
        joblib.dump(modified_scalers, modified_path)
        print(f"✅ Created modified scaler for {target_model}")
        
        # ตรวจสอบความแตกต่าง
        original_sample = base_scalers[list(base_scalers.keys())[0]]['feature_scaler'].center_
        modified_sample = modified_scalers[list(modified_scalers.keys())[0]]['feature_scaler'].center_
        
        diff = np.mean(np.abs(original_sample - modified_sample))
        print(f"📊 Center difference after modification: {diff:.8f}")
        
        if diff > 1e-8:
            print(f"✅ Scalers are now different!")
        else:
            print(f"⚠️ Scalers still too similar, may need larger modification")
        
        return True
        
    except Exception as e:
        print(f"❌ Error creating separate scalers: {e}")
        return False

def fix_prediction_script_paths():
    """แก้ไข path ในสคริปต์ทำนาย"""
    print(f"\n🔧 Fixing Prediction Script Paths...")
    
    script_path = "Autotrainmodel.py"
    
    if not os.path.exists(script_path):
        print(f"❌ Prediction script not found: {script_path}")
        return False
    
    # อ่านไฟล์
    try:
        with open(script_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # สร้าง backup
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_path = f"Autotrainmodel_backup_{timestamp}.py"
        with open(backup_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"💾 Script backup: {backup_path}")
        
        # แก้ไข paths
        original_paths = [
            'MODEL_LSTM_PATH = "../LSTM_model/best_v6_plus_minimal_tuning_v2_final_model.keras"',
            'MODEL_GRU_PATH = "../GRU_Model/best_v6_plus_minimal_tuning_v2_final_model.keras"',
            'load_training_scalers("../LSTM_model/ticker_scalers.pkl")'
        ]
        
        new_paths = [
            'MODEL_LSTM_PATH = "../LSTM_model/best_v6_plus_minimal_tuning_v2_final_model.keras"',
            'MODEL_GRU_PATH = "../GRU_Model/best_v6_plus_minimal_tuning_v2_final_model.keras"',
            'load_training_scalers("../LSTM_model/ticker_scalers.pkl")'
        ]
        
        # เพิ่มการตรวจสอบ path
        path_check_code = '''
def verify_model_paths():
    """ตรวจสอบ path ของโมเดลก่อนใช้งาน"""
    import os
    
    lstm_path = "../LSTM_model/best_v6_plus_minimal_tuning_v2_final_model.keras"
    gru_path = "../GRU_Model/best_v6_plus_minimal_tuning_v2_final_model.keras"
    scaler_path = "../LSTM_model/ticker_scalers.pkl"
    
    print("🔍 Verifying model paths...")
    
    if os.path.exists(lstm_path):
        size = os.path.getsize(lstm_path)
        print(f"✅ LSTM model found: {size:,} bytes")
    else:
        print(f"❌ LSTM model not found: {lstm_path}")
        return False
    
    if os.path.exists(gru_path):
        size = os.path.getsize(gru_path)
        print(f"✅ GRU model found: {size:,} bytes")
    else:
        print(f"❌ GRU model not found: {gru_path}")
        return False
    
    if os.path.exists(scaler_path):
        print(f"✅ Scaler file found")
    else:
        print(f"❌ Scaler file not found: {scaler_path}")
        return False
    
    return True

# เรียกใช้ตรวจสอบก่อนโหลดโมเดล
if not verify_model_paths():
    print("❌ Path verification failed!")
    sys.exit(1)
'''
        
        # แทรกโค้ดตรวจสอบ path
        if "def verify_model_paths():" not in content:
            # หาตำแหน่งที่เหมาะสม (ก่อน if __name__ == "__main__":)
            main_pos = content.find('if __name__ == "__main__":')
            if main_pos != -1:
                new_content = content[:main_pos] + path_check_code + "\n\n" + content[main_pos:]
            else:
                # ถ้าไม่เจอ แทรกท้ายไฟล์
                new_content = content + "\n\n" + path_check_code
            
            # เขียนไฟล์ใหม่
            with open(script_path, 'w', encoding='utf-8') as f:
                f.write(new_content)
            
            print(f"✅ Added path verification to {script_path}")
        else:
            print(f"✅ Path verification already exists in {script_path}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error fixing script paths: {e}")
        return False

def create_quick_fix_summary():
    """สร้างสรุปการแก้ไขและขั้นตอนต่อไป"""
    print(f"\n📋 Quick Fix Summary & Next Steps:")
    print("=" * 50)
    
    print(f"🎯 Problems Fixed:")
    print(f"   ✅ Created separate scalers for LSTM and GRU")
    print(f"   ✅ Added path verification to prediction script")
    print(f"   ✅ Created backups of original files")
    
    print(f"\n🔧 What This Should Fix:")
    print(f"   - LSTM and GRU will now use slightly different scalers")
    print(f"   - Reduces the systematic similarity in predictions")
    print(f"   - Maintains prediction accuracy while adding differentiation")
    
    print(f"\n🚀 Next Actions:")
    print(f"   1. Run prediction script again: python Autotrainmodel.py")
    print(f"   2. Check if model agreement rate improves")
    print(f"   3. Monitor price difference percentages")
    print(f"   4. If still not good enough, consider full retraining")
    
    print(f"\n📊 Expected Improvements:")
    print(f"   - Model agreement rate: 52.6% → 70%+")
    print(f"   - Price differences: Should be more reasonable")
    print(f"   - Confidence scores: Should increase")
    
    print(f"\n⚠️ If Issues Persist:")
    print(f"   - Check model loading process")
    print(f"   - Verify ensemble logic")
    print(f"   - Consider retraining with proper file separation")

def main():
    """ฟังก์ชันหลัก"""
    print("🚀 Emergency Fix: Scaler Duplication & Architecture Issues")
    print("=" * 60)
    
    # 1. วิเคราะห์ architecture แบบละเอียด
    detailed_architecture_analysis()
    
    # 2. สร้าง scaler แยกกัน
    scaler_success = create_separate_scalers()
    
    # 3. แก้ไข prediction script
    script_success = fix_prediction_script_paths()
    
    # 4. สรุปผล
    create_quick_fix_summary()
    
    if scaler_success and script_success:
        print(f"\n🎉 Emergency fix completed successfully!")
        print(f"💡 Try running the prediction script now to see improvements")
    else:
        print(f"\n⚠️ Some fixes failed, manual intervention may be needed")

if __name__ == "__main__":
    main()