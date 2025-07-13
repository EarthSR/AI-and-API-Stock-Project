import numpy as np 
import pandas as pd
import joblib
import xgboost as xgb
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBRegressor, XGBClassifier
from lightgbm import LGBMRegressor, LGBMClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from datetime import datetime

warnings.filterwarnings('ignore')

def create_safe_features(lstm_df, gru_df):
    """สร้าง features ที่ปลอดภัยจาก data leakage"""
    print("🛡️ Creating safe features (no direct price predictions)...")
    
    # Combine results from LSTM and GRU
    df = pd.DataFrame({
        "Ticker": lstm_df["Ticker"],
        "Date": pd.to_datetime(lstm_df["Date"]),
        "Actual_Price": lstm_df["Actual_Price"],
        "Predicted_Price_LSTM": lstm_df["Predicted_Price"],  # ใช้เฉพาะสำหรับคำนวณ derived features
        "Predicted_Price_GRU": gru_df["Predicted_Price"],    # ใช้เฉพาะสำหรับคำนวณ derived features
        "Actual_Direction": lstm_df["Actual_Dir"],
        "Predicted_Dir_LSTM": lstm_df["Predicted_Dir"],
        "Predicted_Dir_GRU": gru_df["Predicted_Dir"]
    })

    print("🔧 Creating derived features...")
    
    # 1. Price-based derived features (NO direct prices)
    df["LSTM_GRU_Price_Diff_Pct"] = abs(df["Predicted_Price_LSTM"] - df["Predicted_Price_GRU"]) / df["Actual_Price"]
    df["LSTM_GRU_Price_Ratio"] = df["Predicted_Price_LSTM"] / df["Predicted_Price_GRU"]
    df["LSTM_GRU_Dir_Match"] = (df["Predicted_Dir_LSTM"] == df["Predicted_Dir_GRU"]).astype(int)
    
    # 2. Prediction error features (lagged)
    df["LSTM_Pred_Error_Pct"] = 0.0
    df["GRU_Pred_Error_Pct"] = 0.0
    df["LSTM_Dir_Accuracy_1d"] = 0.0
    df["GRU_Dir_Accuracy_1d"] = 0.0

    ticker_groups = df.groupby("Ticker")
    for ticker, group in ticker_groups:
        ticker_data = group.sort_values("Date")
        indices = ticker_data.index

        # Lagged prediction errors (ใช้ shift(1) เพื่อหลีกเลี่ยง look-ahead bias)
        lstm_error = abs(ticker_data["Actual_Price"] - ticker_data["Predicted_Price_LSTM"]) / ticker_data["Actual_Price"]
        gru_error = abs(ticker_data["Actual_Price"] - ticker_data["Predicted_Price_GRU"]) / ticker_data["Actual_Price"]
        
        df.loc[indices, "LSTM_Pred_Error_Pct"] = lstm_error.shift(1).values
        df.loc[indices, "GRU_Pred_Error_Pct"] = gru_error.shift(1).values

        # Lagged directional accuracy
        df.loc[indices, "LSTM_Dir_Accuracy_1d"] = (ticker_data["Predicted_Dir_LSTM"].shift(1) == ticker_data["Actual_Direction"].shift(1)).astype(float).values
        df.loc[indices, "GRU_Dir_Accuracy_1d"] = (ticker_data["Predicted_Dir_GRU"].shift(1) == ticker_data["Actual_Direction"].shift(1)).astype(float).values

        # Rolling accuracy (ใช้ shift เพื่อป้องกัน leakage)
        df.loc[indices, "LSTM_Dir_Accuracy_3d"] = (
            (ticker_data["Predicted_Dir_LSTM"].shift(1) == ticker_data["Actual_Direction"].shift(1))
            .rolling(3).mean().values
        )
        df.loc[indices, "GRU_Dir_Accuracy_3d"] = (
            (ticker_data["Predicted_Dir_GRU"].shift(1) == ticker_data["Actual_Direction"].shift(1))
            .rolling(3).mean().values
        )

        # Model agreement features
        df.loc[indices, "Dir_Agreement_3d"] = (
            (ticker_data["Predicted_Dir_LSTM"] == ticker_data["Predicted_Dir_GRU"])
            .rolling(3).mean().shift(1).values
        )

    # 3. Time-based features
    df["DayOfWeek"] = df["Date"].dt.dayofweek
    df["Is_Monday"] = (df["DayOfWeek"] == 0).astype(int)
    df["Is_Friday"] = (df["DayOfWeek"] == 4).astype(int)
    df["Is_WeekEnd"] = (df["DayOfWeek"].isin([5, 6])).astype(int)

    df["DayOfMonth"] = df["Date"].dt.day
    df["IsFirstHalfOfMonth"] = (df["DayOfMonth"] <= 15).astype(int)
    df["IsMonthEnd"] = (df["DayOfMonth"] >= 28).astype(int)

    # 4. Model confidence features
    df["Both_Models_Bullish"] = ((df["Predicted_Dir_LSTM"] == 1) & (df["Predicted_Dir_GRU"] == 1)).astype(int)
    df["Both_Models_Bearish"] = ((df["Predicted_Dir_LSTM"] == 0) & (df["Predicted_Dir_GRU"] == 0)).astype(int)
    df["Models_Disagree"] = (df["Predicted_Dir_LSTM"] != df["Predicted_Dir_GRU"]).astype(int)

    # 5. ลบ direct price predictions (ป้องกัน leakage)
    print("🗑️ Removing direct price predictions to prevent leakage...")
    df = df.drop(['Predicted_Price_LSTM', 'Predicted_Price_GRU'], axis=1)

    # 6. Handle Missing Values
    print("🧹 Handling missing values...")
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    for col in numeric_cols:
        if df[col].isnull().sum() > 0:
            df[col] = df.groupby("Ticker")[col].transform(lambda x: x.fillna(x.median()))

    print(f"✅ Created {len(df.columns)-4} safe features")  # -4 for Ticker, Date, Actual_Price, Actual_Direction
    return df

def handle_outliers(df, cols, lower_quantile=0.01, upper_quantile=0.99):
    """จัดการ outliers ด้วย winsorization (conservative)"""
    df_clean = df.copy()
    for col in cols:
        if col in df_clean.columns:
            lower_bound = df_clean[col].quantile(lower_quantile)
            upper_bound = df_clean[col].quantile(upper_quantile)
            df_clean[col] = df_clean[col].clip(lower_bound, upper_bound)
    return df_clean

def clean_features(df, feature_cols):
    """ทำความสะอาด features"""
    print("🧹 Cleaning features...")
    
    # Remove columns with too many missing values
    cols_to_drop = []
    for col in feature_cols:
        if df[col].isnull().sum() / len(df) > 0.5:  # เข้มงวดขึ้น
            cols_to_drop.append(col)
    
    feature_cols = [col for col in feature_cols if col not in cols_to_drop]
    
    # Remove low variance columns
    variance_threshold = 1e-6  # เข้มงวดขึ้น
    cols_low_variance = []
    for col in feature_cols:
        if df[col].var() < variance_threshold:
            cols_low_variance.append(col)
    
    feature_cols = [col for col in feature_cols if col not in cols_low_variance]
    
    # Remove highly correlated features (ป้องกัน leakage)
    correlation_threshold = 0.95
    corr_matrix = df[feature_cols].corr().abs()
    upper_triangle = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )
    
    high_corr_features = [
        column for column in upper_triangle.columns 
        if any(upper_triangle[column] > correlation_threshold)
    ]
    
    feature_cols = [col for col in feature_cols if col not in high_corr_features]
    
    print(f"Removed {len(cols_to_drop)} columns with missing values")
    print(f"Removed {len(cols_low_variance)} columns with low variance")
    print(f"Removed {len(high_corr_features)} highly correlated features")
    print(f"Final features: {len(feature_cols)}")
    
    return feature_cols

def check_data_leakage_safe(ensemble_features, results_df):
    """ตรวจสอบ data leakage (version ที่ปรับปรุง)"""
    
    print("\n" + "="*60)
    print("🔍 DATA LEAKAGE DETECTION REPORT (FIXED VERSION)")
    print("="*60)
    
    # 1. ตรวจสอบ feature correlations
    print("\n📊 1. FEATURE CORRELATION WITH TARGET")
    print("-" * 40)
    
    feature_cols = [col for col in ensemble_features.columns if col not in 
                   ["Ticker", "Date", "Actual_Price", "Actual_Direction"]]
    
    correlations = []
    for col in feature_cols:
        if ensemble_features[col].dtype in ['float64', 'int64']:
            corr = ensemble_features[col].corr(ensemble_features['Actual_Price'])
            if not np.isnan(corr):
                correlations.append((col, abs(corr)))
    
    correlations.sort(key=lambda x: x[1], reverse=True)
    
    print("🔍 TOP 10 HIGHEST CORRELATIONS WITH ACTUAL PRICE:")
    for i, (feature, corr) in enumerate(correlations[:10]):
        flag = "🔴 SUSPICIOUS!" if corr > 0.7 else "🟡 Check" if corr > 0.5 else "🟢 Normal"
        print(f"{i+1:2d}. {feature:<35} | Correlation: {corr:.4f} {flag}")
    
    # 2. Performance reality check
    print("\n📏 2. PERFORMANCE REALITY CHECK")
    print("-" * 40)
    
    if results_df is not None:
        xgb_r2 = r2_score(results_df['Actual_Price'], results_df['XGBoost_Price_Pred'])
        lgbm_r2 = r2_score(results_df['Actual_Price'], results_df['LightGBM_Price_Pred'])
        
        print(f"XGBoost R²: {xgb_r2:.4f}")
        print(f"LightGBM R²: {lgbm_r2:.4f}")
        
        print("\n📊 Realistic Performance Ranges:")
        print("   🟢 Excellent R²: 0.05 - 0.15")
        print("   🟡 Very Good R²: 0.15 - 0.30")
        print("   🟠 Suspicious R²: 0.30 - 0.50")
        print("   🔴 Likely Leakage R²: > 0.50")
        
        if xgb_r2 > 0.50 or lgbm_r2 > 0.50:
            print("\n🔴 WARNING: R² values are still high - check for remaining leakage")
        elif xgb_r2 > 0.15 or lgbm_r2 > 0.15:
            print("\n🟡 GOOD: R² values are in very good range")
        else:
            print("\n🟢 EXCELLENT: R² values are realistic for stock prediction")
    
    return correlations

def evaluate_models_by_ticker(results_df):
    """ประเมินผลแต่ละหุ้น"""
    tickers = results_df['Ticker'].unique()
    
    print(f"\n📊 Evaluation Results by Ticker (Total: {len(tickers)} Tickers)")
    
    ticker_metrics = {}
    
    for ticker in tickers:
        ticker_data = results_df[results_df['Ticker'] == ticker]
        
        if len(ticker_data) < 5:
            continue
            
        ticker_metrics[ticker] = {
            'price': {},
            'direction': {},
            'sample_size': len(ticker_data)
        }
        
        print(f"\n🔍 Evaluating {ticker} (Sample size: {len(ticker_data)})")
        
        # Price Model Evaluation
        print("\n📈 Price Model Evaluation:")
        for name, pred_col in [('XGBoost', 'XGBoost_Price_Pred'), 
                           ('LightGBM', 'LightGBM_Price_Pred')]:
            
            mae = mean_absolute_error(ticker_data['Actual_Price'], ticker_data[pred_col])
            rmse = np.sqrt(mean_squared_error(ticker_data['Actual_Price'], ticker_data[pred_col]))
            r2 = r2_score(ticker_data['Actual_Price'], ticker_data[pred_col])
            
            ticker_metrics[ticker]['price'][name] = {
                'MAE': mae,
                'RMSE': rmse,
                'R2': r2
            }
            
            print(f"{name} - MAE: {mae:.4f}, RMSE: {rmse:.4f}, R2: {r2:.4f}")
        
        # Direction Model Evaluation
        print("\n📊 Direction Model Evaluation:")
        for name, pred_col in [('XGBoost', 'XGBoost_Dir_Pred'),
                               ('RandomForest', 'RandomForest_Dir_Pred')]:
            
            if len(ticker_data['Actual_Direction'].unique()) < 2:
                print(f"{name} - Cannot evaluate (insufficient class diversity)")
                continue
                
            acc = accuracy_score(ticker_data['Actual_Direction'], ticker_data[pred_col])
            
            try:
                f1 = f1_score(ticker_data['Actual_Direction'], ticker_data[pred_col])
            except:
                f1 = float('nan')
            
            ticker_metrics[ticker]['direction'][name] = {
                'Accuracy': acc,
                'F1': f1
            }
            
            print(f"{name} - Accuracy: {acc:.4f}, F1: {f1:.4f}")
    
    # Find best models
    print("\n🏆 Best Performing Models by Ticker:")
    
    for ticker, metrics in ticker_metrics.items():
        print(f"\n{ticker} (Sample size: {metrics['sample_size']})")
        
        # Best price model (by RMSE)
        if metrics['price']:
            best_price_model = min(metrics['price'].items(), key=lambda x: x[1]['RMSE'])
            print(f"Best Price Model: {best_price_model[0]} - RMSE: {best_price_model[1]['RMSE']:.4f}")
        
        # Best direction model (by accuracy)
        if metrics['direction']:
            best_dir_model = max(metrics['direction'].items(), key=lambda x: x[1]['Accuracy'])
            print(f"Best Direction Model: {best_dir_model[0]} - Accuracy: {best_dir_model[1]['Accuracy']:.4f}")
    
    return ticker_metrics

def plot_results(results_df):
    """สร้างกราฟแสดงผล"""
    # 1. Scatter plot for price predictions
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    models = ['XGBoost_Price_Pred', 'LightGBM_Price_Pred']
    model_names = ['XGBoost', 'LightGBM']
    
    for i, (model, name) in enumerate(zip(models, model_names)):
        axes[i].scatter(results_df['Actual_Price'], results_df[model], alpha=0.6, s=20)
        
        # Perfect prediction line
        min_val = min(results_df['Actual_Price'].min(), results_df[model].min())
        max_val = max(results_df['Actual_Price'].max(), results_df[model].max())
        axes[i].plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8)
        
        axes[i].set_xlabel('Actual Price')
        axes[i].set_ylabel('Predicted Price')
        axes[i].set_title(f'{name} Price Predictions (Fixed)')
        axes[i].grid(True, alpha=0.3)
        
        # R²
        r2 = r2_score(results_df['Actual_Price'], results_df[model])
        color = 'green' if r2 < 0.3 else 'orange' if r2 < 0.5 else 'red'
        axes[i].text(0.05, 0.95, f'R² = {r2:.3f}', 
                    transform=axes[i].transAxes, fontsize=12,
                    bbox=dict(boxstyle='round', facecolor=color, alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('ensemble_price_predictions_fixed.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 2. Overall accuracy by model
    overall_xgb_acc = accuracy_score(results_df['Actual_Direction'], results_df['XGBoost_Dir_Pred'])
    overall_rf_acc = accuracy_score(results_df['Actual_Direction'], results_df['RandomForest_Dir_Pred'])
    
    plt.figure(figsize=(8, 6))
    models = ['XGBoost', 'RandomForest']
    accuracies = [overall_xgb_acc, overall_rf_acc]
    
    bars = plt.bar(models, accuracies, color=['skyblue', 'lightgreen'])
    plt.ylabel('Accuracy')
    plt.title('Direction Prediction Accuracy (Fixed)')
    plt.ylim(0, 1)
    
    # Add values on bars
    for bar, acc in zip(bars, accuracies):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{acc:.3f}', ha='center', va='bottom')
    
    plt.grid(True, alpha=0.3)
    plt.savefig('ensemble_direction_accuracy_fixed.png', dpi=300, bbox_inches='tight')
    plt.show()

def train_test_ensemble_fixed():
    """หลักสูตรการเทรนและเทสแบบไม่มี data leakage"""
    print("🚀 Fixed Ensemble Model Train/Test Evaluation (No Data Leakage)")
    
    try:
        # 1. โหลดข้อมูล
        print("📊 Loading data...")
        predictions_lstm = pd.read_csv("../LSTM_model/all_predictions_per_day_multi_task.csv")
        predictions_gru = pd.read_csv("../GRU_Model/all_predictions_per_day_multi_task.csv")
        print(f"✅ LSTM: {len(predictions_lstm)} rows")
        print(f"✅ GRU: {len(predictions_gru)} rows")
        
        # 2. สร้าง safe features (ไม่มี direct price predictions)
        ensemble_features = create_safe_features(predictions_lstm, predictions_gru)
        
        # 🔧 ลบ features ที่ยังมี leakage สูง
        print("🔧 Removing remaining high correlation features...")
        leakage_features = [
            'Price_Volatility_5d',     # correlation 0.65 - สูงเกินไป
            'Price_Pct_Change_5d',     # อาจมี leakage
            'Price_Pct_Change_1d'      # อาจมี leakage
        ]
        
        for col in leakage_features:
            if col in ensemble_features.columns:
                ensemble_features = ensemble_features.drop(col, axis=1)
                print(f"   Removed: {col}")
        
        # 3. จัดการ outliers (conservative)
        print("🧹 Handling outliers...")
        numeric_cols = ensemble_features.select_dtypes(include=[np.number]).columns
        ensemble_features = handle_outliers(ensemble_features, numeric_cols)
        
        # 4. แบ่ง train/test พร้อม gap
        print("📊 Splitting train/test data with gap...")
        train_cutoff = pd.Timestamp("2024-11-15")  # เพิ่ม gap 2 สัปดาห์
        test_start = pd.Timestamp("2024-12-01")
        
        train_mask = ensemble_features["Date"] < train_cutoff
        test_mask = ensemble_features["Date"] >= test_start
        
        print(f"Train samples: {train_mask.sum()}")
        print(f"Test samples: {test_mask.sum()}")
        print(f"Gap period: {train_cutoff.date()} to {test_start.date()}")
        
        # Check direction balance
        dir_balance = ensemble_features.loc[train_mask, "Actual_Direction"].value_counts(normalize=True)
        print(f"Direction Balance in Training Data: \n{dir_balance}")
        
        # 5. เตรียม features และ targets
        feature_cols = [col for col in ensemble_features.columns if col not in 
                       ["Ticker", "Date", "Actual_Price", "Actual_Direction"]]
        
        # Clean features
        feature_cols = clean_features(ensemble_features, feature_cols)
        
        # Split data
        X_train = ensemble_features.loc[train_mask, feature_cols].copy()
        y_train_price = ensemble_features.loc[train_mask, "Actual_Price"]
        y_train_dir = ensemble_features.loc[train_mask, "Actual_Direction"]
        
        X_test = ensemble_features.loc[test_mask, feature_cols].copy()
        y_test_price = ensemble_features.loc[test_mask, "Actual_Price"]
        y_test_dir = ensemble_features.loc[test_mask, "Actual_Direction"]
        
        print(f"Using {len(feature_cols)} features for training")
        
        # 6. Scale ข้อมูล
        print("⚖️ Scaling data...")
        scaler_features = StandardScaler()
        scaler_target = RobustScaler()
        
        X_train_scaled = pd.DataFrame(
            scaler_features.fit_transform(X_train),
            columns=X_train.columns,
            index=X_train.index
        )
        
        X_test_scaled = pd.DataFrame(
            scaler_features.transform(X_test),
            columns=X_test.columns,
            index=X_test.index
        )
        
        y_train_price_scaled = pd.Series(
            scaler_target.fit_transform(y_train_price.values.reshape(-1, 1)).ravel(),
            index=y_train_price.index
        )
        
        # 7. เทรนโมเดลด้วย regularization เพิ่มเติม
        print("🤖 Training models with stronger regularization...")
        
        # XGBoost Price Model (conservative parameters)
        print("Training XGBoost Price...")
        xgb_price_model = XGBRegressor(
            n_estimators=200,      # ลดลงมากขึ้น
            max_depth=3,           # ลดลงมากขึ้น
            learning_rate=0.03,    # ลดลงมากขึ้น
            subsample=0.6,         # ลดลงมากขึ้น
            colsample_bytree=0.6,  # ลดลงมากขึ้น
            reg_alpha=0.3,         # เพิ่ม regularization
            reg_lambda=0.3,        # เพิ่ม regularization
            random_state=42
        )
        xgb_price_model.fit(X_train_scaled, y_train_price_scaled)
        
        # LightGBM Price Model (conservative parameters)
        print("Training LightGBM Price...")
        lgbm_price_model = LGBMRegressor(
            n_estimators=200,
            max_depth=3,
            learning_rate=0.03,
            num_leaves=8,          # ลดลงมากขึ้น
            subsample=0.6,
            colsample_bytree=0.6,
            reg_alpha=0.3,
            reg_lambda=0.3,
            random_state=42,
            verbose=-1
        )
        lgbm_price_model.fit(X_train_scaled, y_train_price_scaled)
        
        # XGBoost Direction Model
        print("Training XGBoost Direction...")
        xgb_dir_model = XGBClassifier(
            n_estimators=200,
            max_depth=3,
            learning_rate=0.03,
            subsample=0.6,
            colsample_bytree=0.6,
            reg_alpha=0.3,
            reg_lambda=0.3,
            random_state=42
        )
        xgb_dir_model.fit(X_train_scaled, y_train_dir)
        
        # Random Forest Direction Model
        print("Training Random Forest Direction...")
        rf_dir_model = RandomForestClassifier(
            n_estimators=100,      # ลดลงมากขึ้น
            max_depth=5,           # ลดลงมากขึ้น
            min_samples_split=20,  # เพิ่มมากขึ้น
            min_samples_leaf=10,   # เพิ่มมากขึ้น
            max_features='sqrt',
            class_weight='balanced',
            random_state=42
        )
        rf_dir_model.fit(X_train_scaled, y_train_dir)
        
        # 8. ทำนาย
        print("🔮 Making predictions...")
        
        # Price predictions
        xgb_price_pred = xgb_price_model.predict(X_test_scaled)
        lgbm_price_pred = lgbm_price_model.predict(X_test_scaled)
        
        # Convert back to actual prices
        xgb_price_pred = scaler_target.inverse_transform(xgb_price_pred.reshape(-1, 1)).ravel()
        lgbm_price_pred = scaler_target.inverse_transform(lgbm_price_pred.reshape(-1, 1)).ravel()
        
        # Direction predictions
        xgb_dir_pred = xgb_dir_model.predict(X_test_scaled)
        rf_dir_pred = rf_dir_model.predict(X_test_scaled)
        
        xgb_dir_proba = xgb_dir_model.predict_proba(X_test_scaled)[:, 1]
        rf_dir_proba = rf_dir_model.predict_proba(X_test_scaled)[:, 1]
        
        # 9. สร้าง results DataFrame
        results_df = ensemble_features.loc[test_mask].copy()
        results_df['XGBoost_Price_Pred'] = xgb_price_pred
        results_df['LightGBM_Price_Pred'] = lgbm_price_pred
        results_df['XGBoost_Dir_Pred'] = xgb_dir_pred
        results_df['RandomForest_Dir_Pred'] = rf_dir_pred
        results_df['XGBoost_Dir_Proba'] = xgb_dir_proba
        results_df['RandomForest_Dir_Proba'] = rf_dir_proba
        
        # 10. ประเมินผล
        print("📊 Evaluating models...")
        ticker_metrics = evaluate_models_by_ticker(results_df)
        
        # 11. ตรวจสอบ data leakage (version ใหม่)
        print("\n" + "🔍" + " DATA LEAKAGE ANALYSIS (FIXED VERSION) " + "🔍")
        correlations = check_data_leakage_safe(ensemble_features, results_df)
        
        # 12. บันทึกผล
        print("\n💾 Saving results...")
        results_df.to_csv('ensemble_test_results_fixed.csv', index=False)
        
        # Save models for later use
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        joblib.dump(xgb_price_model, f'xgb_price_model_fixed_{timestamp}.pkl')
        joblib.dump(lgbm_price_model, f'lgbm_price_model_fixed_{timestamp}.pkl')
        joblib.dump(xgb_dir_model, f'xgb_dir_model_fixed_{timestamp}.pkl')
        joblib.dump(rf_dir_model, f'rf_dir_model_fixed_{timestamp}.pkl')
        joblib.dump(scaler_features, f'scaler_features_fixed_{timestamp}.pkl')
        joblib.dump(scaler_target, f'scaler_target_fixed_{timestamp}.pkl')
        
        print("📈 Generating visualization...")
        plot_results(results_df)
        print("\n" + "="*60)
        print("📝 FINAL SUMMARY")
        print("="*60)
                # 13. สรุปผลลัพธ์โดยรวม
        print("\n📈 Overall Price Prediction Metrics:")
        for name, pred_col in [('XGBoost', 'XGBoost_Price_Pred'), 
                             ('LightGBM', 'LightGBM_Price_Pred')]:
            mae = mean_absolute_error(results_df['Actual_Price'], results_df[pred_col])
            rmse = np.sqrt(mean_squared_error(results_df['Actual_Price'], results_df[pred_col]))
            r2 = r2_score(results_df['Actual_Price'], results_df[pred_col])
            print(f"{name}:")
            print(f"  MAE: {mae:.4f}")
            print(f"  RMSE: {rmse:.4f}")
            print(f"  R²: {r2:.4f}")

        # Overall direction metrics
        print("\n📊 Overall Direction Prediction Metrics:")
        for name, pred_col in [('XGBoost', 'XGBoost_Dir_Pred'),
                             ('RandomForest', 'RandomForest_Dir_Pred')]:
            acc = accuracy_score(results_df['Actual_Direction'], results_df[pred_col])
            f1 = f1_score(results_df['Actual_Direction'], results_df[pred_col], zero_division=0)
            auc = roc_auc_score(results_df['Actual_Direction'], results_df[f'{name}_Dir_Proba'])
            print(f"{name}:")
            print(f"  Accuracy: {acc:.4f}")
            print(f"  F1-Score: {f1:.4f}")
            print(f"  AUC-ROC: {auc:.4f}")

        # 14. Feature importance
        print("\n🔍 Feature Importance Analysis:")
        feature_importance_xgb = pd.DataFrame({
            'feature': feature_cols,
            'importance': xgb_price_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\nTop 10 Most Important Features (XGBoost Price Model):")
        print(feature_importance_xgb.head(10))

        # 15. Recommendations
        print("\n📋 Recommendations for Model Improvement:")
        if any(corr[1] > 0.5 for corr in correlations):
            print("- Consider removing features with high correlations to target (>0.5)")
        if len(feature_cols) < 10:
            print("- Consider adding more diverse features (current count too low)")
        if dir_balance.max() > 0.7:
            print("- Address class imbalance in direction labels (most common class >70%)")
        if r2_score(results_df['Actual_Price'], results_df['XGBoost_Price_Pred']) > 0.5:
            print("- Investigate potential remaining leakage (R² too high)")
        
        # 16. Return results
        return {
            'results_df': results_df,
            'ticker_metrics': ticker_metrics,
            'feature_importance': feature_importance_xgb,
            'models': {
                'xgb_price': xgb_price_model,
                'lgbm_price': lgbm_price_model,
                'xgb_dir': xgb_dir_model,
                'rf_dir': rf_dir_model
            },
            'scalers': {
                'features': scaler_features,
                'target': scaler_target
            }
        }

    except Exception as e:
        print(f"❌ Error occurred: {str(e)}")
        return None

if __name__ == "__main__":
    results = train_test_ensemble_fixed()
    if results:
        print("\n✅ Ensemble training and evaluation completed successfully!")
    else:
        print("\n❌ Ensemble training failed!")
        