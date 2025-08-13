import mysql.connector
import os
import pandas as pd
import numpy as np
import requests
from dotenv import load_dotenv
import logging

# ✅ Enhanced Trading System ที่ใช้ข้อมูลจากโมเดลที่แก้ไขแล้ว

class EnhancedTradingSystem:
    """ระบบเทรดที่ใช้ข้อมูลจาก Raw Model Data อย่างปลอดภัย"""
    
    def __init__(self, api: 'InnovestXAPI', capital, max_risk_per_trade=0.01, max_positions=5):
        self.api = api
        self.capital = capital
        self.cash = capital
        self.max_risk_per_trade = max_risk_per_trade
        self.max_positions = max_positions
        self.positions = {}
        
        # ✅ Enhanced Risk Management Parameters
        self.min_confidence = 0.6  # ความเชื่อมั่นขั้นต่ำ
        self.min_consistency = 80.0  # consistency ขั้นต่ำ (%)
        self.max_position_risk = 0.05  # ความเสี่ยงสูงสุดต่อตำแหน่ง
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def fetch_enhanced_stock_data(self):
        """ดึงข้อมูลหุ้นพร้อม enhanced columns จากโมเดลที่แก้ไขแล้ว"""
        conn = None
        try:
            conn = mysql.connector.connect(**DB_CONFIG)
            cursor = conn.cursor(dictionary=True)
            
            # ✅ ดึงข้อมูล enhanced columns ด้วย
            query = """
            SELECT 
                Date, StockSymbol, ClosePrice, HighPrice, LowPrice,
                PredictionClose_Ensemble, PredictionTrend_Ensemble,
                XGB_Confidence, Risk_Level, Is_Inconsistent, 
                Suggested_Action, Reliability_Warning, Ensemble_Method,
                Price_Change_Percent, Raw_Prediction_Used
            FROM StockDetail
            WHERE Date = (SELECT MAX(Date) FROM StockDetail)
            AND PredictionClose_Ensemble IS NOT NULL
            AND PredictionTrend_Ensemble IS NOT NULL
            ORDER BY XGB_Confidence DESC
            """
            
            self.logger.info("🔍 กำลังดึงข้อมูลหุ้นล่าสุดพร้อม enhanced data...")
            cursor.execute(query)
            data = cursor.fetchall()
            
            if not data:
                self.logger.warning("❌ ไม่พบข้อมูลการทำนายล่าสุด")
                return pd.DataFrame()
                
            df = pd.DataFrame(data)
            self.logger.info(f"✅ ดึงข้อมูลสำเร็จ: {len(df)} หุ้น")
            
            # แปลงประเภทข้อมูล
            numeric_cols = ['ClosePrice', 'HighPrice', 'LowPrice', 'PredictionClose_Ensemble', 
                          'XGB_Confidence', 'Price_Change_Percent']
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            return df
            
        except mysql.connector.Error as e:
            self.logger.error(f"❌ เกิดข้อผิดพลาดในการเชื่อมต่อฐานข้อมูล: {e}")
            return pd.DataFrame()
        finally:
            if conn and conn.is_connected():
                cursor.close()
                conn.close()

    def filter_safe_stocks(self, df):
        """กรองหุ้นที่ปลอดภัยสำหรับการเทรดตามหลักการ Raw Model"""
        if df.empty:
            return df
        
        self.logger.info("🔍 กำลังกรองหุ้นที่ปลอดภัยสำหรับการเทรด...")
        
        # ✅ เงื่อนไขความปลอดภัยตาม enhanced model
        safe_conditions = (
            # 1. ต้องมีสัญญาณซื้อ
            (df['PredictionTrend_Ensemble'] == 1) &
            
            # 2. ความเชื่อมั่นสูงเพียงพอ
            (df['XGB_Confidence'] >= self.min_confidence) &
            
            # 3. ไม่มี inconsistency (ถ้ามีข้อมูล)
            (df['Is_Inconsistent'].fillna(False) == False) &
            
            # 4. Action ไม่ใช่ AVOID หรือ EXERCISE_EXTREME_CAUTION
            (~df['Suggested_Action'].isin(['AVOID', 'EXERCISE_EXTREME_CAUTION'])) &
            
            # 5. Risk Level ไม่ใช่ HIGH_RISK
            (~df['Risk_Level'].str.contains('HIGH_RISK', na=False)) &
            
            # 6. ราคาปัจจุบันสมเหตุสมผล (> 0)
            (df['ClosePrice'] > 0) &
            
            # 7. การเปลี่ยนแปลงราคาไม่สูงเกินไป (< 10%)
            (df['Price_Change_Percent'].abs() < 10.0)
        )
        
        safe_stocks = df[safe_conditions].copy()
        
        # ✅ เพิ่มการคำนวณ ATR สำหรับ risk management
        safe_stocks = self.calculate_atr_for_stocks(safe_stocks)
        
        # ✅ จัดเรียงตาม confidence และ risk level
        if not safe_stocks.empty:
            safe_stocks['Risk_Score'] = self.calculate_risk_score(safe_stocks)
            safe_stocks = safe_stocks.sort_values(['Risk_Score', 'XGB_Confidence'], 
                                                 ascending=[True, False])
        
        self.logger.info(f"📊 ผลการกรอง:")
        self.logger.info(f"   📈 หุ้นทั้งหมด: {len(df)}")
        self.logger.info(f"   ✅ หุ้นที่ปลอดภัย: {len(safe_stocks)}")
        self.logger.info(f"   🚨 หุ้นที่กรองออก: {len(df) - len(safe_stocks)}")
        
        # แสดงรายละเอียดหุ้นที่กรองออก
        if len(df) > len(safe_stocks):
            filtered_out = df[~safe_conditions]
            self.logger.warning("⚠️ หุ้นที่ถูกกรองออก:")
            for _, row in filtered_out.iterrows():
                reasons = []
                if row['PredictionTrend_Ensemble'] != 1:
                    reasons.append("ไม่มีสัญญาณซื้อ")
                if row['XGB_Confidence'] < self.min_confidence:
                    reasons.append(f"Confidence ต่ำ ({row['XGB_Confidence']:.3f})")
                if row['Is_Inconsistent']:
                    reasons.append("มี Inconsistency")
                if row['Suggested_Action'] in ['AVOID', 'EXERCISE_EXTREME_CAUTION']:
                    reasons.append(f"Action: {row['Suggested_Action']}")
                
                self.logger.warning(f"     {row['StockSymbol']}: {', '.join(reasons)}")
        
        return safe_stocks

    def calculate_atr_for_stocks(self, df):
        """คำนวณ ATR สำหรับการจัดการความเสี่ยง"""
        # ใช้ข้อมูลย้อนหลังจากฐานข้อมูลเพื่อคำนวณ ATR
        conn = None
        try:
            conn = mysql.connector.connect(**DB_CONFIG)
            cursor = conn.cursor(dictionary=True)
            
            symbols = "','".join(df['StockSymbol'].tolist())
            atr_query = f"""
            SELECT StockSymbol, 
                   AVG(HighPrice - LowPrice) as ATR_estimate
            FROM StockDetail 
            WHERE StockSymbol IN ('{symbols}')
            AND Date >= CURDATE() - INTERVAL 14 DAY
            GROUP BY StockSymbol
            """
            
            cursor.execute(atr_query)
            atr_data = cursor.fetchall()
            atr_df = pd.DataFrame(atr_data)
            
            if not atr_df.empty:
                df = df.merge(atr_df, on='StockSymbol', how='left')
                df['ATR'] = df['ATR_estimate'].fillna(df['ClosePrice'] * 0.02)  # 2% fallback
            else:
                df['ATR'] = df['ClosePrice'] * 0.02  # 2% fallback
                
        except Exception as e:
            self.logger.warning(f"⚠️ ไม่สามารถคำนวณ ATR ได้: {e}, ใช้ 2% ของราคาแทน")
            df['ATR'] = df['ClosePrice'] * 0.02
        finally:
            if conn and conn.is_connected():
                cursor.close()
                conn.close()
        
        return df

    def calculate_risk_score(self, df):
        """คำนวณคะแนนความเสี่ยงรวม"""
        risk_scores = []
        
        for _, row in df.iterrows():
            score = 0
            
            # ลดคะแนนตาม confidence (confidence สูง = risk ต่ำ)
            score += (1 - row['XGB_Confidence']) * 50
            
            # เพิ่มคะแนนตาม volatility
            if 'Price_Change_Percent' in row:
                score += abs(row['Price_Change_Percent']) * 2
            
            # เพิ่มคะแนนตาม risk level
            if 'Risk_Level' in row and pd.notna(row['Risk_Level']):
                if 'MEDIUM_RISK' in str(row['Risk_Level']):
                    score += 25
                elif 'HIGH_RISK' in str(row['Risk_Level']):
                    score += 100  # จะถูกกรองออกแล้ว แต่เผื่อไว้
            
            risk_scores.append(score)
        
        return risk_scores

    def calculate_position_size(self, stock_data):
        """คำนวณขนาดตำแหน่งตาม Kelly Criterion และ Risk Management"""
        current_price = stock_data['ClosePrice']
        predicted_price = stock_data['PredictionClose_Ensemble']
        confidence = stock_data['XGB_Confidence']
        atr = stock_data['ATR']
        
        # คำนวณ potential profit และ risk
        expected_return = (predicted_price - current_price) / current_price
        stop_loss_price = current_price - (atr * 2)  # 2 ATR stop loss
        risk_per_share = current_price - stop_loss_price
        
        if risk_per_share <= 0 or expected_return <= 0:
            return 0
        
        # Kelly Criterion แบบ conservative
        win_probability = confidence  # ใช้ model confidence เป็น probability
        avg_win = expected_return
        avg_loss = risk_per_share / current_price
        
        kelly_fraction = (win_probability * avg_win - (1 - win_probability) * avg_loss) / avg_win
        kelly_fraction = max(0, min(kelly_fraction * 0.5, 0.1))  # จำกัดไว้ที่ 10% และลด Kelly ลง 50%
        
        # คำนวณจำนวนหุ้น
        position_value = self.cash * kelly_fraction
        max_risk_value = self.cash * self.max_risk_per_trade
        position_value = min(position_value, max_risk_value)
        
        quantity = int(position_value / current_price)
        
        self.logger.info(f"📊 Position sizing for {stock_data['StockSymbol']}:")
        self.logger.info(f"   💰 Current Price: {current_price:.2f}")
        self.logger.info(f"   🎯 Predicted Price: {predicted_price:.2f}")
        self.logger.info(f"   📈 Expected Return: {expected_return:.2%}")
        self.logger.info(f"   🛡️ Stop Loss: {stop_loss_price:.2f}")
        self.logger.info(f"   🎯 Confidence: {confidence:.3f}")
        self.logger.info(f"   📊 Kelly Fraction: {kelly_fraction:.3%}")
        self.logger.info(f"   📦 Quantity: {quantity}")
        
        return quantity

    def execute_enhanced_trading(self):
        """ดำเนินการเทรดตามข้อมูลจากโมเดลที่แก้ไขแล้ว"""
        self.logger.info("🚀 เริ่มต้นการเทรดด้วย Enhanced Trading System")
        
        # 1. ดึงข้อมูลหุ้น
        stock_data = self.fetch_enhanced_stock_data()
        if stock_data.empty:
            self.logger.warning("❌ ไม่มีข้อมูลหุ้นสำหรับการเทรด")
            return
        
        # 2. กรองหุ้นที่ปลอดภัย
        safe_stocks = self.filter_safe_stocks(stock_data)
        if safe_stocks.empty:
            self.logger.warning("⚠️ ไม่มีหุ้นที่ปลอดภัยสำหรับการเทรด")
            return
        
        # 3. จำกัดจำนวนหุ้นตาม max_positions
        safe_stocks = safe_stocks.head(self.max_positions)
        
        self.logger.info(f"📈 หุ้นที่ผ่านการกรอง: {len(safe_stocks)} หุ้น")
        
        # 4. ดำเนินการซื้อ
        successful_trades = 0
        for _, stock in safe_stocks.iterrows():
            if len(self.positions) >= self.max_positions:
                break
                
            if self.enter_enhanced_position(stock):
                successful_trades += 1
        
        self.logger.info(f"✅ ดำเนินการเทรดสำเร็จ: {successful_trades}/{len(safe_stocks)} หุ้น")
        self.print_portfolio()

    def enter_enhanced_position(self, stock_data):
        """เข้าตำแหน่งตามข้อมูลจากโมเดลที่แก้ไขแล้ว"""
        symbol = stock_data['StockSymbol']
        
        if symbol in self.positions:
            self.logger.warning(f"⚠️ มีตำแหน่ง {symbol} อยู่แล้ว")
            return False
        
        # คำนวณขนาดตำแหน่ง
        quantity = self.calculate_position_size(stock_data)
        
        if quantity <= 0:
            self.logger.warning(f"⚠️ ไม่สามารถคำนวณขนาดตำแหน่งสำหรับ {symbol}")
            return False
        
        current_price = stock_data['ClosePrice']
        atr = stock_data['ATR']
        
        # กำหนด stop loss และ take profit
        stop_loss = current_price - (atr * 2)
        take_profit = stock_data['PredictionClose_Ensemble']
        
        # ตรวจสอบ risk-reward ratio
        risk = current_price - stop_loss
        reward = take_profit - current_price
        risk_reward_ratio = reward / risk if risk > 0 else 0
        
        if risk_reward_ratio < 1.5:  # ต้องการ risk:reward อย่างน้อย 1:1.5
            self.logger.warning(f"⚠️ Risk-reward ratio ต่ำเกินไปสำหรับ {symbol}: {risk_reward_ratio:.2f}")
            return False
        
        # ดำเนินการซื้อ
        self.logger.info(f"📈 กำลังเข้าตำแหน่ง {symbol}:")
        self.logger.info(f"   📦 จำนวน: {quantity} หุ้น")
        self.logger.info(f"   💰 ราคา: {current_price:.2f}")
        self.logger.info(f"   🛡️ Stop Loss: {stop_loss:.2f}")
        self.logger.info(f"   🎯 Take Profit: {take_profit:.2f}")
        self.logger.info(f"   📊 Risk:Reward = 1:{risk_reward_ratio:.2f}")
        
        if self.execute_trade(symbol, quantity, "buy", current_price):
            self.positions[symbol] = {
                'quantity': quantity,
                'entry_price': current_price,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'confidence': stock_data['XGB_Confidence'],
                'predicted_price': stock_data['PredictionClose_Ensemble'],
                'risk_level': stock_data.get('Risk_Level', 'UNKNOWN'),
                'entry_time': pd.Timestamp.now()
            }
            return True
        
        return False

    def execute_trade(self, symbol, quantity, order_type, price):
        """ดำเนินการซื้อขาย (เชื่อมต่อกับ API หรือ simulation)"""
        # ใช้ current_price จากตลาดจริง
        current_price = self.api.get_price(symbol)
        if current_price <= 0:
            current_price = price  # fallback
        
        self.logger.info(f"🔄 ดำเนินการ {order_type} {symbol}: {quantity} หุ้น @ {current_price:.2f}")
        
        # สำหรับ simulation mode
        if hasattr(self.api, 'simulation_mode') and self.api.simulation_mode:
            trade_value = quantity * current_price
            if order_type == "buy":
                if self.cash >= trade_value:
                    self.cash -= trade_value
                    self.logger.info(f"✅ [SIM] ซื้อ {symbol} สำเร็จ")
                    return True
                else:
                    self.logger.error(f"❌ [SIM] เงินสดไม่เพียงพอสำหรับ {symbol}")
                    return False
            elif order_type == "sell":
                self.cash += trade_value
                self.logger.info(f"✅ [SIM] ขาย {symbol} สำเร็จ")
                return True
        else:
            # ใช้ API จริง
            order_result = self.api.place_order(symbol, quantity, order_type, current_price)
            if order_result:
                trade_value = quantity * current_price
                if order_type == "buy":
                    self.cash -= trade_value
                elif order_type == "sell":
                    self.cash += trade_value
                return True
        
        return False

    def print_portfolio(self):
        """แสดงสถานะพอร์ตโดยละเอียด"""
        self.logger.info("\n📊 สถานะพอร์ต Enhanced Trading System:")
        self.logger.info("=" * 80)
        
        if not self.positions:
            self.logger.info("📭 ไม่มีตำแหน่งการลงทุน")
            self.logger.info(f"💵 เงินสดทั้งหมด: {self.cash:,.2f} บาท")
            return
        
        total_value = 0
        total_pnl = 0
        
        for symbol, pos in self.positions.items():
            current_price = self.api.get_price(symbol)
            if current_price <= 0:
                current_price = pos['entry_price']
            
            value = pos['quantity'] * current_price
            total_value += value
            pnl = (current_price - pos['entry_price']) * pos['quantity']
            total_pnl += pnl
            pnl_pct = ((current_price / pos['entry_price']) - 1) * 100
            
            self.logger.info(f"📈 {symbol}:")
            self.logger.info(f"   📦 จำนวน: {pos['quantity']} หุ้น")
            self.logger.info(f"   💰 ราคาเข้า: {pos['entry_price']:.2f} | ปัจจุบัน: {current_price:.2f}")
            self.logger.info(f"   🎯 เป้าหมาย: {pos['take_profit']:.2f} | Stop Loss: {pos['stop_loss']:.2f}")
            self.logger.info(f"   📊 มูลค่า: {value:,.2f} | P&L: {pnl:+,.2f} ({pnl_pct:+.2f}%)")
            self.logger.info(f"   🎯 Confidence: {pos['confidence']:.3f} | Risk: {pos['risk_level']}")
            self.logger.info(f"   ⏰ เข้าตำแหน่ง: {pos['entry_time'].strftime('%Y-%m-%d %H:%M')}")
            self.logger.info("-" * 60)
        
        total_portfolio_value = total_value + self.cash
        total_return_pct = ((total_portfolio_value / self.capital) - 1) * 100
        
        self.logger.info("💼 สรุปพอร์ต:")
        self.logger.info(f"   💰 มูลค่าหุ้นรวม: {total_value:,.2f} บาท")
        self.logger.info(f"   💵 เงินสดคงเหลือ: {self.cash:,.2f} บาท")
        self.logger.info(f"   🏦 มูลค่ารวมทั้งหมด: {total_portfolio_value:,.2f} บาท")
        self.logger.info(f"   📈 กำไร/ขาดทุนรวม: {total_pnl:+,.2f} บาท ({total_return_pct:+.2f}%)")
        self.logger.info("=" * 80)

# ✅ Simulation API สำหรับทดสอบ
class SimulationAPI:
    """API จำลองสำหรับทดสอบระบบ"""
    def __init__(self):
        self.simulation_mode = True
        self.logger = logging.getLogger(__name__)
    
    def get_price(self, symbol):
        """จำลองการดึงราคา (ใช้ราคาจากฐานข้อมูล)"""
        try:
            conn = mysql.connector.connect(**DB_CONFIG)
            cursor = conn.cursor()
            cursor.execute(
                "SELECT ClosePrice FROM StockDetail WHERE StockSymbol = %s ORDER BY Date DESC LIMIT 1",
                (symbol,)
            )
            result = cursor.fetchone()
            return float(result[0]) if result else 0
        except:
            return 0
        finally:
            if conn:
                conn.close()
    
    def place_order(self, symbol, quantity, order_type, price=None):
        """จำลองการส่งคำสั่งซื้อขาย"""
        self.logger.info(f"[SIM] คำสั่ง {order_type}: {symbol} {quantity} หุ้น @ {price:.2f}")
        return {"status": "success", "order_id": f"SIM_{symbol}_{pd.Timestamp.now().strftime('%H%M%S')}"}

# ✅ ฟังก์ชันหลักสำหรับรันระบบ
def run_enhanced_trading_system(capital=1000000, simulation=True):
    """รันระบบเทรดแบบ Enhanced"""
    
    if simulation:
        api = SimulationAPI()
        print("🎮 รันในโหมด Simulation")
    else:
        if not INNOVESTX_API_KEY:
            print("❌ ไม่พบ INNOVESTX_API_KEY สำหรับการเทรดจริง")
            return
        api = InnovestXAPI(INNOVESTX_API_KEY, INNOVESTX_API_URL)
        print("💰 รันในโหมดการเทรดจริง")
    
    # สร้างระบบเทรด
    trading_system = EnhancedTradingSystem(
        api=api,
        capital=capital,
        max_risk_per_trade=0.02,  # เสี่ยง 2% ต่อการเทรด
        max_positions=5           # เก็บได้สูงสุด 5 ตำแหน่ง
    )
    
    # ดำเนินการเทรด
    trading_system.execute_enhanced_trading()
    
    return trading_system

# ✅ Main function
def main():
    print("🚀 Enhanced Trading System - ใช้ข้อมูลจาก Raw Model Data")
    print("=" * 60)
    
    # รันในโหมด simulation ก่อน
    trading_system = run_enhanced_trading_system(
        capital=1000000,
        simulation=True
    )
    
    if trading_system and trading_system.positions:
        print("\n🎯 ระบบพร้อมสำหรับการเทรดจริง!")
        print("💡 เปลี่ยน simulation=False ในการรันครั้งถัดไปเพื่อเทรดจริง")
    else:
        print("\n⚠️ ไม่มีโอกาสการเทรดในวันนี้")

if __name__ == "__main__":
    main()