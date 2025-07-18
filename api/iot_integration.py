"""
IoT Smart Meter Integration with AI/ML
For GSM-based remote energy monitoring system
"""

import json
import sqlite3
import requests
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import pickle
import logging
from flask import Flask, request, jsonify
import threading
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SmartMeterAI:
    def __init__(self, db_path="smart_meter.db"):
        self.db_path = db_path
        self.model = None
        self.scaler = None
        self.setup_database()
        self.load_or_train_model()
        
    def setup_database(self):
        """Setup SQLite database for meter readings"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create tables
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS meter_readings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                meter_id TEXT NOT NULL,
                timestamp DATETIME NOT NULL,
                voltage REAL,
                current REAL,
                power_kw REAL,
                energy_kwh REAL,
                frequency REAL,
                power_factor REAL,
                temperature REAL,
                signal_strength INTEGER,
                battery_level REAL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                meter_id TEXT NOT NULL,
                prediction_time DATETIME NOT NULL,
                predicted_power REAL,
                predicted_cost REAL,
                confidence_score REAL,
                model_used TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                meter_id TEXT NOT NULL,
                alert_type TEXT NOT NULL,
                message TEXT,
                severity TEXT,
                is_resolved BOOLEAN DEFAULT FALSE,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
        logger.info("Database setup completed")
    
    def receive_gsm_data(self, gsm_data):
        """Process incoming GSM data from smart meter"""
        try:
            # Parse GSM data (assuming JSON format)
            if isinstance(gsm_data, str):
                data = json.loads(gsm_data)
            else:
                data = gsm_data
            
            # Store in database
            self.store_reading(data)
            
            # Generate AI predictions
            prediction = self.predict_next_hour(data['meter_id'])
            
            # Check for anomalies
            self.check_anomalies(data)
            
            # Send response back to meter
            response = {
                'status': 'success',
                'prediction': prediction,
                'timestamp': datetime.now().isoformat(),
                'recommendations': self.get_recommendations(data)
            }
            
            return response
            
        except Exception as e:
            logger.error(f"Error processing GSM data: {e}")
            return {'status': 'error', 'message': str(e)}
    
    def store_reading(self, data):
        """Store meter reading in database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO meter_readings 
            (meter_id, timestamp, voltage, current, power_kw, energy_kwh, 
             frequency, power_factor, temperature, signal_strength, battery_level)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            data.get('meter_id'),
            data.get('timestamp', datetime.now()),
            data.get('voltage'),
            data.get('current'),
            data.get('power_kw'),
            data.get('energy_kwh'),
            data.get('frequency'),
            data.get('power_factor'),
            data.get('temperature'),
            data.get('signal_strength'),
            data.get('battery_level')
        ))
        
        conn.commit()
        conn.close()
        logger.info(f"Stored reading for meter {data.get('meter_id')}")
    
    def load_or_train_model(self):
        """Load existing model or train new one"""
        try:
            # Try to load existing model
            with open('smart_meter_model.pkl', 'rb') as f:
                self.model = pickle.load(f)
            logger.info("Loaded existing AI model")
        except FileNotFoundError:
            # Train new model with sample data
            self.train_initial_model()
            logger.info("Trained new AI model")
    
    def train_initial_model(self):
        """Train initial model with historical data"""
        # Load historical data from CSV (your AEP data)
        try:
            df = pd.read_csv('data/AEP_hourly.csv')
            df['Datetime'] = pd.to_datetime(df['Datetime'])
            
            # Create features
            df['hour'] = df['Datetime'].dt.hour
            df['dayofweek'] = df['Datetime'].dt.dayofweek
            df['month'] = df['Datetime'].dt.month
            df['is_weekend'] = (df['dayofweek'] >= 5).astype(int)
            
            # Train model
            features = ['hour', 'dayofweek', 'month', 'is_weekend']
            X = df[features]
            y = df['AEP_MW'] / 1000  # Convert to kW
            
            self.model = RandomForestRegressor(n_estimators=50, random_state=42)
            self.model.fit(X, y)
            
            # Save model
            with open('smart_meter_model.pkl', 'wb') as f:
                pickle.dump(self.model, f)
                
        except Exception as e:
            logger.error(f"Error training model: {e}")
    
    def predict_next_hour(self, meter_id):
        """Predict power consumption for next hour"""
        if not self.model:
            return None
            
        try:
            now = datetime.now()
            next_hour = now + timedelta(hours=1)
            
            # Create features for prediction
            features = [
                next_hour.hour,
                next_hour.weekday(),
                next_hour.month,
                1 if next_hour.weekday() >= 5 else 0
            ]
            
            # Make prediction
            prediction = self.model.predict([features])[0]
            
            # Store prediction
            self.store_prediction(meter_id, next_hour, prediction)
            
            return {
                'predicted_power_kw': round(prediction, 2),
                'predicted_cost': round(prediction * 0.12, 2),  # Assuming $0.12/kWh
                'prediction_time': next_hour.isoformat(),
                'confidence': 0.85  # You can calculate actual confidence
            }
            
        except Exception as e:
            logger.error(f"Error making prediction: {e}")
            return None
    
    def store_prediction(self, meter_id, prediction_time, predicted_power):
        """Store prediction in database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO predictions 
            (meter_id, prediction_time, predicted_power, predicted_cost, confidence_score, model_used)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            meter_id,
            prediction_time,
            predicted_power,
            predicted_power * 0.12,
            0.85,
            'RandomForest'
        ))
        
        conn.commit()
        conn.close()
    
    def check_anomalies(self, data):
        """Check for anomalies in meter readings"""
        try:
            power = data.get('power_kw', 0)
            voltage = data.get('voltage', 0)
            current = data.get('current', 0)
            
            alerts = []
            
            # Check for unusual power consumption
            if power > 10:  # Threshold for high power
                alerts.append({
                    'type': 'HIGH_POWER',
                    'message': f'High power consumption detected: {power} kW',
                    'severity': 'WARNING'
                })
            
            # Check for voltage issues
            if voltage < 200 or voltage > 250:
                alerts.append({
                    'type': 'VOLTAGE_ISSUE',
                    'message': f'Voltage out of range: {voltage}V',
                    'severity': 'CRITICAL'
                })
            
            # Check for low battery
            battery = data.get('battery_level', 100)
            if battery < 20:
                alerts.append({
                    'type': 'LOW_BATTERY',
                    'message': f'Low battery level: {battery}%',
                    'severity': 'WARNING'
                })
            
            # Store alerts
            for alert in alerts:
                self.store_alert(data.get('meter_id'), alert)
                
        except Exception as e:
            logger.error(f"Error checking anomalies: {e}")
    
    def store_alert(self, meter_id, alert):
        """Store alert in database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO alerts (meter_id, alert_type, message, severity)
            VALUES (?, ?, ?, ?)
        ''', (meter_id, alert['type'], alert['message'], alert['severity']))
        
        conn.commit()
        conn.close()
    
    def get_recommendations(self, data):
        """Generate energy efficiency recommendations"""
        recommendations = []
        
        power = data.get('power_kw', 0)
        hour = datetime.now().hour
        
        if power > 5 and 22 <= hour <= 6:  # High power at night
            recommendations.append("Consider shifting high-power activities to daytime for better rates")
        
        if data.get('power_factor', 1) < 0.8:
            recommendations.append("Poor power factor detected. Consider power factor correction")
        
        return recommendations
    
    def get_meter_dashboard_data(self, meter_id):
        """Get dashboard data for specific meter"""
        conn = sqlite3.connect(self.db_path)
        
        # Get recent readings
        recent_readings = pd.read_sql_query('''
            SELECT * FROM meter_readings 
            WHERE meter_id = ? 
            ORDER BY timestamp DESC 
            LIMIT 100
        ''', conn, params=(meter_id,))
        
        # Get recent predictions
        recent_predictions = pd.read_sql_query('''
            SELECT * FROM predictions 
            WHERE meter_id = ? 
            ORDER BY prediction_time DESC 
            LIMIT 24
        ''', conn, params=(meter_id,))
        
        # Get active alerts
        active_alerts = pd.read_sql_query('''
            SELECT * FROM alerts 
            WHERE meter_id = ? AND is_resolved = FALSE
            ORDER BY created_at DESC
        ''', conn, params=(meter_id,))
        
        conn.close()
        
        return {
            'readings': recent_readings.to_dict('records'),
            'predictions': recent_predictions.to_dict('records'),
            'alerts': active_alerts.to_dict('records')
        }

# Flask API for GSM communication
app = Flask(__name__)
smart_meter_ai = SmartMeterAI()

@app.route('/api/meter/data', methods=['POST'])
def receive_meter_data():
    """Endpoint to receive data from GSM meter"""
    try:
        data = request.get_json()
        response = smart_meter_ai.receive_gsm_data(data)
        return jsonify(response)
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/meter/<meter_id>/dashboard', methods=['GET'])
def get_meter_dashboard(meter_id):
    """Get dashboard data for specific meter"""
    try:
        data = smart_meter_ai.get_meter_dashboard_data(meter_id)
        return jsonify(data)
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/meter/<meter_id>/predict', methods=['GET'])
def get_prediction(meter_id):
    """Get next hour prediction for meter"""
    try:
        prediction = smart_meter_ai.predict_next_hour(meter_id)
        return jsonify(prediction)
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

# Simulation function for testing
def simulate_gsm_meter():
    """Simulate GSM meter sending data"""
    import random
    
    while True:
        # Simulate meter data
        meter_data = {
            'meter_id': 'METER_001',
            'timestamp': datetime.now().isoformat(),
            'voltage': random.uniform(220, 240),
            'current': random.uniform(1, 15),
            'power_kw': random.uniform(0.5, 8),
            'energy_kwh': random.uniform(100, 500),
            'frequency': random.uniform(49.5, 50.5),
            'power_factor': random.uniform(0.7, 1.0),
            'temperature': random.uniform(20, 35),
            'signal_strength': random.randint(-80, -40),
            'battery_level': random.uniform(20, 100)
        }
        
        # Send to AI system
        response = smart_meter_ai.receive_gsm_data(meter_data)
        logger.info(f"Simulated meter data processed: {response}")
        
        time.sleep(60)  # Send data every minute

if __name__ == "__main__":
    # Start simulation in background thread
    simulation_thread = threading.Thread(target=simulate_gsm_meter, daemon=True)
    simulation_thread.start()
    
    # Start Flask API
    app.run(host='0.0.0.0', port=5000, debug=True)