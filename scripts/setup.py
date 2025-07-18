#!/usr/bin/env python3
"""
Setup script for Smart Meter AI project
Initializes database, creates directories, and sets up environment
"""

import os
import sys
import sqlite3
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from config.settings import DATABASE, LOGGING

def create_directories():
    """Create necessary project directories"""
    directories = [
        'data',
        'logs',
        'models',
        'temp',
        'data/backup'
    ]
    
    for directory in directories:
        dir_path = project_root / directory
        dir_path.mkdir(parents=True, exist_ok=True)
        print(f"✓ Created directory: {directory}")

def setup_database():
    """Initialize SQLite database with required tables"""
    db_path = DATABASE['path']
    
    # Ensure data directory exists
    db_path.parent.mkdir(parents=True, exist_ok=True)
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Create tables
    tables = [
        '''
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
        ''',
        '''
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
        ''',
        '''
        CREATE TABLE IF NOT EXISTS alerts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            meter_id TEXT NOT NULL,
            alert_type TEXT NOT NULL,
            message TEXT,
            severity TEXT,
            is_resolved BOOLEAN DEFAULT FALSE,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
        ''',
        '''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            role TEXT DEFAULT 'user',
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
        ''',
        '''
        CREATE TABLE IF NOT EXISTS meters (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            meter_id TEXT UNIQUE NOT NULL,
            location TEXT,
            user_id INTEGER,
            installation_date DATE,
            status TEXT DEFAULT 'active',
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
        '''
    ]
    
    for table_sql in tables:
        cursor.execute(table_sql)
        print("✓ Created database table")
    
    # Create indexes for better performance
    indexes = [
        'CREATE INDEX IF NOT EXISTS idx_meter_readings_timestamp ON meter_readings(timestamp)',
        'CREATE INDEX IF NOT EXISTS idx_meter_readings_meter_id ON meter_readings(meter_id)',
        'CREATE INDEX IF NOT EXISTS idx_predictions_meter_id ON predictions(meter_id)',
        'CREATE INDEX IF NOT EXISTS idx_alerts_meter_id ON alerts(meter_id)'
    ]
    
    for index_sql in indexes:
        cursor.execute(index_sql)
        print("✓ Created database index")
    
    conn.commit()
    conn.close()
    print(f"✓ Database initialized: {db_path}")

def create_sample_data():
    """Create sample data for testing"""
    conn = sqlite3.connect(DATABASE['path'])
    cursor = conn.cursor()
    
    # Insert sample meter
    cursor.execute('''
        INSERT OR IGNORE INTO meters (meter_id, location, status)
        VALUES ('METER_001', 'Test Location', 'active')
    ''')
    
    print("✓ Sample data created")
    conn.commit()
    conn.close()

def setup_logging():
    """Setup logging configuration"""
    log_dir = LOGGING['file_path'].parent
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Create empty log file
    LOGGING['file_path'].touch()
    print(f"✓ Logging setup: {LOGGING['file_path']}")

def main():
    """Main setup function"""
    print("🚀 Setting up Smart Meter AI project...")
    print("=" * 50)
    
    try:
        create_directories()
        setup_database()
        create_sample_data()
        setup_logging()
        
        print("=" * 50)
        print("✅ Setup completed successfully!")
        print("\nNext steps:")
        print("1. Install dependencies: pip install -r requirements.txt")
        print("2. Run the dashboard: streamlit run src/enhanced_app.py")
        print("3. Start API server: python api/iot_integration.py")
        print("4. Flash Arduino code to your hardware")
        
    except Exception as e:
        print(f"❌ Setup failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()