"""
Configuration settings for Smart Meter AI project
"""

import os
from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent

# Database Configuration
DATABASE = {
    'type': 'sqlite',
    'path': PROJECT_ROOT / 'data' / 'smart_meter.db',
    'backup_path': PROJECT_ROOT / 'data' / 'backup'
}

# API Configuration
API = {
    'host': '0.0.0.0',
    'port': 5000,
    'debug': True,
    'cors_enabled': True
}

# GSM Configuration
GSM = {
    'apn': 'internet',
    'timeout': 30,
    'retry_attempts': 3,
    'data_interval': 60  # seconds
}

# AI/ML Configuration
ML = {
    'model_path': PROJECT_ROOT / 'models',
    'retrain_interval': 24,  # hours
    'prediction_horizon': 24,  # hours
    'confidence_threshold': 0.8
}

# Alert Configuration
ALERTS = {
    'high_power_threshold': 5.0,  # kW
    'voltage_min': 200,  # V
    'voltage_max': 250,  # V
    'low_battery_threshold': 20,  # %
    'email_enabled': False,
    'sms_enabled': False
}

# Energy Pricing
PRICING = {
    'rate_per_kwh': 0.12,  # USD
    'peak_hours': [18, 19, 20, 21],
    'peak_multiplier': 1.5,
    'currency': 'USD'
}

# Hardware Configuration
HARDWARE = {
    'meter_id_prefix': 'METER_',
    'sensor_calibration': {
        'voltage_multiplier': 1.0,
        'current_multiplier': 1.0,
        'power_factor': 0.9
    },
    'display_modes': 3,
    'lcd_refresh_rate': 2  # seconds
}

# Logging Configuration
LOGGING = {
    'level': 'INFO',
    'file_path': PROJECT_ROOT / 'logs' / 'smart_meter.log',
    'max_file_size': 10,  # MB
    'backup_count': 5
}

# Security Configuration
SECURITY = {
    'api_key_required': True,
    'encryption_enabled': True,
    'max_login_attempts': 3,
    'session_timeout': 3600  # seconds
}