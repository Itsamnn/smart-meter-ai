"""
Basic tests for Smart Meter AI project
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

class TestDataLoading:
    """Test data loading functionality"""
    
    def test_data_file_exists(self):
        """Test if the main data file exists"""
        data_file = project_root / 'data' / 'AEP_hourly.csv'
        assert data_file.exists(), "AEP_hourly.csv file not found"
    
    def test_data_loading(self):
        """Test data loading and basic structure"""
        data_file = project_root / 'data' / 'AEP_hourly.csv'
        df = pd.read_csv(data_file)
        
        # Check if data is not empty
        assert len(df) > 0, "Data file is empty"
        
        # Check required columns
        required_columns = ['Datetime', 'AEP_MW']
        for col in required_columns:
            assert col in df.columns, f"Required column {col} not found"
        
        # Check data types
        df['Datetime'] = pd.to_datetime(df['Datetime'])
        assert df['AEP_MW'].dtype in [np.float64, np.int64], "AEP_MW should be numeric"

class TestFeatureEngineering:
    """Test feature engineering functions"""
    
    def setup_method(self):
        """Setup test data"""
        self.df = pd.DataFrame({
            'Datetime': pd.date_range('2024-01-01', periods=100, freq='h'),
            'AEP_MW': np.random.uniform(1000, 2000, 100)
        })
    
    def test_time_features(self):
        """Test time-based feature creation"""
        # Add time features
        self.df['hour'] = self.df['Datetime'].dt.hour
        self.df['dayofweek'] = self.df['Datetime'].dt.dayofweek
        self.df['month'] = self.df['Datetime'].dt.month
        
        # Check ranges
        assert self.df['hour'].min() >= 0 and self.df['hour'].max() <= 23
        assert self.df['dayofweek'].min() >= 0 and self.df['dayofweek'].max() <= 6
        assert self.df['month'].min() >= 1 and self.df['month'].max() <= 12
    
    def test_weekend_feature(self):
        """Test weekend feature creation"""
        self.df['dayofweek'] = self.df['Datetime'].dt.dayofweek
        self.df['is_weekend'] = (self.df['dayofweek'] >= 5).astype(int)
        
        # Check values are 0 or 1
        assert set(self.df['is_weekend'].unique()).issubset({0, 1})

class TestModelFunctions:
    """Test ML model related functions"""
    
    def test_model_import(self):
        """Test if required ML libraries can be imported"""
        try:
            from sklearn.ensemble import RandomForestRegressor
            from sklearn.metrics import mean_absolute_error, r2_score
            import xgboost as xgb
        except ImportError as e:
            pytest.fail(f"Required ML library import failed: {e}")
    
    def test_basic_model_training(self):
        """Test basic model training"""
        from sklearn.ensemble import RandomForestRegressor
        
        # Create sample data
        X = np.random.rand(100, 4)
        y = np.random.rand(100)
        
        # Train model
        model = RandomForestRegressor(n_estimators=10, random_state=42)
        model.fit(X, y)
        
        # Make prediction
        prediction = model.predict(X[:1])
        assert len(prediction) == 1, "Prediction should return single value"

class TestAPIFunctions:
    """Test API related functions"""
    
    def test_json_creation(self):
        """Test JSON data structure creation"""
        import json
        
        sample_data = {
            'meter_id': 'METER_001',
            'timestamp': '2024-01-15T10:30:00Z',
            'voltage': 230.5,
            'current': 8.2,
            'power_kw': 1.89
        }
        
        # Test JSON serialization
        json_str = json.dumps(sample_data)
        parsed_data = json.loads(json_str)
        
        assert parsed_data['meter_id'] == 'METER_001'
        assert parsed_data['voltage'] == 230.5

class TestConfiguration:
    """Test configuration and settings"""
    
    def test_config_import(self):
        """Test if configuration can be imported"""
        try:
            from config.settings import DATABASE, API, ML
            assert 'type' in DATABASE
            assert 'host' in API
            assert 'model_path' in ML
        except ImportError:
            pytest.fail("Configuration import failed")

if __name__ == "__main__":
    pytest.main([__file__])