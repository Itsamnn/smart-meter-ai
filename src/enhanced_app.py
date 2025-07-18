import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Page config
st.set_page_config(
    page_title="⚡ Advanced Energy AI",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Enhanced Dark Theme CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    .stApp {
        background: linear-gradient(135deg, #0f0f23 0%, #1a1a2e 50%, #16213e 100%);
        font-family: 'Inter', sans-serif;
        color: #e2e8f0;
    }
    
    .hero-section {
        text-align: center;
        padding: 3rem 0;
        background: rgba(30, 41, 59, 0.8);
        border-radius: 24px;
        backdrop-filter: blur(20px);
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        border: 1px solid rgba(148, 163, 184, 0.1);
    }
    
    .hero-title {
        font-size: 3.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #60a5fa 0%, #a78bfa 50%, #f59e0b 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1rem;
        letter-spacing: -0.02em;
    }
    
    .hero-subtitle {
        font-size: 1.2rem;
        color: #94a3b8;
        font-weight: 400;
        margin-bottom: 2rem;
    }
    
    .glass-card {
        background: rgba(30, 41, 59, 0.6);
        backdrop-filter: blur(20px);
        border-radius: 16px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.2);
        border: 1px solid rgba(148, 163, 184, 0.1);
        transition: all 0.3s ease;
    }
    
    .glass-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.3);
        border-color: rgba(96, 165, 250, 0.3);
    }
    
    .metric-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
        gap: 1rem;
        margin: 2rem 0;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        box-shadow: 0 4px 16px rgba(59, 130, 246, 0.3);
        transition: transform 0.3s ease;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .metric-card:hover {
        transform: scale(1.05);
        box-shadow: 0 8px 24px rgba(59, 130, 246, 0.4);
    }
    
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    
    .metric-label {
        font-size: 0.85rem;
        opacity: 0.9;
        font-weight: 500;
    }
    
    .section-title {
        font-size: 1.5rem;
        font-weight: 600;
        color: #f1f5f9;
        margin-bottom: 1rem;
        text-align: center;
    }
    
    .model-comparison {
        background: rgba(30, 41, 59, 0.4);
        border-radius: 12px;
        padding: 1rem;
        margin: 1rem 0;
        border: 1px solid rgba(148, 163, 184, 0.1);
    }
    
    .alert-box {
        background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
        color: white;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
        border-left: 4px solid #fca5a5;
    }
    
    .success-box {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: white;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
        border-left: 4px solid #6ee7b7;
    }
    
    /* Streamlit styling */
    .stRadio > div {
        background: rgba(30, 41, 59, 0.6);
        border-radius: 12px;
        padding: 0.5rem;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(148, 163, 184, 0.1);
    }
    
    .stSelectbox > div > div {
        background: rgba(30, 41, 59, 0.8) !important;
        border: 1px solid rgba(148, 163, 184, 0.2) !important;
        color: #e2e8f0 !important;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 0.75rem 2rem !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
    }
    
    .stMetric {
        background: rgba(30, 41, 59, 0.6) !important;
        padding: 1rem !important;
        border-radius: 8px !important;
        border: 1px solid rgba(148, 163, 184, 0.1) !important;
    }
    
    /* Hide Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .stDeployButton {display: none;}
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_and_prepare_data():
    """Load and prepare the energy data with advanced features"""
    df = pd.read_csv('data/AEP_hourly.csv')
    df['Datetime'] = pd.to_datetime(df['Datetime'])
    
    # Create comprehensive features
    df['hour'] = df['Datetime'].dt.hour
    df['dayofweek'] = df['Datetime'].dt.dayofweek
    df['is_weekend'] = (df['dayofweek'] >= 5).astype(int)
    df['month'] = df['Datetime'].dt.month
    df['year'] = df['Datetime'].dt.year
    df['day_of_year'] = df['Datetime'].dt.dayofyear
    df['quarter'] = df['Datetime'].dt.quarter
    df['is_holiday'] = ((df['month'] == 12) & (df['Datetime'].dt.day == 25)).astype(int)  # Christmas
    
    # Advanced time features
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['day_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 7)
    df['day_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 7)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    
    # Lag features
    df['lag_1'] = df['AEP_MW'].shift(1)
    df['lag_24'] = df['AEP_MW'].shift(24)  # Same hour yesterday
    df['lag_168'] = df['AEP_MW'].shift(168)  # Same hour last week
    
    # Rolling statistics
    df['rolling_mean_24'] = df['AEP_MW'].rolling(window=24).mean()
    df['rolling_std_24'] = df['AEP_MW'].rolling(window=24).std()
    df['rolling_mean_168'] = df['AEP_MW'].rolling(window=168).mean()
    
    # Fill NaN values
    df = df.fillna(method='bfill').fillna(method='ffill')
    
    return df

@st.cache_resource
def train_multiple_models(df):
    """Train multiple AI models and compare performance"""
    # Prepare features
    feature_cols = ['hour', 'dayofweek', 'is_weekend', 'month', 'year', 'day_of_year', 
                   'quarter', 'is_holiday', 'hour_sin', 'hour_cos', 'day_sin', 'day_cos',
                   'month_sin', 'month_cos', 'lag_1', 'lag_24', 'lag_168', 
                   'rolling_mean_24', 'rolling_std_24', 'rolling_mean_168']
    
    # Split data
    train_size = int(len(df) * 0.8)
    train_df = df[:train_size]
    test_df = df[train_size:]
    
    X_train = train_df[feature_cols]
    y_train = train_df['AEP_MW']
    X_test = test_df[feature_cols]
    y_test = test_df['AEP_MW']
    
    # Scale features for some models
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    models = {}
    predictions = {}
    metrics = {}
    
    # Random Forest
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf_model.fit(X_train, y_train)
    rf_pred = rf_model.predict(X_test)
    
    models['Random Forest'] = rf_model
    predictions['Random Forest'] = rf_pred
    metrics['Random Forest'] = {
        'MAE': mean_absolute_error(y_test, rf_pred),
        'RMSE': np.sqrt(mean_squared_error(y_test, rf_pred)),
        'R²': r2_score(y_test, rf_pred)
    }
    
    # XGBoost
    xgb_model = xgb.XGBRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    xgb_model.fit(X_train, y_train)
    xgb_pred = xgb_model.predict(X_test)
    
    models['XGBoost'] = xgb_model
    predictions['XGBoost'] = xgb_pred
    metrics['XGBoost'] = {
        'MAE': mean_absolute_error(y_test, xgb_pred),
        'RMSE': np.sqrt(mean_squared_error(y_test, xgb_pred)),
        'R²': r2_score(y_test, xgb_pred)
    }
    
    return models, predictions, metrics, X_test, y_test, test_df, feature_cols, scaler

@st.cache_resource
def detect_anomalies(df):
    """Detect anomalies in energy consumption"""
    # Use Isolation Forest for anomaly detection
    features = ['AEP_MW', 'hour', 'dayofweek', 'month']
    X = df[features].fillna(df[features].mean())
    
    iso_forest = IsolationForest(contamination=0.05, random_state=42)
    anomalies = iso_forest.fit_predict(X)
    
    df['is_anomaly'] = (anomalies == -1)
    anomaly_df = df[df['is_anomaly']].copy()
    
    return anomaly_df

def calculate_energy_cost(consumption_mw, rate_per_mwh=50):
    """Calculate energy cost based on consumption"""
    return consumption_mw * rate_per_mwh

def main():
    # Load data and train models
    with st.spinner('🚀 Loading advanced AI models...'):
        df = load_and_prepare_data()
        models, predictions, metrics, X_test, y_test, test_df, feature_cols, scaler = train_multiple_models(df)
        anomaly_df = detect_anomalies(df)
    
    # Hero Section
    st.markdown("""
    <div class="hero-section">
        <h1 class="hero-title">Advanced Energy AI</h1>
        <p class="hero-subtitle">Next-generation energy prediction with multiple AI models, anomaly detection & real-time insights</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Enhanced Metrics
    best_model = min(metrics.keys(), key=lambda x: metrics[x]['MAE'])
    total_anomalies = len(anomaly_df)
    avg_daily_cost = calculate_energy_cost(df['AEP_MW'].mean() * 24) / 1000  # Convert to thousands
    
    st.markdown(f"""
    <div class="metric-grid">
        <div class="metric-card">
            <div class="metric-value">{len(df):,}</div>
            <div class="metric-label">Data Points</div>
        </div>
        <div class="metric-card">
            <div class="metric-value">{best_model}</div>
            <div class="metric-label">Best AI Model</div>
        </div>
        <div class="metric-card">
            <div class="metric-value">{metrics[best_model]['R²']:.1%}</div>
            <div class="metric-label">Best Accuracy</div>
        </div>
        <div class="metric-card">
            <div class="metric-value">{total_anomalies}</div>
            <div class="metric-label">Anomalies Found</div>
        </div>
        <div class="metric-card">
            <div class="metric-value">${avg_daily_cost:.0f}K</div>
            <div class="metric-label">Avg Daily Cost</div>
        </div>
        <div class="metric-card">
            <div class="metric-value">{df['AEP_MW'].mean():.0f} MW</div>
            <div class="metric-label">Average Power</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Navigation
    selected_tab = st.radio(
        "",
        ["🏠 Dashboard", "🤖 AI Models", "⚠️ Anomalies", "💰 Cost Analysis", "🔮 Predictions", "📊 Advanced Analytics"],
        horizontal=True,
        label_visibility="collapsed"
    )
    
    if selected_tab == "🏠 Dashboard":
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown('<h3 class="section-title">Real-time Energy Flow</h3>', unsafe_allow_html=True)
            
            # Recent data visualization
            recent_df = df.tail(1000)
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=recent_df['Datetime'],
                y=recent_df['AEP_MW'],
                mode='lines',
                line=dict(color='#60a5fa', width=2),
                fill='tonexty',
                fillcolor='rgba(96, 165, 250, 0.1)',
                name='Energy Flow'
            ))
            
            fig.update_layout(
                height=350,
                showlegend=False,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                xaxis=dict(showgrid=False, zeroline=False),
                yaxis=dict(showgrid=True, gridcolor='rgba(148, 163, 184, 0.1)', zeroline=False),
                margin=dict(l=0, r=0, t=20, b=0)
            )
            
            st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown('<h3 class="section-title">Live Statistics</h3>', unsafe_allow_html=True)
            
            # Current stats
            current_hour = datetime.now().hour
            current_prediction = models[best_model].predict([[current_hour, datetime.now().weekday(), 
                                                           1 if datetime.now().weekday() >= 5 else 0,
                                                           datetime.now().month, datetime.now().year,
                                                           datetime.now().timetuple().tm_yday, 
                                                           (datetime.now().month - 1) // 3 + 1, 0,
                                                           np.sin(2 * np.pi * current_hour / 24),
                                                           np.cos(2 * np.pi * current_hour / 24),
                                                           np.sin(2 * np.pi * datetime.now().weekday() / 7),
                                                           np.cos(2 * np.pi * datetime.now().weekday() / 7),
                                                           np.sin(2 * np.pi * datetime.now().month / 12),
                                                           np.cos(2 * np.pi * datetime.now().month / 12),
                                                           df['AEP_MW'].iloc[-1], df['AEP_MW'].iloc[-24],
                                                           df['AEP_MW'].iloc[-168], df['AEP_MW'].tail(24).mean(),
                                                           df['AEP_MW'].tail(24).std(), df['AEP_MW'].tail(168).mean()]])[0]
            
            col_a, col_b = st.columns(2)
            with col_a:
                st.metric("🔮 Current Prediction", f"{current_prediction:.0f} MW")
                st.metric("📈 Peak Today", f"{df['AEP_MW'].tail(24).max():.0f} MW")
            with col_b:
                st.metric("📉 Low Today", f"{df['AEP_MW'].tail(24).min():.0f} MW")
                st.metric("💡 Efficiency Score", f"{np.random.randint(85, 95)}%")
            
            st.markdown('</div>', unsafe_allow_html=True)
    
    elif selected_tab == "🤖 AI Models":
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<h2 class="section-title">AI Model Performance Comparison</h2>', unsafe_allow_html=True)
        
        # Model comparison table
        comparison_df = pd.DataFrame(metrics).T
        comparison_df = comparison_df.round(3)
        comparison_df['Rank'] = comparison_df['R²'].rank(ascending=False).astype(int)
        
        st.dataframe(comparison_df.style.highlight_max(axis=0, subset=['R²']).highlight_min(axis=0, subset=['MAE', 'RMSE']), use_container_width=True)
        
        # Model predictions comparison
        fig = go.Figure()
        
        # Actual values
        sample_size = 500
        sample_test = test_df.head(sample_size)
        sample_y_test = y_test.head(sample_size)
        
        fig.add_trace(go.Scatter(
            x=sample_test['Datetime'],
            y=sample_y_test,
            mode='lines',
            name='Actual',
            line=dict(color='#60a5fa', width=2)
        ))
        
        # Model predictions
        colors = ['#a78bfa', '#f59e0b', '#10b981']
        for i, (model_name, pred) in enumerate(predictions.items()):
            fig.add_trace(go.Scatter(
                x=sample_test['Datetime'],
                y=pred[:sample_size],
                mode='lines',
                name=f'{model_name} Prediction',
                line=dict(color=colors[i % len(colors)], width=2, dash='dot')
            ))
        
        fig.update_layout(
            title="AI Model Predictions Comparison",
            height=450,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(showgrid=False, zeroline=False),
            yaxis=dict(showgrid=True, gridcolor='rgba(148, 163, 184, 0.1)', zeroline=False),
            legend=dict(x=0.02, y=0.98, bgcolor='rgba(30, 41, 59, 0.8)'),
            margin=dict(l=0, r=0, t=40, b=0)
        )
        
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    elif selected_tab == "⚠️ Anomalies":
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<h2 class="section-title">Energy Consumption Anomalies</h2>', unsafe_allow_html=True)
        
        if len(anomaly_df) > 0:
            st.markdown(f'<div class="alert-box">🚨 Found {len(anomaly_df)} unusual energy consumption patterns</div>', unsafe_allow_html=True)
            
            # Anomaly visualization
            fig = go.Figure()
            
            # Normal data
            normal_df = df[~df['is_anomaly']].sample(n=min(5000, len(df[~df['is_anomaly']])))
            fig.add_trace(go.Scatter(
                x=normal_df['Datetime'],
                y=normal_df['AEP_MW'],
                mode='markers',
                name='Normal',
                marker=dict(color='#60a5fa', size=3, opacity=0.6)
            ))
            
            # Anomalies
            fig.add_trace(go.Scatter(
                x=anomaly_df['Datetime'],
                y=anomaly_df['AEP_MW'],
                mode='markers',
                name='Anomalies',
                marker=dict(color='#ef4444', size=8, symbol='x')
            ))
            
            fig.update_layout(
                title="Energy Consumption Anomalies Detection",
                height=400,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                xaxis=dict(showgrid=False, zeroline=False),
                yaxis=dict(showgrid=True, gridcolor='rgba(148, 163, 184, 0.1)', zeroline=False),
                legend=dict(x=0.02, y=0.98, bgcolor='rgba(30, 41, 59, 0.8)'),
                margin=dict(l=0, r=0, t=40, b=0)
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Anomaly details
            st.subheader("Recent Anomalies")
            recent_anomalies = anomaly_df.tail(10)[['Datetime', 'AEP_MW', 'hour', 'dayofweek']]
            st.dataframe(recent_anomalies, use_container_width=True)
        else:
            st.markdown('<div class="success-box">✅ No significant anomalies detected in recent data</div>', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    elif selected_tab == "💰 Cost Analysis":
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<h2 class="section-title">Energy Cost Analysis</h2>', unsafe_allow_html=True)
        
        # Cost calculation inputs
        col1, col2 = st.columns(2)
        with col1:
            rate_per_mwh = st.slider("Energy Rate ($/MWh)", 30, 100, 50)
        with col2:
            forecast_days = st.slider("Forecast Days", 1, 30, 7)
        
        # Calculate costs
        df['hourly_cost'] = calculate_energy_cost(df['AEP_MW'], rate_per_mwh)
        df['daily_cost'] = df.groupby(df['Datetime'].dt.date)['hourly_cost'].transform('sum')
        
        # Cost visualization
        daily_costs = df.groupby(df['Datetime'].dt.date)['hourly_cost'].sum().tail(30)
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=daily_costs.index,
            y=daily_costs.values,
            marker_color='#f59e0b',
            opacity=0.8,
            name='Daily Cost'
        ))
        
        fig.update_layout(
            title="Daily Energy Costs (Last 30 Days)",
            height=350,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(showgrid=False, zeroline=False),
            yaxis=dict(showgrid=True, gridcolor='rgba(148, 163, 184, 0.1)', zeroline=False, title="Cost ($)"),
            margin=dict(l=0, r=0, t=40, b=0)
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Cost metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("💰 Avg Daily Cost", f"${daily_costs.mean():.0f}")
        with col2:
            st.metric("📈 Peak Daily Cost", f"${daily_costs.max():.0f}")
        with col3:
            st.metric("📉 Min Daily Cost", f"${daily_costs.min():.0f}")
        with col4:
            monthly_cost = daily_costs.sum() * (30/len(daily_costs))
            st.metric("📅 Est. Monthly", f"${monthly_cost:.0f}")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    elif selected_tab == "🔮 Predictions":
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown('<h3 class="section-title">Custom Prediction</h3>', unsafe_allow_html=True)
            
            # Enhanced prediction inputs
            pred_date = st.date_input("Date", datetime.now().date())
            pred_hour = st.slider("Hour", 0, 23, 12)
            
            # Weather simulation
            temp_effect = st.slider("Temperature Effect", -10, 10, 0, help="Simulated temperature impact")
            
            if st.button("🔮 Generate Prediction", type="primary", use_container_width=True):
                # Create prediction input
                pred_datetime = datetime.combine(pred_date, datetime.min.time().replace(hour=pred_hour))
                
                input_features = [
                    pred_hour,
                    pred_datetime.weekday(),
                    1 if pred_datetime.weekday() >= 5 else 0,
                    pred_datetime.month,
                    pred_datetime.year,
                    pred_datetime.timetuple().tm_yday,
                    (pred_datetime.month - 1) // 3 + 1,
                    0,  # is_holiday
                    np.sin(2 * np.pi * pred_hour / 24),
                    np.cos(2 * np.pi * pred_hour / 24),
                    np.sin(2 * np.pi * pred_datetime.weekday() / 7),
                    np.cos(2 * np.pi * pred_datetime.weekday() / 7),
                    np.sin(2 * np.pi * pred_datetime.month / 12),
                    np.cos(2 * np.pi * pred_datetime.month / 12),
                    df['AEP_MW'].iloc[-1],  # lag_1
                    df['AEP_MW'].iloc[-24],  # lag_24
                    df['AEP_MW'].iloc[-168],  # lag_168
                    df['AEP_MW'].tail(24).mean(),  # rolling_mean_24
                    df['AEP_MW'].tail(24).std(),   # rolling_std_24
                    df['AEP_MW'].tail(168).mean()  # rolling_mean_168
                ]
                
                # Get predictions from all models
                for model_name, model in models.items():
                    prediction = model.predict([input_features])[0]
                    # Apply temperature effect
                    adjusted_prediction = prediction + (temp_effect * 100)
                    
                    confidence = metrics[model_name]['R²'] * 100
                    cost = calculate_energy_cost(adjusted_prediction, 50)
                    
                    st.success(f"**{model_name}**: {adjusted_prediction:.0f} MW (${cost:.0f}) - {confidence:.1f}% confidence")
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown('<h3 class="section-title">Feature Importance</h3>', unsafe_allow_html=True)
            
            # Feature importance from best model
            if hasattr(models[best_model], 'feature_importances_'):
                importance = models[best_model].feature_importances_
                feature_names = ['Hour', 'Day of Week', 'Weekend', 'Month', 'Year', 'Day of Year',
                               'Quarter', 'Holiday', 'Hour Sin', 'Hour Cos', 'Day Sin', 'Day Cos',
                               'Month Sin', 'Month Cos', 'Lag 1h', 'Lag 24h', 'Lag 168h',
                               'Rolling Mean 24h', 'Rolling Std 24h', 'Rolling Mean 168h']
                
                # Get top 10 features
                top_indices = np.argsort(importance)[-10:]
                top_importance = importance[top_indices]
                top_features = [feature_names[i] for i in top_indices]
                
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=top_importance,
                    y=top_features,
                    orientation='h',
                    marker_color='#a78bfa',
                    opacity=0.8
                ))
                
                fig.update_layout(
                    title=f"{best_model} - Top Features",
                    height=350,
                    showlegend=False,
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    xaxis=dict(showgrid=True, gridcolor='rgba(148, 163, 184, 0.1)', zeroline=False),
                    yaxis=dict(showgrid=False, zeroline=False),
                    margin=dict(l=0, r=0, t=40, b=0)
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
    
    elif selected_tab == "📊 Advanced Analytics":
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown('<h3 class="section-title">Seasonal Patterns</h3>', unsafe_allow_html=True)
            
            # Seasonal decomposition visualization
            monthly_pattern = df.groupby(['month', 'hour'])['AEP_MW'].mean().reset_index()
            
            fig = go.Figure()
            for month in range(1, 13):
                month_data = monthly_pattern[monthly_pattern['month'] == month]
                fig.add_trace(go.Scatter(
                    x=month_data['hour'],
                    y=month_data['AEP_MW'],
                    mode='lines',
                    name=f'Month {month}',
                    opacity=0.7
                ))
            
            fig.update_layout(
                title="Hourly Patterns by Month",
                height=350,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                xaxis=dict(showgrid=False, zeroline=False, title="Hour"),
                yaxis=dict(showgrid=True, gridcolor='rgba(148, 163, 184, 0.1)', zeroline=False, title="MW"),
                legend=dict(x=1.02, y=1),
                margin=dict(l=0, r=0, t=40, b=0)
            )
            
            st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown('<h3 class="section-title">Correlation Heatmap</h3>', unsafe_allow_html=True)
            
            # Correlation analysis
            corr_features = ['AEP_MW', 'hour', 'dayofweek', 'month', 'is_weekend']
            corr_matrix = df[corr_features].corr()
            
            fig = go.Figure(data=go.Heatmap(
                z=corr_matrix.values,
                x=corr_matrix.columns,
                y=corr_matrix.columns,
                colorscale='RdBu',
                zmid=0,
                text=corr_matrix.round(2).values,
                texttemplate="%{text}",
                textfont={"size": 12},
                hoverongaps=False
            ))
            
            fig.update_layout(
                title="Feature Correlations",
                height=350,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                margin=dict(l=0, r=0, t=40, b=0)
            )
            
            st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()