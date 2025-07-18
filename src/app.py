import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import numpy as np
from datetime import datetime, timedelta

# Page config
st.set_page_config(
    page_title="Energy AI",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Modern Dark Theme CSS
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
        background: linear-gradient(135deg, #60a5fa 0%, #a78bfa 100%);
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
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
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
        font-size: 2rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    
    .metric-label {
        font-size: 0.9rem;
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
    
    /* Streamlit component styling */
    .stRadio > div {
        background: rgba(30, 41, 59, 0.6);
        border-radius: 12px;
        padding: 0.5rem;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(148, 163, 184, 0.1);
    }
    
    .stRadio > div > label {
        background: transparent !important;
        color: #94a3b8 !important;
        border-radius: 8px !important;
        padding: 0.75rem 1rem !important;
        margin: 0 0.25rem !important;
        transition: all 0.3s ease !important;
        font-weight: 500 !important;
    }
    
    .stRadio > div > label:hover {
        background: rgba(59, 130, 246, 0.2) !important;
        color: #e2e8f0 !important;
    }
    
    .stRadio > div > label[data-checked="true"] {
        background: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%) !important;
        color: white !important;
        box-shadow: 0 2px 8px rgba(59, 130, 246, 0.3) !important;
    }
    
    .stSelectbox > div > div {
        background: rgba(30, 41, 59, 0.8) !important;
        border: 1px solid rgba(148, 163, 184, 0.2) !important;
        color: #e2e8f0 !important;
    }
    
    .stSlider > div > div > div {
        background: rgba(30, 41, 59, 0.8) !important;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 0.75rem 2rem !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
        font-family: 'Inter', sans-serif !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 16px rgba(59, 130, 246, 0.4) !important;
    }
    
    .stMetric {
        background: rgba(30, 41, 59, 0.6) !important;
        padding: 1rem !important;
        border-radius: 8px !important;
        border: 1px solid rgba(148, 163, 184, 0.1) !important;
    }
    
    .stMetric > div {
        color: #e2e8f0 !important;
    }
    
    /* Hide Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .stDeployButton {display: none;}
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
    }
    ::-webkit-scrollbar-track {
        background: rgba(30, 41, 59, 0.5);
        border-radius: 4px;
    }
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%);
        border-radius: 4px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(135deg, #2563eb 0%, #7c3aed 100%);
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_and_prepare_data():
    """Load and prepare the energy data"""
    df = pd.read_csv('data/AEP_hourly.csv')
    df['Datetime'] = pd.to_datetime(df['Datetime'])
    
    # Create features
    df['hour'] = df['Datetime'].dt.hour
    df['dayofweek'] = df['Datetime'].dt.dayofweek
    df['is_weekend'] = (df['dayofweek'] >= 5).astype(int)
    df['month'] = df['Datetime'].dt.month
    df['year'] = df['Datetime'].dt.year
    df['day_of_year'] = df['Datetime'].dt.dayofyear
    
    return df

@st.cache_resource
def train_model(df):
    """Train the RandomForest model"""
    features = ['hour', 'dayofweek', 'is_weekend', 'month', 'year', 'day_of_year']
    X = df[features]
    y = df['AEP_MW']
    
    model = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
    model.fit(X, y)
    
    predictions = model.predict(X)
    mae = mean_absolute_error(y, predictions)
    r2 = r2_score(y, predictions)
    
    return model, X, y, predictions, mae, r2, features

def main():
    # Load data
    df = load_and_prepare_data()
    model, X, y, predictions, mae, r2, features = train_model(df)
    
    # Hero Section
    st.markdown("""
    <div class="hero-section">
        <h1 class="hero-title">Energy AI</h1>
        <p class="hero-subtitle">Intelligent energy consumption prediction powered by machine learning</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Key Metrics Cards
    st.markdown("""
    <div class="metric-grid">
        <div class="metric-card">
            <div class="metric-value">{:,}</div>
            <div class="metric-label">Data Points</div>
        </div>
        <div class="metric-card">
            <div class="metric-value">{}</div>
            <div class="metric-label">Years Analyzed</div>
        </div>
        <div class="metric-card">
            <div class="metric-value">{:.0f} MW</div>
            <div class="metric-label">Average Power</div>
        </div>
        <div class="metric-card">
            <div class="metric-value">{:.1%}</div>
            <div class="metric-label">AI Accuracy</div>
        </div>
    </div>
    """.format(len(df), df['year'].nunique(), df['AEP_MW'].mean(), r2), unsafe_allow_html=True)
    
    # Navigation
    selected_tab = st.radio(
        "",
        ["Overview", "Predictions", "Analytics", "Custom Predict"],
        horizontal=True,
        label_visibility="collapsed"
    )
    
    if selected_tab == "Overview":
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<h2 class="section-title">Energy Consumption Timeline</h2>', unsafe_allow_html=True)
        
        # Clean chart
        sample_df = df.sample(n=5000).sort_values('Datetime')
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=sample_df['Datetime'],
            y=sample_df['AEP_MW'],
            mode='lines',
            line=dict(color='#667eea', width=1.5),
            fill='tonexty',
            fillcolor='rgba(102, 126, 234, 0.1)',
            name='Energy'
        ))
        
        fig.update_layout(
            height=400,
            showlegend=False,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(showgrid=False, zeroline=False),
            yaxis=dict(showgrid=True, gridcolor='rgba(0,0,0,0.1)', zeroline=False),
            margin=dict(l=0, r=0, t=20, b=0)
        )
        
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    elif selected_tab == "Predictions":
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<h2 class="section-title">AI vs Reality</h2>', unsafe_allow_html=True)
        
        # Prediction comparison
        sample_df = df.head(3000)
        sample_predictions = predictions[:3000]
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=sample_df['Datetime'],
            y=sample_df['AEP_MW'],
            mode='lines',
            name='Actual',
            line=dict(color='#667eea', width=2)
        ))
        
        fig.add_trace(go.Scatter(
            x=sample_df['Datetime'],
            y=sample_predictions,
            mode='lines',
            name='AI Prediction',
            line=dict(color='#764ba2', width=2, dash='dot')
        ))
        
        fig.update_layout(
            height=450,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(showgrid=False, zeroline=False),
            yaxis=dict(showgrid=True, gridcolor='rgba(0,0,0,0.1)', zeroline=False),
            legend=dict(x=0.02, y=0.98, bgcolor='rgba(255,255,255,0.8)'),
            margin=dict(l=0, r=0, t=20, b=0)
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Performance metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("R² Score", f"{r2:.3f}")
        with col2:
            st.metric("Mean Error", f"{mae:.0f} MW")
        with col3:
            accuracy = (1 - mae/df['AEP_MW'].mean()) * 100
            st.metric("Accuracy", f"{accuracy:.1f}%")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    elif selected_tab == "Analytics":
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown('<h3 class="section-title">Daily Patterns</h3>', unsafe_allow_html=True)
            
            hourly_avg = df.groupby('hour')['AEP_MW'].mean()
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=hourly_avg.index,
                y=hourly_avg.values,
                marker_color='#667eea',
                opacity=0.8
            ))
            
            fig.update_layout(
                height=300,
                showlegend=False,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                xaxis=dict(showgrid=False, zeroline=False, title="Hour"),
                yaxis=dict(showgrid=True, gridcolor='rgba(0,0,0,0.1)', zeroline=False, title="MW"),
                margin=dict(l=0, r=0, t=20, b=0)
            )
            
            st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown('<h3 class="section-title">Weekly Patterns</h3>', unsafe_allow_html=True)
            
            day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
            weekly_avg = df.groupby('dayofweek')['AEP_MW'].mean()
            
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=day_names,
                y=weekly_avg.values,
                marker_color='#764ba2',
                opacity=0.8
            ))
            
            fig.update_layout(
                height=300,
                showlegend=False,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                xaxis=dict(showgrid=False, zeroline=False, title="Day"),
                yaxis=dict(showgrid=True, gridcolor='rgba(0,0,0,0.1)', zeroline=False, title="MW"),
                margin=dict(l=0, r=0, t=20, b=0)
            )
            
            st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
    
    elif selected_tab == "Custom Predict":
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown('<h3 class="section-title">Prediction Input</h3>', unsafe_allow_html=True)
            
            pred_hour = st.slider("Hour", 0, 23, 12)
            pred_day = st.selectbox("Day", ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
            pred_month = st.selectbox("Month", ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'])
            pred_year = st.number_input("Year", 2020, 2030, 2024)
            
            if st.button("Predict Energy", type="primary", use_container_width=True):
                # Convert inputs
                day_mapping = {'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3, 'Friday': 4, 'Saturday': 5, 'Sunday': 6}
                month_mapping = {month: i+1 for i, month in enumerate(['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'])}
                
                pred_dayofweek = day_mapping[pred_day]
                pred_is_weekend = 1 if pred_dayofweek >= 5 else 0
                pred_month_num = month_mapping[pred_month]
                pred_day_of_year = datetime(pred_year, pred_month_num, 1).timetuple().tm_yday
                
                input_data = np.array([[pred_hour, pred_dayofweek, pred_is_weekend, pred_month_num, pred_year, pred_day_of_year]])
                prediction = model.predict(input_data)[0]
                
                st.success(f"Predicted: **{prediction:.0f} MW**")
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown('<h3 class="section-title">Feature Importance</h3>', unsafe_allow_html=True)
            
            importance = model.feature_importances_
            feature_names = ['Hour', 'Day of Week', 'Weekend', 'Month', 'Year', 'Day of Year']
            
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=importance,
                y=feature_names,
                orientation='h',
                marker_color='#667eea',
                opacity=0.8
            ))
            
            fig.update_layout(
                height=300,
                showlegend=False,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                xaxis=dict(showgrid=True, gridcolor='rgba(0,0,0,0.1)', zeroline=False, title="Importance"),
                yaxis=dict(showgrid=False, zeroline=False),
                margin=dict(l=0, r=0, t=20, b=0)
            )
            
            st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()