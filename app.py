"""
🤖 AI Data Analytics Pro - FIXED SPECIFIC INSIGHTS
Fixed: Each insight button now provides UNIQUE, SPECIFIC analysis tailored to the question
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import tempfile
import re
import json
import time
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="🤖 AI Data Analytics Pro",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# FIXED CSS - Perfect horizontal alignment and text display
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

:root {
    --primary-color: #667eea;
    --secondary-color: #764ba2;
    --accent-color: #f093fb;
    --success-color: #4ade80;
    --warning-color: #fbbf24;
    --error-color: #ef4444;
    --info-color: #3b82f6;
    --dark-bg: #1e293b;
    --light-bg: #f8fafc;
    --card-bg: #ffffff;
    --text-primary: #1e293b;
    --text-secondary: #64748b;
    --border-color: #e2e8f0;
    --shadow-light: 0 1px 3px 0 rgba(0, 0, 0, 0.1), 0 1px 2px 0 rgba(0, 0, 0, 0.06);
    --shadow-medium: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
    --shadow-large: 0 20px 25px -5px rgba(0, 0, 0, 0.1), 0 10px 10px -5px rgba(0, 0, 0, 0.04);
}

/* Global Styles */
.main {
    font-family: 'Inter', sans-serif;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    min-height: 100vh;
}

.main .block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    background: var(--light-bg);
    border-radius: 20px;
    margin: 1rem;
    box-shadow: var(--shadow-large);
}

/* Enhanced Header */
.main-header {
    background: linear-gradient(135deg, var(--primary-color) 0%, var(--secondary-color) 50%, var(--accent-color) 100%);
    color: white;
    padding: 2rem;
    border-radius: 15px;
    text-align: center;
    margin-bottom: 2rem;
    box-shadow: var(--shadow-large);
    position: relative;
    overflow: hidden;
}

.main-header h1 {
    font-size: 3rem;
    font-weight: 700;
    margin: 0;
    text-shadow: 0 2px 4px rgba(0,0,0,0.3);
    position: relative;
    z-index: 1;
}

.main-header p {
    font-size: 1.2rem;
    margin-top: 0.5rem;
    opacity: 0.95;
    position: relative;
    z-index: 1;
}

/* FIXED: Perfect Horizontal Metric Cards */
.metric-card {
    background: var(--card-bg);
    border-radius: 15px;
    padding: 1.5rem;
    text-align: center;
    box-shadow: var(--shadow-medium);
    border: 1px solid var(--border-color);
    transition: all 0.3s ease;
    position: relative;
    overflow: hidden;
    height: 120px;
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;
    margin-bottom: 1rem;
}

.metric-card::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    height: 4px;
    background: linear-gradient(90deg, var(--primary-color), var(--secondary-color));
    border-radius: 15px 15px 0 0;
}

.metric-card:hover {
    transform: translateY(-5px);
    box-shadow: var(--shadow-large);
}

/* FIXED: Horizontal text layout */
.metric-card h2 {
    color: var(--text-primary);
    font-size: 2.2rem;
    font-weight: 700;
    margin: 0.3rem 0;
    background: linear-gradient(135deg, var(--primary-color), var(--secondary-color));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1;
    white-space: nowrap;
}

.metric-card h3 {
    color: var(--text-secondary);
    font-size: 0.8rem;
    font-weight: 600;
    margin: 0;
    text-transform: uppercase;
    letter-spacing: 0.3px;
    line-height: 1.2;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    max-width: 100%;
}

/* Enhanced Buttons */
.stButton button {
    background: linear-gradient(135deg, var(--primary-color) 0%, var(--secondary-color) 100%);
    color: white !important;
    border: none;
    border-radius: 10px;
    padding: 0.75rem 1.5rem;
    font-weight: 600;
    font-size: 1rem;
    transition: all 0.3s ease;
    box-shadow: var(--shadow-medium);
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

.stButton button:hover {
    transform: translateY(-2px);
    box-shadow: var(--shadow-large);
    background: linear-gradient(135deg, var(--secondary-color) 0%, var(--accent-color) 100%);
}

/* Enhanced Form Elements */
.stTextArea textarea, .stTextInput input, .stSelectbox select {
    background: white !important;
    color: #1e293b !important;
    border: 2px solid var(--border-color) !important;
    border-radius: 10px !important;
    padding: 1rem !important;
    font-size: 1rem !important;
    font-weight: 500 !important;
    transition: all 0.3s ease;
}

.stTextArea textarea:focus, .stTextInput input:focus, .stSelectbox select:focus {
    border-color: var(--primary-color) !important;
    box-shadow: 0 0 0 0.2rem rgba(102, 126, 234, 0.25) !important;
    background: white !important;
    color: #1e293b !important;
}

.stTextArea textarea::placeholder, .stTextInput input::placeholder {
    color: #64748b !important;
    opacity: 0.8;
}

/* Enhanced Messages */
.stSuccess {
    background: linear-gradient(135deg, #dcfce7 0%, #bbf7d0 100%);
    color: #166534 !important;
    border: 1px solid #86efac;
    border-radius: 10px;
    padding: 1rem;
}

.stError {
    background: linear-gradient(135deg, #fef2f2 0%, #fecaca 100%);
    color: #dc2626 !important;
    border: 1px solid #fca5a5;
    border-radius: 10px;
    padding: 1rem;
}

/* Analysis Response Card */
.analysis-response {
    background: var(--card-bg);
    border-radius: 15px;
    padding: 2rem;
    margin: 1rem 0;
    box-shadow: var(--shadow-medium);
    border: 1px solid var(--border-color);
    position: relative;
    color: #1e293b !important;
}

.analysis-response::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    height: 4px;
    background: linear-gradient(90deg, var(--success-color), var(--info-color));
    border-radius: 15px 15px 0 0;
}

/* Business Intelligence Cards */
.business-card {
    background: var(--card-bg);
    border-radius: 15px;
    padding: 2rem;
    margin: 1rem 0;
    box-shadow: var(--shadow-medium);
    border: 1px solid var(--border-color);
    transition: all 0.3s ease;
    position: relative;
}

.business-card::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    height: 4px;
    background: linear-gradient(90deg, var(--primary-color), var(--secondary-color));
    border-radius: 15px 15px 0 0;
}

.business-card:hover {
    transform: translateY(-3px);
    box-shadow: var(--shadow-large);
}

.business-card h3 {
    color: #1e293b !important;
    margin-bottom: 1rem;
    font-weight: 600;
    font-size: 1.2rem;
}

.business-card h4 {
    color: #1e293b !important;
    margin: 0.5rem 0;
    font-weight: 500;
}

.business-card p, .business-card div, .business-card span {
    color: #1e293b !important;
    line-height: 1.6;
}

.business-card ul, .business-card li {
    color: #1e293b !important;
    margin: 0.25rem 0;
}

.insight-item {
    background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
    padding: 1rem;
    border-radius: 10px;
    margin: 0.5rem 0;
    border-left: 4px solid var(--primary-color);
    color: #1e293b !important;
    font-weight: 500;
}

.insight-item strong {
    color: #1e293b !important;
    font-weight: 600;
}

/* Force all text in cards to be dark */
.business-card * {
    color: #1e293b !important;
}

.insight-item * {
    color: #1e293b !important;
}

/* Quick Analytics Helper Text */
.quick-analytics-help {
    background: linear-gradient(135deg, #e0f2fe 0%, #b3e5fc 100%);
    border-left: 4px solid var(--info-color);
    padding: 1rem;
    border-radius: 10px;
    margin: 1rem 0;
    color: #1e293b !important;
}

.quick-analytics-help h4 {
    color: #1e293b !important;
    margin-bottom: 0.5rem;
    font-weight: 600;
}

.quick-analytics-help p {
    color: #1e293b !important;
    margin: 0.25rem 0;
}

/* Responsive Design */
@media (max-width: 1200px) {
    .metric-card h2 {
        font-size: 1.8rem;
    }
    .metric-card h3 {
        font-size: 0.7rem;
    }
}

@media (max-width: 768px) {
    .metric-card {
        height: 100px;
        padding: 1rem;
    }
    .metric-card h2 {
        font-size: 1.5rem;
    }
    .metric-card h3 {
        font-size: 0.65rem;
    }
    .main-header h1 {
        font-size: 2rem;
    }
}
</style>
""", unsafe_allow_html=True)

# Initialize Gemini AI
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    st.error("🔑 Please set your GEMINI_API_KEY in the .env file!")
    st.info("Get your key from: https://makersuite.google.com/app/apikey")
    st.stop()

try:
    import google.generativeai as genai
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel("gemini-2.5-flash")
except ImportError:
    st.error("Please install: pip install google-generativeai")
    st.stop()
except Exception as e:
    st.error(f"Error initializing Gemini: {e}")

# Initialize session state
if 'df' not in st.session_state:
    st.session_state.df = None
if 'analysis_history' not in st.session_state:
    st.session_state.analysis_history = []
if 'current_question' not in st.session_state:
    st.session_state.current_question = ''

def create_premium_sample_data():
    """Create RANDOM dataset each time with current timestamp as seed"""
    # Use current timestamp for truly random data each time
    np.random.seed(int(time.time()))
    
    products = [
        'iPhone 15 Pro', 'MacBook Air M3', 'iPad Pro', 'Apple Watch Ultra',
        'AirPods Pro 2', 'iMac 24"', 'Mac Studio', 'Apple TV 4K',
        'iPad Air', 'MacBook Pro 16"', 'HomePod mini', 'Magic Keyboard',
        'Vision Pro', 'Studio Display', 'Mac Pro', 'iPad mini'
    ]
    
    regions = [
        'North America', 'Europe', 'Asia Pacific', 'Greater China',
        'Latin America', 'India', 'Japan', 'Rest of Asia Pacific',
        'Middle East', 'Africa', 'Australia', 'South Korea'
    ]
    
    channels = ['Apple Store', 'Online', 'Authorized Reseller', 'Carrier', 'Enterprise', 'Education']
    quarters = ['Q1 2024', 'Q2 2024', 'Q3 2024', 'Q4 2024']
    customer_segments = ['Consumer', 'Education', 'Business', 'Government', 'Healthcare']
    
    data = []
    
    product_tiers = {
        'iPhone 15 Pro': {'price': 1200, 'tier': 'premium'},
        'MacBook Air M3': {'price': 1200, 'tier': 'mid'},
        'iPad Pro': {'price': 900, 'tier': 'premium'},
        'Apple Watch Ultra': {'price': 800, 'tier': 'premium'},
        'AirPods Pro 2': {'price': 250, 'tier': 'mid'},
        'iMac 24"': {'price': 1500, 'tier': 'premium'},
        'Mac Studio': {'price': 2000, 'tier': 'premium'},
        'Apple TV 4K': {'price': 180, 'tier': 'low'},
        'iPad Air': {'price': 600, 'tier': 'mid'},
        'MacBook Pro 16"': {'price': 2500, 'tier': 'premium'},
        'HomePod mini': {'price': 100, 'tier': 'low'},
        'Magic Keyboard': {'price': 300, 'tier': 'low'},
        'Vision Pro': {'price': 3500, 'tier': 'premium'},
        'Studio Display': {'price': 1600, 'tier': 'premium'},
        'Mac Pro': {'price': 7000, 'tier': 'premium'},
        'iPad mini': {'price': 500, 'tier': 'mid'}
    }
    
    # Generate random number of records between 1500-2500
    num_records = np.random.randint(1500, 2500)
    
    for i in range(num_records):
        product = np.random.choice(products)
        region = np.random.choice(regions)
        channel = np.random.choice(channels)
        quarter = np.random.choice(quarters)
        segment = np.random.choice(customer_segments)
        
        base_price = product_tiers[product]['price']
        tier = product_tiers[product]['tier']
        
        # More randomization
        regional_multiplier = {
            'North America': np.random.uniform(0.95, 1.05),
            'Europe': np.random.uniform(1.10, 1.20),
            'Asia Pacific': np.random.uniform(0.90, 1.00),
            'Greater China': np.random.uniform(0.85, 0.95),
            'Latin America': np.random.uniform(0.80, 0.90),
            'India': np.random.uniform(0.75, 0.85),
            'Japan': np.random.uniform(1.05, 1.15),
            'Rest of Asia Pacific': np.random.uniform(0.85, 0.95),
            'Middle East': np.random.uniform(0.90, 1.10),
            'Africa': np.random.uniform(0.70, 0.90),
            'Australia': np.random.uniform(1.00, 1.10),
            'South Korea': np.random.uniform(0.95, 1.05)
        }[region]
        
        channel_multiplier = {
            'Apple Store': np.random.uniform(0.98, 1.02),
            'Online': np.random.uniform(0.95, 1.00),
            'Authorized Reseller': np.random.uniform(1.02, 1.08),
            'Carrier': np.random.uniform(1.00, 1.05),
            'Enterprise': np.random.uniform(0.90, 0.95),
            'Education': np.random.uniform(0.85, 0.90)
        }[channel]
        
        seasonal_multiplier = {
            'Q1 2024': np.random.uniform(0.90, 1.00),
            'Q2 2024': np.random.uniform(0.95, 1.05),
            'Q3 2024': np.random.uniform(1.00, 1.10),
            'Q4 2024': np.random.uniform(1.10, 1.20)
        }[quarter]
        
        price = base_price * regional_multiplier * channel_multiplier * seasonal_multiplier
        
        base_units = {
            'premium': np.random.randint(1, 50),
            'mid': np.random.randint(10, 100),
            'low': np.random.randint(20, 200)
        }[tier]
        
        units = int(base_units * seasonal_multiplier * np.random.uniform(0.7, 1.3))
        units = max(1, units)
        
        revenue = price * units
        
        cost_ratio = {
            'premium': np.random.uniform(0.40, 0.50),
            'mid': np.random.uniform(0.50, 0.60),
            'low': np.random.uniform(0.60, 0.70)
        }[tier]
        
        cost = revenue * cost_ratio
        profit = revenue - cost
        profit_margin = (profit / revenue) * 100
        
        satisfaction = np.random.uniform(3.0, 5.0)
        nps_score = np.random.randint(-20, 100)
        inventory_days = np.random.randint(5, 90)
        lead_time = np.random.randint(1, 30)
        marketing_spend = revenue * np.random.uniform(0.03, 0.08)
        customer_acquisition_cost = marketing_spend / max(1, int(units * 0.6))
        
        warranty_upper_bound = max(1, int(units * 0.2))
        warranty_claims = np.random.randint(0, warranty_upper_bound)
        
        data.append({
            'Product': product,
            'Region': region,
            'Sales_Channel': channel,
            'Quarter': quarter,
            'Customer_Segment': segment,
            'Units_Sold': units,
            'Unit_Price': round(price, 2),
            'Revenue': round(revenue, 2),
            'Cost_of_Goods': round(cost, 2),
            'Gross_Profit': round(profit, 2),
            'Profit_Margin_Percent': round(profit_margin, 2),
            'Customer_Satisfaction': round(satisfaction, 1),
            'NPS_Score': nps_score,
            'Inventory_Days': inventory_days,
            'Lead_Time_Days': lead_time,
            'Marketing_Spend': round(marketing_spend, 2),
            'Customer_Acquisition_Cost': round(customer_acquisition_cost, 2),
            'Product_Tier': tier.title(),
            'Return_Rate_Percent': round(np.random.uniform(0.1, 8.0), 2),
            'Warranty_Claims': warranty_claims
        })
    
    return pd.DataFrame(data)

def display_enhanced_overview(df):
    """Display with horizontal text and correct memory calculation"""
    st.markdown("### 📊 Dataset Intelligence Dashboard")
    
    # Add spacing
    st.markdown("<div style='margin-bottom: 1rem;'></div>", unsafe_allow_html=True)
    
    # Top-level metrics with horizontal text
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        total_records = len(df)
        st.markdown(f"""
        <div class="metric-card">
            <h3>TOTAL RECORDS</h3>
            <h2>{total_records:,}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        total_columns = len(df.columns)
        st.markdown(f"""
        <div class="metric-card">
            <h3>DATA FIELDS</h3>
            <h2>{total_columns}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        st.markdown(f"""
        <div class="metric-card">
            <h3>NUMERIC FIELDS</h3>
            <h2>{len(numeric_cols)}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        data_quality = ((df.shape[0] * df.shape[1]) - df.isnull().sum().sum()) / (df.shape[0] * df.shape[1]) * 100
        st.markdown(f"""
        <div class="metric-card">
            <h3>DATA QUALITY</h3>
            <h2>{data_quality:.1f}%</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col5:
        # FIXED: Correct memory usage calculation
        memory_usage_bytes = df.memory_usage(deep=True).sum()
        memory_usage_mb = memory_usage_bytes / (1024 * 1024)
        st.markdown(f"""
        <div class="metric-card">
            <h3>MEMORY USAGE</h3>
            <h2>{memory_usage_mb:.1f} MB</h2>
        </div>
        """, unsafe_allow_html=True)
    
    # Add spacing after metric cards
    st.markdown("<div style='margin-bottom: 2rem;'></div>", unsafe_allow_html=True)
    
    # Revenue insights if available
    if 'Revenue' in df.columns:
        st.markdown("### 💰 Business Performance Overview")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            total_revenue = df['Revenue'].sum()
            st.markdown(f"""
            <div class="metric-card">
                <h3>TOTAL REVENUE</h3>
                <h2>${total_revenue:,.0f}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            avg_revenue = df['Revenue'].mean()
            st.markdown(f"""
            <div class="metric-card">
                <h3>AVG REVENUE</h3>
                <h2>${avg_revenue:,.0f}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            if 'Gross_Profit' in df.columns:
                total_profit = df['Gross_Profit'].sum()
                st.markdown(f"""
                <div class="metric-card">
                    <h3>TOTAL PROFIT</h3>
                    <h2>${total_profit:,.0f}</h2>
                </div>
                """, unsafe_allow_html=True)
    
    # Add spacing before quick insights
    st.markdown("<div style='margin-bottom: 2rem;'></div>", unsafe_allow_html=True)
    
    # FIXED: Quick visualizations with proper axis labels
    st.markdown("### 📈 Quick Insights")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if 'Revenue' in df.columns and 'Quarter' in df.columns:
            quarterly_revenue = df.groupby('Quarter')['Revenue'].sum().reset_index()
            fig = px.bar(quarterly_revenue, x='Quarter', y='Revenue',
                        title="📊 Revenue by Quarter",
                        color='Revenue',
                        color_continuous_scale='Viridis',
                        labels={
                            'Quarter': 'Quarter',
                            'Revenue': 'Revenue ($)',
                            'x': 'Quarter',
                            'y': 'Revenue ($)'
                        })
            
            # FIXED: Proper axis formatting without duplicate parameters
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='#1e293b',
                title_x=0.5,
                xaxis=dict(
                    title="Quarter",
                    title_font=dict(size=14, color='#1e293b')
                ),
                yaxis=dict(
                    title="Revenue ($)",
                    title_font=dict(size=14, color='#1e293b')
                )
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        if 'Customer_Satisfaction' in df.columns:
            fig = px.histogram(df, x='Customer_Satisfaction',
                             title="😊 Customer Satisfaction Distribution",
                             nbins=20,
                             color_discrete_sequence=['#667eea'],
                             labels={
                                 'Customer_Satisfaction': 'Customer Satisfaction Rating',
                                 'count': 'Number of Customers',
                                 'x': 'Customer Satisfaction Rating',
                                 'y': 'Number of Customers'
                             })
            
            # FIXED: Proper axis formatting without duplicate parameters
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='#1e293b',
                title_x=0.5,
                xaxis=dict(
                    title="Customer Satisfaction Rating (1-5)",
                    title_font=dict(size=14, color='#1e293b')
                ),
                yaxis=dict(
                    title="Number of Customers",
                    title_font=dict(size=14, color='#1e293b')
                )
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    # Data preview
    st.markdown("### 👀 Data Sample")
    st.dataframe(df.head(10), use_container_width=True, height=400)

def create_advanced_visualization(df, chart_type, x_col, y_col, color_col=None, size_col=None):
    """FIXED: Create visualizations with CLEAR axis labels and no duplicate parameters"""
    
    try:
        if chart_type == "Interactive Bar Chart":
            if color_col and color_col in df.columns:
                fig = px.bar(df, x=x_col, y=y_col, color=color_col,
                            title=f"📊 {y_col} by {x_col}",
                            color_discrete_sequence=px.colors.qualitative.Set3,
                            labels={
                                x_col: f'{x_col}',
                                y_col: f'{y_col}',
                                color_col: f'{color_col}'
                            })
            else:
                fig = px.bar(df, x=x_col, y=y_col,
                            title=f"📊 {y_col} by {x_col}",
                            color_discrete_sequence=['#667eea'],
                            labels={
                                x_col: f'{x_col}',
                                y_col: f'{y_col}'
                            })
            
        elif chart_type == "Animated Scatter Plot":
            fig = px.scatter(df, x=x_col, y=y_col, 
                            color=color_col if color_col else None, 
                            size=size_col if size_col else None,
                            title=f"🎯 {y_col} vs {x_col}",
                            hover_data=df.columns[:5].tolist(),
                            labels={
                                x_col: f'{x_col}',
                                y_col: f'{y_col}',
                                color_col: f'{color_col}' if color_col else None,
                                size_col: f'{size_col}' if size_col else None
                            })
            
        elif chart_type == "3D Surface Plot":
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if len(numeric_cols) >= 3 and y_col in numeric_cols:
                z_col = [col for col in numeric_cols if col not in [x_col, y_col]][0] if len(numeric_cols) > 2 else y_col
                fig = px.scatter_3d(df.head(500), x=x_col, y=y_col, z=z_col,
                                  color=color_col if color_col else None,
                                  title=f"🌐 3D: {x_col}, {y_col}, {z_col}",
                                  labels={
                                      x_col: f'{x_col}',
                                      y_col: f'{y_col}',
                                      z_col: f'{z_col}',
                                      color_col: f'{color_col}' if color_col else None
                                  })
            else:
                fig = px.scatter(df, x=x_col, y=y_col, color=color_col if color_col else None,
                               title=f"🎯 {y_col} vs {x_col}",
                               labels={
                                   x_col: f'{x_col}',
                                   y_col: f'{y_col}',
                                   color_col: f'{color_col}' if color_col else None
                               })
                
        elif chart_type == "Heatmap Correlation":
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if len(numeric_cols) > 1:
                corr_matrix = df[numeric_cols].corr()
                fig = px.imshow(corr_matrix,
                              title="🔥 Correlation Heatmap",
                              color_continuous_scale="RdBu",
                              aspect="auto",
                              labels={
                                  'x': 'Variables',
                                  'y': 'Variables',
                                  'color': 'Correlation'
                              })
            else:
                fig = px.bar(df.head(20), x=x_col, y=y_col,
                           title=f"📊 {y_col} by {x_col}",
                           labels={
                               x_col: f'{x_col}',
                               y_col: f'{y_col}'
                           })
                
        elif chart_type == "Box Plot":
            fig = px.box(df, x=x_col, y=y_col, color=color_col if color_col else None,
                        title=f"📦 {y_col} Distribution by {x_col}",
                        labels={
                            x_col: f'{x_col}',
                            y_col: f'{y_col}',
                            color_col: f'{color_col}' if color_col else None
                        })
            
        elif chart_type == "Violin Plot":
            fig = px.violin(df, x=x_col, y=y_col, color=color_col if color_col else None,
                           title=f"🎻 {y_col} Distribution by {x_col}",
                           labels={
                               x_col: f'{x_col}',
                               y_col: f'{y_col}',
                               color_col: f'{color_col}' if color_col else None
                           })
            
        else:  # Default chart
            if df[x_col].dtype == 'object':
                fig = px.bar(df, x=x_col, y=y_col, color=color_col if color_col else None,
                           title=f"📈 {y_col} by {x_col}",
                           labels={
                               x_col: f'{x_col}',
                               y_col: f'{y_col}',
                               color_col: f'{color_col}' if color_col else None
                           })
            else:
                fig = px.line(df, x=x_col, y=y_col, color=color_col if color_col else None,
                            title=f"📈 {y_col} Trend by {x_col}",
                            labels={
                                x_col: f'{x_col}',
                                y_col: f'{y_col}',
                                color_col: f'{color_col}' if color_col else None
                            })
        
        # FIXED: Enhanced styling with CLEAR axis labels (NO DUPLICATE PARAMETERS)
        fig.update_layout(
            plot_bgcolor='rgba(255,255,255,0.9)',
            paper_bgcolor='rgba(255,255,255,0.9)',
            font=dict(color='#1e293b', size=12),
            title_font_size=18,
            title_font_color='#1e293b',
            title_x=0.5,
            height=500,
            showlegend=True,
            hovermode='closest',
            xaxis=dict(
                title=f'{x_col}',
                title_font=dict(size=14, color='#1e293b')
            ),
            yaxis=dict(
                title=f'{y_col}' if y_col else 'Values',
                title_font=dict(size=14, color='#1e293b')
            )
        )
        
        return fig
        
    except Exception as e:
        st.error(f"Error creating visualization: {str(e)}")
        return None

# FIXED: SPECIFIC analysis functions for each insight type
def get_business_overview_analysis(df):
    """Complete Business Overview - SPECIFIC analysis"""
    insights = []
    
    # Financial Analysis
    if 'Revenue' in df.columns:
        total_rev = df['Revenue'].sum()
        avg_rev = df['Revenue'].mean()
        insights.append(f"💰 **Financial Performance**: ${total_rev:,.2f} total revenue, ${avg_rev:,.2f} average per transaction")
        
        if 'Gross_Profit' in df.columns:
            total_profit = df['Gross_Profit'].sum()
            margin = (total_profit/total_rev*100)
            insights.append(f"📈 **Profitability**: ${total_profit:,.2f} gross profit with {margin:.1f}% overall margin")
    
    # Product Portfolio Analysis
    if 'Product' in df.columns:
        product_count = df['Product'].nunique()
        top_product = df['Product'].value_counts().index[0]
        insights.append(f"🛍️ **Product Portfolio**: {product_count} unique products, '{top_product}' leads in transaction volume")
        
        if 'Revenue' in df.columns:
            product_revenue = df.groupby('Product')['Revenue'].sum().sort_values(ascending=False)
            revenue_leader = product_revenue.index[0]
            insights.append(f"🏆 **Revenue Leader**: '{revenue_leader}' generates ${product_revenue.iloc[0]:,.2f} ({product_revenue.iloc[0]/df['Revenue'].sum()*100:.1f}% of total)")
    
    # Market Analysis
    if 'Region' in df.columns:
        regions = df['Region'].nunique()
        top_region = df['Region'].value_counts().index[0]
        insights.append(f"🌍 **Geographic Reach**: Active in {regions} regions, '{top_region}' shows highest activity")
        
        if 'Revenue' in df.columns:
            regional_revenue = df.groupby('Region')['Revenue'].sum().sort_values(ascending=False)
            insights.append(f"💰 **Top Market**: '{regional_revenue.index[0]}' leads with ${regional_revenue.iloc[0]:,.2f} revenue")
    
    # Customer Satisfaction
    if 'Customer_Satisfaction' in df.columns:
        avg_sat = df['Customer_Satisfaction'].mean()
        high_sat = len(df[df['Customer_Satisfaction'] >= 4.0])
        sat_rate = (high_sat/len(df)*100)
        insights.append(f"😊 **Customer Experience**: {avg_sat:.2f}/5.0 average satisfaction, {sat_rate:.1f}% highly satisfied customers")
    
    # Operational Metrics
    if 'Units_Sold' in df.columns:
        total_units = df['Units_Sold'].sum()
        avg_units = df['Units_Sold'].mean()
        insights.append(f"📦 **Volume Metrics**: {total_units:,} total units sold, {avg_units:.1f} average per transaction")
    
    return f"""📊 **Complete Business Intelligence Overview**

🎯 **Executive Summary:**
{chr(10).join(insights)}

📋 **Data Foundation:**
• {len(df):,} transactions analyzed across {len(df.columns)} data dimensions
• {((df.shape[0] * df.shape[1] - df.isnull().sum().sum()) / (df.shape[0] * df.shape[1]) * 100):.1f}% data completeness ensures reliable insights

💡 **Strategic Insights:**
• Strong data foundation supports confident business decisions
• Multiple performance vectors tracked for comprehensive analysis
• Balanced portfolio across products, regions, and customer segments"""

def get_revenue_profitability_analysis(df):
    """Revenue & Profitability Analysis - SPECIFIC analysis"""
    if 'Revenue' not in df.columns:
        return "💰 Revenue analysis requires financial data in your dataset."
    
    total_revenue = df['Revenue'].sum()
    avg_revenue = df['Revenue'].mean()
    revenue_std = df['Revenue'].std()
    
    insights = []
    insights.append(f"💰 **Revenue Overview**: ${total_revenue:,.2f} total, ${avg_revenue:,.2f} average per transaction")
    insights.append(f"📊 **Revenue Distribution**: ${revenue_std:,.2f} standard deviation shows {'high' if revenue_std > avg_revenue else 'moderate'} variability")
    
    # Profitability Analysis
    if 'Gross_Profit' in df.columns:
        total_profit = df['Gross_Profit'].sum()
        avg_margin = (total_profit/total_revenue*100)
        insights.append(f"📈 **Profitability**: ${total_profit:,.2f} gross profit with {avg_margin:.1f}% overall margin")
        
        # Margin by Product
        if 'Product' in df.columns:
            product_margins = df.groupby('Product').apply(lambda x: (x['Gross_Profit'].sum()/x['Revenue'].sum()*100)).sort_values(ascending=False)
            best_margin = product_margins.index[0]
            insights.append(f"🎯 **Best Margins**: '{best_margin}' leads with {product_margins.iloc[0]:.1f}% margin")
    
    # Revenue Concentration
    if 'Product' in df.columns:
        product_revenue = df.groupby('Product')['Revenue'].sum().sort_values(ascending=False)
        top3_share = (product_revenue.head(3).sum()/total_revenue*100)
        insights.append(f"📊 **Revenue Concentration**: Top 3 products account for {top3_share:.1f}% of revenue")
    
    # Customer Value Analysis
    if 'Customer_Segment' in df.columns:
        segment_revenue = df.groupby('Customer_Segment')['Revenue'].sum().sort_values(ascending=False)
        top_segment = segment_revenue.index[0]
        insights.append(f"🎯 **Valuable Segments**: '{top_segment}' generates ${segment_revenue.iloc[0]:,.2f} ({segment_revenue.iloc[0]/total_revenue*100:.1f}% of revenue)")
    
    return f"""💰 **Revenue & Profitability Deep Dive**

🔍 **Financial Performance:**
{chr(10).join(insights)}

📈 **Growth Opportunities:**
• Optimize product mix focusing on high-margin offerings
• Expand successful customer segments
• Address revenue concentration risks through diversification

⚡ **Action Items:**
• Investigate underperforming products for improvement potential
• Scale marketing efforts in high-value customer segments
• Implement pricing strategies to improve overall margins"""

def get_performance_benchmarking_analysis(df):
    """Performance Benchmarking - SPECIFIC analysis"""
    insights = []
    
    # Product Performance Comparison
    if 'Product' in df.columns and 'Revenue' in df.columns:
        product_performance = df.groupby('Product')['Revenue'].agg(['sum', 'mean', 'count']).sort_values('sum', ascending=False)
        top_performer = product_performance.index[0]
        insights.append(f"🏆 **Top Performer**: '{top_performer}' leads with ${product_performance.loc[top_performer, 'sum']:,.2f} total revenue")
        
        # Performance vs Volume
        high_volume = product_performance['count'].max()
        high_value = product_performance['mean'].max()
        volume_leader = product_performance['count'].idxmax()
        value_leader = product_performance['mean'].idxmax()
        insights.append(f"📊 **Volume vs Value**: '{volume_leader}' has highest volume ({high_volume} transactions), '{value_leader}' highest value (${high_value:,.2f} avg)")
    
    # Regional Performance Comparison
    if 'Region' in df.columns and 'Revenue' in df.columns:
        regional_performance = df.groupby('Region')['Revenue'].agg(['sum', 'mean']).sort_values('sum', ascending=False)
        best_region = regional_performance.index[0]
        worst_region = regional_performance.index[-1]
        performance_gap = regional_performance.loc[best_region, 'sum'] - regional_performance.loc[worst_region, 'sum']
        insights.append(f"🌍 **Regional Leaders**: '{best_region}' outperforms '{worst_region}' by ${performance_gap:,.2f}")
    
    # Channel Performance
    if 'Sales_Channel' in df.columns and 'Revenue' in df.columns:
        channel_performance = df.groupby('Sales_Channel')['Revenue'].agg(['sum', 'mean']).sort_values('sum', ascending=False)
        top_channel = channel_performance.index[0]
        insights.append(f"📈 **Channel Performance**: '{top_channel}' leads channels with ${channel_performance.loc[top_channel, 'sum']:,.2f} revenue")
    
    # Customer Segment Benchmarking
    if 'Customer_Segment' in df.columns:
        if 'Revenue' in df.columns:
            segment_performance = df.groupby('Customer_Segment')['Revenue'].agg(['sum', 'mean']).sort_values('sum', ascending=False)
            top_segment = segment_performance.index[0]
            insights.append(f"🎯 **Segment Leaders**: '{top_segment}' segment shows strongest performance")
        
        if 'Customer_Satisfaction' in df.columns:
            segment_satisfaction = df.groupby('Customer_Segment')['Customer_Satisfaction'].mean().sort_values(ascending=False)
            happiest_segment = segment_satisfaction.index[0]
            insights.append(f"😊 **Satisfaction Leaders**: '{happiest_segment}' segment most satisfied ({segment_satisfaction.iloc[0]:.2f}/5.0)")
    
    return f"""🎯 **Performance Benchmarking Analysis**

🏁 **Competitive Performance:**
{chr(10).join(insights)}

📊 **Benchmark Insights:**
• Clear performance differentiation across segments
• Top performers set achievable targets for others
• Performance gaps indicate optimization opportunities

🚀 **Optimization Strategy:**
• Replicate top performer strategies across portfolio
• Address underperforming segments with targeted interventions  
• Balance volume and value for sustainable growth"""

def get_trend_analysis(df):
    """Trend Analysis - SPECIFIC analysis"""
    insights = []
    
    # Quarterly Trends
    if 'Quarter' in df.columns and 'Revenue' in df.columns:
        quarterly_revenue = df.groupby('Quarter')['Revenue'].sum().sort_index()
        q1_rev = quarterly_revenue.get('Q1 2024', 0)
        q4_rev = quarterly_revenue.get('Q4 2024', 0)
        
        if q1_rev > 0 and q4_rev > 0:
            growth = ((q4_rev - q1_rev) / q1_rev * 100)
            trend_direction = "📈 Growing" if growth > 0 else "📉 Declining" if growth < -5 else "➡️ Stable"
            insights.append(f"📅 **Quarterly Trend**: {trend_direction} - {abs(growth):.1f}% change from Q1 to Q4")
        
        best_quarter = quarterly_revenue.idxmax()
        best_performance = quarterly_revenue.max()
        insights.append(f"🏆 **Peak Performance**: {best_quarter} achieved ${best_performance:,.2f} revenue")
        
        # Seasonal patterns
        seasonal_variance = quarterly_revenue.std() / quarterly_revenue.mean() * 100
        insights.append(f"🌊 **Seasonality**: {seasonal_variance:.1f}% variance indicates {'strong' if seasonal_variance > 15 else 'moderate'} seasonal patterns")
    
    # Product Lifecycle Trends
    if 'Product' in df.columns and 'Quarter' in df.columns and 'Revenue' in df.columns:
        product_trends = df.groupby(['Product', 'Quarter'])['Revenue'].sum().unstack(fill_value=0)
        
        # Identify growing products
        growing_products = []
        for product in product_trends.index:
            if 'Q1 2024' in product_trends.columns and 'Q4 2024' in product_trends.columns:
                q1 = product_trends.loc[product, 'Q1 2024']
                q4 = product_trends.loc[product, 'Q4 2024']
                if q1 > 0 and q4 > q1 * 1.1:  # 10% growth threshold
                    growing_products.append(product)
        
        if growing_products:
            insights.append(f"📈 **Growth Stars**: {', '.join(growing_products[:3])} showing strong upward trends")
    
    # Customer Satisfaction Trends
    if 'Customer_Satisfaction' in df.columns and 'Quarter' in df.columns:
        satisfaction_trend = df.groupby('Quarter')['Customer_Satisfaction'].mean().sort_index()
        if len(satisfaction_trend) > 1:
            sat_change = satisfaction_trend.iloc[-1] - satisfaction_trend.iloc[0]
            trend_desc = "improving" if sat_change > 0.1 else "declining" if sat_change < -0.1 else "stable"
            insights.append(f"😊 **Satisfaction Trend**: Customer satisfaction is {trend_desc} ({sat_change:+.2f} change)")
    
    # Regional Growth Patterns
    if 'Region' in df.columns and 'Quarter' in df.columns and 'Revenue' in df.columns:
        regional_growth = df.groupby(['Region', 'Quarter'])['Revenue'].sum().unstack(fill_value=0)
        
        growth_rates = {}
        for region in regional_growth.index:
            if 'Q1 2024' in regional_growth.columns and 'Q4 2024' in regional_growth.columns:
                q1 = regional_growth.loc[region, 'Q1 2024']
                q4 = regional_growth.loc[region, 'Q4 2024']
                if q1 > 0:
                    growth_rates[region] = (q4 - q1) / q1 * 100
        
        if growth_rates:
            fastest_growing = max(growth_rates, key=growth_rates.get)
            insights.append(f"🌍 **Regional Momentum**: '{fastest_growing}' shows fastest growth at {growth_rates[fastest_growing]:+.1f}%")
    
    return f"""📈 **Trend Analysis & Pattern Recognition**

🔍 **Key Trend Insights:**
{chr(10).join(insights)}

📊 **Pattern Analysis:**
• Seasonal patterns provide planning opportunities
• Growth trajectories indicate market momentum
• Performance trends reveal strategic directions

🎯 **Trend-Based Recommendations:**
• Capitalize on seasonal peaks with targeted campaigns
• Invest in growth star products for maximum return
• Address declining trends with intervention strategies
• Expand successful regional patterns to other markets"""

def get_customer_experience_analysis(df):
    """Customer Experience Analysis - SPECIFIC analysis"""
    if 'Customer_Satisfaction' not in df.columns:
        return "😊 Customer experience analysis requires satisfaction data in your dataset."
    
    insights = []
    
    # Overall Satisfaction Analysis
    avg_satisfaction = df['Customer_Satisfaction'].mean()
    satisfaction_std = df['Customer_Satisfaction'].std()
    high_satisfaction = len(df[df['Customer_Satisfaction'] >= 4.0])
    satisfaction_rate = (high_satisfaction/len(df)*100)
    
    insights.append(f"⭐ **Overall Experience**: {avg_satisfaction:.2f}/5.0 average satisfaction with {satisfaction_rate:.1f}% highly satisfied")
    insights.append(f"📊 **Experience Consistency**: {satisfaction_std:.2f} standard deviation shows {'consistent' if satisfaction_std < 0.5 else 'variable'} experience delivery")
    
    # Satisfaction by Product
    if 'Product' in df.columns:
        product_satisfaction = df.groupby('Product')['Customer_Satisfaction'].mean().sort_values(ascending=False)
        best_product = product_satisfaction.index[0]
        worst_product = product_satisfaction.index[-1]
        insights.append(f"🏆 **Experience Champions**: '{best_product}' leads with {product_satisfaction.iloc[0]:.2f}/5.0 satisfaction")
        insights.append(f"⚠️ **Attention Needed**: '{worst_product}' needs improvement at {product_satisfaction.iloc[-1]:.2f}/5.0")
    
    # Satisfaction by Customer Segment
    if 'Customer_Segment' in df.columns:
        segment_satisfaction = df.groupby('Customer_Segment')['Customer_Satisfaction'].agg(['mean', 'count']).sort_values('mean', ascending=False)
        happiest_segment = segment_satisfaction.index[0]
        insights.append(f"😊 **Happiest Customers**: '{happiest_segment}' segment most satisfied ({segment_satisfaction.loc[happiest_segment, 'mean']:.2f}/5.0)")
    
    # NPS Analysis if available
    if 'NPS_Score' in df.columns:
        avg_nps = df['NPS_Score'].mean()
        promoters = len(df[df['NPS_Score'] >= 70])
        detractors = len(df[df['NPS_Score'] <= 30])
        nps_classification = "Excellent" if avg_nps > 50 else "Good" if avg_nps > 0 else "Needs Improvement"
        
        insights.append(f"📈 **NPS Performance**: {avg_nps:.1f} average NPS ({nps_classification})")
        insights.append(f"🎉 **Advocacy**: {promoters} promoters vs {detractors} detractors")
    
    # Regional Experience Variations
    if 'Region' in df.columns:
        regional_satisfaction = df.groupby('Region')['Customer_Satisfaction'].mean().sort_values(ascending=False)
        best_region = regional_satisfaction.index[0]
        experience_gap = regional_satisfaction.iloc[0] - regional_satisfaction.iloc[-1]
        insights.append(f"🌍 **Regional Excellence**: '{best_region}' delivers best experience ({regional_satisfaction.iloc[0]:.2f}/5.0)")
        insights.append(f"📊 **Experience Gap**: {experience_gap:.2f} point difference between best and worst regions")
    
    # Channel Experience
    if 'Sales_Channel' in df.columns:
        channel_satisfaction = df.groupby('Sales_Channel')['Customer_Satisfaction'].mean().sort_values(ascending=False)
        best_channel = channel_satisfaction.index[0]
        insights.append(f"📱 **Channel Excellence**: '{best_channel}' provides superior experience ({channel_satisfaction.iloc[0]:.2f}/5.0)")
    
    return f"""😊 **Customer Experience Deep Dive**

🎯 **Experience Overview:**
{chr(10).join(insights)}

💡 **Experience Insights:**
• Customer satisfaction directly impacts loyalty and retention
• Experience variations highlight improvement opportunities
• Consistent delivery across all touchpoints is crucial

🚀 **Experience Enhancement Strategy:**
• Replicate best-performing product/channel experiences
• Address satisfaction gaps through targeted improvements
• Implement feedback loops for continuous experience optimization
• Develop segment-specific experience strategies"""

def get_operational_efficiency_analysis(df):
    """Operational Efficiency Analysis - SPECIFIC analysis"""
    insights = []
    
    # Inventory Efficiency
    if 'Inventory_Days' in df.columns:
        avg_inventory = df['Inventory_Days'].mean()
        inventory_std = df['Inventory_Days'].std()
        efficiency_rating = "Excellent" if avg_inventory < 30 else "Good" if avg_inventory < 60 else "Needs Improvement"
        insights.append(f"📦 **Inventory Efficiency**: {avg_inventory:.1f} days average cycle ({efficiency_rating})")
        insights.append(f"📊 **Inventory Consistency**: {inventory_std:.1f} days variation shows {'efficient' if inventory_std < 20 else 'inconsistent'} management")
    
    # Lead Time Analysis
    if 'Lead_Time_Days' in df.columns:
        avg_lead_time = df['Lead_Time_Days'].mean()
        lead_time_std = df['Lead_Time_Days'].std()
        insights.append(f"⏱️ **Lead Time Performance**: {avg_lead_time:.1f} days average delivery time")
        insights.append(f"🎯 **Delivery Consistency**: {lead_time_std:.1f} days variation in delivery times")
    
    # Cost Efficiency
    if 'Cost_of_Goods' in df.columns and 'Revenue' in df.columns:
        total_costs = df['Cost_of_Goods'].sum()
        total_revenue = df['Revenue'].sum()
        cost_ratio = (total_costs / total_revenue * 100)
        efficiency_score = "Excellent" if cost_ratio < 60 else "Good" if cost_ratio < 70 else "Review Needed"
        insights.append(f"💰 **Cost Efficiency**: {cost_ratio:.1f}% cost-to-revenue ratio ({efficiency_score})")
    
    # Product Efficiency by Category
    if 'Product_Tier' in df.columns and 'Units_Sold' in df.columns:
        tier_efficiency = df.groupby('Product_Tier')['Units_Sold'].sum().sort_values(ascending=False)
        most_efficient = tier_efficiency.index[0]
        insights.append(f"🏆 **Volume Efficiency**: '{most_efficient}' tier shows highest throughput ({tier_efficiency.iloc[0]:,} units)")
    
    # Channel Efficiency
    if 'Sales_Channel' in df.columns and 'Revenue' in df.columns and 'Units_Sold' in df.columns:
        channel_efficiency = df.groupby('Sales_Channel').agg({
            'Revenue': 'sum',
            'Units_Sold': 'sum'
        })
        channel_efficiency['Revenue_per_Unit'] = channel_efficiency['Revenue'] / channel_efficiency['Units_Sold']
        most_efficient_channel = channel_efficiency['Revenue_per_Unit'].idxmax()
        insights.append(f"📈 **Channel Efficiency**: '{most_efficient_channel}' generates highest revenue per unit (${channel_efficiency.loc[most_efficient_channel, 'Revenue_per_Unit']:.2f})")
    
    # Marketing Efficiency
    if 'Marketing_Spend' in df.columns and 'Revenue' in df.columns:
        total_marketing = df['Marketing_Spend'].sum()
        total_revenue = df['Revenue'].sum()
        marketing_roi = (total_revenue / total_marketing) if total_marketing > 0 else 0
        roi_rating = "Excellent" if marketing_roi > 5 else "Good" if marketing_roi > 3 else "Review Needed"
        insights.append(f"📊 **Marketing ROI**: ${marketing_roi:.1f} return per dollar spent ({roi_rating})")
    
    # Customer Acquisition Efficiency
    if 'Customer_Acquisition_Cost' in df.columns and 'Revenue' in df.columns:
        avg_cac = df['Customer_Acquisition_Cost'].mean()
        avg_revenue = df['Revenue'].mean()
        cac_payback = avg_revenue / avg_cac if avg_cac > 0 else 0
        efficiency_level = "Excellent" if cac_payback > 3 else "Good" if cac_payback > 1.5 else "Review Needed"
        insights.append(f"🎯 **Acquisition Efficiency**: ${avg_cac:.2f} average CAC with {cac_payback:.1f}x payback ({efficiency_level})")
    
    # Return Rate Analysis
    if 'Return_Rate_Percent' in df.columns:
        avg_return_rate = df['Return_Rate_Percent'].mean()
        return_efficiency = "Excellent" if avg_return_rate < 3 else "Good" if avg_return_rate < 6 else "Needs Attention"
        insights.append(f"🔄 **Return Efficiency**: {avg_return_rate:.1f}% average return rate ({return_efficiency})")
    
    return f"""⚙️ **Operational Efficiency Analysis**

🔧 **Efficiency Metrics:**
{chr(10).join(insights)}

📈 **Operational Insights:**
• Efficient operations drive profitability and customer satisfaction
• Consistency in operations builds predictable performance
• Cross-functional efficiency optimization yields compound benefits

🎯 **Efficiency Optimization:**
• Streamline high-impact operational bottlenecks
• Standardize best practices across all channels
• Implement continuous monitoring for efficiency metrics
• Balance cost efficiency with service quality standards"""

# FIXED: Updated get_enhanced_ai_analysis to use specific functions
def get_enhanced_ai_analysis(df, question):
    """Enhanced AI analysis with SPECIFIC responses based on question type"""
    
    # Map question types to specific analysis functions
    if "business intelligence summary" in question.lower() or "complete business overview" in question.lower():
        return get_business_overview_analysis(df)
    elif "revenue" in question.lower() and "profitability" in question.lower():
        return get_revenue_profitability_analysis(df)
    elif "performance benchmarking" in question.lower() or "compare performance" in question.lower():
        return get_performance_benchmarking_analysis(df)
    elif "trend analysis" in question.lower() or "trends" in question.lower() or "growth opportunities" in question.lower():
        return get_trend_analysis(df)
    elif "customer experience" in question.lower() or "customer satisfaction" in question.lower():
        return get_customer_experience_analysis(df)
    elif "operational efficiency" in question.lower() or "supply chain" in question.lower() or "inventory" in question.lower():
        return get_operational_efficiency_analysis(df)
    else:
        # Try Gemini API for other questions
        try:
            sample_data = df.head(10).to_dict('records')
            data_summary = {
                'shape': df.shape,
                'columns': df.columns.tolist(),
                'sample_records': sample_data,
                'statistics': df.describe().to_dict() if len(df.select_dtypes(include=[np.number]).columns) > 0 else {}
            }
            
            prompt = f"""You are an expert data analyst. Analyze this dataset and answer the specific question.

DATASET: {data_summary['shape'][0]} rows × {data_summary['shape'][1]} columns
COLUMNS: {', '.join(data_summary['columns'])}

SAMPLE DATA:
{json.dumps(sample_data, indent=2, default=str)}

STATISTICS:
{json.dumps(data_summary['statistics'], indent=2, default=str)}

QUESTION: {question}

Provide specific, data-driven insights with actual numbers from the dataset."""

            response = model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.1,
                    top_p=0.8,
                    max_output_tokens=2000,
                ),
            )
            return response.text
        except:
            return f"""📊 **Analysis for: "{question}"**

🔍 **Dataset Overview:**
• {len(df):,} records with {len(df.columns)} fields
• Data quality: {((df.shape[0] * df.shape[1] - df.isnull().sum().sum()) / (df.shape[0] * df.shape[1]) * 100):.1f}% complete

📈 **Key Findings:**
Based on your dataset structure, I can provide detailed analysis on:
• Business performance and financial metrics
• Customer experience and satisfaction
• Operational efficiency and trends
• Product and regional performance

💡 **Next Steps:**
Use the specific insight buttons for detailed analysis tailored to your question type."""

def execute_quick_analysis(df, analysis_type):
    """Execute quick analysis that actually works with SPECIFIC responses"""
    
    if analysis_type == "Summarize key business insights":
        return get_business_overview_analysis(df)
    elif analysis_type == "What drives highest revenue?":
        return get_revenue_profitability_analysis(df)
    elif analysis_type == "Show performance trends":
        return get_trend_analysis(df)
    elif analysis_type == "Compare regional performance":
        return get_performance_benchmarking_analysis(df)
    elif analysis_type == "Analyze customer satisfaction":
        return get_customer_experience_analysis(df)
    elif analysis_type == "Identify growth opportunities":
        return get_operational_efficiency_analysis(df)
    else:
        return "🔧 Analysis function not found. Please try one of the specific insight buttons."

def generate_business_insights(df):
    """Generate comprehensive business intelligence dashboard"""
    
    insights = {
        'revenue_insights': [],
        'product_insights': [],
        'regional_insights': [],
        'customer_insights': [],
        'operational_insights': [],
        'recommendations': []
    }
    
    # Revenue Analysis
    if 'Revenue' in df.columns:
        total_revenue = df['Revenue'].sum()
        avg_revenue = df['Revenue'].mean()
        revenue_std = df['Revenue'].std()
        
        insights['revenue_insights'] = [
            f"💰 Total Revenue: ${total_revenue:,.2f}",
            f"📊 Average Transaction: ${avg_revenue:,.2f}",
            f"📈 Revenue Range: ${df['Revenue'].min():,.2f} - ${df['Revenue'].max():,.2f}",
            f"📉 Revenue Variability: ${revenue_std:,.2f} (standard deviation)",
            f"🎯 High-value Transactions: {len(df[df['Revenue'] > avg_revenue]):,} above average"
        ]
    
    # Product Analysis
    if 'Product' in df.columns:
        product_counts = df['Product'].value_counts()
        top_product = product_counts.index[0]
        product_diversity = len(product_counts)
        
        insights['product_insights'] = [
            f"🏆 Top Product: {top_product} ({product_counts.iloc[0]:,} transactions)",
            f"📦 Product Portfolio: {product_diversity} unique products",
            f"📊 Average Transactions/Product: {product_counts.mean():.1f}",
            f"🎯 Product Concentration: Top 3 = {(product_counts.head(3).sum()/product_counts.sum()*100):.1f}% of sales"
        ]
        
        if 'Revenue' in df.columns:
            product_revenue = df.groupby('Product')['Revenue'].sum().sort_values(ascending=False)
            revenue_leader = product_revenue.index[0]
            insights['product_insights'].append(f"💰 Revenue Champion: {revenue_leader} (${product_revenue.iloc[0]:,.2f})")
    
    # Regional Analysis
    if 'Region' in df.columns:
        region_counts = df['Region'].value_counts()
        top_region = region_counts.index[0]
        regional_diversity = len(region_counts)
        
        insights['regional_insights'] = [
            f"🌍 Most Active Region: {top_region} ({region_counts.iloc[0]:,} transactions)",
            f"🗺️ Geographic Reach: {regional_diversity} regions",
            f"📊 Regional Balance Score: {(100-region_counts.std()/region_counts.mean()*100):.1f}/100"
        ]
        
        if 'Revenue' in df.columns:
            regional_revenue = df.groupby('Region')['Revenue'].sum().sort_values(ascending=False)
            revenue_region = regional_revenue.index[0]
            insights['regional_insights'].append(f"💰 Revenue Leader: {revenue_region} (${regional_revenue.iloc[0]:,.2f})")
    
    # Customer Analysis
    if 'Customer_Satisfaction' in df.columns:
        avg_satisfaction = df['Customer_Satisfaction'].mean()
        satisfaction_std = df['Customer_Satisfaction'].std()
        high_satisfaction = (df['Customer_Satisfaction'] >= 4.0).sum()
        
        insights['customer_insights'] = [
            f"⭐ Average Satisfaction: {avg_satisfaction:.2f}/5.0",
            f"📊 Satisfaction Consistency: {satisfaction_std:.2f} (lower is better)",
            f"😊 Highly Satisfied Customers: {high_satisfaction:,} ({(high_satisfaction/len(df)*100):.1f}%)",
            f"🎯 Satisfaction Range: {df['Customer_Satisfaction'].min():.1f} - {df['Customer_Satisfaction'].max():.1f}"
        ]
    
    if 'NPS_Score' in df.columns:
        avg_nps = df['NPS_Score'].mean()
        promoters = (df['NPS_Score'] >= 70).sum()
        insights['customer_insights'].extend([
            f"📈 Net Promoter Score: {avg_nps:.1f}",
            f"🎉 Promoters: {promoters:,} customers ({(promoters/len(df)*100):.1f}%)"
        ])
    
    # Operational Insights
    if 'Units_Sold' in df.columns:
        total_units = df['Units_Sold'].sum()
        avg_units = df['Units_Sold'].mean()
        
        insights['operational_insights'] = [
            f"📦 Total Units Sold: {total_units:,}",
            f"📊 Average Units/Transaction: {avg_units:.1f}",
            f"🎯 Largest Transaction: {df['Units_Sold'].max():,} units",
            f"📈 Volume Distribution: {len(df[df['Units_Sold'] > avg_units]):,} above-average transactions"
        ]
    
    if 'Inventory_Days' in df.columns:
        avg_inventory = df['Inventory_Days'].mean()
        insights['operational_insights'].append(f"⏰ Average Inventory Cycle: {avg_inventory:.1f} days")
    
    # Generate Smart Recommendations
    recommendations = []
    
    if 'Revenue' in df.columns and 'Product' in df.columns:
        product_revenue = df.groupby('Product')['Revenue'].sum().sort_values(ascending=False)
        top_product = product_revenue.index[0]
        bottom_product = product_revenue.index[-1]
        
        recommendations.extend([
            f"🚀 Double down on {top_product} marketing - your revenue champion",
            f"🔍 Analyze {bottom_product} performance - identify improvement opportunities",
            f"💡 Consider bundling high and low performers for cross-selling"
        ])
    
    if 'Customer_Satisfaction' in df.columns:
        avg_satisfaction = df['Customer_Satisfaction'].mean()
        if avg_satisfaction < 4.0:
            recommendations.append("⚠️ Priority: Customer satisfaction below 4.0 - implement feedback program")
        else:
            recommendations.append("✅ Strong satisfaction scores - leverage for testimonials and referrals")
    
    if 'Region' in df.columns and 'Revenue' in df.columns:
        regional_performance = df.groupby('Region')['Revenue'].sum()
        performance_gap = regional_performance.max() - regional_performance.min()
        recommendations.append(f"🌍 Address ${performance_gap:,.2f} regional revenue gap - replicate best practices")
    
    insights['recommendations'] = recommendations if recommendations else [
        "📈 Strong overall performance - focus on scaling successful strategies",
        "🎯 Consider market expansion and product diversification",
        "💡 Implement data-driven decision making processes"
    ]
    
    return insights

# FIXED: Export functionality
def create_export_content(analysis_history, df):
    """Create comprehensive export content"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    content = f"""# AI Data Analytics Report
Generated: {timestamp}

## Dataset Overview
- Records: {len(df):,}
- Fields: {len(df.columns)}
- Data Quality: {((df.shape[0] * df.shape[1] - df.isnull().sum().sum()) / (df.shape[0] * df.shape[1]) * 100):.1f}%
- Memory Usage: {df.memory_usage(deep=True).sum() / (1024 * 1024):.2f} MB

## Column Information
{chr(10).join([f"- {col}: {dtype}" for col, dtype in df.dtypes.items()])}

## Analysis History ({len(analysis_history)} analyses)

"""
    
    for i, analysis in enumerate(analysis_history, 1):
        content += f"""### Analysis {i}: {analysis['question']}
**Timestamp:** {analysis['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}

{analysis['result']}

---

"""
    
    # Add statistical summary if numeric data exists
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if numeric_cols:
        content += """## Statistical Summary

"""
        stats = df[numeric_cols].describe()
        content += stats.to_string()
        content += "\n\n"
    
    return content

def main():
    """Main application with all fixes including WORKING export functionality"""
    
    # Enhanced Header
    st.markdown("""
    <div class="main-header">
        <h1>🤖 AI Data Analytics Professional</h1>
        <p>Powered by Google Gemini 2.5 Flash • Advanced Business Intelligence Platform</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Enhanced Sidebar
    with st.sidebar:
        st.markdown("### 🎛️ Analytics Control Center")
        
        # Status indicator
        if st.session_state.df is not None:
            df = st.session_state.df
            st.success(f"""
            **📊 Dataset Active**
            - Records: {len(df):,}
            - Fields: {len(df.columns)}
            - Quality: {((df.shape[0] * df.shape[1] - df.isnull().sum().sum()) / (df.shape[0] * df.shape[1]) * 100):.1f}%
            """)
            
            # Quick data insights
            st.markdown("**🔍 Data Overview**")
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
            
            if numeric_cols:
                st.info(f"**Numeric**: {', '.join(numeric_cols[:3])}")
            if categorical_cols:
                st.info(f"**Categories**: {', '.join(categorical_cols[:3])}")
        
        else:
            st.warning("⚠️ No dataset loaded")
        
        # Quick Analytics with Clear Explanation
        st.markdown("---")
        st.markdown("### ⚡ Quick Analytics")
        
        # Add explanation for Quick Analytics
        st.markdown("""
        <div class="quick-analytics-help">
            <h4>💡 What is Quick Analytics?</h4>
            <p>One-click analysis buttons that instantly analyze your data and provide specific insights without typing questions.</p>
            <p><strong>Click any button below to get instant results!</strong></p>
        </div>
        """, unsafe_allow_html=True)
        
        example_questions = [
            "📊 Summarize key business insights",
            "💰 What drives highest revenue?",
            "📈 Show performance trends",
            "🎯 Compare regional performance",
            "😊 Analyze customer satisfaction",
            "💡 Identify growth opportunities"
        ]
        
        for i, question in enumerate(example_questions):
            if st.button(question, key=f"sidebar_q_{i}"):
                # Execute quick analysis and show results immediately
                if st.session_state.df is not None:
                    result = execute_quick_analysis(st.session_state.df, question.split(" ", 1)[1])
                    st.session_state.analysis_history.append({
                        'question': question,
                        'result': result,
                        'timestamp': datetime.now(),
                        'data_shape': st.session_state.df.shape,
                        'data_columns': st.session_state.df.columns.tolist()
                    })
                    st.success(f"✅ Analysis complete! Check the AI Analysis tab.")
                    st.rerun()
                else:
                    st.warning("⚠️ Please load data first!")

    # Main content with enhanced tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Data Intelligence", 
        "🤖 AI Analysis", 
        "📈 Advanced Charts", 
        "🎯 Business Insights"
    ])
    
    with tab1:
        st.markdown("## 📊 Data Intelligence Center")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("### 📁 Upload Your Dataset")
            uploaded_file = st.file_uploader(
                "Select your data file",
                type=['csv', 'xlsx', 'xls', 'json', 'parquet'],
                help="Supported formats: CSV, Excel, JSON, Parquet"
            )
            
            if uploaded_file is not None:
                try:
                    with st.spinner("🔄 Processing your data..."):
                        if uploaded_file.name.endswith('.csv'):
                            df = pd.read_csv(uploaded_file)
                        elif uploaded_file.name.endswith('.json'):
                            df = pd.read_json(uploaded_file)
                        elif uploaded_file.name.endswith('.parquet'):
                            df = pd.read_parquet(uploaded_file)
                        else:
                            df = pd.read_excel(uploaded_file)
                        
                        st.session_state.df = df
                        
                    st.success(f"✅ Successfully processed {uploaded_file.name}")
                    display_enhanced_overview(df)
                    
                except Exception as e:
                    st.error(f"❌ Error processing file: {e}")
                    st.info("💡 Please ensure your file format is supported and data is valid")
        
        with col2:
            st.markdown("### 🚀 Premium Sample Data")
            
            # Generate random data each time
            if st.button("📊 Load Premium Dataset", type="primary"):
                with st.spinner("🎲 Generating random business data..."):
                    df = create_premium_sample_data()
                    st.session_state.df = df
                
                st.success("✅ Premium dataset activated!")
                st.info(f"🎲 Generated {len(df):,} random records with {len(df.columns)} fields")
                display_enhanced_overview(df)
            
            st.markdown("---")
            st.markdown("**🎯 Premium Features:**")
            st.markdown("""
            • 1,500-2,500 random business records
            • Multi-dimensional product data  
            • Regional performance metrics
            • Customer satisfaction scores
            • Profit & margin analysis
            • Supply chain metrics
            • Marketing performance data
            • Advanced business KPIs
            • **New random data every time!**
            """)
    
    with tab2:
        if st.session_state.df is None:
            st.warning("⚠️ **Data Required**: Please upload a dataset or load sample data first")
            st.info("💡 Navigate to the Data Intelligence tab to get started")
        else:
            st.markdown("## 🤖 AI-Powered Business Analysis")
            
            df = st.session_state.df
            
            # Enhanced quick analysis grid with working buttons
            st.markdown("### ⚡ Instant Insights")
            
            quick_analyses = [
                ("📊 Complete Business Overview", "Provide a comprehensive business intelligence summary of this dataset"),
                ("💰 Revenue & Profitability Analysis", "Analyze revenue streams, profit margins, and financial performance"),
                ("🎯 Performance Benchmarking", "Compare performance across products, regions, and segments"),
                ("📈 Trend Analysis", "Identify key trends, patterns, and growth opportunities"),
                ("😊 Customer Experience Analysis", "Analyze customer satisfaction, NPS, and experience metrics"),
                ("🔍 Operational Efficiency", "Examine supply chain, inventory, and operational KPIs")
            ]
            
            cols = st.columns(3)
            for i, (title, question) in enumerate(quick_analyses):
                with cols[i % 3]:
                    if st.button(title, key=f"quick_analysis_{i}"):
                        st.session_state.current_question = question
                        # Auto-run analysis when clicked
                        with st.spinner("🧠 Analyzing your data..."):
                            analysis_result = get_enhanced_ai_analysis(df, question)
                            
                            st.session_state.analysis_history.append({
                                'question': title,
                                'result': analysis_result,
                                'timestamp': datetime.now(),
                                'data_shape': df.shape,
                                'data_columns': df.columns.tolist()
                            })
                            
                        st.success("✅ Analysis complete!")
                        st.rerun()
            
            st.markdown("---")
            
            # Enhanced question input
            st.markdown("### 💭 Ask Your Business Question")
            
            question = st.text_area(
                "What would you like to know about your data?",
                value=st.session_state.current_question,
                height=120,
                placeholder="""Ask any business question in natural language...

Examples:
• What product generates the highest profit margin in each region?
• How does customer satisfaction correlate with revenue performance?
• Which sales channels are most effective for premium products?
• What seasonal trends can you identify in the data?
• Compare performance metrics across different customer segments""",
                help="Our AI will analyze your data and provide comprehensive business insights",
                key="business_question_input"
            )
            
            col1, col2, col3 = st.columns([2, 1, 1])
            
            with col1:
                analyze_btn = st.button("🧠 Analyze with AI", type="primary")
            
            with col2:
                # FIXED: Working export button with actual functionality
                if st.session_state.analysis_history:
                    export_content = create_export_content(st.session_state.analysis_history, df)
                    st.download_button(
                        "📄 Export Results",
                        data=export_content,
                        file_name=f"ai_analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                        mime="text/markdown",
                        help="Download comprehensive analysis report as Markdown file"
                    )
                else:
                    st.button("📄 Export Results", disabled=True, help="No analysis results to export")
            
            with col3:
                clear_btn = st.button("🗑️ Clear History")
            
            if clear_btn:
                st.session_state.analysis_history = []
                st.session_state.current_question = ""
                st.rerun()
            
            # Enhanced analysis processing
            if analyze_btn and question.strip():
                with st.spinner("🧠 AI is analyzing your data with actual sample data..."):
                    analysis_result = get_enhanced_ai_analysis(df, question)
                    
                    # Store in enhanced history
                    st.session_state.analysis_history.append({
                        'question': question,
                        'result': analysis_result,
                        'timestamp': datetime.now(),
                        'data_shape': df.shape,
                        'data_columns': df.columns.tolist()
                    })
                    
                    # Display enhanced results
                    st.markdown("### 🎯 AI Analysis Results")
                    st.markdown(f"""
                    <div class="analysis-response">
                        {analysis_result}
                    </div>
                    """, unsafe_allow_html=True)
                    
                st.session_state.current_question = ""
            
            # Enhanced history display
            if st.session_state.analysis_history:
                st.markdown("---")
                st.markdown("### 📚 Analysis History")
                
                for i, analysis in enumerate(reversed(st.session_state.analysis_history[-5:])):
                    with st.expander(
                        f"🔍 {analysis['question'][:80]}... • {analysis['timestamp'].strftime('%H:%M:%S')}",
                        expanded=(i == 0)
                    ):
                        st.markdown(analysis['result'])
                        st.caption(f"📊 Dataset: {analysis['data_shape'][0]} rows × {analysis['data_shape'][1]} columns")
    
    with tab3:
        if st.session_state.df is None:
            st.warning("⚠️ **Data Required**: Please upload a dataset first")
        else:
            st.markdown("## 📈 Advanced Data Visualizations")
            
            df = st.session_state.df
            
            # Chart type selection
            st.markdown("### 🎨 Choose Visualization Type")
            
            chart_options = [
                "Interactive Bar Chart", "Animated Scatter Plot", "3D Surface Plot",
                "Heatmap Correlation", "Box Plot", "Violin Plot"
            ]
            
            cols = st.columns(3)
            selected_chart = None
            
            for i, chart in enumerate(chart_options):
                with cols[i % 3]:
                    if st.button(f"📊 {chart}", key=f"chart_type_{i}"):
                        selected_chart = chart
                        st.session_state.selected_chart_type = chart
            
            # Use session state to remember selection
            if 'selected_chart_type' in st.session_state:
                selected_chart = st.session_state.selected_chart_type
            
            if selected_chart:
                st.markdown(f"### ⚙️ Configure {selected_chart}")
                
                # Better column selection
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    x_col = st.selectbox("📊 X-axis Column", df.columns, key="x_axis_advanced")
                
                with col2:
                    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                    if numeric_cols:
                        y_col = st.selectbox("📈 Y-axis Column", ['None'] + numeric_cols, key="y_axis_advanced")
                        y_col = None if y_col == 'None' else y_col
                    else:
                        y_col = None
                        st.info("No numeric columns available")
                
                with col3:
                    color_col = st.selectbox("🎨 Color by", ['None'] + df.columns.tolist(), key="color_advanced")
                    color_col = None if color_col == 'None' else color_col
                
                with col4:
                    if numeric_cols:
                        size_col = st.selectbox("📏 Size by", ['None'] + numeric_cols, key="size_advanced")
                        size_col = None if size_col == 'None' else size_col
                    else:
                        size_col = None
                        st.info("No numeric columns for sizing")
                
                if st.button("🚀 Create Advanced Visualization", type="primary"):
                    # Validation
                    can_create_chart = True
                    error_msg = ""
                    
                    if selected_chart in ["Box Plot", "Violin Plot", "Animated Scatter Plot", "3D Surface Plot"] and not y_col:
                        can_create_chart = False
                        error_msg = f"❌ {selected_chart} requires a numeric Y-axis column"
                    
                    if can_create_chart:
                        with st.spinner(f"🎨 Creating {selected_chart}..."):
                            fig = create_advanced_visualization(df, selected_chart, x_col, y_col, color_col, size_col)
                            
                            if fig:
                                st.plotly_chart(fig, use_container_width=True)
                                
                                # Chart insights
                                st.success(f"✅ {selected_chart} created successfully!")
                                
                                insight_parts = [f"💡 **Chart Analysis**: Shows relationship between **{x_col}**"]
                                if y_col:
                                    insight_parts.append(f" and **{y_col}**")
                                if color_col:
                                    insight_parts.append(f", grouped by **{color_col}**")
                                if size_col:
                                    insight_parts.append(f", sized by **{size_col}**")
                                
                                st.info("".join(insight_parts))
                            else:
                                st.error("❌ Failed to create visualization. Please try different settings.")
                    else:
                        st.error(error_msg)
    
    with tab4:
        if st.session_state.df is None:
            st.warning("⚠️ **Data Required**: Please upload a dataset first")
            st.info("💡 Navigate to the Data Intelligence tab to load data")
        else:
            st.markdown("## 🎯 Automated Business Intelligence")
            
            df = st.session_state.df
            
            # Generate comprehensive business insights
            with st.spinner("🔄 Generating business intelligence insights..."):
                business_insights = generate_business_insights(df)
            
            # Display KPIs if available
            if all(col in df.columns for col in ['Revenue', 'Gross_Profit', 'Units_Sold']):
                st.markdown("### 📊 Key Performance Indicators")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    total_revenue = df['Revenue'].sum()
                    st.metric("💰 Total Revenue", f"${total_revenue:,.0f}")
                
                with col2:
                    total_profit = df['Gross_Profit'].sum()
                    profit_margin = (total_profit / total_revenue) * 100
                    st.metric("📈 Total Profit", f"${total_profit:,.0f}", f"{profit_margin:.1f}% margin")
                
                with col3:
                    total_units = df['Units_Sold'].sum()
                    st.metric("📦 Units Sold", f"{total_units:,}")
                
                with col4:
                    avg_order_value = total_revenue / len(df)
                    st.metric("💳 Avg Order Value", f"${avg_order_value:,.0f}")
                
                st.markdown("---")
            
            # Display business insights
            st.markdown("### 📈 Intelligence Dashboard")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Revenue Intelligence
                if business_insights['revenue_insights']:
                    st.markdown("""
                    <div class="business-card">
                        <h3>💰 Revenue Intelligence</h3>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    for insight in business_insights['revenue_insights']:
                        st.markdown(f"""
                        <div class="insight-item">
                            {insight}
                        </div>
                        """, unsafe_allow_html=True)
                
                # Product Intelligence
                if business_insights['product_insights']:
                    st.markdown("""
                    <div class="business-card">
                        <h3>🛍️ Product Intelligence</h3>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    for insight in business_insights['product_insights']:
                        st.markdown(f"""
                        <div class="insight-item">
                            {insight}
                        </div>
                        """, unsafe_allow_html=True)
                
                # Operational Intelligence
                if business_insights['operational_insights']:
                    st.markdown("""
                    <div class="business-card">
                        <h3>⚙️ Operational Intelligence</h3>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    for insight in business_insights['operational_insights']:
                        st.markdown(f"""
                        <div class="insight-item">
                            {insight}
                        </div>
                        """, unsafe_allow_html=True)
            
            with col2:
                # Regional Intelligence
                if business_insights['regional_insights']:
                    st.markdown("""
                    <div class="business-card">
                        <h3>🌍 Regional Intelligence</h3>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    for insight in business_insights['regional_insights']:
                        st.markdown(f"""
                        <div class="insight-item">
                            {insight}
                        </div>
                        """, unsafe_allow_html=True)
                
                # Customer Intelligence
                if business_insights['customer_insights']:
                    st.markdown("""
                    <div class="business-card">
                        <h3>😊 Customer Intelligence</h3>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    for insight in business_insights['customer_insights']:
                        st.markdown(f"""
                        <div class="insight-item">
                            {insight}
                        </div>
                        """, unsafe_allow_html=True)
            
            # Strategic Recommendations
            if business_insights['recommendations']:
                st.markdown("---")
                st.markdown("### 💡 Strategic Recommendations")
                
                st.markdown("""
                <div class="business-card">
                    <h3>🎯 Action Items</h3>
                </div>
                """, unsafe_allow_html=True)
                
                for i, rec in enumerate(business_insights['recommendations'], 1):
                    st.markdown(f"""
                    <div class="insight-item">
                        <strong>{i}.</strong> {rec}
                    </div>
                    """, unsafe_allow_html=True)
            
            # FIXED: Performance visualizations with proper axis labels and no duplicate parameters
            if all(col in df.columns for col in ['Product', 'Revenue']):
                st.markdown("---")
                st.markdown("### 📊 Performance Visualizations")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    product_revenue = df.groupby('Product')['Revenue'].sum().sort_values(ascending=False).head(10)
                    fig = px.bar(
                        x=product_revenue.values,
                        y=product_revenue.index,
                        orientation='h',
                        title="🏆 Top Products by Revenue",
                        color=product_revenue.values,
                        color_continuous_scale='Viridis',
                        labels={
                            'x': 'Revenue ($)',
                            'y': 'Product'
                        }
                    )
                    fig.update_layout(
                        plot_bgcolor='rgba(255,255,255,0.9)',
                        paper_bgcolor='rgba(255,255,255,0.9)',
                        font_color='#1e293b',
                        title_x=0.5,
                        # FIXED: Correct syntax without duplicate yaxis parameter
                        yaxis=dict(
                            categoryorder='total ascending',
                            title="Product",
                            title_font=dict(size=14, color='#1e293b')
                        ),
                        xaxis=dict(
                            title="Revenue ($)",
                            title_font=dict(size=14, color='#1e293b')
                        )
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    if 'Customer_Satisfaction' in df.columns:
                        fig = px.histogram(
                            df,
                            x='Customer_Satisfaction',
                            nbins=20,
                            title="😊 Customer Satisfaction Distribution",
                            color_discrete_sequence=['#667eea'],
                            labels={
                                'x': 'Customer Satisfaction Rating',
                                'y': 'Count'
                            }
                        )
                        fig.update_layout(
                            plot_bgcolor='rgba(255,255,255,0.9)',
                            paper_bgcolor='rgba(255,255,255,0.9)',
                            font_color='#1e293b',
                            title_x=0.5,
                            xaxis=dict(
                                title="Customer Satisfaction Rating (1-5)",
                                title_font=dict(size=14, color='#1e293b')
                            ),
                            yaxis=dict(
                                title="Number of Customers",
                                title_font=dict(size=14, color='#1e293b')
                            )
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()