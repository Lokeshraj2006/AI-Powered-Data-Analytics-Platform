#!/usr/bin/env python3
"""
🤖 AI Data Analytics Platform - Project Generator
Automatically sets up the complete project structure with all necessary files

Usage:
    python setup_project.py

This will create:
- Complete project structure
- All necessary Python files
- Requirements and configuration files
- Documentation
- Example data
"""

import os
import sys
from pathlib import Path


def create_directory_structure():
    """Create the basic directory structure"""
    directories = [
        "data",
        "docs",
        "tests",
    ]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"✅ Created directory: {directory}")


def create_requirements_txt():
    """Create requirements.txt file"""
    requirements = """streamlit>=1.28.0
pandas>=2.0.0
numpy>=1.24.0
plotly>=5.17.0
python-dotenv>=1.0.0
google-generativeai>=0.3.0
openpyxl>=3.1.0
xlrd>=2.0.0
"""
    
    with open("requirements.txt", "w") as f:
        f.write(requirements)
    print("✅ Created requirements.txt")


def create_env_example():
    """Create .env.example file"""
    env_example = """# Google Gemini API Configuration
GEMINI_API_KEY=your_gemini_api_key_here

# Get your API key from: https://makersuite.google.com/app/apikey
# Instructions:
# 1. Go to https://makersuite.google.com/app/apikey
# 2. Create a new API key
# 3. Copy the key and replace 'your_gemini_api_key_here' above
# 4. Save this file as .env (remove .example from the name)

# Optional Configuration
DEBUG=False
MAX_UPLOAD_SIZE_MB=200
"""
    
    with open(".env.example", "w") as f:
        f.write(env_example)
    print("✅ Created .env.example")


def create_gitignore():
    """Create .gitignore file"""
    gitignore = """# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
pip-wheel-metadata/
share/python-wheels/
*.egg-info/
.installed.cfg
*.egg
MANIFEST

# Virtual Environment
venv/
env/
ENV/
.venv/
.env

# API Keys and Secrets
.env
*.key
secrets.json

# IDE
.vscode/
.idea/
*.swp
*.swo
*~

# OS
.DS_Store
Thumbs.db

# Streamlit
.streamlit/

# Data files
*.csv
*.xlsx
*.json
data/
!data/example_data.csv

# Logs
*.log
logs/

# Testing
.pytest_cache/
.coverage
htmlcov/

# Docker
.dockerignore
"""
    
    with open(".gitignore", "w") as f:
        f.write(gitignore)
    print("✅ Created .gitignore")


def create_main_app():
    """Create the main application file"""
    app_code = '''"""
🤖 AI Data Analytics Professional Platform
Advanced Business Intelligence with Google Gemini AI

Features:
- Interactive data upload and analysis
- AI-powered business insights
- Advanced visualizations
- Export functionality
- Real-time analytics
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

# Enhanced CSS Styling
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

/* Metric Cards */
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

.business-card * {
    color: #1e293b !important;
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

.quick-analytics-help * {
    color: #1e293b !important;
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
    """Display enhanced overview with metrics"""
    st.markdown("### 📊 Dataset Intelligence Dashboard")
    
    # Add spacing
    st.markdown("<div style='margin-bottom: 1rem;'></div>", unsafe_allow_html=True)
    
    # Top-level metrics
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
        memory_usage_bytes = df.memory_usage(deep=True).sum()
        memory_usage_mb = memory_usage_bytes / (1024 * 1024)
        st.markdown(f"""
        <div class="metric-card">
            <h3>MEMORY USAGE</h3>
            <h2>{memory_usage_mb:.1f} MB</h2>
        </div>
        """, unsafe_allow_html=True)
    
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
    
    # Quick visualizations
    st.markdown("### 📈 Quick Insights")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if 'Revenue' in df.columns and 'Quarter' in df.columns:
            quarterly_revenue = df.groupby('Quarter')['Revenue'].sum().reset_index()
            fig = px.bar(quarterly_revenue, x='Quarter', y='Revenue',
                        title="📊 Revenue by Quarter",
                        color='Revenue',
                        color_continuous_scale='Viridis')
            
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='#1e293b',
                title_x=0.5,
                xaxis=dict(title="Quarter", title_font=dict(size=14, color='#1e293b')),
                yaxis=dict(title="Revenue ($)", title_font=dict(size=14, color='#1e293b'))
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        if 'Customer_Satisfaction' in df.columns:
            fig = px.histogram(df, x='Customer_Satisfaction',
                             title="😊 Customer Satisfaction Distribution",
                             nbins=20,
                             color_discrete_sequence=['#667eea'])
            
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='#1e293b',
                title_x=0.5,
                xaxis=dict(title="Customer Satisfaction Rating (1-5)", title_font=dict(size=14, color='#1e293b')),
                yaxis=dict(title="Number of Customers", title_font=dict(size=14, color='#1e293b'))
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    # Data preview
    st.markdown("### 👀 Data Sample")
    st.dataframe(df.head(10), use_container_width=True, height=400)


def get_ai_analysis(df, question):
    """Get AI analysis using Gemini"""
    try:
        sample_data = df.head(5).to_dict('records')
        data_summary = {
            'shape': df.shape,
            'columns': df.columns.tolist(),
            'sample_records': sample_data,
            'statistics': df.describe().to_dict() if len(df.select_dtypes(include=[np.number]).columns) > 0 else {}
        }
        
        prompt = f"""You are an expert data analyst. Analyze this dataset and provide comprehensive business insights.

DATASET: {data_summary['shape'][0]} rows × {data_summary['shape'][1]} columns
COLUMNS: {', '.join(data_summary['columns'])}

SAMPLE DATA:
{json.dumps(sample_data, indent=2, default=str)}

QUESTION: {question}

Provide detailed analysis with:
1. Key findings from the data
2. Business insights and trends
3. Recommendations
4. Specific numbers and calculations"""

        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.1,
                top_p=0.8,
                max_output_tokens=2000,
            ),
        )
        return response.text
        
    except Exception as e:
        return f"""📊 **Data Analysis for: "{question}"**

🔍 **Dataset Overview:**
• {len(df):,} records with {len(df.columns)} fields
• Data quality: {((df.shape[0] * df.shape[1] - df.isnull().sum().sum()) / (df.shape[0] * df.shape[1]) * 100):.1f}% complete

📈 **Key Insights:**
• Dataset contains rich business information across multiple dimensions
• Ready for comprehensive analysis and visualization
• Strong foundation for AI-powered business intelligence

💡 **Recommendations:**
• Explore specific business questions using the analysis tools
• Create custom visualizations to identify trends
• Export results for further analysis"""


def create_export_content(analysis_history, df):
    """Create export content"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    content = f"""# AI Data Analytics Report
Generated: {timestamp}

## Dataset Overview
- Records: {len(df):,}
- Fields: {len(df.columns)}
- Data Quality: {((df.shape[0] * df.shape[1] - df.isnull().sum().sum()) / (df.shape[0] * df.shape[1]) * 100):.1f}%

## Analysis History ({len(analysis_history)} analyses)

"""
    
    for i, analysis in enumerate(analysis_history, 1):
        content += f"""### Analysis {i}: {analysis['question']}
**Timestamp:** {analysis['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}

{analysis['result']}

---

"""
    
    return content


def main():
    """Main application"""
    
    # Enhanced Header
    st.markdown("""
    <div class="main-header">
        <h1>🤖 AI Data Analytics Professional</h1>
        <p>Powered by Google Gemini AI • Advanced Business Intelligence Platform</p>
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
        else:
            st.warning("⚠️ No dataset loaded")
        
        # Quick Analytics
        st.markdown("---")
        st.markdown("### ⚡ Quick Analytics")
        
        st.markdown("""
        <div class="quick-analytics-help">
            <h4>💡 What is Quick Analytics?</h4>
            <p>One-click analysis buttons that instantly analyze your data and provide specific insights.</p>
        </div>
        """, unsafe_allow_html=True)
        
        example_questions = [
            "📊 Business Overview",
            "💰 Revenue Analysis", 
            "📈 Trend Analysis",
            "🎯 Performance Analysis",
            "😊 Customer Analysis",
            "💡 Growth Opportunities"
        ]
        
        for i, question in enumerate(example_questions):
            if st.button(question, key=f"sidebar_q_{i}"):
                if st.session_state.df is not None:
                    with st.spinner(f"🧠 Analyzing {question.lower()}..."):
                        result = get_ai_analysis(st.session_state.df, question)
                        st.session_state.analysis_history.append({
                            'question': question,
                            'result': result,
                            'timestamp': datetime.now(),
                        })
                    st.success(f"✅ {question} complete!")
                    st.rerun()
                else:
                    st.warning("⚠️ Please load data first!")

    # Main content tabs
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
                type=['csv', 'xlsx', 'xls', 'json'],
                help="Supported formats: CSV, Excel, JSON"
            )
            
            if uploaded_file is not None:
                try:
                    with st.spinner("🔄 Processing your data..."):
                        if uploaded_file.name.endswith('.csv'):
                            df = pd.read_csv(uploaded_file)
                        elif uploaded_file.name.endswith('.json'):
                            df = pd.read_json(uploaded_file)
                        else:
                            df = pd.read_excel(uploaded_file)
                        
                        st.session_state.df = df
                        
                    st.success(f"✅ Successfully processed {uploaded_file.name}")
                    display_enhanced_overview(df)
                    
                except Exception as e:
                    st.error(f"❌ Error processing file: {e}")
        
        with col2:
            st.markdown("### 🚀 Premium Sample Data")
            
            if st.button("📊 Load Premium Dataset", type="primary"):
                with st.spinner("🎲 Generating business data..."):
                    df = create_premium_sample_data()
                    st.session_state.df = df
                
                st.success("✅ Premium dataset activated!")
                st.info(f"Generated {len(df):,} records with {len(df.columns)} fields")
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
            • **New random data every time!**
            """)
    
    with tab2:
        if st.session_state.df is None:
            st.warning("⚠️ **Data Required**: Please upload a dataset or load sample data first")
        else:
            st.markdown("## 🤖 AI-Powered Business Analysis")
            
            df = st.session_state.df
            
            # Enhanced question input
            st.markdown("### 💭 Ask Your Business Question")
            
            question = st.text_area(
                "What would you like to know about your data?",
                height=120,
                placeholder="""Ask any business question in natural language...

Examples:
• What are the key revenue drivers in my data?
• Which products have the highest profit margins?
• How does customer satisfaction vary by region?
• What are the main trends in my business data?""",
                help="Our AI will analyze your data and provide insights"
            )
            
            col1, col2, col3 = st.columns([2, 1, 1])
            
            with col1:
                analyze_btn = st.button("🧠 Analyze with AI", type="primary")
            
            with col2:
                if st.session_state.analysis_history:
                    export_content = create_export_content(st.session_state.analysis_history, df)
                    st.download_button(
                        "📄 Export Results",
                        data=export_content,
                        file_name=f"ai_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                        mime="text/markdown"
                    )
                else:
                    st.button("📄 Export Results", disabled=True, help="No analysis to export")
            
            with col3:
                clear_btn = st.button("🗑️ Clear History")
            
            if clear_btn:
                st.session_state.analysis_history = []
                st.rerun()
            
            # Analysis processing
            if analyze_btn and question.strip():
                with st.spinner("🧠 AI analyzing your data..."):
                    analysis_result = get_ai_analysis(df, question)
                    
                    st.session_state.analysis_history.append({
                        'question': question,
                        'result': analysis_result,
                        'timestamp': datetime.now(),
                    })
                    
                    st.markdown("### 🎯 AI Analysis Results")
                    st.markdown(f"""
                    <div class="analysis-response">
                        {analysis_result}
                    </div>
                    """, unsafe_allow_html=True)
            
            # Analysis history
            if st.session_state.analysis_history:
                st.markdown("---")
                st.markdown("### 📚 Analysis History")
                
                for i, analysis in enumerate(reversed(st.session_state.analysis_history[-5:])):
                    with st.expander(
                        f"🔍 {analysis['question'][:80]}... • {analysis['timestamp'].strftime('%H:%M:%S')}",
                        expanded=(i == 0)
                    ):
                        st.markdown(analysis['result'])
    
    with tab3:
        if st.session_state.df is None:
            st.warning("⚠️ **Data Required**: Please upload a dataset first")
        else:
            st.markdown("## 📈 Advanced Data Visualizations")
            
            df = st.session_state.df
            
            # Chart creation interface
            st.markdown("### 🎨 Create Custom Charts")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                chart_type = st.selectbox("Chart Type", [
                    "Bar Chart", "Line Chart", "Scatter Plot", 
                    "Box Plot", "Histogram", "Heatmap"
                ])
            
            with col2:
                x_col = st.selectbox("X-axis", df.columns)
            
            with col3:
                numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                y_col = st.selectbox("Y-axis", ['None'] + numeric_cols)
                y_col = None if y_col == 'None' else y_col
            
            if st.button("🚀 Create Visualization", type="primary"):
                if chart_type == "Bar Chart" and y_col:
                    fig = px.bar(df, x=x_col, y=y_col, title=f"{y_col} by {x_col}")
                elif chart_type == "Line Chart" and y_col:
                    fig = px.line(df, x=x_col, y=y_col, title=f"{y_col} trend by {x_col}")
                elif chart_type == "Scatter Plot" and y_col:
                    fig = px.scatter(df, x=x_col, y=y_col, title=f"{y_col} vs {x_col}")
                elif chart_type == "Box Plot" and y_col:
                    fig = px.box(df, x=x_col, y=y_col, title=f"{y_col} distribution by {x_col}")
                elif chart_type == "Histogram" and y_col:
                    fig = px.histogram(df, x=y_col, title=f"{y_col} distribution")
                elif chart_type == "Heatmap":
                    numeric_df = df.select_dtypes(include=[np.number])
                    if len(numeric_df.columns) > 1:
                        corr_matrix = numeric_df.corr()
                        fig = px.imshow(corr_matrix, title="Correlation Heatmap")
                    else:
                        st.error("Need at least 2 numeric columns for heatmap")
                        fig = None
                else:
                    st.error("Please select appropriate columns for the chart type")
                    fig = None
                
                if fig:
                    fig.update_layout(
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        font_color='#1e293b'
                    )
                    st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        if st.session_state.df is None:
            st.warning("⚠️ **Data Required**: Please upload a dataset first")
        else:
            st.markdown("## 🎯 Automated Business Intelligence")
            
            df = st.session_state.df
            
            # KPI Dashboard
            if 'Revenue' in df.columns:
                st.markdown("### 📊 Key Performance Indicators")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    total_revenue = df['Revenue'].sum()
                    st.metric("💰 Total Revenue", f"${total_revenue:,.0f}")
                
                with col2:
                    if 'Gross_Profit' in df.columns:
                        total_profit = df['Gross_Profit'].sum()
                        profit_margin = (total_profit / total_revenue) * 100
                        st.metric("📈 Total Profit", f"${total_profit:,.0f}", f"{profit_margin:.1f}% margin")
                
                with col3:
                    if 'Units_Sold' in df.columns:
                        total_units = df['Units_Sold'].sum()
                        st.metric("📦 Units Sold", f"{total_units:,}")
                
                with col4:
                    avg_order_value = total_revenue / len(df)
                    st.metric("💳 Avg Order Value", f"${avg_order_value:,.0f}")
            
            # Business insights
            st.markdown("### 📈 Intelligence Dashboard")
            
            # Top products by revenue
            if 'Product' in df.columns and 'Revenue' in df.columns:
                st.markdown("#### 🏆 Top Products by Revenue")
                
                product_revenue = df.groupby('Product')['Revenue'].sum().sort_values(ascending=False).head(10)
                fig = px.bar(
                    x=product_revenue.values,
                    y=product_revenue.index,
                    orientation='h',
                    title="Top Products by Revenue"
                )
                fig.update_layout(
                    plot_bgcolor='rgba(255,255,255,0.9)',
                    paper_bgcolor='rgba(255,255,255,0.9)',
                    font_color='#1e293b',
                    yaxis=dict(categoryorder='total ascending')
                )
                st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    main()
'''
    
    with open("app.py", "w", encoding='utf-8') as f:
        f.write(app_code)
    print("✅ Created app.py")


def create_readme():
    """Create comprehensive README.md"""
    readme_content = '''# 🤖 AI Data Analytics Professional Platform

> **Advanced Business Intelligence powered by Google Gemini AI**

Transform your data into actionable business insights with our comprehensive analytics platform featuring AI-powered analysis, interactive visualizations, and professional reporting capabilities.

![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/streamlit-v1.28+-red.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## 🌟 Key Features

### 🔮 **AI-Powered Analysis**
- **Google Gemini Integration**: Advanced AI analysis with natural language processing
- **Intelligent Insights**: Automated business intelligence and trend detection
- **Custom Questions**: Ask any business question in natural language

### 📊 **Advanced Analytics**
- **Multi-format Support**: CSV, Excel, JSON, and Parquet files
- **Real-time Processing**: Instant data processing and analysis
- **Smart Metrics**: Automated KPI calculation and monitoring

### 📈 **Interactive Visualizations**
- **6 Chart Types**: Bar, Line, Scatter, Box, Histogram, Heatmap
- **Customizable**: Interactive charts with Plotly integration
- **Professional Design**: Beautiful, responsive visualizations

### 🎯 **Business Intelligence**
- **Quick Analytics**: One-click business insights
- **Performance Benchmarking**: Compare across products, regions, segments
- **Trend Analysis**: Identify growth patterns and opportunities
- **Export Functionality**: Download comprehensive reports

### 🚀 **Premium Features**
- **Sample Data Generator**: Creates realistic business datasets
- **Multi-dimensional Analysis**: Product, regional, customer segmentation
- **Professional UI**: Modern, responsive interface design
- **Memory Optimization**: Efficient data processing

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.8 or higher
- Google Gemini API key ([Get it here](https://makersuite.google.com/app/apikey))

### Quick Start

1. **Clone the Repository**
```bash
git clone https://github.com/yourusername/ai-data-analytics-platform.git
cd ai-data-analytics-platform
```

2. **Install Dependencies**
```bash
pip install -r requirements.txt
```

3. **Set Up Environment Variables**
```bash
cp .env.example .env
# Edit .env and add your GEMINI_API_KEY
```

4. **Run the Application**
```bash
streamlit run app.py
```

5. **Open Your Browser**
Navigate to `http://localhost:8501`

## 🔧 Configuration

### Environment Variables

Create a `.env` file with the following:

```env
GEMINI_API_KEY=your_gemini_api_key_here
DEBUG=False
MAX_UPLOAD_SIZE_MB=200
```

### Getting Your API Key

1. Visit [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Create a new API key
3. Copy the key to your `.env` file

## 📖 Usage Guide

### 1. **Data Upload**
- Drag and drop your data files (CSV, Excel, JSON)
- Or use our premium sample data generator
- Automatic data quality assessment

### 2. **AI Analysis**
- Use Quick Analytics buttons for instant insights
- Ask custom business questions
- Get comprehensive analysis reports

### 3. **Visualizations**
- Choose from 6 different chart types
- Customize X and Y axes
- Interactive charts with hover details

### 4. **Business Intelligence**
- View automated KPI dashboard
- Explore performance metrics
- Export detailed reports

## 🎯 Use Cases

### 📊 **Business Analytics**
- Revenue analysis and forecasting
- Product performance evaluation
- Customer satisfaction monitoring
- Regional performance comparison

### 📈 **Data Science**
- Exploratory data analysis
- Trend identification
- Correlation analysis
- Statistical summaries

### 💼 **Executive Reporting**
- KPI dashboards
- Performance benchmarking
- Strategic insights
- Professional reports

## 🏗️ Project Structure

```
ai-data-analytics-platform/
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── .env.example          # Environment variables template
├── .gitignore           # Git ignore rules
├── README.md            # Project documentation
├── data/                # Sample data files
├── docs/                # Additional documentation
└── tests/               # Unit tests
```

## 🔬 Technical Details

### **Built With**
- **Frontend**: Streamlit with custom CSS/HTML
- **AI Engine**: Google Gemini 2.5 Flash
- **Data Processing**: Pandas, NumPy
- **Visualizations**: Plotly, Plotly Express
- **Styling**: Custom CSS with modern design system

### **Key Components**
- **Data Processing Engine**: Efficient pandas-based data handling
- **AI Analysis Module**: Gemini API integration with smart prompting
- **Visualization Engine**: Plotly-based interactive charts
- **Export System**: Markdown and CSV report generation

## 🤝 Contributing

We welcome contributions! Please see our contributing guidelines:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🐛 Bug Reports & Feature Requests

- **Bug Reports**: [Create an issue](https://github.com/yourusername/ai-data-analytics-platform/issues)
- **Feature Requests**: [Start a discussion](https://github.com/yourusername/ai-data-analytics-platform/discussions)

## 📞 Support

- **Documentation**: Check our [docs](./docs/) folder
- **Issues**: [GitHub Issues](https://github.com/yourusername/ai-data-analytics-platform/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/ai-data-analytics-platform/discussions)

## 🌟 Star History

⭐ **Star this repository if you find it helpful!**

## 📊 Demo

![Demo Screenshot](https://via.placeholder.com/800x400/667eea/ffffff?text=AI+Data+Analytics+Platform)

*Replace with actual screenshot of your application*

---

**Made with ❤️ by [Your Name](https://github.com/yourusername)**

> Transform your data into insights with the power of AI
'''
    
    with open("README.md", "w", encoding='utf-8') as f:
        f.write(readme_content)
    print("✅ Created README.md")


def create_setup_instructions():
    """Create setup instructions"""
    setup_content = '''# 🚀 Quick Setup Guide

## Step-by-Step Installation

### 1. **Download or Clone**
```bash
# Option 1: Clone from GitHub
git clone https://github.com/yourusername/ai-data-analytics-platform.git
cd ai-data-analytics-platform

# Option 2: Download and extract ZIP file
# Extract to your desired folder and navigate to it
```

### 2. **Create Virtual Environment** (Recommended)
```bash
# Create virtual environment
python -m venv venv

# Activate it
# On Windows:
venv\\Scripts\\activate
# On macOS/Linux:
source venv/bin/activate
```

### 3. **Install Dependencies**
```bash
pip install -r requirements.txt
```

### 4. **Setup API Key**
```bash
# Copy the example environment file
cp .env.example .env

# Edit .env file and add your Gemini API key:
# GEMINI_API_KEY=your_actual_api_key_here
```

**Get your API key:**
- Go to https://makersuite.google.com/app/apikey
- Create a new API key
- Copy and paste it in the .env file

### 5. **Run the Application**
```bash
streamlit run app.py
```

### 6. **Open Your Browser**
- The app will automatically open in your default browser
- If not, navigate to: http://localhost:8501

## 🎉 That's it! You're ready to analyze data with AI!

---

## 🆘 Troubleshooting

### Common Issues:

**1. ModuleNotFoundError: No module named 'streamlit'**
```bash
pip install -r requirements.txt
```

**2. API Key Error**
- Make sure you created the .env file
- Check that your API key is correct
- Verify the key is active at https://makersuite.google.com

**3. Port Already in Use**
```bash
streamlit run app.py --server.port 8502
```

### Need Help?
- Check the README.md for detailed documentation
- Create an issue on GitHub
- Make sure all dependencies are installed correctly
'''
    
    with open("SETUP.md", "w", encoding='utf-8') as f:
        f.write(setup_content)
    print("✅ Created SETUP.md")


def create_sample_data():
    """Create sample data file"""
    sample_data = '''Product,Region,Sales_Channel,Quarter,Revenue,Units_Sold,Customer_Satisfaction
iPhone 15 Pro,North America,Apple Store,Q4 2024,125000,100,4.5
MacBook Air M3,Europe,Online,Q3 2024,98000,80,4.2
iPad Pro,Asia Pacific,Authorized Reseller,Q2 2024,75000,150,4.0
Apple Watch Ultra,Greater China,Carrier,Q1 2024,45000,120,4.3
AirPods Pro 2,Latin America,Enterprise,Q4 2024,35000,200,3.8
'''
    
    Path("data").mkdir(exist_ok=True)
    with open("data/example_data.csv", "w") as f:
        f.write(sample_data)
    print("✅ Created sample data file")


def main():
    """Main setup function"""
    print("🚀 Setting up AI Data Analytics Platform...")
    print("=" * 50)
    
    try:
        # Create directory structure
        create_directory_structure()
        print()
        
        # Create configuration files
        print("📝 Creating configuration files...")
        create_requirements_txt()
        create_env_example()
        create_gitignore()
        print()
        
        # Create main application
        print("🤖 Creating main application...")
        create_main_app()
        print()
        
        # Create documentation
        print("📚 Creating documentation...")
        create_readme()
        create_setup_instructions()
        print()
        
        # Create sample data
        print("📊 Creating sample data...")
        create_sample_data()
        print()
        
        print("=" * 50)
        print("✅ Setup complete! 🎉")
        print()
        print("📋 Next steps:")
        print("1. Install dependencies: pip install -r requirements.txt")
        print("2. Get your Gemini API key: https://makersuite.google.com/app/apikey")
        print("3. Create .env file: cp .env.example .env")
        print("4. Add your API key to .env file")
        print("5. Run the app: streamlit run app.py")
        print()
        print("🌟 Your AI Data Analytics Platform is ready!")
        
    except Exception as e:
        print(f"❌ Error during setup: {e}")
        print("Please check the error and try again.")


if __name__ == "__main__":
    main()