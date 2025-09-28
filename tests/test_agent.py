"""
Basic tests for Gemini DataFrame Agent
"""

import pytest
import pandas as pd
import numpy as np
from src.core.dataframe_agent import GeminiDataFrameAgent, create_agent

def test_create_agent():
    """Test agent creation"""
    # This will require API key, so we'll mock it in real tests
    assert callable(create_agent)

def test_sample_data_generation():
    """Test sample data generation"""
    np.random.seed(42)
    
    # Sample data
    data = []
    for i in range(100):
        data.append({
            'Product': 'Test Product',
            'Sales': np.random.randint(10, 500),
            'Revenue': np.random.randint(1000, 50000)
        })
    
    df = pd.DataFrame(data)
    
    assert len(df) == 100
    assert 'Product' in df.columns
    assert 'Sales' in df.columns
    assert 'Revenue' in df.columns

def test_dataframe_preprocessing():
    """Test DataFrame preprocessing logic"""
    # Create test data with duplicates and missing values
    df = pd.DataFrame({
        'A': [1, 2, 2, 4, None],
        'B': ['x', 'y', 'y', 'z', 'w'],
        'date': ['2023-01-01', '2023-01-02', '2023-01-02', '2023-01-04', '2023-01-05']
    })
    
    # Basic preprocessing
    initial_rows = len(df)
    df_processed = df.drop_duplicates()
    
    assert len(df_processed) <= initial_rows

if __name__ == "__main__":
    pytest.main([__file__])