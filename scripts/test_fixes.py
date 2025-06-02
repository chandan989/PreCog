import sys
import os

# Add project root to Python path to allow direct imports
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Import the classes we need to test
from src.precog.analysis.anomaly_detector import SentimentAnomalyDetector
from src.precog.core.data_models import SentimentAlert
import pandas as pd
import numpy as np

# Test SentimentAlert creation
print("Testing SentimentAlert creation...")
alert = SentimentAlert(
    location_name="Test Location",
    timestamp=pd.Timestamp.now(),
    severity="high",
    message="Test message",
    sentiment_score=-0.8,
    contributing_factors=["factor1", "factor2"]
)
print(f"SentimentAlert created successfully: {alert}")

# Test SentimentAnomalyDetector
print("\nTesting SentimentAnomalyDetector...")
detector = SentimentAnomalyDetector()
print(f"SentimentAnomalyDetector created successfully: {detector}")

# Create a test DataFrame
print("\nTesting anomaly detection...")
test_data = pd.DataFrame({
    'location_approx': ['Test Location'] * 5,
    'timestamp': [pd.Timestamp.now()] * 5,
    'sentiment_score': [-0.8, -0.7, -0.6, -0.5, -0.4],
    'misinformation_likelihood': [0.8, 0.7, 0.6, 0.5, 0.4]
})

# Fit the model
detector.fit_baseline(test_data)
print(f"Model fitted: {detector.is_fitted}")

# Detect anomalies
alerts = detector.detect_anomalies(test_data)
print(f"Detected {len(alerts)} anomalies")
if alerts:
    print(f"First alert: {alerts[0]}")

print("\nAll tests passed successfully!")

# Test the bar chart in app.py
print("\nTesting bar chart creation...")
try:
    import plotly.express as px
    
    # Create a test DataFrame
    category_data = pd.DataFrame({
        'category': ['A', 'B', 'C'],
        'sentiment_score': [-0.8, -0.7, -0.6],
        'count': [10, 20, 30]
    })
    
    # Create a bar chart without the 'size' parameter
    fig = px.bar(
        category_data,
        x='category',
        y='sentiment_score',
        color='sentiment_score',
        color_continuous_scale='RdBu',
        title='Average Sentiment by Category',
        labels={'sentiment_score': 'Avg. Sentiment', 'category': 'Category', 'count': 'Number of Posts'}
    )
    print("Bar chart created successfully")
except Exception as e:
    print(f"Error creating bar chart: {e}")