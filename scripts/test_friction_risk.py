import sys
import os

# Add project root to Python path to allow direct imports
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.precog.core.data_models import FrictionRisk
from datetime import datetime

# Create a FrictionRisk object
risk = FrictionRisk(
    location_name="Test Location",
    risk_level=0.8,
    predicted_timeline="short-term",
    primary_contributing_factor="Test Factor",
    explanation="Test Explanation",
    confidence=0.9,
    location_lat=12.34,
    location_lon=56.78
)

# Test accessing the attributes
print(f"Location: {risk.location_name}")
print(f"Risk Level: {risk.risk_level}")
print(f"Timeline: {risk.predicted_timeline}")
print(f"Factor: {risk.primary_contributing_factor}")
print(f"Explanation: {risk.explanation}")
print(f"Confidence: {risk.confidence}")
print(f"Latitude: {risk.location_lat}")
print(f"Longitude: {risk.location_lon}")

# Test the system.py changes
from src.precog.core.system import HyperlocalIntelligenceSystem
import pandas as pd
import numpy as np

# Create a simple test for the _update_system_metrics method
system = HyperlocalIntelligenceSystem()
analysis_results = {
    'timestamp': pd.Timestamp.now(),
    'friction_risks': [risk]
}

# Call the method that was updated
system._update_system_metrics(analysis_results)

# Print the metrics to verify
print("\nSystem Metrics:")
print(f"Friction Risk: {system.system_metrics['friction_risk']}")

# Test the intervention_recommender.py changes
from src.precog.interventions.intervention_recommender import InterventionRecommender

# Create a recommender
recommender = InterventionRecommender()

# Test the _get_issue_type_from_alert method
issue_type = recommender._get_issue_type_from_alert(risk)
print(f"\nIssue Type: {issue_type}")

# Test the _get_context_from_alert method
context = recommender._get_context_from_alert(risk)
print(f"Context: {context}")

# Test the recommend_interventions method
recommendations = recommender.recommend_interventions([], [risk], [])
print(f"Priority Locations: {recommendations['priority_locations']}")

print("\nAll tests passed successfully!")