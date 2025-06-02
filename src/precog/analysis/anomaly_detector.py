from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from datetime import datetime

from ..core.data_models import SentimentAlert # Adjusted import path

class SentimentAnomalyDetector:
    def __init__(self, contamination: float = 0.05, random_state: int = 42):
        """
        Initialize the anomaly detector.
        :param contamination: Expected proportion of outliers in the data.
        """
        self.model = IsolationForest(contamination=contamination, random_state=random_state)
        self.is_fitted = False
        self.baseline_features = ['sentiment_score', 'misinformation_likelihood'] # Example features

    def _extract_features(self, data: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Extract features relevant for anomaly detection from the input DataFrame."""
        if data.empty:
            return None

        features = pd.DataFrame()
        # Ensure required columns exist, fill with defaults if not
        if 'sentiment_score' in data.columns:
            features['sentiment_score'] = data['sentiment_score'].fillna(0)
        else:
            features['sentiment_score'] = 0

        if 'misinformation_likelihood' in data.columns:
            features['misinformation_likelihood'] = data['misinformation_likelihood'].fillna(0)
        else:
            features['misinformation_likelihood'] = 0

        # Example: Add time-based features if 'timestamp' is available
        if 'timestamp' in data.columns:
            try:
                timestamps = pd.to_datetime(data['timestamp'])
                features['hour_of_day'] = timestamps.dt.hour
                features['day_of_week'] = timestamps.dt.dayofweek
            except Exception: # Handle cases where timestamp might not be datetime
                features['hour_of_day'] = 0
                features['day_of_week'] = 0
        else: # Default if no timestamp
            features['hour_of_day'] = 0
            features['day_of_week'] = 0

        # Ensure all baseline features are present, even if with default values
        for f_name in self.baseline_features:
            if f_name not in features.columns:
                features[f_name] = 0 # Default value

        return features[self.baseline_features]

    def fit_baseline(self, baseline_data: pd.DataFrame):
        """
        Fit the Isolation Forest model on baseline (normal) data.
        :param baseline_data: DataFrame containing normal sentiment and activity data.
        """
        features = self._extract_features(baseline_data)
        if features is not None and not features.empty:
            self.model.fit(features)
            self.is_fitted = True
        else:
            # print("Warning: No valid features to fit anomaly detection model.")
            self.is_fitted = False

    def detect_anomalies(self, current_data: pd.DataFrame) -> List[SentimentAlert]:
        """
        Detect anomalies in the current data based on the fitted model.
        :param current_data: DataFrame of current sentiment and activity data.
        :return: List of SentimentAlert objects for detected anomalies.
        """
        if not self.is_fitted:
            # print("Warning: Anomaly detection model not fitted. Returning no anomalies.")
            # Optionally, could try to fit here if enough data, or raise error
            return []
        if current_data.empty:
            return []

        features = self._extract_features(current_data)
        if features is None or features.empty:
            return []

        predictions = self.model.predict(features)
        anomaly_scores = self.model.decision_function(features)

        alerts = []
        for index, row in current_data.iterrows():
            if predictions[index] == -1:  # -1 indicates an anomaly
                score = anomaly_scores[index]
                severity = 'low'
                if score < -0.1: severity = 'medium'
                if score < -0.2: severity = 'high'
                if score < -0.3: severity = 'critical' # Stricter threshold for critical

                description = f"Unusual sentiment activity detected. Score: {score:.2f}."
                if 'sentiment_score' in row and row['sentiment_score'] < -0.5:
                    description += f" Negative sentiment spike ({row['sentiment_score']:.2f})."
                if 'misinformation_likelihood' in row and row['misinformation_likelihood'] > 0.7:
                    description += f" High misinformation likelihood ({row['misinformation_likelihood']:.2f})."

                alerts.append(SentimentAlert(
                    location_name=row.get('location_approx', 'Unknown Location'),
                    timestamp=row.get('timestamp', pd.Timestamp.now()),
                    severity=severity,
                    message=description,
                    sentiment_score=row.get('sentiment_score', 0),
                    contributing_factors=[f for f in self.baseline_features if f in row and abs(row[f]) > 0.5] # Simplified
                ))
        return alerts
