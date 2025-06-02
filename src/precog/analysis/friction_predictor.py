from typing import List, Dict, Any, Tuple, Optional
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib # Added for saving/loading model
import os # Added for path operations
import logging # Added for logging

from ..core.data_models import FrictionRisk
import random # For dummy coordinates

# Dummy coordinates (should be ideally from config or a geo-coding utility)
# These should be at the module level
DUMMY_LAT = 28.6139 # Example: Delhi latitude
DUMMY_LON = 77.2090 # Example: Delhi longitude

logger = logging.getLogger(__name__)

class FrictionPointPredictor:
    def __init__(self, model_path: str = "friction_model.joblib", random_state: int = 42):
        self.model_path = model_path
        self.model = Pipeline([
            ('scaler', StandardScaler()),
            ('rf', RandomForestClassifier(random_state=random_state, class_weight='balanced'))
        ])
        self.is_fitted = False
        # Define features expected by the model. These should align with `_prepare_features`
        self.feature_names = [
            'avg_negative_sentiment', 'sentiment_volatility', 'grievance_post_count',
            'social_tension_keywords', 'misinfo_spread_rate', 'past_incidents_count',
            'population_density_proxy', 'economic_stress_proxy'
        ]
        self._load_model() # Attempt to load model on init

    def _generate_synthetic_training_data(self, num_samples: int = 1000) -> pd.DataFrame:
        """Generates synthetic data for training the friction predictor."""
        data = pd.DataFrame()
        data['avg_negative_sentiment'] = np.random.uniform(-1, 0, num_samples)
        data['sentiment_volatility'] = np.random.uniform(0, 0.5, num_samples)
        data['grievance_post_count'] = np.random.randint(0, 50, num_samples)
        data['social_tension_keywords'] = np.random.randint(0, 10, num_samples)
        data['misinfo_spread_rate'] = np.random.uniform(0, 1, num_samples)
        data['past_incidents_count'] = np.random.randint(0, 5, num_samples)
        data['population_density_proxy'] = np.random.rand(num_samples)
        data['economic_stress_proxy'] = np.random.rand(num_samples)

        # Heuristic for labeling synthetic data (simplified)
        # Higher risk if multiple factors are high
        risk_score_for_labeling = (
            -data['avg_negative_sentiment'] * 0.3 +  # More negative is higher risk
            data['sentiment_volatility'] * 0.1 +
            (data['grievance_post_count'] / 50) * 0.15 +
            (data['social_tension_keywords'] / 10) * 0.2 +
            data['misinfo_spread_rate'] * 0.15 +
            (data['past_incidents_count'] / 5) * 0.1
        )
        data['friction_risk_label'] = (risk_score_for_labeling > 0.4).astype(int) # Threshold for 'high risk'
        return data

    def _prepare_features(self, current_data: pd.DataFrame, for_training: bool = False) -> pd.DataFrame:
        """Prepares features from raw data. current_data is expected to be aggregated per location."""
        if current_data.empty:
            return pd.DataFrame(columns=self.feature_names)

        features = pd.DataFrame(index=current_data.index)

        # Example: these features would ideally come from aggregated analysis of posts per location
        features['avg_negative_sentiment'] = current_data.get('avg_sentiment', 0) # Assuming sentiment is -1 to 1
        features['sentiment_volatility'] = current_data.get('sentiment_std_dev', 0)
        features['grievance_post_count'] = current_data.get('grievance_count', 0)
        features['social_tension_keywords'] = current_data.get('tension_keyword_count', 0)
        features['misinfo_spread_rate'] = current_data.get('misinfo_velocity', 0)
        features['past_incidents_count'] = current_data.get('historical_incidents', 0) # Needs external data
        features['population_density_proxy'] = current_data.get('population_density', 0) # Needs external data
        features['economic_stress_proxy'] = current_data.get('economic_indicators', 0) # Needs external data

        # Fill NaNs that might result from missing source columns
        features = features.fillna(0)

        # Ensure all expected feature_names are present, adding them with 0 if missing
        for col in self.feature_names:
            if col not in features.columns:
                features[col] = 0

        return features[self.feature_names] # Return only the defined features in correct order

    def train_model(self, training_data: Optional[pd.DataFrame] = None, num_synthetic_samples: int = 2000):
        """
        Train the Random Forest model.
        If training_data is not provided, synthetic data will be generated.
        """
        if training_data is None or training_data.empty:
            # print("No training data provided, generating synthetic data.")
            training_df = self._generate_synthetic_training_data(num_synthetic_samples)
        else:
            # Assuming training_data is already processed and has features + 'friction_risk_label'
            training_df = training_data

        X = self._prepare_features(training_df, for_training=True)
        if 'friction_risk_label' not in training_df.columns:
            raise ValueError("Target variable 'friction_risk_label' not found in training data.")
        y = training_df['friction_risk_label']

        if X.empty or len(X) != len(y):
            # print("Warning: Feature preparation resulted in empty data or mismatched lengths. Model not trained.")
            self.is_fitted = False
            return

        # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        # Using all data for training as it's small / synthetic for now
        self.model.fit(X, y)
        self.is_fitted = True
        logger.info("Friction prediction model trained.")
        self._save_model()

    def predict_friction_risk(self, current_data_per_location: pd.DataFrame) -> list[Any] | None:
        """
        Predict friction risk for various locations based on current data.
        :param current_data_per_location: DataFrame with features aggregated per location.
                                          Each row is a location.
        :return: List of FrictionRisk objects.
        """
        if not self.is_fitted:
            # print("Warning: Friction prediction model not fitted. Returning empty predictions.")
            # Consider training a default model or raising an error
            return []
        if current_data_per_location.empty:
            return []

        # The 'current_data_per_location' should have one row per location, with aggregated features.
        # Example: index is location_id, columns are 'avg_sentiment', 'grievance_count', etc.
        features = self._prepare_features(current_data_per_location)

        if features.empty:
            return []

        probabilities = self.model.predict_proba(features)[:, 1]  # Probability of class 1 (high risk)
        risks = []

        for i, prob_score in enumerate(probabilities):
            # Ensure location_name is correctly derived from features.index or current_data_per_location.index
            # current_data_per_location was the input to _prepare_features which created 'features'
            # So, features.index should correspond to current_data_per_location.index
            location_name = features.index[i] if features.index.name else f"Location_{i}"

            # Determine risk level string and explanation (simplified)
            if prob_score > 0.7:
                timeline_str = "short-term" # Or "High"
                explanation_str = f"High friction risk ({prob_score:.2f}) predicted for {location_name}."
            elif prob_score > 0.4:
                timeline_str = "medium-term" # Or "Medium"
                explanation_str = f"Medium friction risk ({prob_score:.2f}) predicted for {location_name}."
            else:
                timeline_str = "long-term" # Or "Low"
                explanation_str = f"Low friction risk ({prob_score:.2f}) predicted for {location_name}."

            # Get primary contributing factor (placeholder)
            primary_factor_str = self.feature_names[0] if self.feature_names else "N/A"
            # A more robust method (e.g., SHAP) would be needed for accurate instance-level explanations.
            # For now, we can try to list features with high values for this instance if available before scaling.
            # Or, if the model has feature_importances_ (like RF), use that (but it's global).
            # Simplistic: if model has coef_ (linear models), try to use that with feature values.
            # This part is complex and model-dependent for true accuracy.
            # Sticking to a simple placeholder for now.
            instance_features = features.iloc[i]
            try:
                if hasattr(self.model.named_steps['rf'], 'feature_importances_'):
                    importances = self.model.named_steps['rf'].feature_importances_
                    sorted_indices = np.argsort(importances)[::-1]
                    primary_factor_str = self.feature_names[sorted_indices[0]]
                elif hasattr(self.model.named_steps['rf'], 'coef_'): # For linear models in pipeline
                     # This needs unscaled features for interpretability or inverse transform coefs
                    primary_factor_str = self.feature_names[np.argmax(np.abs(instance_features.values * self.model.named_steps['rf'].coef_[0]))]
            except Exception: # Fallback if the above fails
                primary_factor_str = self.feature_names[0] if self.feature_names else "N/A"

            risks.append(FrictionRisk(
                location_name=location_name,
                risk_level=float(prob_score), 
                predicted_timeline=timeline_str,
                primary_contributing_factor=primary_factor_str, 
                explanation=explanation_str,
                confidence=float(abs(prob_score - 0.5) * 2),  # Simplistic confidence
                location_lat=DUMMY_LAT + (random.random() - 0.5) * 0.01,
                location_lon=DUMMY_LON + (random.random() - 0.5) * 0.01
            ))
        return risks

    def _save_model(self):
        """Saves the trained model, feature names, and fitted status to disk."""
        if not self.is_fitted:
            logger.warning("Model is not fitted. Nothing to save.")
            return
        try:
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            joblib.dump({
                'model_pipeline': self.model,
                'feature_names': self.feature_names,
                'is_fitted': self.is_fitted
            }, self.model_path)
            logger.info(f"Friction model saved to {self.model_path}")
        except Exception as e:
            logger.error(f"Error saving model to {self.model_path}: {e}")

    def _load_model(self):
        """Loads the model, feature names, and fitted status from disk if it exists."""
        if os.path.exists(self.model_path):
            try:
                model_data = joblib.load(self.model_path)
                self.model = model_data['model_pipeline']
                self.feature_names = model_data['feature_names']
                self.is_fitted = model_data['is_fitted']
                logger.info(f"Friction model loaded from {self.model_path}")
            except Exception as e:
                logger.error(f"Error loading model from {self.model_path}: {e}. A new model will be initialized.")
                self.is_fitted = False # Ensure it's marked as not fitted if loading fails
        else:
            logger.info(f"No pre-trained model found at {self.model_path}. A new model will be initialized.")
            self.is_fitted = False

