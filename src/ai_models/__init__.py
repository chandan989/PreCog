# src/ai_models/__init__.py

from .nlp_processor import (
    preprocess_text_series,
    process_text_data,
    analyze_sentiment_vader,
    batch_analyze_sentiment,
    extract_topics_tfidf,
    batch_extract_topics,
    download_nltk_resources # If this should be callable e.g. for setup
)

from .anomaly_detector import (
    detect_sentiment_anomalies,
    detect_misinformation_keyword_velocity
)

from .intervention_recommender import (
    recommend_interventions,
    InterventionRecommendation # Export the class as well
)

__all__ = [
    # NLP Processor
    "preprocess_text_series",
    "process_text_data",
    "analyze_sentiment_vader",
    "batch_analyze_sentiment",
    "extract_topics_tfidf",
    "batch_extract_topics",
    "download_nltk_resources",
    # Anomaly Detector
    "detect_sentiment_anomalies",
    "detect_misinformation_keyword_velocity",
    # Intervention Recommender
    "recommend_interventions",
    "InterventionRecommendation"
]

# print("ai_models package initialized") # Optional