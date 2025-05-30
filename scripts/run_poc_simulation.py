# scripts/run_poc_simulation.py

import sys
import os
from typing import List

import pandas as pd
from datetime import datetime, timedelta
import json

# Add src directory to Python path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(PROJECT_ROOT)

from src.data_ingestion.data_loader import (
    create_dummy_data_files as create_ingestion_dummy_files,
    load_social_media_data, 
    load_news_data,
    load_crowdsourced_reports
)
from src.ai_models.nlp_processor import (
    process_text_data,
    batch_analyze_sentiment,
    batch_extract_topics, preprocess_text_series
)
from src.ai_models.anomaly_detector import (
    detect_sentiment_anomalies,
    detect_misinformation_keyword_velocity
)
from src.ai_models.intervention_recommender import recommend_interventions, InterventionRecommendation
from src.utils.helpers import get_current_timestamp, format_data_for_logging

# --- Configuration for PoC --- 
DATA_DIR = os.path.join(PROJECT_ROOT, 'data/raw')
SIMULATED_SOCIAL_MEDIA_FILE = os.path.join(DATA_DIR, 'synthetic_social_media.csv')
SIMULATED_NEWS_FILE = os.path.join(DATA_DIR, 'synthetic_news_articles.csv')
SIMULATED_CROWD_REPORTS_FILE = os.path.join(DATA_DIR, 'crowd_synthetic.csv')


def run_simulation():
    """Runs a simulation of the PreCog PoC pipeline with enhanced modules."""
    print(f"--- PreCog PoC Simulation Started at {get_current_timestamp()} ---")

    # Ensure dummy data files exist (using the one from data_loader now)
    create_ingestion_dummy_files(data_dir=DATA_DIR)

    # 1. Data Ingestion
    print("\n--- 1. Data Ingestion ---")
    social_df = load_social_media_data(SIMULATED_SOCIAL_MEDIA_FILE)
    news_df = load_news_data(SIMULATED_NEWS_FILE)
    crowd_df = load_crowdsourced_reports(SIMULATED_CROWD_REPORTS_FILE)


    if social_df.empty and news_df.empty and crowd_df.empty:
        print("No data loaded. Exiting simulation.")
        return

    # Combine data sources into one DataFrame for this simulation
    # In a real system, they might be processed differently or at different times
    all_data_list = []
    if not social_df.empty:
        social_df['source'] = 'social_media'
        all_data_list.append(social_df)
    if not news_df.empty:
        news_df['source'] = 'news'
        all_data_list.append(news_df)
    if not crowd_df.empty:
        crowd_df['source'] = 'crowdsourced_report'
        all_data_list.append(crowd_df)
    
    if not all_data_list:
        print("No data to process after attempting to load all sources.")
        return

    combined_df = pd.concat(all_data_list, ignore_index=True)
    # Ensure 'timestamp' is datetime
    combined_df['timestamp'] = pd.to_datetime(combined_df['timestamp'], errors='coerce')
    combined_df.dropna(subset=['text', 'timestamp'], inplace=True)
    combined_df.sort_values(by='timestamp', inplace=True)

    print(f"Loaded a total of {len(combined_df)} records from all sources.")
    print("Sample of combined data:")
    print(combined_df.head())

    # 2. AI Processing
    print("\n--- 2. AI Processing ---")
    # Text preprocessing (cleaning, tokenization, etc.)
    combined_df['processed_text'] = preprocess_text_series(combined_df['text'])
    processed_df = combined_df
    # Sentiment Analysis
    processed_df = batch_analyze_sentiment(processed_df) # Adds 'sentiment_compound', 'sentiment_label'
    # Topic Extraction
    processed_df = batch_extract_topics(processed_df, num_topics=3) # Adds 'topics'


    print("\nSample of processed data with NLP features:")
    print(processed_df)

    # 3. Anomaly Detection
    print("\n--- 3. Anomaly Detection ---")
    # Detect Sentiment Anomalies
    # Using 'sentiment_compound' and 'timestamp' from processed_df
    sentiment_anomalies = detect_sentiment_anomalies(
        processed_df.copy(), 
        sentiment_col='sentiment_compound', 
        timestamp_col='timestamp',
        threshold=-0.3, # More sensitive for demo
        window_size=5, 
        min_anomalous_in_window=2
    )
    
    # Detect Misinformation Keyword Velocity
    misinfo_keywords = ['protest', 'riot', 'fake news', 'conspiracy', 'urgent warning']
    misinfo_anomalies = detect_misinformation_keyword_velocity(
        processed_df.copy(),
        text_col='processed_text', # or 'topics'
        timestamp_col='timestamp',
        keywords=misinfo_keywords,
        time_window_minutes=120, # 2 hours
        velocity_threshold=2 # 2 mentions in 2 hours
    )

    all_detected_issues = sentiment_anomalies + misinfo_anomalies

    if all_detected_issues:
        print(f"\nFound {len(all_detected_issues)} total potential anomalies/issues:")
        for i, issue in enumerate(all_detected_issues):
            print(f"Issue {i+1}: Type: {issue.get('type', 'N/A')}, Description: {issue.get('description', 'N/A')[:100]}...")
            # print(format_data_for_logging(issue, indent=2))
    else:
        print("No significant anomalies detected in this run.")

    # 4. Intervention Recommendation
    print("\n--- 4. Intervention Recommendation ---")
    if all_detected_issues:
        recommendations: List[InterventionRecommendation] = recommend_interventions(all_detected_issues)
        if recommendations:
            print(f"\nGenerated {len(recommendations)} recommendations:")
            for i, rec_obj in enumerate(recommendations):
                rec = rec_obj.to_dict() # Convert Pydantic model to dict for printing
                print(f"\nRecommendation {i+1} for: {rec['issue_trigger_summary']}")
                print(f"  Category: {rec['category']}, Priority: {rec['priority']}")
                print(f"  Responsible: {', '.join(rec['responsible_actors'])}")
                if rec['short_term_actions']:
                    print(f"  Short-term: {'; '.join(rec['short_term_actions'])}")
                if rec['medium_term_actions']:
                    print(f"  Medium-term: {'; '.join(rec['medium_term_actions'])}")
                if rec['long_term_actions']:
                    print(f"  Long-term: {'; '.join(rec['long_term_actions'])}")
                print(f"  Monitor: {', '.join(rec['monitoring_indicators'])}")
                # print("  Triggering Issue Details:")
                # print(format_data_for_logging(rec['issue_details'], indent=2))
        else:
            print("No specific interventions recommended for the detected issues.")
    else:
        print("No issues detected that warrant intervention recommendations in this run.")

    print(f"\n--- PreCog PoC Simulation Ended at {get_current_timestamp()} ---")

if __name__ == '__main__':
    run_simulation()