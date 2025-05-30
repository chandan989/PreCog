# src/ai_models/anomaly_detector.py
import pandas as pd
from typing import List, Dict, Any
from datetime import datetime, timedelta
from collections import Counter

# Assuming nlp_processor.py provides preprocess_text for keyword extraction if needed
# from .nlp_processor import preprocess_text 

def detect_sentiment_anomalies(data_df: pd.DataFrame, 
                               sentiment_col: str = 'sentiment_compound', 
                               threshold: float = -0.5, 
                               window_size: int = 10, 
                               min_anomalous_in_window: int = 3) -> List[Dict[str, Any]]:
    """
    Detects anomalies based on spikes in negative sentiment using a rolling window approach.
    `data_df` is expected to be a pandas DataFrame with a sentiment score column.
    `sentiment_col` specifies the column name containing sentiment scores (e.g., compound score from VADER).
    `threshold` is the sentiment score below which an item is considered highly negative.
    `window_size` is the number of data points in the rolling window.
    `min_anomalous_in_window` is the minimum number of highly negative items in a window to flag an anomaly.
    """
    anomalies = []
    if not isinstance(data_df, pd.DataFrame) or data_df.empty or sentiment_col not in data_df.columns:
        print(f"Data format incorrect or empty for sentiment anomaly detection. Expected DataFrame with '{sentiment_col}' column.")
        return anomalies

    # Ensure the sentiment column is numeric
    try:
        data_df[sentiment_col] = pd.to_numeric(data_df[sentiment_col])
    except ValueError:
        print(f"Error: Sentiment column '{sentiment_col}' could not be converted to numeric.")
        return anomalies

    data_df['is_highly_negative'] = data_df[sentiment_col] < threshold
    
    # Use rolling window to count highly negative items
    rolling_negative_count = data_df['is_highly_negative'].rolling(window=window_size, min_periods=1).sum()

    for i in range(len(data_df)):
        if rolling_negative_count.iloc[i] >= min_anomalous_in_window:
            # Check if this anomaly (based on end of window) is a new one or continuation
            # This logic avoids reporting overlapping anomalies repeatedly
            # More sophisticated grouping of continuous anomalies could be added
            is_new_anomaly = True
            if anomalies:
                last_anomaly = anomalies[-1]
                # If current anomaly start is within the last reported anomaly's window, consider it part of it
                if i < last_anomaly['index_end'] + window_size / 2 : # Heuristic to merge close anomalies
                    is_new_anomaly = False 
                    # Optionally, update the end of the last anomaly if this extends it
                    # anomalies[-1]['index_end'] = i 
                    # anomalies[-1]['description'] = f"Extended sentiment anomaly..."
            
            if is_new_anomaly:
                window_start_index = max(0, i - window_size + 1)
                current_window_data = data_df.iloc[window_start_index : i + 1]
                actual_negative_in_window = current_window_data['is_highly_negative'].sum()

                if actual_negative_in_window >= min_anomalous_in_window: # Final check on the specific window
                    anomaly_info = {
                        "type": "Sentiment Spike",
                        "index_start": window_start_index,
                        "index_end": i,
                        "description": f"Potential sentiment anomaly: {actual_negative_in_window}/{len(current_window_data)} items highly negative (score < {threshold}) in window ending at index {i}.",
                        "triggering_data_indices": current_window_data[current_window_data['is_highly_negative']].index.tolist(),
                        # "items_in_window": current_window_data.to_dict('records') # Can be large
                    }
                    anomalies.append(anomaly_info)
                    # print(f"Sentiment anomaly detected: Window [{window_start_index}-{i}], {actual_negative_in_window} negative items.")
    
    data_df.drop(columns=['is_highly_negative'], inplace=True)
    return anomalies

def detect_misinformation_keyword_velocity(data_df: pd.DataFrame, 
                                         text_col: str = 'processed_text', 
                                         timestamp_col: str = 'timestamp', 
                                         keywords: List[str] = None, 
                                         time_window_minutes: int = 60, 
                                         velocity_threshold: int = 5,
                                         min_keyword_length: int = 3) -> List[Dict[str, Any]]:
    """
    Detects potential misinformation based on rapid spread (velocity) of specific keywords or general frequent terms.
    `data_df` is a pandas DataFrame with a text column (preferably preprocessed tokens) and a timestamp column.
    `text_col` is the column with text or list of tokens.
    `timestamp_col` is the column with datetime objects or parseable timestamp strings.
    `keywords` is a list of specific keywords to track. If None, will look for general high-frequency terms.
    `time_window_minutes` defines the period over which to calculate velocity.
    `velocity_threshold` is the minimum number of keyword occurrences in the time window to flag an anomaly.
    `min_keyword_length` is the minimum length for a token to be considered a keyword if `keywords` is None.
    """
    anomalies = []
    if not isinstance(data_df, pd.DataFrame) or data_df.empty or text_col not in data_df.columns or timestamp_col not in data_df.columns:
        print(f"Data format incorrect or empty for misinformation detection. Expected DataFrame with '{text_col}' and '{timestamp_col}'.")
        return anomalies

    # Ensure timestamp column is datetime
    try:
        data_df[timestamp_col] = pd.to_datetime(data_df[timestamp_col])
    except Exception as e:
        print(f"Error parsing timestamp column '{timestamp_col}': {e}. Ensure it's in a recognizable format.")
        return anomalies

    # Sort by timestamp to process chronologically
    df_sorted = data_df.sort_values(by=timestamp_col).copy()
    df_sorted.reset_index(drop=True, inplace=True)

    time_delta = timedelta(minutes=time_window_minutes)

    for i in range(len(df_sorted)):
        current_time = df_sorted[timestamp_col].iloc[i]
        window_start_time = current_time - time_delta
        
        # Get data within the current time window (ending at current_item)
        window_df = df_sorted[(df_sorted[timestamp_col] >= window_start_time) & (df_sorted[timestamp_col] <= current_time)]
        
        if window_df.empty:
            continue

        all_tokens_in_window = []
        for text_content in window_df[text_col]:
            if isinstance(text_content, list): # Assuming preprocessed tokens
                all_tokens_in_window.extend(text_content)
            elif isinstance(text_content, str): # If raw text, basic split (ideally preprocess)
                all_tokens_in_window.extend([word.lower() for word in text_content.split() if len(word) >= min_keyword_length])
        
        if not all_tokens_in_window:
            continue
            
        token_counts_in_window = Counter(all_tokens_in_window)

        tracked_keywords = keywords
        if not tracked_keywords: # If no specific keywords, look for any high-frequency terms
            # Consider terms that appear frequently in this window as potential keywords of interest
            # This is a heuristic and might need refinement
            tracked_keywords = [token for token, count in token_counts_in_window.items() if count >= velocity_threshold and len(token) >= min_keyword_length]

        for keyword in tracked_keywords:
            if token_counts_in_window.get(keyword, 0) >= velocity_threshold:
                # Avoid re-flagging the exact same keyword spike for overlapping windows if the core items are the same
                # This is a simple check; more robust de-duplication might be needed
                is_new_spike = True
                for anom in reversed(anomalies):
                    if anom['keyword'] == keyword and (current_time - anom['last_timestamp_in_anomaly']).total_seconds() < time_window_minutes * 60 / 2:
                        # If a similar keyword spike was recently logged, potentially merge or ignore
                        # For simplicity, we'll just skip if too recent to avoid floods of similar alerts
                        is_new_spike = False
                        break
                
                if is_new_spike:
                    anomaly_info = {
                        "type": "Keyword Velocity Spike",
                        "keyword": keyword,
                        "count_in_window": token_counts_in_window[keyword],
                        "time_window_minutes": time_window_minutes,
                        "start_time_of_window": window_start_time.isoformat(),
                        "end_time_of_window": current_time.isoformat(), # This is also the timestamp of the last item triggering it
                        "description": f"Keyword '{keyword}' appeared {token_counts_in_window[keyword]} times in {time_window_minutes} mins ending at {current_time.isoformat()}.",
                        "triggering_data_indices": window_df[window_df[text_col].apply(lambda x: keyword in (x if isinstance(x,list) else str(x).lower().split()) )].index.tolist(),
                        "last_timestamp_in_anomaly": current_time # Helper for de-duplication
                    }
                    anomalies.append(anomaly_info)
                    # print(f"Misinfo keyword velocity: '{keyword}' {token_counts_in_window[keyword]} times.")
    return anomalies

if __name__ == '__main__':
    # --- Sentiment Anomaly Example ---
    print("\n--- Sentiment Anomaly Detection Example ---")
    sentiment_data_list = [
        {'id': 1, 'text': 'Everything is fine', 'sentiment_compound': 0.1, 'timestamp': '2023-01-01T10:00:00Z'},
        {'id': 2, 'text': 'Slight issue here', 'sentiment_compound': -0.2, 'timestamp': '2023-01-01T10:05:00Z'},
        {'id': 3, 'text': 'This is a bad problem', 'sentiment_compound': -0.7, 'timestamp': '2023-01-01T10:10:00Z'},
        {'id': 4, 'text': 'Terrible situation unfolding', 'sentiment_compound': -0.8, 'timestamp': '2023-01-01T10:15:00Z'},
        {'id': 5, 'text': 'Another very bad report', 'sentiment_compound': -0.6, 'timestamp': '2023-01-01T10:20:00Z'},
        {'id': 6, 'text': 'Okay, but still concerning', 'sentiment_compound': -0.55, 'timestamp': '2023-01-01T10:25:00Z'},
        {'id': 7, 'text': 'Things are looking up', 'sentiment_compound': 0.3, 'timestamp': '2023-01-01T10:30:00Z'},
        {'id': 8, 'text': 'All good now', 'sentiment_compound': 0.5, 'timestamp': '2023-01-01T10:35:00Z'},
        {'id': 9, 'text': 'Horrible news again!', 'sentiment_compound': -0.9, 'timestamp': '2023-01-01T11:00:00Z'},
        {'id': 10, 'text': 'This is unacceptable and bad', 'sentiment_compound': -0.75, 'timestamp': '2023-01-01T11:05:00Z'},
    ]
    sample_sentiment_df = pd.DataFrame(sentiment_data_list)
    
    sentiment_anomalies = detect_sentiment_anomalies(sample_sentiment_df, 
                                                     sentiment_col='sentiment_compound', 
                                                     threshold=-0.5, 
                                                     window_size=4, 
                                                     min_anomalous_in_window=2)
    if sentiment_anomalies:
        print(f"Found {len(sentiment_anomalies)} sentiment anomalies:")
        for anomaly in sentiment_anomalies:
            print(f"  - {anomaly['description']} (Indices: {anomaly['triggering_data_indices']})")
    else:
        print("No sentiment anomalies detected in sample data.")

    # --- Misinformation Keyword Velocity Example ---
    print("\n--- Misinformation Keyword Velocity Detection Example ---")
    # Assume preprocess_text from nlp_processor would create 'processed_text' column with lists of tokens
    misinfo_data_list = [
        {'id': 1, 'raw_text': 'Rumor about water contamination spreading fast in Sector A!', 'timestamp': '2023-01-01T10:00:00Z', 'processed_text': ['rumor', 'water', 'contamin', 'spread', 'fast', 'sector', 'a']},
        {'id': 2, 'raw_text': 'Heard the water is unsafe in Zone B due to contamination', 'timestamp': '2023-01-01T10:05:00Z', 'processed_text': ['heard', 'water', 'unsaf', 'zone', 'b', 'due', 'contamin']},
        {'id': 3, 'raw_text': 'Is the water contamination story true?', 'timestamp': '2023-01-01T10:10:00Z', 'processed_text': ['water', 'contamin', 'stori', 'true']},
        {'id': 4, 'raw_text': 'Official statement: Water is safe. Please ignore rumors.', 'timestamp': '2023-01-01T11:00:00Z', 'processed_text': ['offici', 'statement', 'water', 'safe', 'pleas', 'ignor', 'rumor']},
        {'id': 5, 'raw_text': 'Another claim about water issues and contamination in Zone B, this is serious', 'timestamp': '2023-01-01T10:12:00Z', 'processed_text': ['anoth', 'claim', 'water', 'issu', 'contamin', 'zone', 'b', 'seriou']},
        {'id': 6, 'raw_text': 'Contamination confirmed by unofficial source, be careful with water.', 'timestamp': '2023-01-01T10:15:00Z', 'processed_text': ['contamin', 'confirm', 'unoffici', 'sourc', 'care', 'water']},
        {'id': 7, 'raw_text': 'My neighbor said the water is definitely contaminated.', 'timestamp': '2023-01-01T10:18:00Z', 'processed_text': ['neighbor', 'said', 'water', 'definit', 'contamin']}
    ]
    sample_misinfo_df = pd.DataFrame(misinfo_data_list)

    # Example 1: Tracking specific keywords
    specific_keywords = ['contamin', 'rumor']
    print(f"Tracking specific keywords: {specific_keywords}")
    misinfo_anomalies_specific = detect_misinformation_keyword_velocity(sample_misinfo_df, 
                                                                  text_col='processed_text', 
                                                                  timestamp_col='timestamp', 
                                                                  keywords=specific_keywords, 
                                                                  time_window_minutes=15, 
                                                                  velocity_threshold=3)
    if misinfo_anomalies_specific:
        print(f"Found {len(misinfo_anomalies_specific)} misinformation patterns (specific keywords):")
        for anomaly in misinfo_anomalies_specific:
            print(f"  - {anomaly['description']}") # Indices: {anomaly['triggering_data_indices']}
    else:
        print("No misinformation patterns detected for specific keywords.")

    # Example 2: Letting the function find high-velocity general terms
    print(f"\nTracking general high-velocity keywords (min_len=4, threshold=3 in 20 mins):")
    misinfo_anomalies_general = detect_misinformation_keyword_velocity(sample_misinfo_df, 
                                                                  text_col='processed_text', 
                                                                  timestamp_col='timestamp', 
                                                                  keywords=None, # Let it find frequent terms
                                                                  time_window_minutes=20, 
                                                                  velocity_threshold=3,
                                                                  min_keyword_length=4)
    if misinfo_anomalies_general:
        print(f"Found {len(misinfo_anomalies_general)} misinformation patterns (general keywords):")
        for anomaly in misinfo_anomalies_general:
            print(f"  - {anomaly['description']}")
    else:
        print("No misinformation patterns detected for general keywords.")