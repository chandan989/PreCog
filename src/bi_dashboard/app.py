# src/bi_dashboard/app.py
import streamlit as st
import pandas as pd
import json
from datetime import datetime

# Attempt to import project modules
# These imports assume the dashboard is run from the project root (e.g., `streamlit run src/bi_dashboard/app.py`)
# or that the `src` directory is in PYTHONPATH.
try:
    from src.data_ingestion.data_loader import DataManager, load_social_media_data_from_df, load_news_data_from_df, load_crowdsourced_reports_from_df
    from src.ai_models.nlp_processor import process_text_data, batch_analyze_sentiment, batch_extract_topics
    from src.ai_models.anomaly_detector import detect_sentiment_anomalies, detect_misinformation_keyword_velocity
    from src.ai_models.intervention_recommender import recommend_interventions
except ImportError:
    # Fallback for direct execution if imports fail (e.g. if run directly from bi_dashboard folder)
    # This is less ideal as it duplicates paths. Proper PYTHONPATH setup is better.
    import sys
    import os
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
    from src.data_ingestion.data_loader import DataManager, load_social_media_data_from_df, load_news_data_from_df, load_crowdsourced_reports_from_df
    from src.ai_models.nlp_processor import process_text_data, batch_analyze_sentiment, batch_extract_topics
    from src.ai_models.anomaly_detector import detect_sentiment_anomalies, detect_misinformation_keyword_velocity
    from src.ai_models.intervention_recommender import recommend_interventions


def preprocess_text_series(text_series: pd.Series) -> pd.Series:
    """
    Applies the process_text_data function to each entry in the text_series.
    Ensures NaNs are handled and returns a Series of processed text strings.
    """
    # Replace NaN with empty string to avoid errors
    filled = text_series.fillna("")
    # Apply the NLP processing to each text entry
    return filled.apply(lambda x: process_text_data(x))


# Helper function to load data from uploaded file
def load_data_from_upload(uploaded_file):
    if uploaded_file is None:
        return None
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith('.json') or uploaded_file.name.endswith('.jsonl'):
            # For JSON, assume a list of records or one record per line (JSONL)
            try:
                df = pd.read_json(uploaded_file)
            except ValueError:
                uploaded_file.seek(0)
                records = [json.loads(line) for line in uploaded_file.readlines()]
                df = pd.DataFrame(records)
        else:
            st.error("Unsupported file type. Please upload CSV or JSON/JSONL.")
            return None

        # Basic validation: ensure 'text' and 'timestamp' columns exist
        if 'text' not in df.columns or 'timestamp' not in df.columns:
            st.error("Uploaded data must contain 'text' and 'timestamp' columns.")
            # Show columns found for debugging
            st.write("Columns found:", df.columns.tolist())
            # Attempt to use first string column as text and first datetime-like as timestamp if available
            if 'text' not in df.columns:
                str_cols = df.select_dtypes(include='object').columns
                if len(str_cols) > 0:
                    df.rename(columns={str_cols[0]: 'text'}, inplace=True)
                    st.warning(f"Used column '{str_cols[0]}' as 'text'. Please verify.")
                else:
                    return None
            if 'timestamp' not in df.columns:
                # Try to infer timestamp from common names or convert first suitable column
                potential_ts_cols = [col for col in df.columns if 'time' in col.lower() or 'date' in col.lower()]
                if potential_ts_cols:
                    df.rename(columns={potential_ts_cols[0]: 'timestamp'}, inplace=True)
                    st.warning(f"Used column '{potential_ts_cols[0]}' as 'timestamp'. Please verify.")
                else:  # if no suitable column, create a dummy one for PoC
                    df['timestamp'] = pd.to_datetime(datetime.now())
                    st.warning("No 'timestamp' column found. Added current time as dummy timestamp.")

        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        df.dropna(subset=['timestamp', 'text'], inplace=True)
        return df

    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

def main_dashboard():
    st.set_page_config(layout="wide", page_title="PreCog Dashboard")
    st.title("PreCog - Social Conflict Prevention Dashboard")

    st.sidebar.header("1. Load Data")
    uploaded_file = st.sidebar.file_uploader("Upload data (CSV or JSON/JSONL with 'text' and 'timestamp' columns)", type=['csv', 'json', 'jsonl'])

    if 'processed_data' not in st.session_state:
        st.session_state.processed_data = None
    if 'anomalies' not in st.session_state:
        st.session_state.anomalies = None
    if 'recommendations' not in st.session_state:
        st.session_state.recommendations = None

    raw_df = None
    if uploaded_file:
        raw_df = load_data_from_upload(uploaded_file)
    else:
        st.sidebar.info("No file uploaded. Attempting to load latest data from `data/raw` directory.")
        try:
            data_manager = DataManager(base_dir="data") # Assumes 'data' is in the same root as 'src'
            loaded_data_dict = data_manager.load_raw_data()
            if loaded_data_dict:
                # Combine all loaded dataframes into one. 
                # This assumes they have compatible 'text' and 'timestamp' columns or can be processed to have them.
                # For simplicity, we'll concatenate and then rely on later processing to handle potential schema differences.
                all_dfs = []
                for name, df_item in loaded_data_dict.items():
                    # Ensure 'text' and 'timestamp' columns exist, similar to load_data_from_upload
                    if 'text' not in df_item.columns or 'timestamp' not in df_item.columns:
                        st.warning(f"Dataset '{name}' is missing 'text' or 'timestamp'. Trying to adapt.")
                        if 'text' not in df_item.columns:
                            str_cols = df_item.select_dtypes(include='object').columns
                            if len(str_cols) > 0:
                                df_item.rename(columns={str_cols[0]: 'text'}, inplace=True)
                        if 'timestamp' not in df_item.columns:
                            potential_ts_cols = [col for col in df_item.columns if 'time' in col.lower() or 'date' in col.lower()]
                            if potential_ts_cols:
                                df_item.rename(columns={potential_ts_cols[0]: 'timestamp'}, inplace=True)
                            else:
                                df_item['timestamp'] = pd.to_datetime(datetime.now()) # Dummy if not found
                    
                    if 'text' in df_item.columns and 'timestamp' in df_item.columns:
                        df_item['timestamp'] = pd.to_datetime(df_item['timestamp'], errors='coerce')
                        df_item.dropna(subset=['timestamp', 'text'], inplace=True)
                        all_dfs.append(df_item)
                    else:
                        st.warning(f"Skipping dataset '{name}' due to missing critical columns after adaptation.")

                if all_dfs:
                    raw_df = pd.concat(all_dfs, ignore_index=True)
                    st.sidebar.success(f"Loaded {len(raw_df)} records from `data/raw`.")
                else:
                    st.sidebar.warning("No usable data found in `data/raw` or datasets were incompatible.")
            else:
                st.sidebar.warning("No data found in `data/raw` directory.")
        except Exception as e:
            st.sidebar.error(f"Error loading data from directory: {e}")

    if raw_df is not None and not raw_df.empty:
        # st.sidebar.success(f"Loaded {len(raw_df)} records.") # Moved up for clarity
        if st.sidebar.button("Process Data for Analysis"):
            with st.spinner("Processing data..."):
                # 1. NLP Processing (using 'text' column)
                raw_df['processed_text'] = preprocess_text_series(raw_df['text'])
                sentiment_results = batch_analyze_sentiment(raw_df['processed_text'].tolist())
                sentiment_df = pd.DataFrame(sentiment_results)
                # print(sentiment_df.head())
                processed_df = pd.concat([raw_df, sentiment_df], axis=1)
                st.session_state.processed_data = processed_df
                st.success("NLP processing complete!")

    if st.session_state.processed_data is not None:
        processed_df = st.session_state.processed_data
        st.header("Data Overview & NLP Results")
        st.dataframe(processed_df)

        st.sidebar.header("2. Anomaly Detection Settings")
        sentiment_threshold = st.sidebar.slider("Negative Sentiment Threshold", -1.0, 0.0, -0.5, 0.05)
        sentiment_window = st.sidebar.slider("Sentiment Anomaly Window Size", 5, 50, 10, 1)
        min_anomalous_in_sentiment_window = st.sidebar.slider("Min Negative in Window (Sentiment)", 1, sentiment_window, max(1, sentiment_window//2), 1)
        
        misinfo_keywords_input = st.sidebar.text_input("Keywords for Misinfo (comma-separated)", "rumor,fake,hoax")
        misinfo_keywords = [k.strip() for k in misinfo_keywords_input.split(',') if k.strip()] if misinfo_keywords_input else None
        misinfo_window_minutes = st.sidebar.slider("Misinfo Keyword Window (Minutes)", 10, 360, 60, 10)
        misinfo_velocity_threshold = st.sidebar.slider("Misinfo Keyword Velocity Threshold", 2, 20, 3, 1)

        if st.sidebar.button("Detect Anomalies"):
            with st.spinner("Detecting anomalies..."):
                sentiment_anomalies_list = detect_sentiment_anomalies(
                    processed_df.copy(), 
                    sentiment_col='compound',
                    threshold=sentiment_threshold, 
                    window_size=sentiment_window,
                    min_anomalous_in_window=min_anomalous_in_sentiment_window
                )
                
                misinfo_anomalies_list = detect_misinformation_keyword_velocity(
                    processed_df.copy(),
                    text_col='processed_text', # or 'topics' if that's preferred for keywords
                    timestamp_col='timestamp',
                    keywords=misinfo_keywords,
                    time_window_minutes=misinfo_window_minutes,
                    velocity_threshold=misinfo_velocity_threshold
                )
                st.session_state.anomalies = sentiment_anomalies_list + misinfo_anomalies_list
                st.success(f"Anomaly detection complete! Found {len(st.session_state.anomalies)} potential issues.")

    if st.session_state.anomalies is not None:
        st.header("Detected Anomalies")
        if not st.session_state.anomalies:
            st.info("No anomalies detected with current settings.")
        else:
            for i, anomaly in enumerate(st.session_state.anomalies):
                with st.expander(f"Anomaly {i+1}: {anomaly.get('type', 'Unknown')} - {anomaly.get('description', 'N/A')[:100]}..."):
                    st.json(anomaly) # Display full anomaly details
            
            st.sidebar.header("3. Generate Interventions")
            if st.sidebar.button("Recommend Interventions"):
                with st.spinner("Generating recommendations..."):
                    st.session_state.recommendations = recommend_interventions(st.session_state.anomalies)
                    st.success(f"Generated {len(st.session_state.recommendations)} recommendations.")

    if st.session_state.recommendations is not None:
        st.header("Intervention Recommendations")
        if not st.session_state.recommendations:
            st.info("No interventions recommended for the detected anomalies.")
        else:
            for i, rec in enumerate(st.session_state.recommendations):
                st.subheader(f"Recommendation {i+1}: For '{rec['issue_trigger_summary'][:100]}...' ({rec['category']})")
                st.markdown(f"**Priority:** {rec['priority']}")
                st.markdown(f"**Responsible Actors:** {', '.join(rec['responsible_actors'])}")
                
                if rec['short_term_actions']:
                    st.markdown("**Short-term Actions:**")
                    for action in rec['short_term_actions']:
                        st.markdown(f"- {action}")
                if rec['medium_term_actions']:
                    st.markdown("**Medium-term Actions:**")
                    for action in rec['medium_term_actions']:
                        st.markdown(f"- {action}")
                if rec['long_term_actions']:
                    st.markdown("**Long-term Actions:**")
                    for action in rec['long_term_actions']:
                        st.markdown(f"- {action}")
                st.markdown(f"**Monitoring Indicators:** {', '.join(rec['monitoring_indicators'])}")
                with st.expander("View Triggering Issue Details"):
                    st.json(rec['issue_details'])
                st.markdown("---    ")
    else:
        if uploaded_file and st.session_state.processed_data is None:
            st.info("Click 'Process Data for Analysis' to proceed.")
        elif st.session_state.processed_data is not None and st.session_state.anomalies is None:
            st.info("Click 'Detect Anomalies' to identify potential issues.")
        elif st.session_state.anomalies is not None and st.session_state.recommendations is None:
            st.info("Click 'Recommend Interventions' to get suggestions.")
        elif not uploaded_file:
            st.info("Upload a data file to begin the analysis workflow.")

if __name__ == '__main__':
    main_dashboard()