import streamlit as st
import pandas as pd
import sys
import os

# Debugging path issues
print(f"DEBUG: Initial CWD: {os.getcwd()}")
print(f"DEBUG: Initial sys.path: {sys.path}")

# Add project root to Python path to allow direct imports
# app.py is in src/precog/dashboard/
# PROJECT_ROOT should be the parent of 'src'
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
print(f"DEBUG: Calculated PROJECT_ROOT: {PROJECT_ROOT}")

# It's generally better to append to sys.path unless there's a conflict
# But for ensuring our project's 'src' is found, insert(0) is common.
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
    print(f"DEBUG: PROJECT_ROOT inserted into sys.path.")
else:
    print(f"DEBUG: PROJECT_ROOT already in sys.path.")


print(f"DEBUG: Modified sys.path: {sys.path}")
print(f"DEBUG: Current CWD after path changes: {os.getcwd()}") # Check if CWD changed

# Verify existence of key directories and files based on PROJECT_ROOT
print(f"DEBUG: Does PROJECT_ROOT ({PROJECT_ROOT}) exist? {os.path.exists(PROJECT_ROOT)}")
src_dir_path = os.path.join(PROJECT_ROOT, 'src')
print(f"DEBUG: Does src dir ({src_dir_path}) exist? {os.path.exists(src_dir_path)}")
precog_dir_path = os.path.join(src_dir_path, 'precog')
print(f"DEBUG: Does precog dir ({precog_dir_path}) exist? {os.path.exists(precog_dir_path)}")
# Corrected path for system.py
system_file_path = os.path.join(precog_dir_path, 'core', 'system.py') 
print(f"DEBUG: Does core/system.py file ({system_file_path}) exist? {os.path.exists(system_file_path)}")
# Corrected path for config.py
config_file_path = os.path.join(precog_dir_path, 'config', 'config.py') 
print(f"DEBUG: Does config/config.py file ({config_file_path}) exist? {os.path.exists(config_file_path)}")

# Try importing after debug prints
print("DEBUG: Attempting to import HyperlocalIntelligenceSystem and get_config...")
# Corrected import for HyperlocalIntelligenceSystem
from src.precog.core.system import HyperlocalIntelligenceSystem 
# Corrected import for get_config
from src.precog.config.config import get_config 
print("DEBUG: Imports successful.")

# --- Page Configuration --- #
st.set_page_config(
    page_title="PreCog Dashboard",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Caching System Initialization --- #
@st.cache_resource # Use cache_resource for non-data objects like system instances
def load_system():
    print("Loading HyperlocalIntelligenceSystem...")
    # Load default config, can be made configurable later
    config = get_config()
    # Ensure data directories exist or handle appropriately
    os.makedirs(config.get('DEFAULT_DATA_DIR', 'data'), exist_ok=True)
    os.makedirs(os.path.join(config.get('DEFAULT_DATA_DIR', 'data'), 'synthetic'), exist_ok=True)
    os.makedirs(os.path.join(config.get('DEFAULT_DATA_DIR', 'data'), 'reports'), exist_ok=True)
    
    # Initialize system (adjust parameters as needed, e.g., from config or UI)
    # Forcing LLM usage to False for now to simplify dashboard startup
    # and avoid API key issues during initial dashboard bring-up.
    # These can be made configurable via sidebar later.
    config['USE_LLM_FOR_SENTIMENT'] = False
    config['USE_LLM_FOR_NARRATIVES'] = False
    config['USE_LLM_FOR_COUNTER_NARRATIVES'] = False
    config['USE_LLM_FOR_INTERVENTIONS'] = False

    # Create a dummy config object or dictionary to pass if system expects it
    # For now, assuming system can be initialized without explicit config if it uses get_config internally
    # or if we pass the modified config dict.
    # The HyperlocalIntelligenceSystem constructor doesn't take config directly in current version.
    # It relies on environment variables or get_config().
    # We've modified get_config to be influenced by env vars in main.py, but here we modify it directly.
    # This is a bit of a hack; ideally, system would take a config dict.
    
    # Update environment variables based on the modified config for the system to pick up
    # This is still not ideal, but a workaround if system strictly uses get_config() which reads env vars.
    # A better way would be to pass the config dict to the system's constructor if it supported it.
    # For now, let's assume the system's internal get_config() will be called and we can't easily override it here
    # without more significant refactoring of the system class or config loading.
    # The simplest path for the dashboard is to initialize the system and let it use its default config loading.
    # The LLM flags are part of the config dict returned by get_config, so if the system uses that, it should work.

    system = HyperlocalIntelligenceSystem(data_dir=config.get('DEFAULT_DATA_DIR', 'data'))
    # Models are trained during HyperlocalIntelligenceSystem initialization.
    # The friction_predictor.is_fitted check can be used for verification if needed.
    if not system.friction_predictor.is_fitted:
        print("WARNING: Friction model was not fitted after system initialization!")
    # else:
    #     print("System initialized and friction model is fitted.")
    return system

# --- Main Dashboard Logic --- #
def main():
    st.title("PreCog: Hyperlocal Intelligence Dashboard")

    # --- Sidebar Controls --- #
    st.sidebar.header("Controls")
    if st.sidebar.button("Refresh Data & Analysis"):
        st.cache_resource.clear() # Clear cache to reload system and data
        st.rerun()

    # Load the system
    system = load_system()

    # Run analysis (or load latest analysis results)
    # For simplicity, running a single cycle. In a real app, this might be triggered or scheduled.
    print("Running analysis for dashboard...")
    # The analyze_current_situation might need specific inputs or use defaults
    # It returns a tuple: (sentiment_alerts, misinformation_alerts, friction_risks, intervention_recommendations)
    # We need to ensure the system is ready (models trained, etc.)
    # The load_system function now attempts to train if not fitted.
    # The analyze_current_situation method expects dataframes, not paths or flags to generate them.
    # Data loading/generation should happen before calling analyze_current_situation.
    # For now, let's assume the system's default data loading/generation is sufficient
    # or that we need to call a different method to prepare data first.
    # The HyperlocalIntelligenceSystem.run_simulation_cycle() seems more appropriate as it handles data loading.

    # Let's try using run_simulation_cycle instead, which seems to handle data loading internally.
    # It also returns the same set of results.
    # We need to ensure the LLM flags are set in the config within load_system if run_simulation_cycle uses them.
    # The LLM flags are already set to False in load_system's config modification.
    print("Loading data for dashboard analysis...")
    # Call load_and_prepare_data, then analyze_current_situation
    # Defaulting to synthetic data for the dashboard for now.
    data = system.load_and_prepare_data(
        fetch_live=False, 
        generate_synthetic=True,
        social_media_path="/Users/chandan/Documents/Elykid Private Limited/Products/PreCog/data/raw/synthetic_social_media.csv", # Let it use synthetic
        news_articles_path="/Users/chandan/Documents/Elykid Private Limited/Products/PreCog/data/raw/synthetic_news_articles.csv" # Let it use synthetic
    )
    social_df = data.get('social_media', pd.read_csv("/Users/chandan/Documents/Elykid Private Limited/Products/PreCog/data/raw/synthetic_social_media.csv"))
    news_df = data.get('news_articles', pd.read_csv("/Users/chandan/Documents/Elykid Private Limited/Products/PreCog/data/raw/synthetic_news_articles.csv"))

    print("Running analysis for dashboard...")
    # analyze_current_situation expects two positional DataFrame arguments
    # The LLM flags are read from system.config_data internally by analyze_current_situation
    analysis_output_dict = system.analyze_current_situation(social_df, news_df)

    # Extract results from the dictionary returned by analyze_current_situation
    # The keys in analysis_output_dict might be 'sentiment_alerts', 'misinformation_alerts', etc.
    # or they might be nested under other keys like 'anomaly_alerts', 'friction_risks'.
    # Based on system.py, analyze_current_situation returns a dict where keys are like 'sentiment_analysis', 'misinformation_alerts', etc.
    # The dashboard expects specific alert objects.
    # We need to map the output of analyze_current_situation to what the dashboard expects.
    # For now, let's assume the dashboard's original expectation of direct alert lists is what we need to reconstruct.
    # However, analyze_current_situation returns a dictionary of results, not a tuple of lists.

    # Let's adjust to what analyze_current_situation actually returns and what the dashboard needs.
    # The dashboard expects: sentiment_alerts, misinformation_alerts, friction_risks, intervention_recommendations
    # analyze_current_situation returns a dict like:
    # { 'sentiment_analysis': DataFrame, 'anomaly_alerts': List[SentimentAlert], 'friction_risks': List[FrictionRisk], 
    #   'misinformation_alerts': List[MisinformationAlert], 'intervention_recommendations': List[InterventionAction], ... }

    sentiment_alerts = analysis_output_dict.get('anomaly_alerts', []) # anomaly_alerts are based on sentiment
    misinformation_alerts = analysis_output_dict.get('misinformation_alerts', [])
    friction_risks = analysis_output_dict.get('friction_risks', [])
    intervention_recommendations = analysis_output_dict.get('intervention_recommendations', [])

    # If sentiment_alerts should come from 'sentiment_analysis' DataFrame, further processing is needed here.
    # For now, using 'anomaly_alerts' as a proxy for sentiment-related alerts.

    sentiment_alerts, misinformation_alerts, friction_risks, intervention_recommendations = analysis_output_dict.get('anomaly_alerts', []), analysis_output_dict.get('misinformation_alerts', []), analysis_output_dict.get('friction_risks', []), analysis_output_dict.get('intervention_recommendations', [])
    print(f"Dashboard: Sentiment Alerts: {len(sentiment_alerts)}, Misinfo: {len(misinformation_alerts)}, Friction: {len(friction_risks)}")


    # --- Key Metrics Display --- # 
    st.header("Key Metrics Overview")
    col1, col2, col3 = st.columns(3)
    col1.metric("Sentiment Alerts", len(sentiment_alerts) if sentiment_alerts else 0, help="Number of active high-severity sentiment alerts.")
    col2.metric("Misinformation Threats", len(misinformation_alerts) if misinformation_alerts else 0, help="Number of detected misinformation campaigns/narratives.")
    col3.metric("Friction Risk Zones", len(friction_risks) if friction_risks else 0, help="Number of locations identified with high friction risk.")

    st.markdown("---_More sections to be added: Map Visualization, Detailed Alerts, Recommendations_---")

    # --- Real-time Map Visualization --- #
    st.header("Geospatial Hotspot Map")
    map_data_points = []
    if sentiment_alerts:
        for alert in sentiment_alerts:
            if alert.location_lat is not None and alert.location_lon is not None:
                map_data_points.append({'lat': alert.location_lat, 'lon': alert.location_lon, 'size': 10, 'color': [255, 0, 0, 160]}) # Red for sentiment
    
    if misinformation_alerts:
        for alert in misinformation_alerts:
            # Using origin_location_lat/lon for misinformation alerts if available
            if alert.origin_location_lat is not None and alert.origin_location_lon is not None:
                map_data_points.append({'lat': alert.origin_location_lat, 'lon': alert.origin_location_lon, 'size': 15, 'color': [0, 0, 255, 160]}) # Blue for misinfo

    if friction_risks:
        for risk in friction_risks:
            if risk.location_lat is not None and risk.location_lon is not None:
                map_data_points.append({'lat': risk.location_lat, 'lon': risk.location_lon, 'size': risk.risk_level * 20, 'color': [255, 165, 0, 160]}) # Orange for friction

    if map_data_points:
        map_df = pd.DataFrame(map_data_points)
        # Ensure 'size' and 'color' columns exist if using them with st.map, or adjust st.map call
        # st.map(map_df, latitude='lat', longitude='lon', size='size', color='color') # This is for st.pydeck_chart or similar
        st.map(map_df[['lat', 'lon']]) # Basic st.map only takes lat/lon
        st.caption("Red: Sentiment Alerts, Blue: Misinformation Origins, Orange: Friction Risks (size may vary by risk level if using advanced map)")
    else:
        st.info("No geolocated data available for map display. Ensure latitude/longitude are populated in alerts.")

    # --- Detailed Alerts --- #
    st.header("Alerts & Risks Details")

    if sentiment_alerts:
        st.subheader("Sentiment Alerts")
        sa_data = [{'Timestamp': alert.timestamp, 'Location': alert.location_name, 'Severity': alert.severity, 'Message': alert.message, 'Keywords': ", ".join(alert.contributing_factors)} for alert in sentiment_alerts]
        st.dataframe(pd.DataFrame(sa_data))
    else:
        st.info("No active sentiment alerts.")

    if misinformation_alerts:
        st.subheader("Misinformation Alerts")
        ma_data = [{'Timestamp': alert.timestamp, 'Narrative': alert.narrative, 'Severity': alert.severity, 'Source': alert.source_type, 'Affected Locations': ", ".join(alert.affected_locations), 'Counter Narrative': alert.counter_narrative_suggestion, 'Origin': alert.origin_location_name} for alert in misinformation_alerts]
        st.dataframe(pd.DataFrame(ma_data))
    else:
        st.info("No active misinformation alerts.")

    if friction_risks:
        st.subheader("Friction Risks")
        fr_data = [{'Location': risk.location_name, 'Risk Level': f"{risk.risk_level:.2f}", 'Primary Factor': risk.primary_contributing_factor, 'Details': risk.explanation} for risk in friction_risks]
        st.dataframe(pd.DataFrame(fr_data))
    else:
        st.info("No active friction risks.")
    
    # --- Intervention Recommendations --- # 
    st.header("Intervention Recommendations")
    if intervention_recommendations:
        st.subheader("Suggested Actions")
        # Assuming intervention_recommendations is a list of InterventionAction objects or strings
        ir_display_data = []
        for action in intervention_recommendations:
            if hasattr(action, 'priority') and hasattr(action, 'action_description'): # Check if it's an object with expected attributes
                ir_display_data.append({
                    'Priority': action.priority,
                    'Action': action.action_description,
                    'Target Area': getattr(action, 'target_area', 'N/A'),
                    'Rationale': getattr(action, 'rationale', 'N/A'),
                    'Resources': ", ".join(getattr(action, 'required_resources', []))
                })
            elif isinstance(action, str):
                ir_display_data.append({'Priority': 'N/A', 'Action': action, 'Target Area': 'N/A', 'Rationale': 'N/A', 'Resources': 'N/A'})
            else:
                ir_display_data.append({'Priority': 'N/A', 'Action': 'Unknown recommendation format', 'Target Area': 'N/A', 'Rationale': 'N/A', 'Resources': 'N/A'})
        if ir_display_data:
            st.dataframe(pd.DataFrame(ir_display_data))
        else:
            st.info("No specific intervention recommendations at this time or recommendations are not in the expected format.")
    else:
        st.info("No specific intervention recommendations at this time.")

if __name__ == "__main__":
    main()