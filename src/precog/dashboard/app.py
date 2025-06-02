import streamlit as st
import pandas as pd
import numpy as np
import sys
import os
import plotly.express as px
import plotly.graph_objects as go
import altair as alt
from datetime import datetime, timedelta
import folium
from streamlit_folium import folium_static
import json

# Add project root to Python path to allow direct imports
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Import required modules
from src.precog.core.system import HyperlocalIntelligenceSystem
from src.precog.config.config import get_config

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
    st.title("🧠 PreCog: Hyperlocal Intelligence Dashboard")
    st.markdown("### Illuminating Paths to Social Harmony – A Game-Changing Hyperlocal Intelligence System")

    # --- Sidebar Controls --- #
    st.sidebar.image("https://img.icons8.com/color/96/000000/artificial-intelligence.png", width=80)
    st.sidebar.title("Dashboard Controls")

    # Data source selection
    data_source = st.sidebar.radio(
        "Select Data Source",
        ["Synthetic Data", "Upload Custom Data", "Use Existing Data"]
    )

    # LLM usage toggles
    st.sidebar.markdown("### AI Enhancement Settings")
    use_llm_sentiment = st.sidebar.toggle("Use LLM for Sentiment Analysis", value=False)
    use_llm_narratives = st.sidebar.toggle("Use LLM for Narrative Detection", value=False)
    use_llm_counter = st.sidebar.toggle("Use LLM for Counter-Narratives", value=False)
    use_llm_interventions = st.sidebar.toggle("Use LLM for Intervention Suggestions", value=False)

    # Time period selection
    st.sidebar.markdown("### Time Period")
    time_period = st.sidebar.selectbox(
        "Analysis Time Period",
        ["Last 24 Hours", "Last 7 Days", "Last 30 Days", "Custom Range"]
    )

    if time_period == "Custom Range":
        start_date = st.sidebar.date_input("Start Date", datetime.now() - timedelta(days=7))
        end_date = st.sidebar.date_input("End Date", datetime.now())

    # Refresh button
    if st.sidebar.button("🔄 Refresh Data & Analysis", use_container_width=True):
        st.cache_resource.clear()
        st.rerun()

    # Load the system with progress indicator
    with st.spinner("Loading PreCog Intelligence System..."):
        system = load_system()

    # Configure system based on sidebar settings
    config = get_config()
    config['USE_LLM_FOR_SENTIMENT'] = use_llm_sentiment
    config['USE_LLM_FOR_NARRATIVES'] = use_llm_narratives
    config['USE_LLM_FOR_COUNTER_NARRATIVES'] = use_llm_counter
    config['USE_LLM_FOR_INTERVENTIONS'] = use_llm_interventions

    # Data loading based on selection
    with st.spinner("Loading and analyzing data..."):
        if data_source == "Upload Custom Data":
            social_media_file = st.sidebar.file_uploader("Upload Social Media Data (CSV)", type="csv")
            news_articles_file = st.sidebar.file_uploader("Upload News Articles Data (CSV)", type="csv")

            if social_media_file and news_articles_file:
                social_df = pd.read_csv(social_media_file)
                news_df = pd.read_csv(news_articles_file)
            else:
                st.warning("Please upload both data files or select another data source.")
                return
        else:
            # Use synthetic or existing data
            data = system.load_and_prepare_data(
                fetch_live=False,
                generate_synthetic=(data_source == "Synthetic Data"),
                social_media_path="data/raw/synthetic_social_media.csv",
                news_articles_path="data/raw/synthetic_news_articles.csv"
            )
            social_df = data.get('social_media_feed', pd.DataFrame())
            news_df = data.get('news_articles', pd.DataFrame())

        # Run analysis
        analysis_output_dict = system.analyze_current_situation(social_df, news_df)

    # Extract results
    sentiment_alerts = analysis_output_dict.get('anomaly_alerts', [])
    misinformation_alerts = analysis_output_dict.get('misinformation_alerts', [])
    friction_risks = analysis_output_dict.get('friction_risks', [])
    intervention_recommendations = analysis_output_dict.get('intervention_recommendations', [])
    sentiment_df = analysis_output_dict.get('sentiment_analysis', pd.DataFrame())

    # --- Dashboard Tabs --- #
    tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Sentiment Analysis", "Misinformation", "Interventions"])

    with tab1:
        # --- Key Metrics Display --- #
        st.header("Key Metrics Overview")

        # Create metrics with delta indicators
        col1, col2, col3, col4 = st.columns(4)

        # Sample delta values (in a real app, these would be calculated from historical data)
        sentiment_delta = 2 if len(sentiment_alerts) > 3 else -1
        misinfo_delta = 1 if len(misinformation_alerts) > 2 else -2
        friction_delta = 3 if len(friction_risks) > 2 else 0

        col1.metric(
            "Sentiment Alerts", 
            len(sentiment_alerts) if sentiment_alerts else 0,
            delta=sentiment_delta,
            delta_color="inverse",
            help="Number of active high-severity sentiment alerts."
        )
        col2.metric(
            "Misinformation Threats", 
            len(misinformation_alerts) if misinformation_alerts else 0,
            delta=misinfo_delta,
            delta_color="inverse",
            help="Number of detected misinformation campaigns/narratives."
        )
        col3.metric(
            "Friction Risk Zones", 
            len(friction_risks) if friction_risks else 0,
            delta=friction_delta,
            delta_color="inverse",
            help="Number of locations identified with high friction risk."
        )

        # Calculate overall risk score (example algorithm)
        risk_score = (len(sentiment_alerts) * 2 + len(misinformation_alerts) * 3 + len(friction_risks) * 4) / 10
        risk_score = min(10, max(1, risk_score))  # Scale between 1-10

        col4.metric(
            "Overall Risk Score",
            f"{risk_score:.1f}/10",
            help="Composite risk score based on all alerts and risks."
        )

        # --- Interactive Map Visualization --- #
        st.subheader("Geospatial Hotspot Map")

        # Create a Folium map centered on India
        m = folium.Map(location=[20.5937, 78.9629], zoom_start=5, tiles="CartoDB positron")

        # Add markers for different alert types
        if sentiment_alerts:
            sentiment_group = folium.FeatureGroup(name="Sentiment Alerts")
            for alert in sentiment_alerts:
                if alert.location_lat is not None and alert.location_lon is not None:
                    # Create popup content with alert details
                    popup_content = f"""
                    <div style="width:250px">
                        <h4>{alert.location_name}</h4>
                        <p><b>Severity:</b> {alert.severity}</p>
                        <p><b>Message:</b> {alert.message}</p>
                        <p><b>Keywords:</b> {', '.join(alert.contributing_factors[:3])}</p>
                        <p><b>Timestamp:</b> {alert.timestamp}</p>
                    </div>
                    """
                    # Add marker with popup
                    folium.CircleMarker(
                        location=[alert.location_lat, alert.location_lon],
                        radius=8,
                        color='red',
                        fill=True,
                        fill_color='red',
                        fill_opacity=0.7,
                        popup=folium.Popup(popup_content, max_width=300)
                    ).add_to(sentiment_group)
            sentiment_group.add_to(m)

        if misinformation_alerts:
            misinfo_group = folium.FeatureGroup(name="Misinformation Alerts")
            for alert in misinformation_alerts:
                if hasattr(alert, 'origin_location_lat') and hasattr(alert, 'origin_location_lon') and alert.origin_location_lat and alert.origin_location_lon:
                    # Create popup content
                    popup_content = f"""
                    <div style="width:250px">
                        <h4>Misinformation Alert</h4>
                        <p><b>Narrative:</b> {alert.narrative}</p>
                        <p><b>Severity:</b> {alert.severity}</p>
                        <p><b>Source:</b> {alert.source_type}</p>
                        <p><b>Origin:</b> {alert.origin_location_name}</p>
                    </div>
                    """
                    # Add marker
                    folium.CircleMarker(
                        location=[alert.origin_location_lat, alert.origin_location_lon],
                        radius=10,
                        color='blue',
                        fill=True,
                        fill_color='blue',
                        fill_opacity=0.7,
                        popup=folium.Popup(popup_content, max_width=300)
                    ).add_to(misinfo_group)
            misinfo_group.add_to(m)

        if friction_risks:
            friction_group = folium.FeatureGroup(name="Friction Risk Zones")
            for risk in friction_risks:
                if risk.location_lat is not None and risk.location_lon is not None:
                    # Scale radius by risk level
                    radius = 5 + (risk.risk_level * 10)

                    # Create popup content
                    popup_content = f"""
                    <div style="width:250px">
                        <h4>{risk.location_name}</h4>
                        <p><b>Risk Level:</b> {risk.risk_level:.2f}</p>
                        <p><b>Primary Factor:</b> {risk.primary_contributing_factor}</p>
                        <p><b>Details:</b> {risk.explanation[:100]}...</p>
                    </div>
                    """
                    # Add marker
                    folium.CircleMarker(
                        location=[risk.location_lat, risk.location_lon],
                        radius=radius,
                        color='orange',
                        fill=True,
                        fill_color='orange',
                        fill_opacity=0.7,
                        popup=folium.Popup(popup_content, max_width=300)
                    ).add_to(friction_group)
            friction_group.add_to(m)

        # Add layer control to toggle different alert types
        folium.LayerControl().add_to(m)

        # Display the map
        if sentiment_alerts or misinformation_alerts or friction_risks:
            folium_static(m)
            st.caption("Interactive map showing sentiment alerts (red), misinformation origins (blue), and friction risk zones (orange)")
        else:
            st.info("No geolocated data available for map display. Ensure latitude/longitude are populated in alerts.")

        # --- Trend Analysis --- #
        st.subheader("Trend Analysis")

        # Create sample trend data (in a real app, this would come from historical data)
        if not sentiment_df.empty and 'timestamp' in sentiment_df.columns:
            # Group by date and calculate average sentiment
            sentiment_df['date'] = pd.to_datetime(sentiment_df['timestamp']).dt.date
            daily_sentiment = sentiment_df.groupby('date')['sentiment_score'].mean().reset_index()

            # Create line chart
            fig = px.line(
                daily_sentiment, 
                x='date', 
                y='sentiment_score',
                title='Average Sentiment Score Over Time',
                labels={'sentiment_score': 'Avg. Sentiment Score', 'date': 'Date'},
                markers=True
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        else:
            # Create dummy data for demonstration
            dates = pd.date_range(end=datetime.now(), periods=14).tolist()
            sentiment_values = [-0.2, -0.3, -0.25, -0.1, 0.1, 0.05, -0.15, -0.4, -0.5, -0.3, -0.2, -0.1, 0.0, 0.1]
            misinfo_values = [2, 3, 5, 4, 3, 2, 4, 7, 8, 6, 5, 4, 3, 2]
            friction_values = [1, 1, 2, 2, 3, 2, 2, 4, 5, 3, 2, 2, 1, 1]

            trend_df = pd.DataFrame({
                'date': dates,
                'sentiment': sentiment_values,
                'misinfo': misinfo_values,
                'friction': friction_values
            })

            # Create multi-line chart
            fig = px.line(
                trend_df, 
                x='date', 
                y=['sentiment', 'misinfo', 'friction'],
                title='Trend Analysis (Sample Data)',
                labels={'value': 'Score/Count', 'date': 'Date', 'variable': 'Metric'},
                color_discrete_map={
                    'sentiment': 'red',
                    'misinfo': 'blue',
                    'friction': 'orange'
                }
            )
            fig.update_layout(height=400, legend_title_text='Metric')
            st.plotly_chart(fig, use_container_width=True)
            st.caption("Sample trend data for demonstration purposes. In a production environment, this would show actual historical data.")

    # Content for the Sentiment Analysis tab
    with tab2:
        st.header("Sentiment Analysis Dashboard")

        # Sentiment distribution chart
        st.subheader("Sentiment Distribution")

        if not sentiment_df.empty and 'sentiment_score' in sentiment_df.columns:
            # Create histogram of sentiment scores
            fig = px.histogram(
                sentiment_df, 
                x='sentiment_score',
                nbins=20,
                color_discrete_sequence=['#3366CC'],
                title='Distribution of Sentiment Scores',
                labels={'sentiment_score': 'Sentiment Score', 'count': 'Number of Posts'}
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

            # Sentiment by category if available
            if 'category' in sentiment_df.columns:
                st.subheader("Sentiment by Category")
                category_sentiment = sentiment_df.groupby('category')['sentiment_score'].mean().reset_index()
                category_counts = sentiment_df['category'].value_counts().reset_index()
                category_counts.columns = ['category', 'count']
                category_data = pd.merge(category_sentiment, category_counts, on='category')

                fig = px.bar(
                    category_data,
                    x='category',
                    y='sentiment_score',
                    color='sentiment_score',
                    color_continuous_scale='RdBu',
                    title='Average Sentiment by Category',
                    labels={'sentiment_score': 'Avg. Sentiment', 'category': 'Category', 'count': 'Number of Posts'}
                )
                fig.update_layout(height=500)
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No sentiment data available for visualization.")

        # Sentiment alerts table with filtering
        st.subheader("Sentiment Alerts")
        if sentiment_alerts:
            sa_data = [{'Timestamp': alert.timestamp, 'Location': alert.location_name, 'Severity': alert.severity, 'Message': alert.message, 'Keywords': ", ".join(alert.contributing_factors)} for alert in sentiment_alerts]
            sa_df = pd.DataFrame(sa_data)

            # Add filtering options
            col1, col2 = st.columns(2)
            with col1:
                if 'Severity' in sa_df.columns:
                    severity_filter = st.multiselect("Filter by Severity", options=sorted(sa_df['Severity'].unique()), default=sorted(sa_df['Severity'].unique()))
                else:
                    severity_filter = []

            with col2:
                if 'Location' in sa_df.columns:
                    location_filter = st.multiselect("Filter by Location", options=sorted(sa_df['Location'].unique()), default=sorted(sa_df['Location'].unique()))
                else:
                    location_filter = []

            # Apply filters
            filtered_df = sa_df
            if severity_filter:
                filtered_df = filtered_df[filtered_df['Severity'].isin(severity_filter)]
            if location_filter:
                filtered_df = filtered_df[filtered_df['Location'].isin(location_filter)]

            # Display filtered dataframe with expanded height
            st.dataframe(filtered_df, height=400, use_container_width=True)
        else:
            st.info("No active sentiment alerts.")

    # Content for the Misinformation tab
    with tab3:
        st.header("Misinformation Analysis")

        # Misinformation metrics
        if misinformation_alerts:
            # Summary metrics
            col1, col2, col3 = st.columns(3)

            # Count alerts by severity
            severity_counts = {}
            for alert in misinformation_alerts:
                severity_counts[alert.severity] = severity_counts.get(alert.severity, 0) + 1

            high_count = severity_counts.get('high', 0)
            medium_count = severity_counts.get('medium', 0)
            low_count = severity_counts.get('low', 0)

            col1.metric("High Severity", high_count, help="Number of high severity misinformation alerts")
            col2.metric("Medium Severity", medium_count, help="Number of medium severity misinformation alerts")
            col3.metric("Low Severity", low_count, help="Number of low severity misinformation alerts")

            # Narrative spread visualization
            st.subheader("Narrative Spread Analysis")

            # Create data for visualization
            narrative_data = []
            for alert in misinformation_alerts:
                narrative_data.append({
                    'Narrative': alert.narrative[:50] + "..." if len(alert.narrative) > 50 else alert.narrative,
                    'Spread Velocity': getattr(alert, 'spread_velocity', 0),
                    'Severity': alert.severity,
                    'Confidence': getattr(alert, 'confidence', 0.5)
                })

            narrative_df = pd.DataFrame(narrative_data)

            # Create bubble chart
            if not narrative_df.empty:
                fig = px.scatter(
                    narrative_df,
                    x='Spread Velocity',
                    y='Confidence',
                    size='Spread Velocity',
                    color='Severity',
                    hover_name='Narrative',
                    title='Misinformation Narratives by Spread and Confidence',
                    color_discrete_map={'high': 'red', 'medium': 'orange', 'low': 'yellow'},
                    size_max=50
                )
                fig.update_layout(height=500)
                st.plotly_chart(fig, use_container_width=True)

            # Detailed misinformation alerts
            st.subheader("Misinformation Alerts")
            ma_data = []
            for alert in misinformation_alerts:
                ma_data.append({
                    'Timestamp': alert.timestamp,
                    'Narrative': alert.narrative,
                    'Severity': alert.severity,
                    'Source': alert.source_type,
                    'Affected Locations': ", ".join(alert.affected_locations),
                    'Counter Narrative': alert.counter_narrative_suggestion,
                    'Origin': alert.origin_location_name
                })

            # Create expandable sections for each alert
            for i, alert_data in enumerate(ma_data):
                with st.expander(f"Alert {i+1}: {alert_data['Narrative'][:50]}... ({alert_data['Severity'].upper()})"):
                    st.markdown(f"**Source:** {alert_data['Source']}")
                    st.markdown(f"**Origin:** {alert_data['Origin']}")
                    st.markdown(f"**Affected Locations:** {alert_data['Affected Locations']}")
                    st.markdown(f"**Timestamp:** {alert_data['Timestamp']}")
                    st.markdown("### Counter Narrative")
                    st.info(alert_data['Counter Narrative'])
        else:
            st.info("No active misinformation alerts.")

    # Content for the Interventions tab
    with tab4:
        st.header("Intervention Recommendations")

        if intervention_recommendations:
            # Organize interventions by priority
            high_priority = []
            medium_priority = []
            low_priority = []

            for action in intervention_recommendations:
                if hasattr(action, 'priority'):
                    if action.priority == 'high':
                        high_priority.append(action)
                    elif action.priority == 'medium':
                        medium_priority.append(action)
                    else:
                        low_priority.append(action)
                elif isinstance(action, str):
                    low_priority.append(action)

            # Display high priority interventions
            if high_priority:
                st.subheader("🔴 High Priority Actions")
                for i, action in enumerate(high_priority):
                    with st.container():
                        st.markdown(f"### Action {i+1}: {action.action_description}")
                        cols = st.columns(3)
                        cols[0].markdown(f"**Target Area:** {getattr(action, 'target_area', 'N/A')}")
                        cols[1].markdown(f"**Timeline:** {getattr(action, 'timeline', 'Immediate')}")
                        cols[2].markdown(f"**Resources:** {', '.join(getattr(action, 'required_resources', []))}")
                        st.markdown(f"**Rationale:** {getattr(action, 'rationale', 'N/A')}")
                        st.markdown("---")

            # Display medium priority interventions
            if medium_priority:
                st.subheader("🟠 Medium Priority Actions")
                for i, action in enumerate(medium_priority):
                    with st.container():
                        st.markdown(f"### Action {i+1}: {action.action_description}")
                        cols = st.columns(3)
                        cols[0].markdown(f"**Target Area:** {getattr(action, 'target_area', 'N/A')}")
                        cols[1].markdown(f"**Timeline:** {getattr(action, 'timeline', 'Short-term')}")
                        cols[2].markdown(f"**Resources:** {', '.join(getattr(action, 'required_resources', []))}")
                        st.markdown(f"**Rationale:** {getattr(action, 'rationale', 'N/A')}")
                        st.markdown("---")

            # Display low priority interventions
            if low_priority:
                st.subheader("🟡 Low Priority Actions")
                for i, action in enumerate(low_priority):
                    if isinstance(action, str):
                        st.markdown(f"- {action}")
                    else:
                        with st.container():
                            st.markdown(f"### Action {i+1}: {action.action_description}")
                            cols = st.columns(3)
                            cols[0].markdown(f"**Target Area:** {getattr(action, 'target_area', 'N/A')}")
                            cols[1].markdown(f"**Timeline:** {getattr(action, 'timeline', 'Long-term')}")
                            cols[2].markdown(f"**Resources:** {', '.join(getattr(action, 'required_resources', []))}")
                            st.markdown(f"**Rationale:** {getattr(action, 'rationale', 'N/A')}")
                            st.markdown("---")

            # Intervention effectiveness tracking (placeholder)
            st.subheader("Intervention Effectiveness Tracking")
            st.markdown("This section would track the implementation and effectiveness of past interventions.")

            # Create sample data for demonstration
            effectiveness_data = {
                'Intervention': ['Community Outreach', 'Media Campaign', 'Leader Engagement', 'Resource Allocation', 'Educational Program'],
                'Implementation': [0.9, 0.7, 0.5, 0.8, 0.3],
                'Effectiveness': [0.8, 0.6, 0.7, 0.5, 0.4],
                'Status': ['Completed', 'In Progress', 'In Progress', 'Completed', 'Planning']
            }

            effectiveness_df = pd.DataFrame(effectiveness_data)

            # Create radar chart for intervention effectiveness
            fig = go.Figure()

            fig.add_trace(go.Scatterpolar(
                r=effectiveness_df['Implementation'],
                theta=effectiveness_df['Intervention'],
                fill='toself',
                name='Implementation'
            ))

            fig.add_trace(go.Scatterpolar(
                r=effectiveness_df['Effectiveness'],
                theta=effectiveness_df['Intervention'],
                fill='toself',
                name='Effectiveness'
            ))

            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 1]
                    )
                ),
                showlegend=True,
                title="Intervention Implementation vs. Effectiveness"
            )

            st.plotly_chart(fig, use_container_width=True)
            st.caption("Sample intervention tracking data for demonstration purposes.")
        else:
            st.info("No intervention recommendations available at this time.")

if __name__ == "__main__":
    main()
