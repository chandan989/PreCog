import os
import pandas as pd
import numpy as np
from datetime import datetime
import time
import logging
from typing import Dict, Any, Optional, List
from collections import Counter

# Precog components - Adjust import paths based on final structure
from ..core.data_models import SentimentAlert, FrictionRisk, MisinformationAlert
from ..core.synthetic_data_generator import SyntheticDataGenerator
from ..analysis.sentiment_analyzer import HyperlocalSentimentAnalyzer
from ..analysis.anomaly_detector import SentimentAnomalyDetector
from ..analysis.friction_predictor import FrictionPointPredictor
from ..analysis.misinformation_detector import MisinformationDetector
from ..interventions.intervention_recommender import InterventionRecommender
from ..config.config import get_config
from .llm_clients import get_llm_client # Import the LLM client factory

logger = logging.getLogger(__name__)

class HyperlocalIntelligenceSystem:
    def __init__(self, config_path: Optional[str] = None, data_dir: Optional[str] = None, cli_config_overrides: Optional[Dict[str, Any]] = None):
        self.config_data = get_config() # Load default config
        if config_path:
            # Logic to load and merge custom config from path (e.g., JSON, YAML)
            # For simplicity, we'll assume get_config() handles environment variables or has fixed defaults
            pass
        if cli_config_overrides:
            self.config_data.update(cli_config_overrides) # Apply CLI overrides 

        # Override data_dir if provided
        self.data_dir = data_dir if data_dir else self.config_data.get('DEFAULT_DATA_DIR', 'data')
        self.models_dir = os.path.join(self.data_dir, 'models')
        self.reports_dir = os.path.join(self.data_dir, 'reports')
        self.raw_data_dir = os.path.join(self.data_dir, 'raw') # For loading data
        self.processed_data_dir = os.path.join(self.data_dir, 'processed') # For saving processed data

        # Create data_dirs dictionary for compatibility with methods that expect it
        self.data_dirs = {
            'models': self.models_dir,
            'reports': self.reports_dir,
            'raw': self.raw_data_dir,
            'processed': self.processed_data_dir
        }

        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.reports_dir, exist_ok=True)
        os.makedirs(self.raw_data_dir, exist_ok=True)
        os.makedirs(self.processed_data_dir, exist_ok=True)

        # Initialize LLM Client (can be None if not configured or SDKs missing)
        # You might want to add a config option to specify 'azure' or 'vertex' preference
        preferred_llm = self.config_data.get('PREFERRED_LLM_PROVIDER', 'azure') # Default to Azure
        self.llm_client = get_llm_client(self.config_data, preferred_client=preferred_llm)
        if self.llm_client:
            logger.info(f"Successfully initialized LLM client: {type(self.llm_client).__name__}")
        else:
            logger.warning("LLM client could not be initialized. LLM-specific features will be disabled.")

        # Initialize AI components, passing the LLM client
        self.synthetic_data_generator = SyntheticDataGenerator(seed=self.config_data.get('RANDOM_SEED'))
        self.sentiment_analyzer = HyperlocalSentimentAnalyzer(
            lang=self.config_data.get('SENTIMENT_ANALYZER_LANGUAGE', 'en_hinglish'),
            llm_client=self.llm_client
        )
        self.anomaly_detector = SentimentAnomalyDetector(
            contamination=self.config_data.get('ANOMALY_DETECTOR_CONTAMINATION', 0.05)
        )
        self.friction_predictor = FrictionPointPredictor(
            model_path=os.path.join(self.models_dir, self.config_data.get('FRICTION_MODEL_PATH'))
        )
        self.misinformation_detector = MisinformationDetector(
            config=self.config_data,
            llm_client=self.llm_client
        )
        self.intervention_recommender = InterventionRecommender(llm_client=self.llm_client)

        self.system_metrics = {
            'alert_counts': [],
            'friction_risk': [],
            'misinformation': []
        }

        # Train models during initialization
        data = self.load_and_prepare_data(generate_synthetic=True)
        self.train_models(data)

    def _ensure_data_dirs_exist(self):
        for _, dir_path in self.data_dirs.items():
            os.makedirs(dir_path, exist_ok=True)

    def load_and_prepare_data(
        self, 
        fetch_live: bool = False, 
        generate_synthetic: bool = False, # Default changed as per user's intent
        social_media_path: Optional[str] = None,
        news_articles_path: Optional[str] = None,
        force_no_synthetic: bool = False # New parameter added
    ) -> Dict[str, pd.DataFrame]:
        """Loads data from files or generates synthetic data."""
        datasets = {}

        if fetch_live:
            logger.info("Live data fetching not implemented yet. Using synthetic/local files.")
            # Placeholder for live data ingestion (e.g., API calls)
            # For now, it will fall back to synthetic or provided files.

        # Load social media data
        if social_media_path and os.path.exists(social_media_path):
            try:
                datasets['social_media_feed'] = pd.read_csv(social_media_path)
                logger.info(f"Loaded social media data from {social_media_path}")
            except Exception as e:
                logger.error(f"Error loading social media data from {social_media_path}: {e}")
                if generate_synthetic and not force_no_synthetic:
                    logger.info("Generating synthetic social media data as fallback.")
                    datasets['social_media_feed'] = self.synthetic_data_generator.generate_social_media_data()
                else:
                    datasets['social_media_feed'] = pd.DataFrame() # Empty if path error and no synthetic fallback
        elif generate_synthetic and not force_no_synthetic:
            datasets['social_media_feed'] = self.synthetic_data_generator.generate_social_media_data()
            logger.info("Generated synthetic social media data.")
        else:
            datasets['social_media_feed'] = pd.DataFrame() # Empty if no path and no synthetic

        # Load news articles data
        if news_articles_path and os.path.exists(news_articles_path):
            try:
                datasets['news_articles'] = pd.read_csv(news_articles_path)
                logger.info(f"Loaded news articles data from {news_articles_path}")
            except Exception as e:
                logger.error(f"Error loading news articles data from {news_articles_path}: {e}")
                if generate_synthetic and not force_no_synthetic:
                    logger.info("Generating synthetic news articles data as fallback.")
                    datasets['news_articles'] = self.synthetic_data_generator.generate_news_articles_data()
                else:
                    datasets['news_articles'] = pd.DataFrame() # Empty if path error and no synthetic fallback
        elif generate_synthetic and not force_no_synthetic:
            datasets['news_articles'] = self.synthetic_data_generator.generate_news_articles_data()
            logger.info("Generated synthetic news articles data.")
        else:
            datasets['news_articles'] = pd.DataFrame() # Empty

        # Basic preprocessing (example)
        for name, df in datasets.items():
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
            if 'publish_date' in df.columns:
                df['publish_date'] = pd.to_datetime(df['publish_date'], errors='coerce')

        # Save raw loaded/generated data (optional)
        # for name, df in datasets.items():
        #     if not df.empty:
        #         df.to_csv(os.path.join(self.data_dirs['raw'], f"raw_{name}_{datetime.now():%Y%m%d%H%M%S}.csv"), index=False)

        return datasets

    def train_models(self, data: Dict[str, pd.DataFrame]):
        """Train all relevant AI models with the provided data."""
        logger.info("Starting model training process...")
        social_data = data.get('social_media_feed', pd.DataFrame())
        # news_data = data.get('news_articles', pd.DataFrame())

        if not social_data.empty:
            # Fit anomaly detector baseline
            # Requires 'sentiment_score' and 'misinformation_likelihood' - these might need to be pre-calculated
            # For initial training, we might use dummy values or a preliminary sentiment pass.
            if 'sentiment_score' not in social_data.columns:
                 social_data['sentiment_score'] = np.random.uniform(-1, 1, len(social_data))
            if 'misinformation_likelihood' not in social_data.columns:
                 social_data['misinformation_likelihood'] = np.random.uniform(0, 1, len(social_data))
            self.anomaly_detector.fit_baseline(social_data)
            logger.info("Sentiment Anomaly Detector baseline fitted.")

            # Train friction predictor
            # This expects aggregated data per location with specific features and a 'friction_risk_label'
            # For now, we use its internal synthetic data generation for training if no specific training data is passed.
            self.friction_predictor.train_model() # Uses synthetic data by default
            logger.info("Friction Point Predictor model trained (using synthetic data by default).")
        else:
            logger.warning("Social media data is empty. Some models may not be trained effectively.")

        # Other models (sentiment, misinfo detector) are rule-based or don't require explicit batch training in this version.
        logger.info("Model training process completed.")

    def analyze_current_situation(self, social_media_df: pd.DataFrame, news_articles_df: pd.DataFrame) -> Dict[str, Any]:
        logger.info("Starting current situation analysis...")
        analysis_results = {'timestamp': pd.Timestamp.now()}

        # --- Sentiment Analysis ---
        logger.info("Performing sentiment analysis...")
        # Add a config flag to use LLM for sentiment
        use_llm_sentiment = self.config_data.get('USE_LLM_FOR_SENTIMENT', False)
        if not social_media_df.empty and 'text' in social_media_df.columns:
            sentiment_results = []
            for index, row in social_media_df.iterrows():
                # Pass the use_llm flag to the analyzer
                analysis = self.sentiment_analyzer.analyze_single_text(row['text'], use_llm=use_llm_sentiment)
                sentiment_results.append({
                    'id': row.get('id', index),
                    'text': row['text'],
                    'location': row.get('location_approx'),  # Use location_approx instead of location
                    'timestamp': row.get('timestamp'),
                    'sentiment_score': analysis['sentiment_score'],
                    'sentiment_label': analysis['sentiment_label'],
                    'category': analysis.get('category', 'unknown'), # from LLM or rule-based
                    'keywords': analysis.get('keywords', [])
                })
            sentiment_df = pd.DataFrame(sentiment_results)
            analysis_results['sentiment_analysis'] = sentiment_df
            # Add sentiment scores to the main social media dataframe for other components
            if not sentiment_df.empty:
                 social_media_df = social_media_df.merge(sentiment_df[['id', 'sentiment_score', 'sentiment_label', 'category']], on='id', how='left')
            logger.info(f"Sentiment analysis completed. Analyzed {len(sentiment_df)} posts.")
        else:
            logger.warning("Social media data is empty or 'text' column is missing. Skipping sentiment analysis.")
            analysis_results['sentiment_analysis'] = pd.DataFrame()
            # Ensure social_media_df has expected columns even if empty, for downstream components
            if 'sentiment_score' not in social_media_df.columns: social_media_df['sentiment_score'] = 0.0
            if 'category' not in social_media_df.columns: social_media_df['category'] = 'unknown'

        # --- Anomaly Detection ---
        if not social_media_df.empty:
            try:
                anomaly_alerts = self.anomaly_detector.detect_anomalies(social_media_df)
                analysis_results['anomaly_alerts'] = anomaly_alerts
                logger.info(f"Detected {len(anomaly_alerts)} sentiment anomalies.")
            except Exception as e:
                logger.error(f"Error in anomaly detection: {e}")

        # Friction Prediction
        # Requires data aggregated by location. For now, we'll use a placeholder or assume social_media_df can be used directly if small.
        # A proper implementation would aggregate features from social_media_df per location.
        if not social_media_df.empty:
            # Simplistic: use unique locations from social_media_df to create dummy aggregated data
            # In a real scenario, you'd aggregate features like avg_sentiment, grievance_counts etc. per location.
            if 'location_approx' in social_media_df.columns:
                unique_locations_df = social_media_df.groupby('location_approx').agg(
                    avg_sentiment=('sentiment_score', 'mean'),
                    # Add other aggregations needed for friction_predictor._prepare_features
                ).reset_index().set_index('location_approx')

                if not unique_locations_df.empty:
                    try:
                        friction_risks = self.friction_predictor.predict_friction_risk(unique_locations_df)
                        analysis_results['friction_risks'] = friction_risks
                        logger.info(f"Identified {len(friction_risks)} friction risks.")
                    except Exception as e:
                        logger.error(f"Error in friction prediction: {e}")
                else:
                    logger.warning("No location data to aggregate for friction prediction.")
            else:
                logger.warning("Missing 'location_approx' in social_media_df for friction prediction.")

        # Misinformation Detection
        if not social_media_df.empty or not news_articles_df.empty:
            try:
                # Add config flags for LLM usage in misinformation detection
                use_llm_narratives = self.config_data.get('USE_LLM_FOR_NARRATIVES', False)
                use_llm_counter_narratives = self.config_data.get('USE_LLM_FOR_COUNTER_NARRATIVES', False)

                # Convert DataFrames to list of dicts as expected by MisinformationDetector
                social_media_list = social_media_df.to_dict(orient='records') if not social_media_df.empty else []
                news_articles_list = news_articles_df.to_dict(orient='records') if not news_articles_df.empty else []

                misinfo_alerts = self.misinformation_detector.detect_misinformation(
                    social_media_list,
                    news_articles_list,
                    use_llm_for_narratives=use_llm_narratives,
                    use_llm_for_counter_narratives=use_llm_counter_narratives
                )
                analysis_results['misinformation_alerts'] = misinfo_alerts
                logger.info(f"Misinformation detection completed. Found {len(misinfo_alerts)} alerts.")
            except Exception as e:
                logger.error(f"Error in misinformation detection: {e}")

        # --- Intervention Recommendations ---
        logger.info("Generating intervention recommendations...")
        # Add a config flag to use LLM for intervention suggestions
        use_llm_interventions = self.config_data.get('USE_LLM_FOR_INTERVENTIONS', False)
        recommendations = self.intervention_recommender.recommend_interventions(
            analysis_results.get('anomaly_alerts', []),
            analysis_results.get('friction_risks', []),
            analysis_results.get('misinformation_alerts', []),
            use_llm_for_suggestions=use_llm_interventions
        )
        analysis_results['intervention_recommendations'] = recommendations

        self._update_system_metrics(analysis_results)
        self.last_analysis_time = analysis_results.get('timestamp', pd.Timestamp.now()) # Use .get for safety
        logger.info("Analysis cycle complete.")
        return analysis_results

    def _update_system_metrics(self, analysis_results: Dict[str, Any]):
        """Update system performance metrics."""
        timestamp = analysis_results['timestamp']

        alert_counts = {'low': 0, 'medium': 0, 'high': 0, 'critical': 0}
        for alert_list_key in ['anomaly_alerts', 'misinformation_alerts']:
            for alert in analysis_results.get(alert_list_key, []):
                if hasattr(alert, 'severity'): # Check if alert has severity
                    alert_counts[alert.severity] = alert_counts.get(alert.severity,0) + 1
        self.system_metrics['alert_counts'].append({'timestamp': timestamp, **alert_counts})

        risk_levels = [risk.risk_level for risk in analysis_results.get('friction_risks', []) if hasattr(risk, 'risk_level')]
        avg_risk = np.mean(risk_levels) if risk_levels else 0
        self.system_metrics['friction_risk'].append({
            'timestamp': timestamp, 
            'avg_risk_score': avg_risk,
            'high_risk_locations': len([r for r in risk_levels if r > 0.6])
        })

        misinfo_alerts_list = analysis_results.get('misinformation_alerts', [])
        misinfo_velocities = [alert.spread_velocity for alert in misinfo_alerts_list if hasattr(alert, 'spread_velocity')]
        avg_velocity = np.mean(misinfo_velocities) if misinfo_velocities else 0
        self.system_metrics['misinformation'].append({
            'timestamp': timestamp, 
            'avg_spread_velocity': avg_velocity,
            'total_alerts': len(misinfo_alerts_list)
        })

    def generate_report(self, analysis_results: Dict[str, Any]) -> str:
        """Generate comprehensive analysis report."""
        report = []
        report.append("=" * 60)
        report.append("HYPERLOCAL INTELLIGENCE ANALYSIS REPORT".center(60))
        report.append("=" * 60)
        report.append(f"Generated: {analysis_results.get('timestamp', datetime.now()).strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("" * 2)

        # Sentiment Analysis Summary
        report.append("--- SENTIMENT ANALYSIS SUMMARY ---")
        sentiment_df = analysis_results.get('sentiment_analysis', pd.DataFrame())
        if not sentiment_df.empty and 'location' in sentiment_df.columns:
            # Group by location and calculate average sentiment
            location_groups = sentiment_df.groupby('location')
            for location, group in list(location_groups)[:5]:  # Top 5 locations
                if location and not pd.isna(location):  # Skip empty or NaN locations
                    avg_sentiment = group['sentiment_score'].mean()
                    post_count = len(group)
                    # Determine alert level based on average sentiment
                    alert_level = 'NORMAL'
                    if avg_sentiment < -0.5:
                        alert_level = 'HIGH'
                    elif avg_sentiment < -0.3:
                        alert_level = 'MEDIUM'
                    elif avg_sentiment < -0.1:
                        alert_level = 'LOW'

                    report.append(f"Location: {location}")
                    report.append(f"  Avg Sentiment: {avg_sentiment:.3f}, Alert Level: {alert_level}, Posts: {post_count}")

                    # Extract keywords if available
                    if 'keywords' in group.columns:
                        all_keywords = []
                        for keywords_list in group['keywords']:
                            if isinstance(keywords_list, list):
                                all_keywords.extend(keywords_list)
                        keyword_counts = Counter(all_keywords)
                        top_keywords = keyword_counts.most_common(3)
                        if top_keywords:
                            top_issue_str = ", ".join([f"{kw[0]} ({kw[1]})" for kw in top_keywords])
                            report.append(f"  Top Issues: {top_issue_str}")
                    report.append("")
        else: report.append("No sentiment data available or location information missing.\n")

        # Anomaly Alerts
        report.append("--- ANOMALY ALERTS ---")
        alerts = analysis_results.get('anomaly_alerts', [])
        if alerts:
            crit_count = len([a for a in alerts if hasattr(a,'severity') and a.severity == 'critical'])
            high_count = len([a for a in alerts if hasattr(a,'severity') and a.severity == 'high'])
            report.append(f"Total: {len(alerts)} (Critical: {crit_count}, High: {high_count})")
            for alert in alerts[:3]: # Show top 3
                report.append(f"  🚨 {getattr(alert,'severity','N/A').upper()} @ {getattr(alert,'location','N/A')}: {getattr(alert,'message','N/A')[:100]}")
            report.append("")
        else: report.append("No anomaly alerts detected.\n")

        # Friction Risks
        report.append("--- FRICTION RISK ASSESSMENT ---")
        risks = analysis_results.get('friction_risks', [])
        if risks:
            high_risks_count = len([r for r in risks if hasattr(r,'risk_level') and r.risk_level > 0.6])
            report.append(f"Total Risk Areas: {len(risks)}, High Risk Areas: {high_risks_count}")
            for risk in sorted(risks, key=lambda x: getattr(x, 'risk_level', 0), reverse=True)[:3]:
                primary_factor = getattr(risk, 'primary_contributing_factor', 'N/A')
                report.append(f"  ⚠️ {getattr(risk,'location_name','N/A')} - Score: {getattr(risk,'risk_level',0):.2f}, Timeline: {getattr(risk,'predicted_timeline','N/A')}, Factor: {primary_factor}")
            report.append("")
        else: report.append("No significant friction risks identified.\n")

        # Misinformation Alerts
        report.append("--- MISINFORMATION MONITORING ---")
        misinfo = analysis_results.get('misinformation_alerts', [])
        if misinfo:
            report.append(f"Active Misinformation Threads: {len(misinfo)}")
            for alert in misinfo[:2]: # Show top 2
                locs = ", ".join(getattr(alert,'affected_locations',[])[:2])
                report.append(f"  📢 Narrative: {getattr(alert,'narrative','N/A')}, Velocity: {getattr(alert,'spread_velocity',0):.2f}, Locations: {locs}, Credibility: {getattr(alert,'credibility_score',0):.2f}")
            report.append("")
        else: report.append("No significant misinformation detected.\n")

        # Recommendations
        report.append("--- INTERVENTION RECOMMENDATIONS ---")
        recs = analysis_results.get('intervention_recommendations', {})
        if recs.get('immediate_actions'):
            report.append(" IMMEDIATE ACTIONS:")
            for action in recs['immediate_actions'][:3]: report.append(f"    • {action}")
        if recs.get('priority_locations'):
            report.append(" PRIORITY LOCATIONS:")
            report.append(f"    • {', '.join(list(recs['priority_locations'])[:5])}")
        if recs.get('resource_allocation'):
            report.append(" RESOURCE NEEDS (Examples):")
            for res_type, tasks in list(recs['resource_allocation'].items())[:2]:
                if tasks: report.append(f"    • {res_type.title()}: {', '.join(list(tasks)[:2])}")
        if not recs or not any(recs.values()): report.append("No specific recommendations generated at this time.")

        report.append("\n" + "=" * 60)
        report.append("End of Report".center(60))
        report.append("=" * 60)
        return "\n".join(report)

    def run_continuous_monitoring(self, interval_minutes: int = 30, run_once: bool = False):
        logger.info(f"Starting continuous monitoring (interval: {interval_minutes} minutes). Run once: {run_once}")
        while True:
            try:
                data = self.load_and_prepare_data(fetch_live=False, generate_synthetic=True) # Modify for live data
                if not any(not df.empty for df in data.values()):
                    logger.warning("No data loaded or generated. Skipping analysis cycle.")
                    if run_once: break
                    time.sleep(interval_minutes * 60)
                    continue

                # Train models (e.g., if they need periodic retraining or first-time training)
                # For now, let's assume models are trained once initially or use pre-trained versions.
                # self.train_models(data) # Potentially heavy for each cycle

                social_df = data.get('social_media_feed', pd.DataFrame())
                news_df = data.get('news_articles', pd.DataFrame())
                results = self.analyze_current_situation(social_df, news_df)
                report_str = self.generate_report(results)

                timestamp_str = results.get('timestamp', datetime.now()).strftime("%Y%m%d_%H%M%S")
                report_path = os.path.join(self.data_dirs['reports'], f"analysis_report_{timestamp_str}.txt")
                with open(report_path, 'w', encoding='utf-8') as f:
                    f.write(report_str)
                logger.info(f"Analysis complete. Report saved to {report_path}")

                critical_alerts = [a for a in results.get('anomaly_alerts', []) if hasattr(a,'severity') and a.severity == 'critical']
                if critical_alerts:
                    logger.warning(f"🚨 CRITICAL ALERTS DETECTED: {len(critical_alerts)}")
                    # Implement notification logic here

                if run_once: break
                logger.info(f"Waiting for {interval_minutes} minutes for the next cycle...")
                time.sleep(interval_minutes * 60)

            except KeyboardInterrupt:
                logger.info("Monitoring stopped by user.")
                break
            except Exception as e:
                logger.error(f"Error in monitoring cycle: {e}", exc_info=True)
                if run_once: break
                time.sleep(60) # Wait a bit before retrying on error
