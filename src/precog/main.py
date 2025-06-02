import argparse
import os
import time
import logging
import pandas as pd

from src.precog.core.system import HyperlocalIntelligenceSystem
from src.precog.config.config import get_config

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Determine the base directory of the project (PreCog)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
DEFAULT_DATA_DIR = os.path.join(PROJECT_ROOT, 'data')

def main():
    parser = argparse.ArgumentParser(description="PreCog Hyperlocal Intelligence System", add_help=False)
    parser.add_argument('mode', nargs='?', choices=['monitor', 'report_once', 'train', 'generate_synthetic_data'])
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--interval', type=int)
    parser.add_argument('--social_media_csv', type=str)
    parser.add_argument('--news_articles_csv', type=str)
    parser.add_argument('--no_synthetic', action='store_true')
    parser.add_argument('--disable-synthetic-data', action='store_true')
    parser.add_argument('--preferred-llm', type=str, choices=['azure', 'vertex', 'none'])
    parser.add_argument('--use-llm-sentiment', action='store_true')
    parser.add_argument('--use-llm-narratives', action='store_true')
    parser.add_argument('--use-llm-counter-narratives', action='store_true')
    parser.add_argument('--use-llm-interventions', action='store_true')
    parser.add_argument('--help', action='help', help='Show this help message and exit.')

    args = parser.parse_args()

    # Interactive prompts for missing arguments
    if not args.mode:
        args.mode = input("Select mode [monitor, report_once, train, generate_synthetic_data]: ").strip()
        if args.mode not in ['monitor', 'report_once', 'train', 'generate_synthetic_data']:
            print("Invalid mode. Exiting.")
            return

    if not args.data_dir:
        args.data_dir = input(f"Enter data directory (default: {DEFAULT_DATA_DIR}): ").strip() or DEFAULT_DATA_DIR

    if args.mode == 'monitor' and not args.interval:
        interval_input = input("Enter monitoring interval in minutes (default: 30): ").strip()
        args.interval = int(interval_input) if interval_input else 30

    # Prompt for CSV paths (blank means synthetic)
    if not args.social_media_csv:
        social_input = input("Enter path to social media CSV (or leave blank to use synthetic data): ").strip()
        args.social_media_csv = social_input or None
    if not args.news_articles_csv:
        news_input = input("Enter path to news articles CSV (or leave blank to use synthetic data): ").strip()
        args.news_articles_csv = news_input or None

    # Optional LLM flags prompts
    if args.preferred_llm is None:
        # llm_choice = input("Preferred LLM provider? [azure, vertex, none] (leave blank for none): ").strip()
        args.preferred_llm = 'azure'

    def yes_no_prompt(prompt):
        return input(prompt + " [y/N]: ").strip().lower() == 'y'

    if not args.use_llm_sentiment:
        args.use_llm_sentiment = True
    if not args.use_llm_narratives:
        args.use_llm_narratives = True
    if not args.use_llm_counter_narratives:
        args.use_llm_counter_narratives = True
    if not args.use_llm_interventions:
        args.use_llm_interventions = True

    # Prepare LLM config overrides from args
    llm_config_overrides = {
        'PREFERRED_LLM_PROVIDER': args.preferred_llm,
        'USE_LLM_FOR_SENTIMENT': args.use_llm_sentiment,
        'USE_LLM_FOR_NARRATIVES': args.use_llm_narratives,
        'USE_LLM_FOR_COUNTER_NARRATIVES': args.use_llm_counter_narratives,
        'USE_LLM_FOR_INTERVENTIONS': args.use_llm_interventions
    }

    system = HyperlocalIntelligenceSystem(data_dir=args.data_dir, cli_config_overrides=llm_config_overrides)
    data_dir_abs = os.path.abspath(args.data_dir)

    # Handle explicit synthetic generation mode
    if args.mode == 'generate_synthetic_data':
        logger.info(f"Generating synthetic data in {data_dir_abs}/raw...")
        os.makedirs(os.path.join(data_dir_abs, 'raw'), exist_ok=True)
        social_df = system.synthetic_data_generator.generate_social_media_data(num_records=1000)
        news_df = system.synthetic_data_generator.generate_news_articles_data(num_records=100)
        social_path = os.path.join(data_dir_abs, 'raw', 'synthetic_social_media.csv')
        news_path = os.path.join(data_dir_abs, 'raw', 'synthetic_news_articles.csv')
        social_df.to_csv(social_path, index=False)
        news_df.to_csv(news_path, index=False)
        logger.info(f"Synthetic social media data saved to {social_path}")
        logger.info(f"Synthetic news articles data saved to {news_path}")
        return

    # Log arguments before checking for on-the-fly synthetic data generation
    logger.info(f"Pre-synthetic check: disable_synthetic_data={args.disable_synthetic_data}, "
                f"social_media_csv='{args.social_media_csv}' (type: {type(args.social_media_csv)}), "
                f"news_articles_csv='{args.news_articles_csv}' (type: {type(args.news_articles_csv)}))")

    # Automatically generate synthetic data if no CSVs provided
    if not args.disable_synthetic_data and not args.social_media_csv and not args.news_articles_csv:
        logger.info("Condition met: Entering on-the-fly synthetic data generation block...")
        logger.info("No CSVs provided; generating synthetic data on-the-fly...")
        os.makedirs(os.path.join(data_dir_abs, 'raw'), exist_ok=True)
        social_df = system.synthetic_data_generator.generate_social_media_data(num_records=1000)
        news_df = system.synthetic_data_generator.generate_news_articles_data(num_records=100)
        social_path = os.path.join(data_dir_abs, 'raw', 'synthetic_social_media.csv')
        news_path = os.path.join(data_dir_abs, 'raw', 'synthetic_news_articles.csv')
        social_df.to_csv(social_path, index=False)
        news_df.to_csv(news_path, index=False)
        args.social_media_csv = social_path
        args.news_articles_csv = news_path
        logger.info(f"On-the-fly synthetic social media data saved to {social_path}")
        logger.info(f"On-the-fly synthetic news articles data saved to {news_path}")
    else:
        logger.info("Condition NOT met: Skipping on-the-fly synthetic data generation block.")
        if args.disable_synthetic_data:
            logger.info("Reason: args.disable_synthetic_data is True.")
        if args.social_media_csv:
            logger.info(f"Reason: args.social_media_csv is '{args.social_media_csv}' (truthy).")
        if args.news_articles_csv:
            logger.info(f"Reason: args.news_articles_csv is '{args.news_articles_csv}' (truthy).")

    # Load data
    logger.info(f"Preparing to load data. Current args.social_media_csv='{args.social_media_csv}', "
                f"args.news_articles_csv='{args.news_articles_csv}'")
    social_media_path_abs = os.path.abspath(args.social_media_csv) if args.social_media_csv else None
    news_articles_path_abs = os.path.abspath(args.news_articles_csv) if args.news_articles_csv else None
    logger.info(f"Calling load_and_prepare_data with social_media_path='{social_media_path_abs}', "
                f"news_articles_path='{news_articles_path_abs}'")
    # Determine if we should generate synthetic data
    should_generate_synthetic = not args.disable_synthetic_data and (not social_media_path_abs or not news_articles_path_abs)
    logger.info(f"Setting generate_synthetic={should_generate_synthetic} for load_and_prepare_data")

    data = system.load_and_prepare_data(
        social_media_path=social_media_path_abs,
        news_articles_path=news_articles_path_abs,
        generate_synthetic=should_generate_synthetic,  # Generate synthetic data if needed
        force_no_synthetic=args.no_synthetic # Pass the flag to prevent synthetic data generation if paths are bad
    )

    logger.info(f"Data loaded. Type: {type(data)}. Keys: {list(data.keys()) if isinstance(data, dict) else 'Not a dict or None'}")
    if isinstance(data, dict) and 'social_media_feed' in data and isinstance(data['social_media_feed'], pd.DataFrame):
        logger.info(f"Social media feed type: {type(data['social_media_feed'])}. Empty: {data['social_media_feed'].empty}")
    elif isinstance(data, dict) and 'social_media_feed' in data:
        logger.info(f"Social media feed type: {type(data['social_media_feed'])}. Not a DataFrame.")
    else:
        logger.info("Social media feed not found in data or data is not a dict.")

    # Robustly check if social_media_feed is usable
    social_media_feed_is_problematic = True  # Assume problematic by default
    if isinstance(data, dict):
        social_feed_item = data.get('social_media_feed') # Safely get item, defaults to None if key missing
        if isinstance(social_feed_item, pd.DataFrame) and not social_feed_item.empty:
            social_media_feed_is_problematic = False # It's a valid, non-empty DataFrame
    # If data is not a dict, or social_feed_item is None, not a DataFrame, or an empty DataFrame,
    # then social_media_feed_is_problematic remains True.

    if not data or social_media_feed_is_problematic:  # Exit if data is falsey (None, empty dict) or social media feed is problematic
        logger.error("Critical data (e.g., social media feed) is missing, empty, or invalid, or main data structure not loaded. Exiting. Provide CSV files or allow synthetic data generation.")
        return

    # Execute selected mode
    if args.mode == 'train':
        logger.info("Starting model training...")
        system.train_models(data)
        logger.info("Model training finished.")

    elif args.mode == 'report_once':
        logger.info("Running single analysis cycle and generating report...")
        # if not system.models_trained():
        #     logger.info("Models not trained or missing. Training first...")
        #     system.train_models(data)
        analysis_results = system.analyze_current_situation(data['social_media_feed'],data['news_articles'])
        report_str = system.generate_report(analysis_results)
        logger.info("\n" + report_str)
        reports_dir = os.path.join(data_dir_abs, 'reports')
        os.makedirs(reports_dir, exist_ok=True)
        timestamp = system.last_analysis_time.strftime('%Y%m%d_%H%M%S') if system.last_analysis_time else time.strftime('%Y%m%d_%H%M%S')
        report_filename = f"analysis_report_manual_{timestamp}.txt"
        report_path = os.path.join(reports_dir, report_filename)
        try:
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(report_str)
            logger.info(f"Report saved to {report_path}")
        except IOError as e:
            logger.error(f"Failed to save report to {report_path}: {e}")

    elif args.mode == 'monitor':
        logger.info(f"Starting continuous monitoring. Interval: {args.interval} minutes.")
        if not system.models_trained():
            logger.info("Models not trained or missing. Training first...")
            system.train_models(data)
        system.run_continuous_monitoring(interval_minutes=args.interval, run_once=False)

if __name__ == "__main__":
    main()