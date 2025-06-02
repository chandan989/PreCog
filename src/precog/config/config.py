# PreCog Configuration File
import os

# --- General Settings ---
APP_NAME = "PreCog Hyperlocal Intelligence System"
VERSION = "0.1.0-alpha"

# --- Data Paths (relative to project root or absolute) ---
# These can be overridden by command-line arguments in main.py
# Or, if main.py's data_dir is used, these might not be needed here directly
# but can serve as defaults if the system components were used independently.
# PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')) # src/precog/config -> PreCog
# DATA_DIR_BASE = os.path.join(PROJECT_ROOT, 'data')
# RAW_DATA_DIR = os.path.join(DATA_DIR_BASE, 'raw')
# PROCESSED_DATA_DIR = os.path.join(DATA_DIR_BASE, 'processed')
# MODELS_DIR = os.path.join(DATA_DIR_BASE, 'models')
# REPORTS_DIR = os.path.join(DATA_DIR_BASE, 'reports')

# Default data directory path
DEFAULT_DATA_DIR = 'data'

# --- AI Model Settings ---
# Sentiment Analysis
SENTIMENT_ANALYZER_LANGUAGE = 'en_hinglish' # 'en', 'hi', 'en_hinglish'

# Anomaly Detection
ANOMALY_DETECTOR_CONTAMINATION = 0.05 # Expected proportion of outliers

# Friction Predictor
FRICTION_MODEL_PATH = "default_friction_model.joblib" # To be saved in MODELS_DIR
SYNTHETIC_FRICTION_DATA_SIZE = 1000

# Misinformation Detector
# Add any specific model paths or thresholds if they become configurable

# --- External API Keys (Placeholders - DO NOT COMMIT REAL KEYS) ---
# Example for Azure OpenAI (if used in the future)
AZURE_OPENAI_API_KEY = os.environ.get("AZURE_OPENAI_API_KEY", "b9ba1b47e18f48dc9b86248752c80395")
AZURE_OPENAI_ENDPOINT = os.environ.get("AZURE_OPENAI_ENDPOINT", "https://ravager-base.openai.azure.com/")
AZURE_OPENAI_API_VERSION = "2025-01-01-preview"
AZURE_OPENAI_DEPLOYMENT_NAME = "gpt-4.1" # e.g., gpt-35-turbo or gpt-4

# Example for a generic News API (if used for live news fetching)
NEWS_API_KEY = os.environ.get("NEWS_API_KEY", "YOUR_NEWS_API_KEY_HERE")

# Example for Google Vertex AI (if used in the future)
VERTEX_AI_PROJECT_ID = os.environ.get("VERTEX_AI_PROJECT_ID", "YOUR_VERTEX_AI_PROJECT_ID_HERE")
VERTEX_AI_LOCATION = os.environ.get("VERTEX_AI_LOCATION", "YOUR_VERTEX_AI_LOCATION_HERE") # e.g., us-central1
VERTEX_AI_MODEL_NAME = os.environ.get("VERTEX_AI_MODEL_NAME", "gemini-pro") # e.g., gemini-pro, text-bison
# For authentication, Vertex AI typically uses Application Default Credentials (ADC)
# or a service account key JSON file (path set via GOOGLE_APPLICATION_CREDENTIALS env var).

# --- Logging ---
LOG_LEVEL = "INFO" # DEBUG, INFO, WARNING, ERROR, CRITICAL
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

# --- System Behavior --- 
DEFAULT_MONITORING_INTERVAL_MINUTES = 30
RANDOM_SEED = 42 # For reproducibility in synthetic data generation, model training etc.

# --- LLM Usage Control Flags ---
PREFERRED_LLM_PROVIDER = os.environ.get("PREFERRED_LLM_PROVIDER", "azure") # 'azure' or 'vertex', or 'none'
USE_LLM_FOR_SENTIMENT = os.environ.get("USE_LLM_FOR_SENTIMENT", "False").lower() == 'true'
USE_LLM_FOR_NARRATIVES = os.environ.get("USE_LLM_FOR_NARRATIVES", "False").lower() == 'true'
USE_LLM_FOR_COUNTER_NARRATIVES = os.environ.get("USE_LLM_FOR_COUNTER_NARRATIVES", "False").lower() == 'true'
USE_LLM_FOR_INTERVENTIONS = os.environ.get("USE_LLM_FOR_INTERVENTIONS", "False").lower() == 'true'

# --- Placeholder for future AI integration specifics ---
# For example, if using local LLMs:
# LOCAL_LLM_MODEL_PATH = "/path/to/your/local/llm/model"
# LOCAL_LLM_TYPE = "llama2-7b-chat" # or similar identifier

# It's good practice to load sensitive keys from environment variables.
# The defaults above are placeholders and should be managed securely.

def get_config() -> dict:
    """Returns a dictionary of all configuration settings."""
    return {
        "APP_NAME": APP_NAME,
        "VERSION": VERSION,
        "DEFAULT_DATA_DIR": DEFAULT_DATA_DIR,
        "SENTIMENT_ANALYZER_LANGUAGE": SENTIMENT_ANALYZER_LANGUAGE,
        "ANOMALY_DETECTOR_CONTAMINATION": ANOMALY_DETECTOR_CONTAMINATION,
        "FRICTION_MODEL_PATH": FRICTION_MODEL_PATH,
        "SYNTHETIC_FRICTION_DATA_SIZE": SYNTHETIC_FRICTION_DATA_SIZE,
        "AZURE_OPENAI_API_KEY": AZURE_OPENAI_API_KEY,
        "AZURE_OPENAI_ENDPOINT": AZURE_OPENAI_ENDPOINT,
        "AZURE_OPENAI_API_VERSION": AZURE_OPENAI_API_VERSION,
        "AZURE_OPENAI_DEPLOYMENT_NAME": AZURE_OPENAI_DEPLOYMENT_NAME,
        "NEWS_API_KEY": NEWS_API_KEY,
        "VERTEX_AI_PROJECT_ID": VERTEX_AI_PROJECT_ID,
        "VERTEX_AI_LOCATION": VERTEX_AI_LOCATION,
        "VERTEX_AI_MODEL_NAME": VERTEX_AI_MODEL_NAME,
        "LOG_LEVEL": LOG_LEVEL,
        "LOG_FORMAT": LOG_FORMAT,
        "DEFAULT_MONITORING_INTERVAL_MINUTES": DEFAULT_MONITORING_INTERVAL_MINUTES,
        "RANDOM_SEED": RANDOM_SEED,
        "PREFERRED_LLM_PROVIDER": PREFERRED_LLM_PROVIDER,
        "USE_LLM_FOR_SENTIMENT": USE_LLM_FOR_SENTIMENT,
        "USE_LLM_FOR_NARRATIVES": USE_LLM_FOR_NARRATIVES,
        "USE_LLM_FOR_COUNTER_NARRATIVES": USE_LLM_FOR_COUNTER_NARRATIVES,
        "USE_LLM_FOR_INTERVENTIONS": USE_LLM_FOR_INTERVENTIONS,
        # "LOCAL_LLM_MODEL_PATH": LOCAL_LLM_MODEL_PATH,
        # "LOCAL_LLM_TYPE": LOCAL_LLM_TYPE,
    }

if __name__ == '__main__':
    # Example of how to access the config
    config = get_config()
    print(f"Running {config['APP_NAME']} v{config['VERSION']}")
    if config['AZURE_OPENAI_API_KEY'] == "YOUR_AZURE_OPENAI_API_KEY_HERE":
        print("Warning: Azure OpenAI API Key is a placeholder. Please set it via environment variable or in config.py.")
    if config['VERTEX_AI_PROJECT_ID'] == "YOUR_VERTEX_AI_PROJECT_ID_HERE":
        print("Warning: Vertex AI Project ID is a placeholder. Please set it via environment variable or in config.py.")
