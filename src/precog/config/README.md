# Config Package

This package manages the settings and parameters for the PreCog application through the `config.py` module.

## `config.py`

This central configuration file defines various parameters essential for the application's operation. It includes:

- **General Settings**: Application name (`APP_NAME`) and version (`VERSION`).
- **Data Paths**: Default directory for application data (`DEFAULT_DATA_DIR`). Specific paths for raw, processed, model, and report data can be derived or overridden.
- **AI Model Settings**:
    - `SENTIMENT_ANALYZER_LANGUAGE`: Specifies the language model for sentiment analysis (e.g., 'en_hinglish').
    - `ANOMALY_DETECTOR_CONTAMINATION`: Sets the expected proportion of outliers for the anomaly detection model.
    - `FRICTION_MODEL_PATH`: Default path for the saved friction prediction model.
    - `SYNTHETIC_FRICTION_DATA_SIZE`: Number of samples for generating synthetic data for the friction model.
- **External API Keys**: Placeholders and environment variable lookups for services like Azure OpenAI (`AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_ENDPOINT`, etc.) and Google Vertex AI (`VERTEX_AI_PROJECT_ID`, `VERTEX_AI_LOCATION`, etc.). It emphasizes that real keys should not be hardcoded but managed via environment variables.
- **Logging**: Configuration for log level (`LOG_LEVEL`) and format (`LOG_FORMAT`).
- **System Behavior**:
    - `DEFAULT_MONITORING_INTERVAL_MINUTES`: Default interval for the application's monitoring mode.
    - `RANDOM_SEED`: Seed for random number generation to ensure reproducibility.
- **LLM Usage Control Flags**:
    - `PREFERRED_LLM_PROVIDER`: Specifies the preferred Large Language Model provider (e.g., 'azure', 'vertex', 'none').
    - Flags to enable/disable LLM usage for specific tasks like sentiment analysis (`USE_LLM_FOR_SENTIMENT`), narrative extraction (`USE_LLM_FOR_NARRATIVES`), counter-narrative generation (`USE_LLM_FOR_COUNTER_NARRATIVES`), and intervention recommendations (`USE_LLM_FOR_INTERVENTIONS`).

The module provides a `get_config()` function that returns a dictionary of all these settings, making them easily accessible throughout the application. It also includes a basic check to warn if placeholder API keys are still in use.