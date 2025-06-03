# Core Package

This package houses the core business logic, fundamental data structures, and essential algorithms of the PreCog application. It forms the backbone of the application, orchestrating various analytical components.

## Modules

- **`system.py`**: This module defines the `HyperlocalIntelligenceSystem` class, which is the central orchestrator for the PreCog application. 
    - **Initialization**: The class constructor (`__init__`) loads configuration (default and custom), sets up data directories (raw, processed, models, reports), and initializes various AI components. These components include:
        - `SyntheticDataGenerator`: For creating sample data.
        - `HyperlocalSentimentAnalyzer`: For analyzing sentiment in text, with optional LLM integration.
        - `SentimentAnomalyDetector`: For identifying unusual sentiment patterns.
        - `FrictionPointPredictor`: For predicting social friction risks.
        - `MisinformationDetector`: For detecting potential misinformation, with optional LLM integration.
        - `InterventionRecommender`: For suggesting interventions based on analysis, with optional LLM integration.
        - It also initializes an LLM client (e.g., Azure, Vertex AI) based on the configuration, which can be used by the other components.
    - **Data Handling**: The `load_and_prepare_data` method is responsible for loading data from specified file paths (e.g., social media feeds, news articles) or generating synthetic data if real data is unavailable or explicitly requested. It performs basic preprocessing like timestamp conversion.
    - **Model Training**: The `train_models` method orchestrates the training of relevant models. For instance, it fits the baseline for the `SentimentAnomalyDetector` and trains the `FrictionPointPredictor` (often using its internal synthetic data generation by default).
    - **Analysis Workflow**: The `analyze_current_situation` method processes current data (social media, news) through the various analytical components. It performs sentiment analysis, detects anomalies, identifies misinformation, and predicts friction risks. The results, including alerts and risk assessments, are aggregated.
    - **Reporting**: The `generate_report` method compiles the analysis results into a structured report, which can include overall summaries, lists of alerts (sentiment, misinformation), friction risk predictions, and recommended interventions. The report can be saved to a file.
    - **Continuous Monitoring**: The `monitor_continuously` method implements a loop for ongoing analysis. It periodically fetches/simulates new data, runs the analysis pipeline, generates reports, and logs key metrics. The monitoring interval is configurable.
    - **LLM Integration**: The system is designed to leverage Large Language Models (LLMs) for various tasks, controlled by configuration flags. The `get_llm_client` utility is used to instantiate the appropriate LLM client.

In essence, `system.py` ties together the data ingestion, processing, analysis, and reporting functionalities of the PreCog application, providing a cohesive framework for hyperlocal intelligence.