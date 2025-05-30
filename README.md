# PreCog - Hyperlocal Intelligence System

PreCog is a Python-based system designed for hyperlocal intelligence analysis. It processes social media and news data to identify sentiment trends, detect anomalies, predict friction points, monitor misinformation, and recommend interventions.

This project has been recently refactored into a more modular structure.

## Project Structure

```
PreCog/
├── data/                     # Data directory (raw, processed, models, reports)
│   ├── raw/
│   ├── processed/
│   ├── models/
│   └── reports/
├── notebooks/                # Jupyter notebooks for experimentation
├── scripts/                  # Utility scripts (e.g., old run_poc_simulation.py)
├── src/
│   └── precog/               # Main source code for the PreCog application
│       ├── __init__.py
│       ├── analysis/           # Modules for various analytical tasks
│       │   ├── __init__.py
│       │   ├── anomaly_detector.py
│       │   ├── friction_predictor.py
│       │   ├── misinformation_detector.py
│       │   └── sentiment_analyzer.py
│       ├── config/             # Configuration files
│       │   ├── __init__.py
│       │   └── config.py
│       ├── core/               # Core components like data models and the main system orchestrator
│       │   ├── __init__.py
│       │   ├── data_models.py
│       │   ├── synthetic_data_generator.py
│       │   └── system.py
│       ├── interventions/      # Modules related to recommending interventions
│       │   ├── __init__.py
│       │   └── intervention_recommender.py
│       └── main.py             # Main entry point for the application
├── tests/                    # Unit and integration tests (to be developed)
│   └── __init__.py
├── .gitignore
├── requirements.txt          # Python package dependencies
├── README.md                 # This file
└── ... (other project files)
```

## Setup

1.  **Clone the repository (if applicable).**
2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    Ensure `nltk` data is downloaded if you haven't already (for sentiment analysis):
    ```python
    import nltk
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('vader_lexicon')
    nltk.download('wordnet') # For lemmatization in sentiment analysis
    nltk.download('averaged_perceptron_tagger') # For TextBlob POS tagging
    ```

3.  **Configuration (Optional):**
    Review and update `src/precog/config/config.py` if needed. For example, to use actual API keys for services like Azure OpenAI (currently placeholders), set the corresponding environment variables or update the file directly (not recommended for sensitive keys in version control).

## Running the Application

The main entry point is `src/precog/main.py`.

Navigate to the project root directory (`PreCog/`) in your terminal.

**Available Modes:**

1.  **Generate Synthetic Data:**
    Creates sample social media and news CSV files in the `data/raw/` directory.
    ```bash
    python -m src.precog.main generate_synthetic_data --data_dir ./data
    ```

2.  **Train Models:**
    Trains the analytical models (e.g., friction predictor) using available data. It will use synthetic data if no specific CSVs are provided and synthetic data generation is enabled.
    ```bash
    python -m src.precog.main train --data_dir ./data
    ```
    To use your own data:
    ```bash
    python -m src.precog.main train --data_dir ./data --social_media_csv /path/to/your/social_media.csv --news_articles_csv /path/to/your/news.csv
    ```

3.  **Run Single Analysis & Report:**
    Performs one cycle of data loading, analysis, and report generation. The report is printed to the console and saved in `data/reports/`.
    ```bash
    python -m src.precog.main report_once --data_dir ./data
    ```

4.  **Continuous Monitoring:**
    Runs the system in a continuous loop, periodically fetching/generating data, analyzing it, and saving reports.
    ```bash
    python -m src.precog.main monitor --data_dir ./data --interval 30 
    ```
    (`--interval` is in minutes)

**Command-line Arguments:**

*   `mode`: `generate_synthetic_data`, `train`, `report_once`, `monitor` (required)
*   `--data_dir`: Path to the base data directory (default: `./data` relative to project root).
*   `--interval`: Monitoring interval in minutes for `monitor` mode (default: 30).
*   `--social_media_csv`: Path to your social media data CSV file.
*   `--news_articles_csv`: Path to your news articles data CSV file.
*   `--no_synthetic`: Disable automatic generation of synthetic data if input CSVs are missing or fail to load.

Example using generated synthetic data for a single report:
```bash
# First, generate some data if you don't have any
python -m src.precog.main generate_synthetic_data --data_dir ./data

# Then, train the models (uses synthetic data by default if no CSVs specified)
python -m src.precog.main train --data_dir ./data

# Finally, run a single report
python -m src.precog.main report_once --data_dir ./data
```

## Next Steps & AI Integration

*   **Develop Unit and Integration Tests:** Populate the `tests/` directory.
*   **Enhance AI Capabilities:**
    *   Integrate Large Language Models (LLMs) like Azure OpenAI or local models for:
        *   More nuanced sentiment analysis.
        *   Sophisticated narrative extraction and misinformation detection.
        *   Generating detailed counter-narratives.
        *   Context-aware intervention recommendations.
    *   Improve existing models with more complex features and techniques.
*   **Live Data Ingestion:** Implement robust connectors for real-time data sources (social media APIs, news feeds).
*   **Scalability:** Optimize for larger datasets and more frequent analyses (e.g., using distributed computing frameworks if necessary).
*   **Dashboarding:** Connect outputs to a BI dashboard for visualization.

This refactoring provides a solid foundation for these future enhancements.