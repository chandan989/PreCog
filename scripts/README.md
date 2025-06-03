# Scripts Folder

This folder contains various Python scripts for testing different components and functionalities of the PreCog application.

- `test_dashboard.py`: This script tests the initialization of the main dashboard application (`src.precog.dashboard.app.main`). It ensures that the dashboard application can be loaded without immediate errors.

- `test_enhancements.py`: This script focuses on testing enhanced analytical components:
    - `HyperlocalSentimentAnalyzer`: Tests sentiment analysis on texts in English, Hindi, and Hinglish, using both rule-based and batch analysis.
    - `MisinformationDetector`: Tests misinformation detection on sample social media posts and news articles, including the calculation of misinformation scores and generation of alerts.
    - `HyperlocalIntelligenceSystem`: Tests the integration of these enhanced components by running a system-level analysis on sample social media and news data.

- `test_fixes.py`: This script is designed to test specific bug fixes or verify the functionality of certain modules:
    - `SentimentAlert`: Tests the creation and attributes of the `SentimentAlert` data model.
    - `SentimentAnomalyDetector`: Tests the fitting of a baseline model and detection of anomalies using sample data.
    - It also includes a test for creating a bar chart using `plotly.express`, likely to verify a fix or functionality in the dashboard's visualization components (`app.py`).

- `test_friction_risk.py`: This script tests components related to friction risk assessment and intervention recommendations:
    - `FrictionRisk`: Tests the creation and attributes of the `FrictionRisk` data model.
    - `HyperlocalIntelligenceSystem`: Specifically tests the `_update_system_metrics` method with friction risk data.
    - `InterventionRecommender`: Tests methods like `_get_issue_type_from_alert`, `_get_context_from_alert`, and the main `recommend_interventions` method.