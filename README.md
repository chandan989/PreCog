# PreCog AI (Saarthi): Illuminating Paths to Social Harmony – A Game-Changing Hyperlocal Intelligence System

**PreCog AI**, affectionately nicknamed **Saarthi (सारथी - The Charioteer/Guide)**, is a revolutionary Hyperlocal Intelligence System engineered to be a cornerstone in proactive de-escalation, social cohesion, and community well-being. In a world grappling with misinformation and complex social dynamics, particularly within diverse societies like India, PreCog AI shifts the paradigm from reactive crisis management to proactive, data-informed foresight. By harnessing the power of advanced AI, Natural Language Processing (NLP) for vernacular languages, and Big Data analytics, PreCog AI empowers local authorities, community leaders, and public service organizations with the critical insights needed to anticipate, understand, and mitigate potential social friction before it escalates.

This is not just a technology; it's a commitment to building resilient communities, fostering understanding, and ensuring efficient, empathetic governance. PreCog AI aims to be the silent guardian, the intelligent guide that helps navigate the complexities of modern societal challenges, making our communities safer and more harmonious.

## The Profound Challenge: Navigating Social Complexities in the Digital Age

India's rich tapestry of cultures, languages, and communities is a source of immense strength. However, this diversity can also present unique challenges:

*   **Localized Friction:** Misunderstandings, resource competition, or unaddressed grievances can lead to localized social friction.
*   **Misinformation & Rumors:** The rapid spread of misinformation and rumors, often amplified by social media, can quickly escalate tensions and erode trust.
*   **Reactive Governance:** Authorities often find themselves reacting to incidents after they have occurred, limiting their ability to prevent escalation and address root causes proactively.
*   **Information Overload:** Sifting through vast amounts of data from diverse sources (social media, news, public grievances) to find actionable intelligence is a monumental task.

There is an urgent need for tools that can cut through the noise, identify emerging risks with precision, and guide timely, effective interventions.

## Our Solution: PreCog AI – Your Proactive Intelligence Partner

PreCog AI offers a multi-faceted solution designed to address these challenges head-on:

1.  **Hyperlocal Sentiment & Misinformation Anomaly Detector:**
    *   Analyzes vast streams of anonymized public social media trends, local vernacular news, and crowdsourced civic reports.
    *   Utilizes advanced NLP to understand nuanced sentiment, identify emerging narratives, and detect unusual spikes in negative sentiment or the rapid spread of unverified claims (especially those with potential to cause alarm or friction).
    *   **Ethical AI Core:** Focuses strictly on *topics* of concern and the *velocity/spread* of narratives, not on individuals, ensuring privacy and preventing profiling.

2.  **Friction Point Predictor (Probabilistic):**
    *   Employs machine learning models trained on historical (anonymized, aggregated) data of social disturbances, correlated with past grievance reports, civic issue patterns, and relevant socio-economic stress indicators.
    *   Identifies geographic areas or specific conditions that might have a *higher probability* of developing social friction if underlying issues are not addressed proactively.
    *   Acts as an early *risk-flagging* system, enabling preventative resource allocation.

3.  **Intelligent Intervention Recommender:**
    *   Based on the nature and severity of detected issues (e.g., specific misinformation, type of civic grievance, inter-group misunderstanding), the AI suggests a range of proactive interventions for local authorities and community leaders.
    *   Recommendations can include: targeted public information campaigns, prioritization of civic issue resolution, facilitated community dialogue sessions, or deployment of community liaison officers for on-ground assessment and engagement.

4.  **Community Pulse Dashboard (BI & Visualization):**
    *   A dynamic, intuitive dashboard providing a real-time (or near real-time) geospatial view of anonymized sentiment hotspots, trending civic concerns, and areas flagged by the AI for potential friction.
    *   Allows authorized users (local authorities, trained community leaders) to drill down into the *types* of issues being discussed (e.g., water, electricity, specific rumors) – **never individual posts or users**.
    *   Tracks the implementation and impact of interventions, enabling a continuous feedback loop for improving strategies.

## Key Features & Capabilities

*   **Vernacular NLP:** Specialized models for understanding multiple Indian languages and their nuances (including Hinglish).
*   **Advanced Anomaly Detection:** Identifies statistically significant deviations from baseline sentiment and information spread.
*   **Predictive Analytics:** Probabilistic forecasting of potential friction points.
*   **Ethical by Design:** Built-in safeguards for data anonymization, bias detection, and privacy preservation.
*   **Modular Architecture:** Flexible and extensible to incorporate new data sources and analytical modules.
*   **Actionable Insights:** Focuses on providing clear, concise, and actionable recommendations.

## Why PreCog AI is a Game Changer for Social Issues

PreCog AI is more than just a data analytics platform; it's a transformative approach to social governance:

*   **From Reactive to Proactive:** Fundamentally shifts the approach from crisis response to proactive prevention and de-escalation.
*   **Empowering Local Heroes:** Provides local authorities and community leaders with the tools they need to be more effective and targeted in their efforts.
*   **Building Trust & Transparency:** Facilitates better communication between citizens and authorities by identifying and helping address concerns promptly.
*   **Combating Misinformation at Scale:** Offers a powerful tool to detect and counter the spread of harmful misinformation that can destabilize communities.
*   **Optimized Resource Allocation:** Ensures that limited resources (personnel, funds, attention) are directed to where they are most needed, maximizing impact.
*   **Fostering Social Resilience:** By addressing grievances early and promoting understanding, PreCog AI helps build more resilient and cohesive communities.

## Alternative Business Use Case: Hyperlocal Public Health Sentinel

Beyond social cohesion, the core capabilities of PreCog AI can be adapted for other critical public service domains. One compelling alternative use case is as a **Hyperlocal Public Health Sentinel**:

*   **Early Outbreak Detection:** Monitor anonymized public discussions (social media, local forums, news) for unusual clusters of symptom-related keywords or spikes in health concerns in specific localities.
*   **Track Health Sentiment:** Analyze public sentiment regarding health services, vaccination campaigns, or emerging diseases.
*   **Misinformation in Health:** Identify and track the spread of health-related misinformation or rumors (e.g., about vaccine side-effects, ineffective remedies).
*   **Resource Prioritization:** Help public health officials identify areas needing urgent attention, targeted awareness campaigns, or increased medical supply/personnel deployment.

This adaptation would leverage the same data ingestion, NLP, anomaly detection, and dashboarding framework, but with a focus on health-related ontologies and intervention strategies, providing a powerful tool for proactive public health management.

## Deploying and Scaling with Databricks

PreCog AI is designed to leverage the power and scalability of the **Databricks Lakehouse Platform**, making it an ideal solution for handling the volume, velocity, and variety of data involved in hyperlocal intelligence.

*   **Unified Data Management:** Utilize **Delta Lake** to store raw, processed, and curated data (social media feeds, news articles, grievance reports, demographic data) in an open, reliable, and high-performance format.
*   **Large-Scale NLP Processing:** Employ **Apache Spark** and **Spark NLP** on Databricks clusters for efficient, distributed processing of massive text datasets in multiple Indian languages. This includes tokenization, sentiment analysis, topic modeling, and named entity recognition.
*   **Advanced Machine Learning with MLflow:** Train, manage, and deploy machine learning models (for anomaly detection, friction prediction, recommendation systems) using **MLflow** integrated within Databricks. This ensures reproducibility, versioning, and seamless deployment.
*   **Real-Time Analytics & Dashboards:** Power the **Community Pulse Dashboard** using **Databricks SQL**, enabling fast, interactive queries on large datasets. Visualizations can be built directly in Databricks or integrated with tools like Streamlit, Tableau, or Power BI.
*   **Stream Processing:** Implement real-time data ingestion and analysis pipelines using **Structured Streaming** to detect emerging trends and anomalies as they happen.
*   **Workflow Orchestration:** Use **Databricks Workflows** to schedule and manage complex data pipelines, model retraining jobs, and automated reporting.
*   **Collaboration & Security:** Leverage Databricks' collaborative notebooks, role-based access control, and enterprise-grade security features to ensure a secure and efficient development and operational environment.

By building on Databricks, PreCog AI can scale to handle data from numerous cities or regions, process information in near real-time, and continuously improve its AI models, delivering robust and impactful hyperlocal intelligence.

## Project Structure

```
PreCog/
├── data/                     # Data directory (raw, processed, models, reports)
│   ├── raw/
│   ├── processed/
│   ├── models/
│   └── reports/
├── notebooks/                # Jupyter notebooks for experimentation
├── scripts/                  # Utility scripts
├── src/
│   ├── ai_models/            # Core AI model implementations
│   ├── bi_dashboard/         # Streamlit dashboard application
│   ├── data_ingestion/       # Data loading and preprocessing modules
│   ├── precog/               # Main application logic (CLI, system orchestration)
│   │   ├── analysis/
│   │   ├── config/
│   │   ├── core/
│   │   ├── interventions/
│   │   └── main.py
│   └── utils/                # Helper utilities
├── tests/                    # Unit and integration tests
├── .gitignore
├── requirements.txt          # Python package dependencies
├── README.md                 # This file
├── idea.markdown             # Original ideation document
└── ... (other project files)
```

## Setup and Installation

1.  **Clone the Repository:**
    ```bash
    git clone <repository_url>
    cd PreCog
    ```

2.  **Create and Activate a Virtual Environment (Recommended):**
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Download NLTK Data:**
    Run the following Python script or execute these lines in a Python interpreter:
    ```python
    import nltk
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('vader_lexicon')
    nltk.download('wordnet')
    nltk.download('averaged_perceptron_tagger')
    ```

## Configuration

PreCog AI's behavior can be customized through environment variables and the configuration file `src/precog/config/config.py`.

**Key Configuration Points:**

*   **API Keys for External Services (LLMs, News APIs):**
    *   PreCog AI can integrate with Large Language Models (e.g., Azure OpenAI, Google Vertex AI) for enhanced NLP capabilities and news APIs for live data feeds.
    *   **IMPORTANT:** Never hardcode API keys directly into `config.py` or commit them to version control.
    *   Set API keys as **environment variables**. The `config.py` file is set up to read these variables.
        *   Example for Azure OpenAI:
            ```bash
            export AZURE_OPENAI_API_KEY="your_actual_api_key"
            export AZURE_OPENAI_ENDPOINT="your_azure_openai_endpoint"
            ```
        *   Refer to `src/precog/config/config.py` for the specific environment variable names expected (e.g., `NEWS_API_KEY`, `VERTEX_AI_PROJECT_ID`).
*   **LLM Provider Preference:**
    *   You can specify your preferred LLM provider (e.g., 'azure', 'vertex', or 'none') via the `PREFERRED_LLM_PROVIDER` environment variable.
    *   Flags like `USE_LLM_FOR_SENTIMENT`, `USE_LLM_FOR_NARRATIVES`, etc., (also configurable via environment variables) control whether LLMs are used for specific tasks. See `config.py` for details.
*   **Data Paths & Model Settings:**
    *   Default paths for data directories and model files are set in `config.py` but can often be overridden by command-line arguments when running specific modules (e.g., `src.precog.main`).
*   **Logging Levels:**
    *   Adjust `LOG_LEVEL` in `config.py` or via environment variable to control verbosity (e.g., "INFO", "DEBUG").

**Recommendation:** Review `src/precog/config/config.py` to understand all available settings and their corresponding environment variables. For production or sensitive deployments, manage these configurations using secure secret management tools.

## Running the Application

PreCog AI offers a command-line interface (CLI) for its core functionalities and a Streamlit-based dashboard for visualization and interactive analysis.

**1. Running Core System (CLI via `src.precog.main`):**

Navigate to the project root directory (`PreCog/`) in your terminal.

*   **Generate Synthetic Data:**
    Creates sample social media and news CSV files in the `data/raw/` directory for testing and development.
    ```bash
    python -m src.precog.main generate_synthetic_data --data_dir ./data
    ```

*   **Train Models:**
    Trains analytical models (e.g., friction predictor) using available data. Uses synthetic data by default if specific CSVs are not provided and synthetic generation is enabled.
    ```bash
    python -m src.precog.main train --data_dir ./data
    ```
    To use your own data:
    ```bash
    python -m src.precog.main train --data_dir ./data --social_media_csv /path/to/your/social_media.csv --news_articles_csv /path/to/your/news.csv
    ```

*   **Run Single Analysis & Report:**
    Performs one cycle of data loading, analysis, and report generation. The report is printed to the console and saved in `data/reports/`.
    ```bash
    python -m src.precog.main report_once --data_dir ./data
    ```

*   **Continuous Monitoring:**
    Runs the system in a continuous loop, periodically fetching/generating data, analyzing it, and saving reports.
    ```bash
    python -m src.precog.main monitor --data_dir ./data --interval 30
    ```
    (`--interval` is in minutes, default is 30 as per `config.py`)

**Command-line Arguments for `src.precog.main`:**

*   `mode`: `generate_synthetic_data`, `train`, `report_once`, `monitor` (required)
*   `--data_dir`: Path to the base data directory (default: `./data`).
*   `--interval`: Monitoring interval in minutes for `monitor` mode.
*   `--social_media_csv`: Path to your social media data CSV file.
*   `--news_articles_csv`: Path to your news articles data CSV file.
*   `--no_synthetic`: Disable automatic generation of synthetic data if input CSVs are missing.

**Example Workflow (CLI):**
```bash
# 1. Generate synthetic data (if you don't have your own)
python -m src.precog.main generate_synthetic_data --data_dir ./data

# 2. Train models (uses synthetic data by default if no specific CSVs provided)
python -m src.precog.main train --data_dir ./data

# 3. Run a single analysis cycle and generate a report
python -m src.precog.main report_once --data_dir ./data
```

**2. Running the Community Pulse Dashboard (Streamlit):**

The interactive dashboard provides visualizations of detected anomalies, sentiment trends, and intervention recommendations.

*   Ensure you are in the project root directory (`PreCog/`).
*   Run the following command in your terminal:
    ```bash
    streamlit run src/bi_dashboard/app.py
    ```
*   This will typically open the dashboard in your default web browser (e.g., at `http://localhost:8501`).
*   The dashboard allows you to upload data files (CSV, JSON/JSONL with 'text' and 'timestamp' columns) or attempt to load the latest data from the `data/raw` directory. You can then process this data, detect anomalies, and view recommendations interactively.

## Roadmap & Future Enhancements

PreCog AI is an evolving platform with an ambitious roadmap:

*   **Enhanced Vernacular NLP:** Continuously improve support for more Indian languages and dialects.
*   **Advanced Misinformation Models:** Develop more sophisticated models to detect nuanced misinformation, deepfakes (textual cues), and coordinated inauthentic behavior.
*   **Causal Inference:** Explore causal inference techniques to better understand the drivers of social friction.
*   **Integration with Alerting Systems:** Connect PreCog AI outputs to existing alerting and dispatch systems used by local authorities.
*   **Mobile Interface:** Develop a simplified mobile interface for community leaders and field officers.
*   **Federated Learning:** Investigate federated learning approaches to train models on decentralized data while preserving privacy.
*   **Comprehensive Ethical AI Auditing:** Implement regular, independent audits of AI models and data handling practices.

## Contributing

We welcome contributions from the community! If you're interested in helping improve PreCog AI, please refer to our (forthcoming) `CONTRIBUTING.md` guide for details on how to submit issues, feature requests, and pull requests.

## License

This project is licensed under the [MIT License](LICENSE) (or specify your chosen license).

---

**PreCog AI (Saarthi) - Building a Safer, More Cohesive Future, Together.**
