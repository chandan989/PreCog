# PreCog Application Package

This is the main package for the PreCog application. It contains the core logic and modules.

## Sub-packages:

- `analysis/`: Contains modules related to data analysis and insights generation.
- `config/`: Likely holds configuration files or modules for managing application settings.
- `core/`: Contains the core business logic and fundamental components of the application.
- `dashboard/`: Modules responsible for the user interface and dashboard presentation.
- `interventions/`: Contains logic related to suggesting or implementing interventions based on risk assessment.

## Files:

- `main.py`: This is the primary command-line interface (CLI) entry point for the PreCog application. It handles various operational modes and configurations:
    - **Modes of Operation**: 
        - `monitor`: Runs the system in a continuous monitoring mode, analyzing data at specified intervals.
        - `report_once`: Performs a single analysis cycle and generates a report.
        - `train`: Initiates the training process for the system's machine learning models.
        - `generate_synthetic_data`: Creates synthetic datasets for social media and news articles for testing or demonstration purposes.
    - **Data Handling**: 
        - Allows specification of a data directory.
        - Can take paths to CSV files for social media and news articles.
        - If no data paths are provided, it can automatically generate and use synthetic data (unless disabled).
    - **LLM Configuration**: 
        - Allows selection of a preferred Large Language Model (LLM) provider (e.g., Azure, Vertex, or none).
        - Provides flags to enable/disable LLM usage for specific tasks like sentiment analysis, narrative generation, counter-narrative suggestions, and intervention recommendations.
    - **Interactive Prompts**: If required arguments are not provided via the command line, the script interactively prompts the user for them.
    - **Initialization**: It initializes the `HyperlocalIntelligenceSystem` with the specified configurations and data.