import sys
import os

# Add project root to Python path to allow direct imports
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Import the dashboard app
from src.precog.dashboard.app import main

if __name__ == "__main__":
    print("Testing dashboard app...")
    try:
        # This will initialize the system but won't actually run the Streamlit app
        # since we're not using streamlit run
        main()
        print("Dashboard app initialized successfully!")
    except Exception as e:
        print(f"Error initializing dashboard app: {e}")
        import traceback
        traceback.print_exc()