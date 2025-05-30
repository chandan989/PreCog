# src/utils/helpers.py

import datetime
import json
import re
import logging
from typing import Any, Dict, List, Optional

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_current_timestamp() -> str:
    """Returns the current timestamp in ISO format."""
    return datetime.datetime.now(datetime.timezone.utc).isoformat()

def format_data_for_logging(data: Dict[str, Any], indent: Optional[int] = None) -> str:
    """Formats a dictionary for pretty logging using JSON.
    
    Args:
        data: The dictionary to format.
        indent: JSON indentation level. None for compact, or an int (e.g., 2 or 4) for pretty print.
    Returns:
        A JSON string representation of the data.
    """
    try:
        return json.dumps(data, indent=indent, default=str) # default=str handles non-serializable types like datetime
    except TypeError as e:
        logging.error(f"Error formatting data for logging: {e}")
        return str(data) # Fallback to simple string representation

def clean_text_data(text: str) -> str:
    """Cleans text data by removing HTML tags, extra whitespace, and non-alphanumeric characters (except spaces).
    
    Args:
        text: The input string to clean.
    Returns:
        The cleaned string.
    """
    if not isinstance(text, str):
        logging.warning(f"clean_text_data received non-string input: {type(text)}. Returning as is.")
        return str(text) # Or raise error, depending on desired strictness

    # Remove HTML tags (simple regex, might need a library like BeautifulSoup for complex HTML)
    text = re.sub(r'<[^>]+>', '', text)
    # Normalize whitespace (replace multiple spaces/newlines with a single space)
    text = re.sub(r'\s+', ' ', text).strip()
    # Optional: Remove special characters, keeping basic punctuation if needed for NLP
    # text = re.sub(r'[^a-zA-Z0-9\s.,!?-]', '', text) # Example: keep some punctuation
    # For this PoC, let's be more aggressive for simplicity in later stages if not using advanced NLP
    # text = re.sub(r'[^a-zA-Z0-9\s]', '', text) # Keeps only alphanumeric and spaces
    return text

def validate_data_schema(data_item: Dict[str, Any], required_fields: List[str]) -> bool:
    """Validates if a data item (dictionary) contains all required fields.

    Args:
        data_item: The dictionary to validate.
        required_fields: A list of strings representing the keys that must be present.
    Returns:
        True if all required fields are present, False otherwise.
    """
    if not isinstance(data_item, dict):
        logging.error(f"Invalid data_item type for schema validation: {type(data_item)}")
        return False
    missing_fields = [field for field in required_fields if field not in data_item]
    if missing_fields:
        logging.warning(f"Data item missing required fields: {', '.join(missing_fields)}. Item: {data_item}")
        return False
    return True


# Example: Configuration loading (simple version)
def load_config(config_path: str = "config.json") -> Dict[str, Any]:
    """Loads a JSON configuration file.

    Args:
        config_path: Path to the JSON configuration file.
    Returns:
        A dictionary with configuration parameters or an empty dict if an error occurs.
    """
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        logging.info(f"Configuration loaded from {config_path}")
        return config
    except FileNotFoundError:
        logging.warning(f"Configuration file {config_path} not found. Using default or empty config.")
        return {}
    except json.JSONDecodeError as e:
        logging.error(f"Error decoding JSON from {config_path}: {e}")
        return {}

if __name__ == '__main__':
    print(f"Current UTC Timestamp: {get_current_timestamp()}")
    
    sample_data = {'event': 'test_event', 'value': 123, 'status': 'success', 'timestamp': datetime.datetime.now()}
    print(f"Formatted Log (compact): {format_data_for_logging(sample_data)}")
    print(f"Formatted Log (pretty): \n{format_data_for_logging(sample_data, indent=2)}")

    raw_text_html = "  <p>This is <b>some messy text</b> with extra spaces.  \nAnd a <a href='#'>link</a>.  " 
    print(f"Original Text: '{raw_text_html}'")
    print(f"Cleaned Text: '{clean_text_data(raw_text_html)}'")
    print(f"Cleaned Text (None input): '{clean_text_data(None)}'")

    valid_item = {'id': 1, 'text': 'Valid data', 'user': 'test'}
    invalid_item = {'id': 2, 'user': 'test_no_text'}
    required = ['id', 'text']
    print(f"Validation for valid_item ({required}): {validate_data_schema(valid_item, required)}")
    print(f"Validation for invalid_item ({required}): {validate_data_schema(invalid_item, required)}")
    print(f"Validation for non-dict: {validate_data_schema(None, required)}")

    # Example config loading (create a dummy config.json for this to work)
    # with open("config.json", "w") as f_conf:
    #     json.dump({"api_key": "your_key", "timeout": 30}, f_conf)
    # app_config = load_config()
    # if app_config:
    #     print(f"Loaded API Key from config: {app_config.get('api_key')}")
    # else:
    #     print("Could not load config or config is empty.")