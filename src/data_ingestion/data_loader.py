import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import os
import json
import requests
import pickle
from datetime import datetime, timedelta
import random
from faker import Faker
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, LabelEncoder
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Faker for synthetic data generation
fake = Faker(['en_IN', 'hi_IN'])  # Indian locales


# ----------- Data Persistence Utils -----------

class DataManager:
    """Handles saving and loading data for AI models"""

    def __init__(self, base_dir: str = "data"):
        self.base_dir = base_dir
        self.raw_dir = os.path.join(base_dir, "raw")
        self.processed_dir = os.path.join(base_dir, "processed")
        self.models_dir = os.path.join(base_dir, "models")
        self.features_dir = os.path.join(base_dir, "features")

        # Create directories
        for dir_path in [self.raw_dir, self.processed_dir, self.models_dir, self.features_dir]:
            os.makedirs(dir_path, exist_ok=True)

    def save_raw_data(self, data: Dict[str, pd.DataFrame], timestamp: str = None) -> str:
        """Save raw data with timestamp"""
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        save_path = os.path.join(self.raw_dir, f"raw_data_{timestamp}")
        os.makedirs(save_path, exist_ok=True)

        metadata = {"timestamp": timestamp, "datasets": {}}

        for name, df in data.items():
            file_path = os.path.join(save_path, f"{name}.csv")
            df.to_csv(file_path, index=False)
            metadata["datasets"][name] = {
                "shape": df.shape,
                "columns": list(df.columns),
                "file_path": file_path
            }
            logger.info(f"Saved {name}: {df.shape}")

        # Save metadata
        with open(os.path.join(save_path, "metadata.json"), 'w') as f:
            json.dump(metadata, f, indent=2)

        return save_path

    def load_raw_data(self, timestamp: str = None) -> Dict[str, pd.DataFrame]:
        """Load raw data by timestamp (latest if None)"""
        if timestamp is None:
            # Find latest timestamp
            timestamps = [d for d in os.listdir(self.raw_dir) if d.startswith("raw_data_")]
            if not timestamps:
                return {}
            timestamp = max(timestamps).replace("raw_data_", "")

        load_path = os.path.join(self.raw_dir, f"raw_data_{timestamp}")
        if not os.path.exists(load_path):
            logger.error(f"Data path not found: {load_path}")
            return {}

        # Load metadata
        metadata_path = os.path.join(load_path, "metadata.json")
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        data = {}
        for name, info in metadata["datasets"].items():
            file_path = os.path.join(load_path, f"{name}.csv")
            data[name] = pd.read_csv(file_path)
            logger.info(f"Loaded {name}: {data[name].shape}")

        return data

    def save_processed_data(self, data: Dict[str, Any], name: str) -> str:
        """Save processed data for AI models"""
        file_path = os.path.join(self.processed_dir, f"{name}.pkl")
        with open(file_path, 'wb') as f:
            pickle.dump(data, f)
        logger.info(f"Saved processed data: {file_path}")
        return file_path

    def load_processed_data(self, name: str) -> Dict[str, Any]:
        """Load processed data"""
        file_path = os.path.join(self.processed_dir, f"{name}.pkl")
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        logger.info(f"Loaded processed data: {file_path}")
        return data

    def save_model(self, model: Any, name: str, metadata: Dict = None) -> str:
        """Save AI model with metadata"""
        model_path = os.path.join(self.models_dir, f"{name}.pkl")
        joblib.dump(model, model_path)

        if metadata:
            metadata_path = os.path.join(self.models_dir, f"{name}_metadata.json")
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)

        logger.info(f"Saved model: {model_path}")
        return model_path

    def load_model(self, name: str) -> Tuple[Any, Dict]:
        """Load AI model with metadata"""
        model_path = os.path.join(self.models_dir, f"{name}.pkl")
        metadata_path = os.path.join(self.models_dir, f"{name}_metadata.json")

        model = joblib.load(model_path)

        metadata = {}
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)

        logger.info(f"Loaded model: {model_path}")
        return model, metadata


# ----------- Placeholder Data Loaders for Dashboard ----------- 

def load_social_media_data(data_dir: Optional[str] = None, from_synthetic: bool = True) -> pd.DataFrame:
    logger.info("Placeholder: load_social_media_data called")
    if from_synthetic:
        # Example: Generate some synthetic data if requested
        # gen = SyntheticDataGenerator()
        # return gen.generate_social_media_data(num_records=100)
        pass # Returning empty for now
    # Add logic here to load actual social media data from data_dir if from_synthetic is False
    return pd.read_csv(data_dir)

def load_news_data(data_dir: Optional[str] = None, from_synthetic: bool = True) -> pd.DataFrame:
    logger.info("Placeholder: load_news_data called")
    if from_synthetic:
        # gen = SyntheticDataGenerator()
        # return gen.generate_news_articles(num_records=50)
        pass
    return pd.read_csv(data_dir)

def load_crowdsourced_reports(data_dir: Optional[str] = None, from_synthetic: bool = True) -> pd.DataFrame:
    logger.info("Placeholder: load_crowdsourced_reports called")
    if from_synthetic:
        # gen = SyntheticDataGenerator()
        # return gen.generate_crowdsourced_reports(num_records=20)
        pass
    return pd.read_csv(data_dir)

# Placeholder for loading from existing DataFrames (if needed by __init__.py)
def load_social_media_data_from_df(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Placeholder: load_social_media_data_from_df called")
    return df

def load_news_data_from_df(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Placeholder: load_news_data_from_df called")
    return df

def load_crowdsourced_reports_from_df(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Placeholder: load_crowdsourced_reports_from_df called")
    return df

# Placeholder for creating dummy files (if needed by __init__.py)
def create_dummy_data_files(data_dir: str):
    logger.info(f"Placeholder: create_dummy_data_files called for {data_dir}")
    # Example: Create empty CSVs or use SyntheticDataGenerator to populate them
    # dm = DataManager(base_dir=os.path.dirname(data_dir)) # Assuming data_dir is like 'data/raw'
    # syn_gen = SyntheticDataGenerator()
    # raw_data_payload = {
    #     "social_media_posts": syn_gen.generate_social_media_data(10),
    #     "news_articles": syn_gen.generate_news_articles(5),
    #     "crowdsourced_reports": syn_gen.generate_crowdsourced_reports(3)
    # }
    # dm.save_raw_data(raw_data_payload, timestamp="dummy_data")
    pass


# ----------- Synthetic Data Generators -----------

class SyntheticDataGenerator:
    """Generate synthetic social media and news data"""

    def __init__(self):
        self.fake = Faker(['en_IN', 'hi_IN'])

        # Indian cities and states
        self.indian_locations = [
            "Mumbai", "Delhi", "Bangalore", "Hyderabad", "Ahmedabad", "Chennai",
            "Kolkata", "Surat", "Pune", "Jaipur", "Lucknow", "Kanpur", "Nagpur",
            "Visakhapatnam", "Indore", "Thane", "Bhopal", "Pimpri-Chinchwad",
            "Patna", "Vadodara", "Ghaziabad", "Ludhiana", "Agra", "Nashik"
        ]

        # Common issues and topics
        self.civic_issues = [
            "road repair", "water shortage", "power outage", "garbage collection",
            "street lighting", "traffic congestion", "public transport", "healthcare",
            "education", "corruption", "pollution", "monsoon flooding", "housing"
        ]

        self.news_categories = [
            "Politics", "Sports", "Entertainment", "Technology", "Business",
            "Health", "Education", "Environment", "Crime", "Infrastructure"
        ]

        self.sentiment_words = {
            "positive": ["great", "excellent", "amazing", "wonderful", "fantastic", "good", "happy"],
            "negative": ["terrible", "awful", "bad", "horrible", "disappointed", "angry", "frustrated"],
            "neutral": ["okay", "fine", "normal", "regular", "standard", "average"]
        }

    def generate_social_media_data(self, num_records: int = 1000) -> pd.DataFrame:
        """Generate synthetic social media posts"""
        data = []

        for _ in range(num_records):
            # Random timestamp within last 30 days
            timestamp = self.fake.date_time_between(start_date='-30d', end_date='now')

            # Random user
            user_id = f"user_{random.randint(1000, 99999)}"

            # Random location
            location = random.choice(self.indian_locations)

            # Generate text based on civic issues
            issue = random.choice(self.civic_issues)
            sentiment = random.choice(["positive", "negative", "neutral"])
            sentiment_word = random.choice(self.sentiment_words[sentiment])

            # Generate realistic post text
            post_templates = [
                f"The {issue} situation in {location} is {sentiment_word}. #civic #india",
                f"Can someone help with {issue} problem in {location}? It's really {sentiment_word}.",
                f"Update on {issue} in {location} - things are looking {sentiment_word} now!",
                f"Why is {issue} always {sentiment_word} in {location}? Need better governance.",
                f"Grateful for the {issue} improvements in {location}. Feels {sentiment_word}!"
            ]

            text = random.choice(post_templates)

            # Add engagement metrics
            likes = random.randint(0, 500)
            shares = random.randint(0, 100)
            comments = random.randint(0, 50)

            data.append({
                'timestamp': timestamp,
                'user_id': user_id,
                'text': text,
                'location_approx': location,
                'sentiment': sentiment,
                'issue_category': issue,
                'likes': likes,
                'shares': shares,
                'comments': comments,
                'engagement_score': likes + shares * 2 + comments * 3
            })

        return pd.DataFrame(data)

    def generate_news_data(self, num_records: int = 500) -> pd.DataFrame:
        """Generate synthetic news articles"""
        data = []

        for _ in range(num_records):
            timestamp = self.fake.date_time_between(start_date='-30d', end_date='now')

            source = random.choice([
                "Times of India", "Hindu", "Indian Express", "NDTV", "CNN-News18",
                "Republic TV", "Aaj Tak", "ABP News", "Zee News", "News18"
            ])

            category = random.choice(self.news_categories)
            location = random.choice(self.indian_locations)

            # Generate headline and content based on category
            if category == "Politics":
                headlines = [
                    f"New policy announced for {location}",
                    f"Election preparations underway in {location}",
                    f"Political rally draws thousands in {location}"
                ]
            elif category == "Infrastructure":
                headlines = [
                    f"New metro line approved for {location}",
                    f"Road construction project launched in {location}",
                    f"Smart city initiative expands to {location}"
                ]
            else:
                headlines = [
                    f"{category} news from {location}",
                    f"Latest {category} updates in {location}",
                    f"{category} development in {location}"
                ]

            headline = random.choice(headlines)

            # Generate content
            content = f"{headline}. " + " ".join([
                self.fake.sentence() for _ in range(random.randint(3, 8))
            ])

            # Add credibility score
            credibility = random.uniform(0.3, 1.0)

            data.append({
                'timestamp': timestamp,
                'source': source,
                'headline': headline,
                'content': content,
                'category': category,
                'location': location,
                'credibility_score': credibility,
                'word_count': len(content.split()),
                'has_image': random.choice([True, False]),
                'shares': random.randint(10, 1000)
            })

        return pd.DataFrame(data)

    def generate_crowdsourced_reports(self, num_records: int = 300) -> pd.DataFrame:
        """Generate synthetic crowdsourced reports"""
        data = []

        report_types = ["complaint", "suggestion", "emergency", "information", "appreciation"]
        urgency_levels = ["low", "medium", "high", "critical"]

        for _ in range(num_records):
            timestamp = self.fake.date_time_between(start_date='-30d', end_date='now')

            report_id = f"RPT_{random.randint(100000, 999999)}"
            report_type = random.choice(report_types)
            location = random.choice(self.indian_locations)
            issue = random.choice(self.civic_issues)
            urgency = random.choice(urgency_levels)

            # Generate description
            descriptions = [
                f"Urgent {issue} issue reported in {location}",
                f"Community feedback on {issue} in {location}",
                f"Request for {issue} improvement in {location}",
                f"Emergency situation regarding {issue} in {location}"
            ]

            description = random.choice(descriptions)

            # Add status and verification
            status = random.choice(["pending", "in_progress", "resolved", "closed"])
            verified = random.choice([True, False])

            data.append({
                'timestamp': timestamp,
                'report_id': report_id,
                'report_type': report_type,
                'description': description,
                'location': location,
                'urgency': urgency,
                'issue_category': issue,
                'status': status,
                'verified': verified,
                'upvotes': random.randint(0, 100),
                'priority_score': random.uniform(0, 1)
            })

        return pd.DataFrame(data)


# ----------- AI Model Preparation -----------

class AIDataPreprocessor:
    """Preprocess data for AI models"""

    def __init__(self):
        self.text_vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        self.label_encoders = {}
        self.scalers = {}

    def prepare_text_features(self, texts: List[str]) -> np.ndarray:
        """Convert text to numerical features"""
        return self.text_vectorizer.fit_transform(texts).toarray()

    def prepare_categorical_features(self, data: pd.DataFrame, categorical_cols: List[str]) -> pd.DataFrame:
        """Encode categorical variables"""
        processed_data = data.copy()

        for col in categorical_cols:
            if col in processed_data.columns:
                if col not in self.label_encoders:
                    self.label_encoders[col] = LabelEncoder()
                processed_data[col] = self.label_encoders[col].fit_transform(processed_data[col].astype(str))

        return processed_data

    def prepare_numerical_features(self, data: pd.DataFrame, numerical_cols: List[str]) -> pd.DataFrame:
        """Scale numerical features"""
        processed_data = data.copy()

        for col in numerical_cols:
            if col in processed_data.columns:
                if col not in self.scalers:
                    self.scalers[col] = StandardScaler()
                processed_data[col] = self.scalers[col].fit_transform(processed_data[[col]])

        return processed_data

    def create_ai_dataset(self, social_media_df: pd.DataFrame, news_df: pd.DataFrame,
                          reports_df: pd.DataFrame) -> Dict[str, Any]:
        """Create consolidated dataset for AI training"""

        # Process social media data
        if not social_media_df.empty:
            social_features = self.prepare_text_features(social_media_df['text'].fillna(''))
            social_df = self.prepare_categorical_features(
                social_media_df, ['sentiment', 'issue_category', 'location_approx']
            )
            social_df = self.prepare_numerical_features(
                social_df, ['likes', 'shares', 'comments', 'engagement_score']
            )

        # Process news data
        if not news_df.empty:
            news_features = self.prepare_text_features(news_df['content'].fillna(''))
            news_df = self.prepare_categorical_features(
                news_df, ['category', 'source', 'location']
            )
            news_df = self.prepare_numerical_features(
                news_df, ['credibility_score', 'word_count', 'shares']
            )

        # Process reports data
        if not reports_df.empty:
            reports_features = self.prepare_text_features(reports_df['description'].fillna(''))
            reports_df = self.prepare_categorical_features(
                reports_df, ['report_type', 'urgency', 'issue_category', 'status', 'location']
            )
            reports_df = self.prepare_numerical_features(
                reports_df, ['upvotes', 'priority_score']
            )

        return {
            'social_media': {
                'features': social_features if not social_media_df.empty else np.array([]),
                'data': social_df if not social_media_df.empty else pd.DataFrame(),
                'text_vectorizer': self.text_vectorizer
            },
            'news': {
                'features': news_features if not news_df.empty else np.array([]),
                'data': news_df if not news_df.empty else pd.DataFrame(),
            },
            'reports': {
                'features': reports_features if not reports_df.empty else np.array([]),
                'data': reports_df if not reports_df.empty else pd.DataFrame(),
            },
            'label_encoders': self.label_encoders,
            'scalers': self.scalers
        }


# ----------- Original Loaders (Enhanced) -----------

def load_simulated_social_media_data(file_path: str) -> pd.DataFrame:
    """Loads simulated social media data from a given CSV file path."""
    if not os.path.exists(file_path):
        return pd.DataFrame(columns=['timestamp', 'user_id', 'text', 'location_approx'])
    return pd.read_csv(file_path)


def load_simulated_news_data(file_path: str) -> pd.DataFrame:
    if not os.path.exists(file_path):
        return pd.DataFrame(columns=['timestamp', 'source', 'headline', 'content', 'category'])
    if file_path.endswith('.jsonl'):
        return pd.read_json(file_path, orient='records', lines=True)
    return pd.read_json(file_path, orient='records')


def load_simulated_crowdsourced_reports(file_path: str) -> pd.DataFrame:
    if not os.path.exists(file_path):
        return pd.DataFrame(columns=['timestamp', 'report_id', 'report_type', 'description', 'location', 'urgency'])
    return pd.read_csv(file_path)


def fetch_data_gov_dataset(resource_id: str, limit: int = 1000) -> pd.DataFrame:
    """Fetches a dataset by resource ID from data.gov.in API."""
    endpoint = f"https://api.data.gov.in/resource/{resource_id}?api-key=579b464db66ec23bdd0000011c96fb42a43f4ccd7b511f167c1570a7&format=json&limit={limit}"
    try:
        r = requests.get(endpoint, timeout=30)
        r.raise_for_status()
        records = r.json().get('records', [])
        return pd.DataFrame(records)
    except Exception as e:
        logger.error(f"Error fetching Data.gov.in dataset {resource_id}: {e}")
        return pd.DataFrame()


def fetch_state_portal_data(portal_url: str, resource_path: str, params: Dict[str, Any] = None) -> pd.DataFrame:
    """Fetches data from a state portal's API endpoint."""
    url = portal_url.rstrip('/') + '/' + resource_path.lstrip('/')
    try:
        r = requests.get(url, params=params or {}, timeout=10)
        r.raise_for_status()
        data = r.json()
        records = data.get('data', data)
        return pd.DataFrame(records)
    except Exception as e:
        logger.error(f"Error fetching state portal data from {url}: {e}")
        return pd.DataFrame()


def fetch_ncrb_data(dataset_url: str, params: Dict[str, Any] = None) -> pd.DataFrame:
    """Fetches aggregated crime/disturbance data from NCRB endpoints."""
    try:
        r = requests.get(dataset_url, params=params or {}, timeout=10)
        r.raise_for_status()
        data = r.json()
        return pd.DataFrame(data)
    except Exception as e:
        logger.error(f"Error fetching NCRB data: {e}")
        return pd.DataFrame()


def fetch_census_data(dataset_url: str, params: Dict[str, Any] = None) -> pd.DataFrame:
    """Fetches demographic data from Census API or data dumps."""
    try:
        r = requests.get(dataset_url, params=params or {}, timeout=10)
        r.raise_for_status()
        data = r.json()
        return pd.DataFrame(data)
    except Exception as e:
        logger.error(f"Error fetching Census data: {e}")
        return pd.DataFrame()


def fetch_social_media_trends(api_endpoint: str, keywords: List[str]) -> pd.DataFrame:
    try:
        resp = requests.get(api_endpoint, params={'keywords': ','.join(keywords)}, timeout=10)
        resp.raise_for_status()
        return pd.DataFrame(resp.json().get('trends', []))
    except Exception as e:
        logger.error(f"Error fetching social media trends: {e}")
        return pd.DataFrame()


def fetch_local_news(api_endpoint: str, locale: str) -> pd.DataFrame:
    try:
        resp = requests.get(api_endpoint, params={'locale': locale}, timeout=10)
        resp.raise_for_status()
        return pd.DataFrame(resp.json().get('articles', []))
    except Exception as e:
        logger.error(f"Error fetching local news: {e}")
        return pd.DataFrame()


# ----------- Enhanced Unified Loader -----------

def load_all_data(
        data_dir: str = 'data',
        fetch_live: bool = False,
        generate_synthetic: bool = True,
        sources: Dict[str, Dict[str, Any]] = None,
        synthetic_config: Dict[str, int] = None
) -> Dict[str, pd.DataFrame]:
    """Enhanced loader with synthetic data generation and AI model support"""

    data_manager = DataManager(data_dir)
    out = {}

    # Load existing simulated data
    out['social_media_sim'] = load_simulated_social_media_data(
        os.path.join(data_dir, 'raw', 'simulated_social_media.csv')
    )
    out['news_sim'] = load_simulated_news_data(
        os.path.join(data_dir, 'raw', 'simulated_news.jsonl')
    )
    out['crowd_sim'] = load_simulated_crowdsourced_reports(
        os.path.join(data_dir, 'raw', 'simulated_crowd_reports.csv')
    )

    # Generate synthetic data if requested
    if generate_synthetic:
        synthetic_config = synthetic_config or {
            'social_media': 1000,
            'news': 500,
            'crowdsourced': 300
        }

        generator = SyntheticDataGenerator()

        logger.info("Generating synthetic data...")
        out['social_media_synthetic'] = generator.generate_social_media_data(
            synthetic_config.get('social_media', 1000)
        )
        out['news_synthetic'] = generator.generate_news_data(
            synthetic_config.get('news', 500)
        )
        out['crowd_synthetic'] = generator.generate_crowdsourced_reports(
            synthetic_config.get('crowdsourced', 300)
        )

    # Fetch live data if requested
    if fetch_live and sources:
        logger.info("Fetching live data...")

        if 'pgr' in sources:
            out['public_grievances'] = fetch_data_gov_dataset(**sources['pgr'])
        if 'infrastructure' in sources:
            out['infra_projects'] = fetch_data_gov_dataset(**sources['infrastructure'])
        if 'demographics' in sources:
            out['demographics'] = fetch_data_gov_dataset(**sources['demographics'])
        if 'ncrb' in sources:
            out['ncrb'] = fetch_ncrb_data(**sources['ncrb'])
        if 'census' in sources:
            out['census'] = fetch_census_data(**sources['census'])
        if 'state_portals' in sources:
            for name, cfg in sources['state_portals'].items():
                out[f'state_{name}'] = fetch_state_portal_data(**cfg)
        if 'social_trends' in sources:
            out['social_trends'] = fetch_social_media_trends(**sources['social_trends'])
        if 'local_news' in sources:
            out['local_news'] = fetch_local_news(**sources['local_news'])
        if 'crowd_reports' in sources:
            out['crowd_live'] = fetch_state_portal_data(**sources['crowd_reports'])

    # Save all data
    save_path = data_manager.save_raw_data(out)
    logger.info(f"Data saved to: {save_path}")

    return out


def prepare_ai_dataset(data: Dict[str, pd.DataFrame], data_dir: str = 'data') -> str:
    """Prepare and save dataset for AI model training"""

    data_manager = DataManager(data_dir)
    preprocessor = AIDataPreprocessor()

    # Combine social media data
    social_dfs = [df for name, df in data.items() if 'social_media' in name and not df.empty]
    social_combined = pd.concat(social_dfs, ignore_index=True) if social_dfs else pd.DataFrame()

    # Combine news data
    news_dfs = [df for name, df in data.items() if 'news' in name and not df.empty]
    news_combined = pd.concat(news_dfs, ignore_index=True) if news_dfs else pd.DataFrame()

    # Combine crowdsourced data
    crowd_dfs = [df for name, df in data.items() if 'crowd' in name and not df.empty]
    crowd_combined = pd.concat(crowd_dfs, ignore_index=True) if crowd_dfs else pd.DataFrame()

    # Prepare AI dataset
    ai_dataset = preprocessor.create_ai_dataset(social_combined, news_combined, crowd_combined)

    # Save processed data
    save_path = data_manager.save_processed_data(ai_dataset, "ai_training_dataset")
    logger.info(f"AI dataset prepared and saved: {save_path}")

    return save_path


if __name__ == '__main__':
    # Configuration
    base_dir = "data"

    # Load all data with synthetic generation
    data = load_all_data(
        data_dir=base_dir,
        fetch_live=True,
        generate_synthetic=True,
        sources={
            'pgr': {'resource_id': '15150682-a9ed-475d-b0e3-67b292e90d22'},
            'infrastructure': {'resource_id': '3b01bcb8-0b14-4abf-b6f2-c1bfd384ba69'},
            'demographics': {'resource_id': '232c50b1-b245-4226-bd69-5e42c1fb7b71'},
        },
        synthetic_config={
            'social_media': 2000,
            'news': 1000,
            'crowdsourced': 500
        }
    )

    # Print data summary
    for k, v in data.items():
        print(f"{k}: {len(v)} records")
        if len(v) > 0:
            print(f"  Columns: {list(v.columns)}")
        print()

    # Prepare AI dataset
    ai_dataset_path = prepare_ai_dataset(data, base_dir)
    print(f"AI dataset prepared at: {ai_dataset_path}")

    # Example: Load the prepared AI dataset
    data_manager = DataManager(base_dir)
    ai_data = data_manager.load_processed_data("ai_training_dataset")

    print("\nAI Dataset Summary:")
    for key, value in ai_data.items():
        if isinstance(value, dict) and 'features' in value:
            print(f"{key} features shape: {value['features'].shape}")
        elif isinstance(value, dict):
            print(f"{key}: {len(value)} items")

    print("\nData pipeline setup complete!")