import re
from collections import Counter, defaultdict
from typing import List, Dict, Tuple, Any, Optional
import os
import ssl

import nltk
import numpy as np
from textblob import TextBlob
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Ensure NLTK resources are available
try:
    # Try to create an unverified SSL context to work around certificate issues
    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        pass
    else:
        ssl._create_default_https_context = _create_unverified_https_context
    
    nltk.data.find('sentiment/vader_lexicon.zip')
except Exception as e:
    print(f"Attempting to download vader_lexicon: {e}")
    try:
        nltk.download('vader_lexicon', quiet=True)
    except Exception as download_error:
        print(f"Error downloading vader_lexicon: {download_error}")
        # Create a directory for NLTK data if it doesn't exist
        nltk_data_dir = os.path.expanduser('~/nltk_data/sentiment')
        os.makedirs(nltk_data_dir, exist_ok=True)
        print(f"Created NLTK data directory: {nltk_data_dir}")

try:
    nltk.data.find('tokenizers/punkt')
except Exception as e:
    print(f"Attempting to download punkt: {e}")
    try:
        nltk.download('punkt', quiet=True)
    except Exception as download_error:
        print(f"Error downloading punkt: {download_error}")

# Constants for Indian context sentiment analysis
INDIAN_CONTEXT_KEYWORDS = {
    'positive': ['अच्छा', 'बढ़िया', 'शानदार', 'उत्कृष्ट', 'महान', 'खुश', 'धन्यवाद', 'स्वागत', 'support', 'good', 'great', 'happy', 'thanks', 'excellent', 'wonderful'],
    'negative': ['बुरा', 'खराब', 'समस्या', 'चिंता', 'दुखी', 'angry', 'sad', 'bad', 'problem', 'concern', 'issue', 'protest', 'strike', 'bandh'],
    'neutral': ['ठीक', 'सामान्य', 'okay', 'normal', 'average'],
    'grievance': ['शिकायत', 'पानी नहीं', 'बिजली नहीं', 'सड़क खराब', 'जाम', 'pollution', 'corruption', 'scam', 'complaint', 'grievance', 'no water', 'no electricity', 'bad road', 'traffic jam'],
    'social_tension': ['दंगा', 'हिंसा', 'तनाव', 'झगड़ा', 'riot', 'violence', 'tension', 'conflict', 'clash', 'dispute']
}

SENTIMENT_MODIFIERS = {
    'very': 1.5,
    'अत्यंत': 1.5,
    'थोड़ा': 0.5,
    'somewhat': 0.7,
    'not': -1.0, # Basic negation handling
    'नहीं': -1.0
}

from ..config.config import get_config # Adjusted import path
from ..core.llm_clients import BaseLLMClient # New import for LLM

class HyperlocalSentimentAnalyzer:
    def __init__(self, lang: str = 'en_hinglish', custom_rules: Optional[Dict] = None, llm_client: Optional[BaseLLMClient] = None):
        self.lang = lang
        self.sid = SentimentIntensityAnalyzer()
        self.custom_rules = custom_rules if custom_rules else {}
        self.llm_client = llm_client # Store LLM client

        # Ensure NLTK data is available
        
    def _preprocess_text(self, text: str) -> str:
        text = text.lower()
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE) # Remove URLs
        text = re.sub(r'\@\w+|#\w+', '', text) # Remove mentions and hashtags for pure sentiment
        text = re.sub(r'[^ऀ-ॿA-Za-z\s]', '', text) # Keep Devanagari, English chars and spaces
        text = text.strip()
        return text

    def _get_textblob_sentiment(self, text: str) -> float:
        # TextBlob primarily supports English. For Hinglish, it might be less accurate.
        # Consider using a multilingual model for better Hinglish/Indian language support.
        return TextBlob(text).sentiment.polarity

    def _get_vader_sentiment(self, text: str) -> float:
        # VADER is good for English social media text, includes emojis, etc.
        return self.sid.polarity_scores(text)['compound']

    def _apply_custom_rules(self, text: str, base_sentiment: float) -> float:
        sentiment = base_sentiment
        words = text.split()
        modifier_effect = 1.0

        for i, word in enumerate(words):
            if word in SENTIMENT_MODIFIERS:
                if SENTIMENT_MODIFIERS[word] < 0: # Negation
                    # Basic negation: flip sentiment of next few words or clause
                    # This is a simplification; true negation is complex.
                    sentiment *= SENTIMENT_MODIFIERS[word]
                else:
                    modifier_effect *= SENTIMENT_MODIFIERS[word]

        sentiment *= modifier_effect

        for category, keywords in INDIAN_CONTEXT_KEYWORDS.items():
            for keyword in keywords:
                if keyword in text:
                    if category == 'positive': sentiment += 0.1
                    elif category == 'negative': sentiment -= 0.1
                    elif category == 'grievance': sentiment -= 0.15 # Grievances are typically negative
                    elif category == 'social_tension': sentiment -= 0.25 # Social tension is strongly negative
        return np.clip(sentiment, -1.0, 1.0)

    def analyze_single_text(self, text: str, use_llm: bool = False) -> Dict[str, Any]:
        """Analyzes a single text for sentiment, category, and keywords."""
        if not text or not isinstance(text, str):
            return {
                'text': text,
                'sentiment_score': 0.0,
                'sentiment_label': 'neutral',
                'category': 'unknown',
                'keywords': [],
                'processed_text': '',
                'error': 'Input text is empty or not a string'
            }

        processed_text = self._preprocess_text(text)

        if use_llm and self.llm_client:
            llm_analysis = self.analyze_text_with_llm(processed_text)
            if llm_analysis: # If LLM provides a valid analysis, use it
                return {
                    'text': text,
                    'sentiment_score': llm_analysis.get('sentiment_score', 0.0),
                    'sentiment_label': llm_analysis.get('sentiment_label', 'neutral'),
                    'category': llm_analysis.get('category', 'unknown'),
                    'keywords': llm_analysis.get('keywords', self._extract_keywords(processed_text)), # Fallback for keywords
                    'processed_text': processed_text
                }
            else:
                logger.warning(f"LLM analysis failed for text: {text[:50]}... Falling back to rule-based.")

        # Fallback to existing rule-based analysis
        tb_sentiment = self._get_textblob_sentiment(processed_text)
        vader_sentiment = self._get_vader_sentiment(processed_text)

        # Simple averaging, can be weighted based on model performance on validation data
        combined_sentiment = (tb_sentiment + vader_sentiment) / 2

        # Apply custom rules and Indian context keywords
        final_sentiment = self._apply_custom_rules(processed_text, combined_sentiment)

        # Categorize sentiment
        if final_sentiment > 0.3:
            category = 'positive'
        elif final_sentiment < -0.3:
            category = 'negative'
        else:
            category = 'neutral'

        # Extract keywords (simple match for now)
        keywords_found = []
        for cat, kw_list in INDIAN_CONTEXT_KEYWORDS.items():
            for kw in kw_list:
                if kw in processed_text:
                    keywords_found.append(kw)

        return {
            'sentiment_score': final_sentiment,
            'sentiment_label': category, # Use category as sentiment_label
            'category': category,
            'keywords_found': list(set(keywords_found)) # Unique keywords
        }

    def analyze_text_batch(self, texts: List[str], locations: List[str]) -> Dict[str, Dict[str, Any]]:
        results_by_location = defaultdict(lambda: {'total_posts': 0, 'sentiment_scores': [], 'issues': Counter()})

        for i, text in enumerate(texts):
            location = locations[i] if i < len(locations) else "Unknown"
            analysis = self.analyze_single_text(text)

            results_by_location[location]['total_posts'] += 1
            results_by_location[location]['sentiment_scores'].append(analysis['sentiment_score'])

            # Track issues based on keywords
            for kw_type, kw_list in INDIAN_CONTEXT_KEYWORDS.items():
                if kw_type in ['grievance', 'social_tension']:
                    for kw in kw_list:
                        if kw in analysis['keywords_found']:
                            results_by_location[location]['issues'][kw] +=1

        aggregated_results = {}
        for loc, data in results_by_location.items():
            avg_sentiment = np.mean(data['sentiment_scores']) if data['sentiment_scores'] else 0
            alert_level = self._calculate_alert_level(avg_sentiment, data['total_posts'], data['issues'])
            top_issues = data['issues'].most_common(3)

            aggregated_results[loc] = {
                'avg_sentiment': avg_sentiment,
                'total_posts': data['total_posts'],
                'alert_level': alert_level,
                'top_issues': top_issues,
                'sentiment_distribution': np.histogram(data['sentiment_scores'], bins=5, range=(-1,1))[0].tolist() if data['sentiment_scores'] else [0,0,0,0,0]
            }
        return aggregated_results

    def _calculate_alert_level(self, avg_sentiment: float, num_posts: int, issues: Counter) -> str:
        # More sophisticated alerting based on volume, velocity, and specific keywords
        if avg_sentiment < -0.5 and num_posts > 10:
            return 'high'
        elif avg_sentiment < -0.3 and num_posts > 5:
            return 'medium'
        elif any(issues.get(kw, 0) > 3 for kw_cat in ['social_tension'] for kw in INDIAN_CONTEXT_KEYWORDS.get(kw_cat,[])):
             return 'high' # Elevated alert for social tension keywords
        elif avg_sentiment < -0.1:
            return 'low'
        return 'normal'

    def analyze_text_with_llm(self, text: str) -> Optional[Dict[str, Any]]:
        """Analyzes text using an LLM for richer sentiment and categorization."""
        if not self.llm_client:
            logger.info("LLM client not available for sentiment analysis.")
            return None

        prompt = f"""
        Analyze the sentiment of the following text from a social media post in an Indian context.
        Provide the overall sentiment score (between -1.0 for very negative and 1.0 for very positive).
        Provide a sentiment label (positive, negative, neutral).
        Categorize the text into one of the following: 'general_statement', 'civic_grievance', 'social_tension_indicator', 'misinformation_rumor', 'positive_feedback', 'other'.
        Extract up to 5 key terms or phrases that are most relevant to the sentiment and category.
        
        Text: "{text}"
        
        Return your analysis as a JSON object with keys: "sentiment_score", "sentiment_label", "category", "keywords".
        Example for keywords: ["term1", "term2"]
        """
        try:
            response_text = self.llm_client.generate_text(prompt, max_tokens=250)
            if response_text:
                import json
                try:
                    # Handle potential markdown code block ```json ... ``` in LLM response
                    if response_text.strip().startswith("```json"):
                        response_text = response_text.strip()[7:-3].strip()
                    elif response_text.strip().startswith("```"):
                         response_text = response_text.strip()[3:-3].strip()

                    analysis_result = json.loads(response_text)
                    
                    # Basic validation of expected keys
                    if not all(k in analysis_result for k in ["sentiment_score", "sentiment_label", "category", "keywords"]):
                        logger.error(f"LLM response missing expected keys: {analysis_result}")
                        return None # Or handle partial data
                        
                    # Ensure score is float, label and category are strings, keywords is list
                    if not isinstance(analysis_result["sentiment_score"], (float, int)):
                        logger.warning(f"LLM sentiment_score is not a number: {analysis_result['sentiment_score']}. Defaulting to 0.0")
                        analysis_result["sentiment_score"] = 0.0 
                    if not isinstance(analysis_result["sentiment_label"], str):
                        logger.warning(f"LLM sentiment_label is not a string: {analysis_result['sentiment_label']}. Defaulting to 'neutral'")
                        analysis_result["sentiment_label"] = "neutral"
                    if not isinstance(analysis_result["category"], str):
                        logger.warning(f"LLM category is not a string: {analysis_result['category']}. Defaulting to 'other'")
                        analysis_result["category"] = "other"
                    if not isinstance(analysis_result["keywords"], list):
                        logger.warning(f"LLM keywords is not a list: {analysis_result['keywords']}. Defaulting to []")
                        analysis_result["keywords"] = []
                    return analysis_result
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse LLM JSON response: {e}. Response: {response_text}")
            else:
                logger.warning("LLM returned empty response for sentiment analysis.")
            return None
        except Exception as e:
            logger.error(f"Error during LLM sentiment analysis: {e}")
            return None