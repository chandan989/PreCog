import re
import logging
from collections import Counter, defaultdict
from typing import List, Dict, Tuple, Any, Optional
import os
import ssl

import nltk
import numpy as np
from textblob import TextBlob
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# Set up logging
logger = logging.getLogger(__name__)

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
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except Exception as e:
    logger.info(f"Downloading NLTK resources: {e}")
    try:
        nltk.download('vader_lexicon', quiet=True)
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
    except Exception as download_error:
        logger.error(f"Error downloading NLTK resources: {download_error}")
        # Create a directory for NLTK data if it doesn't exist
        nltk_data_dir = os.path.expanduser('~/nltk_data')
        os.makedirs(nltk_data_dir, exist_ok=True)
        logger.info(f"Created NLTK data directory: {nltk_data_dir}")

# Enhanced constants for Indian context sentiment analysis
INDIAN_CONTEXT_KEYWORDS = {
    'positive': [
        'अच्छा', 'बढ़िया', 'शानदार', 'उत्कृष्ट', 'महान', 'खुश', 'धन्यवाद', 'स्वागत', 
        'support', 'good', 'great', 'happy', 'thanks', 'excellent', 'wonderful', 'progress',
        'improvement', 'success', 'achievement', 'beneficial', 'helpful', 'impressive',
        'सफलता', 'प्रगति', 'विकास', 'सहायक', 'लाभदायक', 'प्रभावशाली', 'सुधार'
    ],
    'negative': [
        'बुरा', 'खराब', 'समस्या', 'चिंता', 'दुखी', 'angry', 'sad', 'bad', 'problem', 
        'concern', 'issue', 'protest', 'strike', 'bandh', 'failure', 'disappointment',
        'poor', 'terrible', 'horrible', 'awful', 'frustrating', 'annoying', 'useless',
        'असफलता', 'निराशा', 'गरीब', 'भयानक', 'निराशाजनक', 'परेशान', 'बेकार'
    ],
    'neutral': [
        'ठीक', 'सामान्य', 'okay', 'normal', 'average', 'moderate', 'standard', 'typical',
        'regular', 'common', 'usual', 'ordinary', 'मध्यम', 'सामान्य', 'आम'
    ],
    'grievance': [
        'शिकायत', 'पानी नहीं', 'बिजली नहीं', 'सड़क खराब', 'जाम', 'pollution', 'corruption', 
        'scam', 'complaint', 'grievance', 'no water', 'no electricity', 'bad road', 'traffic jam',
        'garbage', 'waste', 'sewage', 'drainage', 'pothole', 'delay', 'overcharge', 'bribe',
        'कचरा', 'अपशिष्ट', 'सीवेज', 'जल निकासी', 'गड्ढा', 'देरी', 'अधिक शुल्क', 'रिश्वत',
        'infrastructure', 'public service', 'government failure', 'civic issue', 'municipal',
        'बुनियादी ढांचा', 'सार्वजनिक सेवा', 'सरकारी विफलता', 'नागरिक मुद्दा', 'नगरपालिका'
    ],
    'social_tension': [
        'दंगा', 'हिंसा', 'तनाव', 'झगड़ा', 'riot', 'violence', 'tension', 'conflict', 'clash', 
        'dispute', 'protest', 'demonstration', 'unrest', 'agitation', 'mob', 'attack',
        'communal', 'sectarian', 'religious', 'caste', 'ethnic', 'discrimination', 'prejudice',
        'सांप्रदायिक', 'धार्मिक', 'जाति', 'जातीय', 'भेदभाव', 'पूर्वाग्रह', 'विरोध', 'प्रदर्शन',
        'अशांति', 'आंदोलन', 'भीड़', 'हमला', 'धार्मिक तनाव', 'जातीय संघर्ष'
    ],
    'civic_improvement': [
        'development', 'initiative', 'scheme', 'project', 'plan', 'proposal', 'investment',
        'infrastructure', 'facility', 'amenity', 'service', 'public', 'community', 'welfare',
        'विकास', 'पहल', 'योजना', 'परियोजना', 'प्रस्ताव', 'निवेश', 'बुनियादी ढांचा', 
        'सुविधा', 'सेवा', 'सार्वजनिक', 'समुदाय', 'कल्याण'
    ],
    'emergency': [
        'emergency', 'urgent', 'critical', 'crisis', 'disaster', 'accident', 'incident',
        'fire', 'flood', 'earthquake', 'storm', 'epidemic', 'outbreak', 'collapse',
        'आपातकालीन', 'तत्काल', 'महत्वपूर्ण', 'संकट', 'आपदा', 'दुर्घटना', 'घटना',
        'आग', 'बाढ़', 'भूकंप', 'तूफान', 'महामारी', 'प्रकोप', 'पतन'
    ]
}

# Enhanced sentiment modifiers with more nuanced negation handling
SENTIMENT_MODIFIERS = {
    # Intensifiers
    'very': 1.5,
    'extremely': 1.8,
    'highly': 1.6,
    'absolutely': 1.7,
    'completely': 1.6,
    'totally': 1.5,
    'अत्यंत': 1.7,
    'बहुत': 1.5,
    'काफी': 1.4,
    'पूरी तरह से': 1.6,

    # Diminishers
    'somewhat': 0.7,
    'slightly': 0.6,
    'a bit': 0.7,
    'a little': 0.6,
    'marginally': 0.5,
    'थोड़ा': 0.6,
    'कुछ': 0.7,
    'थोड़ा सा': 0.5,

    # Negations
    'not': -1.0,
    "don't": -1.0,
    "doesn't": -1.0,
    "didn't": -1.0,
    "won't": -1.0,
    "can't": -1.0,
    "cannot": -1.0,
    "isn't": -1.0,
    "aren't": -1.0,
    "wasn't": -1.0,
    "weren't": -1.0,
    "haven't": -1.0,
    "hasn't": -1.0,
    "hadn't": -1.0,
    "shouldn't": -1.0,
    "wouldn't": -1.0,
    "couldn't": -1.0,
    'never': -1.0,
    'no': -1.0,
    'नहीं': -1.0,
    'ना': -1.0,
    'नही': -1.0,
    'बिलकुल नहीं': -1.0,
    'कभी नहीं': -1.0
}

# Regional language support - common phrases with sentiment values
REGIONAL_PHRASES = {
    # Hindi
    'बहुत अच्छा': 0.8,  # Very good
    'बहुत बुरा': -0.8,   # Very bad
    'ठीक है': 0.2,      # It's okay
    'बहुत खराब': -0.7,   # Very poor
    'मुझे पसंद है': 0.6, # I like it
    'मुझे नापसंद है': -0.6, # I dislike it

    # Tamil
    'மிகவும் நன்று': 0.8, # Very good
    'மிகவும் மோசமானது': -0.8, # Very bad
    'பரவாயில்லை': 0.2, # It's okay

    # Bengali
    'খুব ভালো': 0.8, # Very good
    'খুব খারাপ': -0.8, # Very bad
    'ঠিক আছে': 0.2, # It's okay

    # Telugu
    'చాలా బాగుంది': 0.8, # Very good
    'చాలా చెడ్డది': -0.8, # Very bad
    'పర్వాలేదు': 0.2, # It's okay

    # Marathi
    'खूप छान': 0.8, # Very good
    'खूप वाईट': -0.8, # Very bad
    'ठीक आहे': 0.2, # It's okay
}

# Common civic issues with sentiment values
CIVIC_ISSUES = {
    'water shortage': -0.7,
    'power cut': -0.6,
    'electricity problem': -0.6,
    'road condition': -0.5,
    'garbage collection': -0.6,
    'traffic jam': -0.5,
    'public transport': -0.4,
    'corruption': -0.8,
    'bribe': -0.8,
    'pollution': -0.7,
    'noise': -0.5,
    'crime': -0.8,
    'safety': -0.6,
    'पानी की कमी': -0.7,
    'बिजली कटौती': -0.6,
    'सड़क की स्थिति': -0.5,
    'कचरा संग्रह': -0.6,
    'ट्रैफिक जाम': -0.5,
    'सार्वजनिक परिवहन': -0.4,
    'भ्रष्टाचार': -0.8,
    'रिश्वत': -0.8,
    'प्रदूषण': -0.7,
    'शोर': -0.5,
    'अपराध': -0.8,
    'सुरक्षा': -0.6
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
        """
        Preprocess text for sentiment analysis.
        - Converts to lowercase
        - Removes URLs, mentions, hashtags
        - Preserves Devanagari and English characters
        - Handles emojis (converts to text representation)
        """
        if not text:
            return ""

        # Convert to lowercase
        text = text.lower()

        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)

        # Remove mentions and hashtags for pure sentiment (but store hashtags for topic analysis)
        hashtags = re.findall(r'#\w+', text)
        text = re.sub(r'\@\w+|#\w+', '', text)

        # Keep punctuation for better sentiment analysis (especially for negation detection)
        # But remove excessive punctuation
        text = re.sub(r'([!?.])\1+', r'\1', text)

        # Keep Devanagari, English chars, punctuation, and spaces
        # text = re.sub(r'[^ऀ-ॿA-Za-z\s.,!?;:\'\"-]', '', text)

        # Strip extra whitespace
        text = ' '.join(text.split())

        return text.strip()

    def _extract_keywords(self, text: str) -> List[str]:
        """Extract important keywords from text."""
        # Tokenize
        tokens = word_tokenize(text)

        # Remove stopwords (English)
        try:
            stop_words = set(stopwords.words('english'))
            tokens = [word for word in tokens if word.lower() not in stop_words]
        except:
            logger.warning("Could not load English stopwords, continuing without stopword removal")

        # Count word frequencies
        word_freq = Counter(tokens)

        # Return top keywords (excluding very short words)
        return [word for word, count in word_freq.most_common(10) if len(word) > 2]

    def _get_textblob_sentiment(self, text: str) -> float:
        """Get sentiment score using TextBlob."""
        try:
            return TextBlob(text).sentiment.polarity
        except Exception as e:
            logger.error(f"Error in TextBlob sentiment analysis: {e}")
            return 0.0

    def _get_vader_sentiment(self, text: str) -> float:
        """Get sentiment score using VADER."""
        try:
            return self.sid.polarity_scores(text)['compound']
        except Exception as e:
            logger.error(f"Error in VADER sentiment analysis: {e}")
            return 0.0

    def _check_regional_phrases(self, text: str) -> Optional[float]:
        """Check for regional language phrases with known sentiment values."""
        for phrase, sentiment_value in REGIONAL_PHRASES.items():
            if phrase in text:
                return sentiment_value
        return None

    def _check_civic_issues(self, text: str) -> List[Tuple[str, float]]:
        """Identify civic issues mentioned in the text and their sentiment impact."""
        issues_found = []
        for issue, sentiment_value in CIVIC_ISSUES.items():
            if issue in text:
                issues_found.append((issue, sentiment_value))
        return issues_found

    def _apply_custom_rules(self, text: str, base_sentiment: float) -> Dict[str, Any]:
        """
        Apply custom rules for Indian context sentiment analysis.
        Returns a dictionary with:
        - sentiment_score: The adjusted sentiment score
        - category: The detected category
        - keywords_found: List of keywords found
        """
        sentiment = base_sentiment
        words = text.split()
        modifier_effect = 1.0
        negation_active = False
        negation_window = 0

        # Check for regional phrases first
        regional_sentiment = self._check_regional_phrases(text)
        if regional_sentiment is not None:
            # If a strong regional phrase is found, weight it heavily
            sentiment = (sentiment + regional_sentiment * 2) / 3

        # Check for civic issues
        civic_issues = self._check_civic_issues(text)
        for issue, issue_sentiment in civic_issues:
            sentiment = (sentiment + issue_sentiment) / 2

        # Process modifiers and negations with a sliding window
        for i, word in enumerate(words):
            # Handle negation window
            if negation_active and negation_window > 0:
                negation_window -= 1
                if negation_window == 0:
                    negation_active = False

            # Check for modifiers
            if word in SENTIMENT_MODIFIERS:
                if SENTIMENT_MODIFIERS[word] < 0:  # Negation
                    negation_active = True
                    negation_window = min(4, len(words) - i - 1)  # Look ahead up to 4 words
                else:
                    modifier_effect *= SENTIMENT_MODIFIERS[word]

        # Apply modifier effect
        if negation_active:
            sentiment *= -0.7  # Negation doesn't always completely reverse sentiment
        else:
            sentiment *= modifier_effect

        # Track keywords found and their categories
        keywords_found = []
        categories_found = Counter()

        # Check for keywords in each category
        for category, keywords in INDIAN_CONTEXT_KEYWORDS.items():
            for keyword in keywords:
                if keyword in text:
                    keywords_found.append(keyword)
                    categories_found[category] += 1

                    # Adjust sentiment based on category
                    if category == 'positive': 
                        sentiment += 0.1
                    elif category == 'negative': 
                        sentiment -= 0.1
                    elif category == 'grievance': 
                        sentiment -= 0.15  # Grievances are typically negative
                    elif category == 'social_tension': 
                        sentiment -= 0.25  # Social tension is strongly negative
                    elif category == 'emergency':
                        sentiment -= 0.2   # Emergencies are negative
                    elif category == 'civic_improvement':
                        sentiment += 0.05  # Civic improvements are slightly positive

        # Determine the primary category
        primary_category = 'general'
        if categories_found:
            primary_category = categories_found.most_common(1)[0][0]

        # Ensure sentiment is within bounds
        final_sentiment = np.clip(sentiment, -1.0, 1.0)

        return {
            'sentiment_score': final_sentiment,
            'category': primary_category,
            'keywords_found': list(set(keywords_found))
        }

    def analyze_single_text(self, text: str, use_llm: bool = False) -> Dict[str, Any]:
        """
        Analyzes a single text for sentiment, category, and keywords.

        Args:
            text: The text to analyze
            use_llm: Whether to use LLM for enhanced analysis

        Returns:
            Dictionary with sentiment analysis results including:
            - sentiment_score: Float between -1.0 (negative) and 1.0 (positive)
            - sentiment_label: String label (positive, negative, neutral)
            - category: Detected category of the text
            - keywords: List of important keywords
            - civic_issues: List of civic issues mentioned (if any)
            - regional_context: Detected regional context (if any)
        """
        # Handle invalid input
        if not text or not isinstance(text, str):
            return {
                'text': text,
                'sentiment_score': 0.0,
                'sentiment_label': 'neutral',
                'category': 'unknown',
                'keywords': [],
                'civic_issues': [],
                'regional_context': None,
                'processed_text': '',
                'error': 'Input text is empty or not a string'
            }

        # Preprocess text
        processed_text = self._preprocess_text(text)

        # Try LLM analysis if requested and available
        if use_llm and self.llm_client:
            logger.info(f"Using LLM for sentiment analysis of: {processed_text[:50]}...")
            llm_analysis = self.analyze_text_with_llm(processed_text)

            if llm_analysis:  # If LLM provides a valid analysis, use it
                # Extract keywords if not provided by LLM
                keywords = llm_analysis.get('keywords', [])
                if not keywords:
                    keywords = self._extract_keywords(processed_text)

                # Check for civic issues regardless of LLM analysis
                civic_issues = self._check_civic_issues(processed_text)
                civic_issue_names = [issue[0] for issue in civic_issues]

                # Detect regional context
                regional_context = None
                for phrase in REGIONAL_PHRASES:
                    if phrase in processed_text:
                        # Determine language based on the phrase
                        if any(char in phrase for char in 'अआइईउऊएऐओऔकखगघ'):
                            regional_context = 'Hindi'
                        elif any(char in phrase for char in 'அஆஇஈஉஊஎஏஐஒஓஔ'):
                            regional_context = 'Tamil'
                        elif any(char in phrase for char in 'অআইঈউঊএঐওঔকখগঘ'):
                            regional_context = 'Bengali'
                        elif any(char in phrase for char in 'అఆఇఈఉఊఎఏఐఒఓఔకఖగఘ'):
                            regional_context = 'Telugu'
                        elif any(char in phrase for char in 'अआइईउऊएऐओऔकखगघ'):
                            regional_context = 'Marathi'
                        break

                return {
                    'text': text,
                    'sentiment_score': llm_analysis.get('sentiment_score', 0.0),
                    'sentiment_label': llm_analysis.get('sentiment_label', 'neutral'),
                    'category': llm_analysis.get('category', 'unknown'),
                    'keywords': keywords,
                    'civic_issues': civic_issue_names,
                    'regional_context': regional_context,
                    'processed_text': processed_text,
                    'analysis_method': 'llm'
                }
            else:
                logger.warning(f"LLM analysis failed for text: {text[:50]}... Falling back to rule-based.")

        # Rule-based analysis (fallback or default)
        logger.info(f"Using rule-based sentiment analysis for: {processed_text[:50]}...")

        # Get base sentiment scores
        tb_sentiment = self._get_textblob_sentiment(processed_text)
        vader_sentiment = self._get_vader_sentiment(processed_text)

        # Weight VADER more heavily for social media text
        combined_sentiment = (tb_sentiment + vader_sentiment * 2) / 3

        # Apply custom rules and get enhanced results
        custom_analysis = self._apply_custom_rules(processed_text, combined_sentiment)
        final_sentiment = custom_analysis['sentiment_score']
        category = custom_analysis['category']
        keywords_found = custom_analysis['keywords_found']

        # Extract additional keywords using NLP
        extracted_keywords = self._extract_keywords(processed_text)
        all_keywords = list(set(keywords_found + extracted_keywords))

        # Check for civic issues
        civic_issues = self._check_civic_issues(processed_text)
        civic_issue_names = [issue[0] for issue in civic_issues]

        # Determine sentiment label
        if final_sentiment > 0.3:
            sentiment_label = 'positive'
        elif final_sentiment < -0.3:
            sentiment_label = 'negative'
        else:
            sentiment_label = 'neutral'

        # Detect regional context
        regional_context = None
        for phrase in REGIONAL_PHRASES:
            if phrase in processed_text:
                # Determine language based on the phrase
                if any(char in phrase for char in 'अआइईउऊएऐओऔकखगघ'):
                    regional_context = 'Hindi'
                elif any(char in phrase for char in 'அஆஇஈஉஊஎஏஐஒஓஔ'):
                    regional_context = 'Tamil'
                elif any(char in phrase for char in 'অআইঈউঊএঐওঔকখগঘ'):
                    regional_context = 'Bengali'
                elif any(char in phrase for char in 'అఆఇఈఉఊఎఏఐఒఓఔకఖగఘ'):
                    regional_context = 'Telugu'
                elif any(char in phrase for char in 'अआइईउऊएऐओऔकखगघ'):
                    regional_context = 'Marathi'
                break

        return {
            'text': text,
            'sentiment_score': final_sentiment,
            'sentiment_label': sentiment_label,
            'category': category,
            'keywords': all_keywords[:10],  # Limit to top 10 keywords
            'civic_issues': civic_issue_names,
            'regional_context': regional_context,
            'processed_text': processed_text,
            'analysis_method': 'rule_based'
        }

    def analyze_text_batch(self, texts: List[str], locations: List[str], use_llm: bool = False) -> Dict[str, Dict[str, Any]]:
        """
        Analyze a batch of texts with their associated locations.

        Args:
            texts: List of text strings to analyze
            locations: List of location names corresponding to each text
            use_llm: Whether to use LLM for enhanced analysis

        Returns:
            Dictionary with aggregated results by location
        """
        if not texts:
            logger.warning("Empty text list provided for batch analysis")
            return {}

        # Initialize results structure with enhanced metrics
        results_by_location = defaultdict(lambda: {
            'total_posts': 0, 
            'sentiment_scores': [], 
            'issues': Counter(),
            'categories': Counter(),
            'civic_issues': Counter(),
            'regional_contexts': Counter(),
            'keywords': Counter(),
            'positive_count': 0,
            'negative_count': 0,
            'neutral_count': 0,
            'texts': []  # Store sample texts for reference
        })

        # Process each text
        for i, text in enumerate(texts):
            location = locations[i] if i < len(locations) else "Unknown"

            # Get comprehensive analysis
            analysis = self.analyze_single_text(text, use_llm=use_llm)

            # Update location-specific metrics
            loc_data = results_by_location[location]
            loc_data['total_posts'] += 1
            loc_data['sentiment_scores'].append(analysis['sentiment_score'])

            # Track sentiment distribution
            sentiment_label = analysis['sentiment_label']
            if sentiment_label == 'positive':
                loc_data['positive_count'] += 1
            elif sentiment_label == 'negative':
                loc_data['negative_count'] += 1
            else:
                loc_data['neutral_count'] += 1

            # Track categories
            loc_data['categories'][analysis['category']] += 1

            # Track civic issues
            for issue in analysis.get('civic_issues', []):
                loc_data['civic_issues'][issue] += 1

            # Track regional context
            if analysis.get('regional_context'):
                loc_data['regional_contexts'][analysis['regional_context']] += 1

            # Track keywords
            for keyword in analysis.get('keywords', []):
                loc_data['keywords'][keyword] += 1

            # Store sample texts (up to 5 per location)
            if len(loc_data['texts']) < 5:
                loc_data['texts'].append({
                    'text': text[:100] + ('...' if len(text) > 100 else ''),
                    'sentiment_score': analysis['sentiment_score'],
                    'sentiment_label': sentiment_label,
                    'category': analysis['category']
                })

            # Track issues based on keywords (for backward compatibility)
            for kw_type, kw_list in INDIAN_CONTEXT_KEYWORDS.items():
                if kw_type in ['grievance', 'social_tension', 'emergency']:
                    for kw in kw_list:
                        if kw in analysis.get('keywords', []):
                            loc_data['issues'][kw] += 1

        # Aggregate results for each location
        aggregated_results = {}
        for loc, data in results_by_location.items():
            # Calculate average sentiment
            avg_sentiment = np.mean(data['sentiment_scores']) if data['sentiment_scores'] else 0

            # Calculate alert level using enhanced method
            alert_level = self._calculate_alert_level(
                avg_sentiment, 
                data['total_posts'], 
                data['issues'],
                data['civic_issues'],
                data['categories']
            )

            # Get top issues, categories, and keywords
            top_issues = data['issues'].most_common(5)
            top_civic_issues = data['civic_issues'].most_common(5)
            top_categories = data['categories'].most_common(3)
            top_keywords = data['keywords'].most_common(10)

            # Calculate sentiment distribution
            sentiment_distribution = np.histogram(
                data['sentiment_scores'], 
                bins=5, 
                range=(-1, 1)
            )[0].tolist() if data['sentiment_scores'] else [0, 0, 0, 0, 0]

            # Calculate sentiment trend (placeholder - in a real system, this would compare to historical data)
            sentiment_trend = 'stable'
            if avg_sentiment < -0.3:
                sentiment_trend = 'declining'
            elif avg_sentiment > 0.3:
                sentiment_trend = 'improving'

            # Determine dominant regional context if any
            dominant_regional_context = None
            if data['regional_contexts']:
                dominant_regional_context = data['regional_contexts'].most_common(1)[0][0]

            # Create comprehensive result
            aggregated_results[loc] = {
                'avg_sentiment': avg_sentiment,
                'total_posts': data['total_posts'],
                'alert_level': alert_level,
                'sentiment_trend': sentiment_trend,
                'top_issues': top_issues,
                'top_civic_issues': top_civic_issues,
                'top_categories': top_categories,
                'top_keywords': top_keywords,
                'sentiment_distribution': sentiment_distribution,
                'sentiment_counts': {
                    'positive': data['positive_count'],
                    'negative': data['negative_count'],
                    'neutral': data['neutral_count']
                },
                'dominant_regional_context': dominant_regional_context,
                'sample_texts': data['texts']
            }

        return aggregated_results

    def _calculate_alert_level(self, avg_sentiment: float, num_posts: int, issues: Counter, 
                              civic_issues: Counter = None, categories: Counter = None) -> str:
        """
        Calculate alert level based on multiple factors.

        Args:
            avg_sentiment: Average sentiment score
            num_posts: Number of posts
            issues: Counter of issues
            civic_issues: Counter of civic issues
            categories: Counter of categories

        Returns:
            Alert level as string: 'high', 'medium', 'low', or 'normal'
        """
        # Initialize score-based system (0-100)
        alert_score = 0

        # Factor 1: Sentiment intensity and volume
        if avg_sentiment < -0.7:
            alert_score += 40
        elif avg_sentiment < -0.5:
            alert_score += 30
        elif avg_sentiment < -0.3:
            alert_score += 20
        elif avg_sentiment < -0.1:
            alert_score += 10

        # Factor 2: Volume amplification
        if num_posts > 50:
            alert_score += 20
        elif num_posts > 20:
            alert_score += 15
        elif num_posts > 10:
            alert_score += 10
        elif num_posts > 5:
            alert_score += 5

        # Factor 3: Critical issues
        social_tension_count = sum(issues.get(kw, 0) for kw_cat in ['social_tension'] 
                                  for kw in INDIAN_CONTEXT_KEYWORDS.get(kw_cat, []))
        emergency_count = sum(issues.get(kw, 0) for kw_cat in ['emergency'] 
                             for kw in INDIAN_CONTEXT_KEYWORDS.get(kw_cat, []))

        if social_tension_count > 5:
            alert_score += 30
        elif social_tension_count > 3:
            alert_score += 20
        elif social_tension_count > 1:
            alert_score += 10

        if emergency_count > 3:
            alert_score += 30
        elif emergency_count > 1:
            alert_score += 15

        # Factor 4: Civic issues (if available)
        if civic_issues:
            critical_civic_issues = ['water shortage', 'power cut', 'corruption', 'crime', 'safety',
                                    'पानी की कमी', 'बिजली कटौती', 'भ्रष्टाचार', 'अपराध', 'सुरक्षा']
            critical_count = sum(civic_issues.get(issue, 0) for issue in critical_civic_issues)

            if critical_count > 5:
                alert_score += 20
            elif critical_count > 2:
                alert_score += 10

        # Factor 5: Categories (if available)
        if categories:
            if categories.get('social_tension', 0) > 3:
                alert_score += 15
            if categories.get('emergency', 0) > 2:
                alert_score += 15
            if categories.get('grievance', 0) > 5:
                alert_score += 10

        # Determine alert level based on score
        if alert_score >= 60:
            return 'high'
        elif alert_score >= 30:
            return 'medium'
        elif alert_score >= 15:
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
