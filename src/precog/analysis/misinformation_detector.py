from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import re
from collections import defaultdict
import logging

from ..core.data_models import MisinformationAlert, NewsArticle # Ensure NewsArticle is imported if used here
from ..core.llm_clients import BaseLLMClient # New import for LLM

# Set up logging
logger = logging.getLogger(__name__)

# Enhanced Constants for Misinformation Detection (Indian Context)
MISINFO_INDICATORS_INDIAN_CONTEXT = {
    'sensational_language': [
        'SHOCKING', 'BREAKING', 'EXCLUSIVE', '!', 'MUST WATCH', 'ज़रूर देखें', 'सनसनीखेज',
        'UNBELIEVABLE', 'MIND-BLOWING', 'NEVER SEEN BEFORE', 'BANNED', 'CENSORED',
        'अविश्वसनीय', 'दिमाग हिला देने वाला', 'पहले कभी नहीं देखा', 'प्रतिबंधित', 'सेंसर किया गया',
        'VIRAL', 'EXPOSED', 'LEAKED', 'SECRET', 'HIDDEN TRUTH', 'THEY DON\'T WANT YOU TO KNOW',
        'वायरल', 'भंडाफोड़', 'लीक', 'रहस्य', 'छिपा सच', 'वे नहीं चाहते कि आप जानें'
    ],

    'urgency_claims': [
        'SHARE NOW', 'FORWARDED AS RECEIVED', 'URGENT', 'तुरंत शेयर करें',
        'SHARE BEFORE DELETED', 'SHARE WITH EVERYONE YOU KNOW', 'DON\'T IGNORE',
        'हटाए जाने से पहले शेयर करें', 'अपने सभी परिचितों के साथ शेयर करें', 'अनदेखा न करें',
        'IMPORTANT ALERT', 'EMERGENCY BROADCAST', 'FORWARD TO ALL GROUPS',
        'महत्वपूर्ण अलर्ट', 'आपातकालीन प्रसारण', 'सभी ग्रुप्स को फॉरवर्ड करें',
        'BEFORE IT\'S TOO LATE', 'TIME SENSITIVE', 'ACT NOW',
        'देर होने से पहले', 'समय संवेदनशील', 'अभी कार्रवाई करें'
    ],

    'unverified_sources': [
        'sources say', 'according to a WhatsApp forward', 'सूत्रों के अनुसार',
        'I heard from', 'someone told me', 'my friend who works at', 'insider information',
        'मैंने सुना है', 'किसी ने मुझे बताया', 'मेरा दोस्त जो काम करता है', 'अंदरूनी जानकारी',
        'reliable sources confirm', 'leaked document shows', 'anonymous source',
        'विश्वसनीय सूत्रों की पुष्टि', 'लीक दस्तावेज़ दिखाता है', 'अनाम स्रोत',
        'experts believe', 'studies suggest', 'research indicates', 'according to reports',
        'विशेषज्ञों का मानना है', 'अध्ययन बताते हैं', 'शोध से पता चलता है', 'रिपोर्ट्स के अनुसार'
    ],

    'emotional_appeals': [
        'danger', 'threat', 'conspiracy', 'साजिश', 'खतरा',
        'fear', 'panic', 'terror', 'horror', 'dread', 'alarm', 'anxiety',
        'डर', 'घबराहट', 'आतंक', 'भय', 'चिंता', 'अलार्म', 'बेचैनी',
        'outrage', 'anger', 'fury', 'rage', 'hatred', 'disgust', 'contempt',
        'आक्रोश', 'क्रोध', 'गुस्सा', 'रोष', 'नफरत', 'घृणा', 'तिरस्कार',
        'betrayal', 'treachery', 'treason', 'deception', 'fraud', 'scam',
        'धोखा', 'विश्वासघात', 'देशद्रोह', 'छल', 'धोखाधड़ी', 'घोटाला'
    ],

    'political_bias_extreme': [
        'anti-national', 'traitor', 'enemy of the state', 'foreign agent', 'spy',
        'देशद्रोही', 'गद्दार', 'राज्य का दुश्मन', 'विदेशी एजेंट', 'जासूस',
        'corrupt', 'criminal', 'thief', 'looter', 'scammer', 'fraudster',
        'भ्रष्ट', 'अपराधी', 'चोर', 'लूटेरा', 'घोटालेबाज', 'धोखेबाज',
        'radical', 'extremist', 'terrorist', 'militant', 'fundamentalist',
        'कट्टरपंथी', 'चरमपंथी', 'आतंकवादी', 'उग्रवादी', 'कट्टरवादी'
    ],

    'health_misinfo_cues': [
        'miracle cure', 'government hiding this', 'आयुर्वेदिक रामबाण',
        'doctors won\'t tell you', 'big pharma doesn\'t want you to know', 'secret remedy',
        'डॉक्टर आपको नहीं बताएंगे', 'बड़ी फार्मा कंपनियां नहीं चाहतीं कि आप जानें', 'गुप्त उपचार',
        'ancient remedy', 'natural cure', 'alternative medicine', 'traditional healing',
        'प्राचीन उपचार', 'प्राकृतिक इलाज', 'वैकल्पिक चिकित्सा', 'पारंपरिक उपचार',
        'proven to work', '100% effective', 'guaranteed results', 'instant relief',
        'काम करना सिद्ध', '100% प्रभावी', 'गारंटीड परिणाम', 'तत्काल राहत'
    ],

    'conspiracy_theories': [
        'new world order', 'illuminati', 'deep state', 'government cover-up', 'mind control',
        'न्यू वर्ल्ड ऑर्डर', 'इल्लुमिनाती', 'डीप स्टेट', 'सरकारी षड्यंत्र', 'दिमागी नियंत्रण',
        'chemtrails', 'microchips', 'surveillance', 'tracking', 'monitoring', 'spying',
        'केमट्रेल्स', 'माइक्रोचिप्स', 'निगरानी', 'ट्रैकिंग', 'मॉनिटरिंग', 'जासूसी',
        'secret society', 'hidden agenda', 'population control', 'depopulation',
        'गुप्त समाज', 'छिपा एजेंडा', 'जनसंख्या नियंत्रण', 'जनसंख्या कम करना'
    ],

    'religious_divisiveness': [
        'they are destroying our religion', 'attack on our faith', 'religious conversion',
        'वे हमारे धर्म को नष्ट कर रहे हैं', 'हमारे विश्वास पर हमला', 'धर्म परिवर्तन',
        'blasphemy', 'sacrilege', 'heresy', 'apostasy', 'infidel', 'kafir',
        'धर्मनिंदा', 'अपवित्रता', 'धर्मविरोध', 'धर्मत्याग', 'काफिर', 'नास्तिक',
        'religious conspiracy', 'jihad', 'crusade', 'holy war', 'religious cleansing',
        'धार्मिक षड्यंत्र', 'जिहाद', 'धर्मयुद्ध', 'पवित्र युद्ध', 'धार्मिक सफाई'
    ]
}

# Enhanced list of known fake news domains
KNOWN_FAKE_NEWS_DOMAINS_PATTERNS = [
    # Fictional examples (for demonstration)
    r'fakenewssite\.com',
    r'satireofficial\.net',
    r'whatsapp-forward-only\.info',
    r'conspiracy-news\.(com|org|net)',
    r'truth-exposed\.(com|org|net)',
    r'secret-news\.(com|org|net)',
    r'viral-truth\.(com|org|net)',
    r'hidden-facts\.(com|org|net)',
    r'uncensored-news\.(com|org|net)',
    r'real-truth\.(com|org|net)',
    r'alternative-facts\.(com|org|net)',
    r'breaking-truth\.(com|org|net)',
    r'news-they-hide\.(com|org|net)',
    r'suppressed-news\.(com|org|net)',
    r'truth-seeker\.(com|org|net)',
    r'news-the-elite-fear\.(com|org|net)',
    r'news-they-dont-want-you-to-see\.(com|org|net)',
    r'the-truth-movement\.(com|org|net)',
    r'wake-up-world\.(com|org|net)',
    r'freedom-news\.(com|org|net)'
]

# Enhanced news source credibility database
# In a real system, this would be much more comprehensive and regularly updated
NEWS_SOURCE_CREDIBILITY = {
    # High credibility sources (examples)
    'bbc.com': 0.9,
    'reuters.com': 0.9,
    'ap.org': 0.9,
    'thehindu.com': 0.85,
    'indianexpress.com': 0.85,
    'ndtv.com': 0.8,
    'hindustantimes.com': 0.8,
    'timesofindia.indiatimes.com': 0.75,
    'theprint.in': 0.75,
    'thequint.com': 0.75,
    'scroll.in': 0.75,
    'news18.com': 0.7,
    'indiatoday.in': 0.7,

    # Medium credibility sources (examples)
    'firstpost.com': 0.65,
    'dnaindia.com': 0.65,
    'deccanherald.com': 0.65,
    'telegraphindia.com': 0.65,
    'tribuneindia.com': 0.65,
    'theweek.in': 0.6,
    'outlookindia.com': 0.6,
    'thewire.in': 0.6,
    'opindia.com': 0.5,
    'swarajyamag.com': 0.5,

    # Lower credibility sources (examples)
    'questionable_source.net': 0.3,
    'known_bias_source.com': 0.4,
    'unverified_blog.wordpress.com': 0.2,

    # Default for unknown sources
    'default': 0.4
}

# Dummy coordinates (should be ideally from config or a geo-coding utility)
DUMMY_LAT = 28.6139 # Example: Delhi latitude
DUMMY_LON = 77.2090 # Example: Delhi longitude

class MisinformationDetector:
    def __init__(self, llm_client=None, config=None):
        self.config = config if config else {}
        self.narrative_tracker = defaultdict(lambda: {
            'mentions': 0,
            'first_seen': None, # Timestamp of first occurrence
            'last_seen': None,  # Timestamp of most recent occurrence
            'spread_rate': 0.0, # Mentions per hour (or other unit)
            'locations': set()  # Set of locations where narrative appeared
        })
        self.llm_client = llm_client # Store LLM client

    def _clean_text(self, text: str) -> str:
        """Placeholder for narrative extraction. Could use topic modeling or keyword sets."""
        # Simple keyword-based narrative for now
        text_lower = text.lower()
        if 'election fraud' in text_lower or 'चुनाव में धांधली' in text_lower:
            return 'election_fraud_claims'
        if 'vaccine side effects severe' in text_lower or 'वैक्सीन के दुष्प्रभाव' in text_lower:
            return 'vaccine_danger_claims'
        if 'community conflict' in text_lower or 'सांप्रदायिक तनाव' in text_lower:
            return 'community_conflict_rumors'
        # Fallback to a generic hash or first few salient words if no specific narrative found
        return "generic_narrative_" + str(hash(text[:50]))

    def _calculate_misinfo_score(self, text: str, source: Optional[str] = None) -> Dict[str, Any]:
        """
        Calculate a misinformation likelihood score based on indicators.

        Args:
            text: The text to analyze
            source: Optional source URL or name

        Returns:
            Dictionary with:
            - score: Overall misinformation likelihood score (0-1)
            - indicators: Dictionary of indicator categories and their counts
            - source_credibility: Credibility score of the source (if provided)
            - explanation: Human-readable explanation of the score
        """
        score = 0.0
        text_lower = text.lower()
        indicators_found = defaultdict(list)
        explanation_parts = []

        # Check for indicators in each category
        for category, indicators in MISINFO_INDICATORS_INDIAN_CONTEXT.items():
            category_count = 0
            for indicator in indicators:
                indicator_lower = indicator.lower()
                if indicator_lower in text_lower:
                    indicators_found[category].append(indicator)
                    category_count += 1

            # Apply category-specific weights
            if category_count > 0:
                if category == 'sensational_language':
                    category_score = min(0.05 * category_count, 0.2)  # Cap at 0.2
                    score += category_score
                    explanation_parts.append(f"Contains sensational language ({category_count} instances)")

                elif category == 'urgency_claims':
                    category_score = min(0.1 * category_count, 0.3)  # Cap at 0.3
                    score += category_score
                    explanation_parts.append(f"Contains urgency claims ({category_count} instances)")

                elif category == 'unverified_sources':
                    category_score = min(0.15 * category_count, 0.3)  # Cap at 0.3
                    score += category_score
                    explanation_parts.append(f"References unverified sources ({category_count} instances)")

                elif category == 'emotional_appeals':
                    category_score = min(0.05 * category_count, 0.2)  # Cap at 0.2
                    score += category_score
                    explanation_parts.append(f"Contains emotional appeals ({category_count} instances)")

                elif category == 'political_bias_extreme':
                    category_score = min(0.1 * category_count, 0.3)  # Cap at 0.3
                    score += category_score
                    explanation_parts.append(f"Contains extreme political bias ({category_count} instances)")

                elif category == 'health_misinfo_cues':
                    category_score = min(0.15 * category_count, 0.4)  # Cap at 0.4
                    score += category_score
                    explanation_parts.append(f"Contains health misinformation cues ({category_count} instances)")

                elif category == 'conspiracy_theories':
                    category_score = min(0.15 * category_count, 0.4)  # Cap at 0.4
                    score += category_score
                    explanation_parts.append(f"Contains conspiracy theory elements ({category_count} instances)")

                elif category == 'religious_divisiveness':
                    category_score = min(0.15 * category_count, 0.4)  # Cap at 0.4
                    score += category_score
                    explanation_parts.append(f"Contains religiously divisive content ({category_count} instances)")

        # Check source credibility
        source_credibility = 0.5  # Default for unknown sources
        if source:
            source_lower = source.lower()

            # Check for known fake news domains
            for pattern in KNOWN_FAKE_NEWS_DOMAINS_PATTERNS:
                if re.search(pattern, source_lower):
                    score += 0.3  # Higher penalty for known fake news domains
                    explanation_parts.append(f"Source matches known fake news pattern: {pattern}")
                    source_credibility = 0.1
                    break

            # Check source credibility database
            for domain, credibility in NEWS_SOURCE_CREDIBILITY.items():
                if domain in source_lower:
                    source_credibility = credibility

                    # Adjust score based on source credibility
                    if credibility < 0.4:
                        score += (0.4 - credibility) * 0.5  # Low credibility increases misinfo score
                        explanation_parts.append(f"Source has low credibility rating: {credibility:.2f}")
                    elif credibility > 0.7:
                        score -= (credibility - 0.7) * 0.3  # High credibility decreases misinfo score
                        explanation_parts.append(f"Source has high credibility rating: {credibility:.2f}")
                    break
            else:
                # If no match found, use default
                source_credibility = NEWS_SOURCE_CREDIBILITY.get('default', 0.4)

        # Text length penalty (very short texts with high indicator density are suspicious)
        text_length = len(text.split())
        if text_length < 20 and score > 0.3:
            score += 0.1
            explanation_parts.append("Very short text with high indicator density")

        # Final score with explanation
        final_score = np.clip(score, 0, 1.0)

        return {
            'score': final_score,
            'indicators': dict(indicators_found),
            'source_credibility': source_credibility,
            'explanation': "; ".join(explanation_parts) if explanation_parts else "No misinformation indicators detected"
        }

    def _analyze_news_credibility(self, news_articles: pd.DataFrame) -> pd.DataFrame:
        """Analyze credibility of news articles."""
        if news_articles.empty:
            return news_articles

        def get_credibility(url_or_source_name: str) -> float:
            if not isinstance(url_or_source_name, str): return 0.3 # Default for unknown
            # Try to match domain from URL
            domain_match = re.search(r'https?:\/\/(www\.)?([^\/]+)', url_or_source_name)
            domain = domain_match.group(2) if domain_match else url_or_source_name.lower()

            return NEWS_SOURCE_CREDIBILITY.get(domain, 0.3) # Default for unknown sources

        if 'url' in news_articles.columns:
            news_articles['credibility_score'] = news_articles['url'].apply(get_credibility)
        elif 'source' in news_articles.columns: # If only source name is available
            news_articles['credibility_score'] = news_articles['source'].apply(get_credibility)
        else:
            news_articles['credibility_score'] = 0.3 # Default if no source info

        return news_articles

    def _extract_narratives_from_text(self, text: str, use_llm: bool = False) -> List[str]:
        """Extracts potential narratives/claims from text. Enhanced with LLM if available."""
        if use_llm and self.llm_client:
            prompt = f"""
            Analyze the following text from a social media post or news snippet in an Indian context.
            Identify and extract key narratives or claims being made. A narrative is a central theme or story being conveyed.
            Focus on claims that could be verifiable or have public impact.
            Return a JSON list of strings, where each string is a concise summary of a distinct narrative/claim.
            Example: ["Claim about new local policy X", "Rumor about event Y affecting community Z"]
            If no clear narratives or claims are found, return an empty list.

            Text: "{text}"
            """
            try:
                response_text = self.llm_client.generate_text(prompt, max_tokens=150)
                if response_text:
                    import json
                    try:
                        if response_text.strip().startswith("```json"):
                            response_text = response_text.strip()[7:-3].strip()
                        elif response_text.strip().startswith("```"):
                            response_text = response_text.strip()[3:-3].strip()

                        narratives = json.loads(response_text)
                        if isinstance(narratives, list) and all(isinstance(n, str) for n in narratives):
                            if narratives: # Only return if LLM found something
                                return narratives
                        else:
                            logger.warning(f"LLM narrative extraction did not return a list of strings: {narratives}")
                    except json.JSONDecodeError as e:
                        logger.error(f"Failed to parse LLM JSON response for narratives: {e}. Response: {response_text}")
            except Exception as e:
                logger.error(f"Error during LLM narrative extraction: {e}")
            # Fallback to keyword-based if LLM fails, not used, or returns empty useful list
            logger.info("Falling back to keyword-based narrative extraction for: " + text[:50] + "...")

        # Fallback: Simple keyword-based narrative extraction
        narratives = []
        text_lower = self._clean_text(text).lower()
        # Example keywords, expand significantly for real use
        if 'election' in text_lower and ('fraud' in text_lower or 'rigged' in text_lower or 'धांधली' in text_lower):
            narratives.append('Potential election integrity concern')
        if 'vaccine' in text_lower and ('unsafe' in text_lower or 'side effect' in text_lower or 'खतरनाक' in text_lower or 'दुष्प्रभाव' in text_lower):
            narratives.append('Potential vaccine safety concern')
        if ('riot' in text_lower or 'violence' in text_lower or 'दंगा' in text_lower or 'हिंसा' in text_lower) and ('community' in text_lower or 'group' in text_lower or 'सांप्रदायिक' in text_lower):
            narratives.append('Potential inter-group conflict/violence narrative')
        if not narratives: # If no specific keywords, use a generic placeholder if text is substantial
            if len(text_lower) > 20 : # Arbitrary length to avoid empty/too short texts as narratives
                 narratives.append(f"Uncategorized narrative starting with: {text_lower[:30]}...")
        return narratives

    def _calculate_misinformation_likelihood(self, text: str, source_credibility: float = 0.5) -> float:
        """Calculate a misinformation likelihood score based on indicators."""
        score = 0.0
        text_lower = text.lower()

        for category, indicators in MISINFO_INDICATORS_INDIAN_CONTEXT.items():
            for indicator in indicators:
                if indicator.lower() in text_lower:
                    score += 0.1 # Increment score for each indicator found

        if source:
            source_lower = source.lower()
            for pattern in KNOWN_FAKE_NEWS_DOMAINS_PATTERNS:
                if re.search(pattern, source_lower):
                    score += 0.3 # Higher penalty for known fake news domains
                    break
        return np.clip(score, 0, 1.0)

    def _analyze_news_credibility(self, news_articles: pd.DataFrame) -> pd.DataFrame:
        """Analyze credibility of news articles."""
        if news_articles.empty:
            return news_articles

        def get_credibility(url_or_source_name: str) -> float:
            if not isinstance(url_or_source_name, str): return 0.3 # Default for unknown
            # Try to match domain from URL
            domain_match = re.search(r'https?:\/\/(www\.)?([^\/]+)', url_or_source_name)
            domain = domain_match.group(2) if domain_match else url_or_source_name.lower()

            return NEWS_SOURCE_CREDIBILITY.get(domain, 0.3) # Default for unknown sources

        if 'url' in news_articles.columns:
            news_articles['credibility_score'] = news_articles['url'].apply(get_credibility)
        elif 'source' in news_articles.columns: # If only source name is available
            news_articles['credibility_score'] = news_articles['source'].apply(get_credibility)
        else:
            news_articles['credibility_score'] = 0.3 # Default if no source info

        return news_articles

    def _generate_counter_narrative_stub(self, narrative: str, use_llm: bool = False, indicators: Dict[str, List[str]] = None) -> str:
        """
        Generates a basic counter-narrative suggestion. Enhanced with LLM.

        Args:
            narrative: The narrative to generate a counter-narrative for
            use_llm: Whether to use LLM for enhanced generation
            indicators: Dictionary of misinformation indicators found in the narrative

        Returns:
            A counter-narrative suggestion as a string
        """
        if use_llm and self.llm_client:
            # Build prompt with indicators if available
            indicator_text = ""
            if indicators:
                indicator_categories = []
                for category, items in indicators.items():
                    if items:  # Only include non-empty categories
                        indicator_categories.append(f"{category} ({', '.join(items[:3])}{'...' if len(items) > 3 else ''})")
                if indicator_categories:
                    indicator_text = f"\nThe narrative contains these misinformation indicators: {'; '.join(indicator_categories[:3])}."

            prompt = f"""
            A potentially problematic narrative circulating is: "{narrative}"
            This narrative is relevant to an Indian context.{indicator_text}
            Suggest a concise, factual, and neutral counter-narrative or clarification point that authorities or trusted sources could use.
            The counter-narrative should aim to debunk misinformation or provide accurate context if the narrative is misleading.
            Focus on being constructive and informative. Limit to 1-2 sentences.
            Return only the suggested counter-narrative text, no extra formatting or explanation.
            Example: "Official sources confirm that policy X aims to improve Y, not cause Z as rumored. Check [official website/source] for details."
            """
            try:
                response_text = self.llm_client.generate_text(prompt, max_tokens=150) # Increased tokens for better response
                if response_text and response_text.strip():
                    # Clean up potential LLM pleasantries if any, though prompt asks for text only
                    cleaned_response = response_text.strip()
                    # Remove common conversational prefixes if LLM adds them despite instructions
                    prefixes_to_remove = ["Certainly, here's a counter-narrative:", "Here's a suggested counter-narrative:", "A possible counter-narrative could be:"]
                    for prefix in prefixes_to_remove:
                        if cleaned_response.lower().startswith(prefix.lower()):
                            cleaned_response = cleaned_response[len(prefix):].strip()
                    return cleaned_response
                else:
                    logger.warning(f"LLM returned empty response for counter-narrative generation for: {narrative}")
            except Exception as e:
                logger.error(f"Error during LLM counter-narrative generation: {e}")
            # Fallback if LLM fails or returns empty
            logger.info("Falling back to basic counter-narrative stub generation for: " + narrative)

        # Fallback: Basic stub based on narrative keywords
        if 'election' in narrative.lower():
            return "Authorities are ensuring a fair and transparent election process. Report any concerns to official channels."
        if 'vaccine' in narrative.lower():
            return "COVID-19 vaccines are safe and effective. Consult healthcare providers for reliable information."
        if 'conflict' in narrative.lower() or 'violence' in narrative.lower():
            return "Maintain peace and harmony. Do not believe or spread rumors that incite hatred or violence."
        return f"Verify information about '{narrative[:30]}...' from trusted sources before sharing."

    def detect_misinformation(
        self,
        social_media_data: List[Dict[str, Any]], 
        news_articles_data: List[Dict[str, Any]],
        use_llm_for_narratives: bool = False, 
        use_llm_for_counter_narratives: bool = False 
    ) -> List[MisinformationAlert]:
        """
        Detect potential misinformation in social media and news article data.

        Args:
            social_media_data: List of social media posts as dictionaries
            news_articles_data: List of news articles as dictionaries
            use_llm_for_narratives: Whether to use LLM for narrative extraction
            use_llm_for_counter_narratives: Whether to use LLM for counter-narrative generation

        Returns:
            List of MisinformationAlert objects
        """
        alerts: List[MisinformationAlert] = []
        now = datetime.now()

        # Log analysis start
        logger.info(f"Starting misinformation detection on {len(social_media_data)} social media posts and {len(news_articles_data)} news articles")

        # Convert lists of dicts to DataFrames
        social_media_df = pd.DataFrame(social_media_data if social_media_data else [])
        news_articles_df = pd.DataFrame(news_articles_data if news_articles_data else [])

        # Analyze news credibility first
        news_articles_df = self._analyze_news_credibility(news_articles_df)

        # Track narratives across all data for cross-referencing
        all_narratives = set()
        narrative_sources = defaultdict(list)
        narrative_locations = defaultdict(set)

        # Process social media data
        if not social_media_df.empty:
            for index, post in social_media_df.iterrows():
                text = post.get('text', '')
                if not text:  # Skip if text is empty
                    continue

                source_platform = post.get('source_platform', 'social_media')
                timestamp = pd.to_datetime(post.get('timestamp', now))
                location = post.get('location_approx', 'Unknown')

                # Get location coordinates (for geospatial analysis)
                location_lat = post.get('location_lat', DUMMY_LAT)
                location_lon = post.get('location_lon', DUMMY_LON)

                # Extract narratives using enhanced method
                extracted_narratives = self._extract_narratives_from_text(text, use_llm=use_llm_for_narratives)

                # Get comprehensive misinformation score
                misinfo_analysis = self._calculate_misinfo_score(text, post.get('source_url'))
                misinfo_score = misinfo_analysis['score']
                indicators = misinfo_analysis['indicators']
                source_credibility = misinfo_analysis['source_credibility']
                explanation = misinfo_analysis['explanation']

                # Process each extracted narrative
                for single_narrative in extracted_narratives:
                    if not single_narrative:  # Skip empty narratives if any
                        continue

                    # Add to global tracking
                    all_narratives.add(single_narrative)
                    narrative_sources[single_narrative].append(source_platform)
                    narrative_locations[single_narrative].add(location)

                    # Update narrative tracker
                    tracker_entry = self.narrative_tracker[single_narrative]
                    tracker_entry['mentions'] += 1

                    # Track first seen time
                    if tracker_entry['first_seen'] is None:
                        tracker_entry['first_seen'] = timestamp

                    # Calculate spread rate
                    if tracker_entry['first_seen']:
                        time_delta_hours = (timestamp - tracker_entry['first_seen']).total_seconds() / 3600 + 1e-6
                        if time_delta_hours > 0:
                            tracker_entry['spread_rate'] = tracker_entry['mentions'] / time_delta_hours
                        else: 
                            tracker_entry['spread_rate'] = float(tracker_entry['mentions'])
                    else: 
                        tracker_entry['spread_rate'] = 0.0

                    # Update last seen time and locations
                    tracker_entry['last_seen'] = timestamp
                    tracker_entry['locations'].add(location)

                    # Store indicators for this narrative
                    if 'indicators' not in tracker_entry:
                        tracker_entry['indicators'] = defaultdict(int)

                    for category, found_indicators in indicators.items():
                        tracker_entry['indicators'][category] += len(found_indicators)

                    # Check if this narrative appears in news articles
                    news_credibility_for_narrative = 0.5
                    news_sources_for_narrative = []

                    if not news_articles_df.empty and 'title' in news_articles_df.columns:
                        for _, news_row in news_articles_df.iterrows():
                            news_title = news_row.get('title', '')
                            news_content = news_row.get('content', '')

                            # Check if narrative appears in title or content
                            narrative_words = single_narrative.replace('_', ' ').lower().split()
                            if (any(word in news_title.lower() for word in narrative_words) or 
                                (news_content and any(word in news_content.lower() for word in narrative_words))):
                                news_credibility = news_row.get('credibility_score', 0.3)
                                news_sources_for_narrative.append({
                                    'source': news_row.get('source', 'Unknown'),
                                    'credibility': news_credibility,
                                    'url': news_row.get('url', '')
                                })

                                # Use the lowest credibility score found (worst case)
                                if news_credibility < news_credibility_for_narrative:
                                    news_credibility_for_narrative = news_credibility

                    # Determine severity based on multiple factors
                    current_spread_rate = tracker_entry.get('spread_rate', 0)
                    indicator_count = sum(len(inds) for inds in indicators.values())

                    # Enhanced severity calculation
                    severity = 'low'

                    # High severity conditions
                    if (misinfo_score > 0.85 or 
                        current_spread_rate > 50 or 
                        (misinfo_score > 0.7 and current_spread_rate > 30) or
                        (indicator_count > 10 and news_credibility_for_narrative < 0.3)):
                        severity = 'high'
                    # Medium severity conditions
                    elif (misinfo_score > 0.7 or 
                          current_spread_rate > 20 or 
                          (misinfo_score > 0.5 and current_spread_rate > 10) or
                          (indicator_count > 5 and news_credibility_for_narrative < 0.4)):
                        severity = 'medium'
                    # Low severity conditions
                    elif (misinfo_score > 0.5 or 
                          current_spread_rate > 10 or
                          (indicator_count > 3 and news_credibility_for_narrative < 0.5)):
                        severity = 'low'
                    else:
                        # Skip creating an alert if below thresholds
                        continue

                    # Generate potential impact based on severity and indicators
                    potential_impact = self._generate_potential_impact(
                        severity, 
                        indicators, 
                        current_spread_rate,
                        list(tracker_entry['locations'])
                    )

                    # Generate counter-narrative
                    counter_narrative = self._generate_counter_narrative_stub(
                        single_narrative, 
                        use_llm=use_llm_for_counter_narratives,
                        indicators=indicators
                    )

                    # Create alert with enhanced information
                    alerts.append(MisinformationAlert(
                        narrative=single_narrative,
                        source_type=source_platform,  # Using source_type as per class definition
                        timestamp=timestamp,
                        severity=severity,
                        spread_velocity=current_spread_rate,
                        potential_impact=potential_impact,
                        confidence=misinfo_score,
                        affected_locations=list(tracker_entry['locations']),
                        counter_narrative_suggestion=counter_narrative,
                        credibility_score=news_credibility_for_narrative,
                        origin_location_name=location,
                        origin_location_lat=location_lat,
                        origin_location_lon=location_lon
                        # Removed explanation, indicators_found, and news_sources as they're not in the class definition
                    ))

        # Prune old narratives from tracker (optional)
        for narrative, data in list(self.narrative_tracker.items()):
            if (now - data['last_seen']) > timedelta(days=7):
                del self.narrative_tracker[narrative]

        logger.info(f"Misinformation detection complete. Found {len(alerts)} potential misinformation alerts.")
        return alerts

    def _generate_potential_impact(self, severity: str, indicators: Dict[str, List[str]], 
                                 spread_rate: float, affected_locations: List[str]) -> str:
        """
        Generate a detailed potential impact statement based on severity and indicators.

        Args:
            severity: The severity level (high, medium, low)
            indicators: Dictionary of indicator categories and their instances
            spread_rate: The spread velocity of the narrative
            affected_locations: List of affected locations

        Returns:
            A detailed potential impact statement
        """
        impact_parts = []

        # Base impact by severity
        if severity == 'high':
            impact_parts.append("High risk of causing significant social disruption")
        elif severity == 'medium':
            impact_parts.append("Moderate risk of causing social tension or confusion")
        else:
            impact_parts.append("Low risk of causing minor confusion or concern")

        # Add specifics based on indicators
        if 'religious_divisiveness' in indicators:
            impact_parts.append("may inflame religious tensions")

        if 'conspiracy_theories' in indicators:
            impact_parts.append("could erode trust in institutions")

        if 'health_misinfo_cues' in indicators:
            impact_parts.append("might lead to harmful health decisions")

        if 'emotional_appeals' in indicators and len(indicators['emotional_appeals']) > 2:
            impact_parts.append("likely to trigger strong emotional responses")

        # Add spread dynamics
        if spread_rate > 30:
            impact_parts.append(f"spreading extremely rapidly (velocity: {spread_rate:.1f})")
        elif spread_rate > 15:
            impact_parts.append(f"spreading rapidly (velocity: {spread_rate:.1f})")
        elif spread_rate > 5:
            impact_parts.append(f"spreading steadily (velocity: {spread_rate:.1f})")

        # Add location context if available
        if len(affected_locations) > 3:
            impact_parts.append(f"affecting multiple locations ({len(affected_locations)} areas)")
        elif len(affected_locations) > 1:
            impact_parts.append(f"affecting {', '.join(affected_locations[:3])}")

        return ". This ".join(impact_parts) + "."
