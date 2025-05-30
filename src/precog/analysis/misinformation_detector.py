from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import re
from collections import defaultdict
import logging

from ..core.data_models import MisinformationAlert, NewsArticle # Ensure NewsArticle is imported if used here
from ..core.llm_clients import BaseLLMClient # New import for LLM

# Constants for Misinformation Detection (Indian Context)
MISINFO_INDICATORS_INDIAN_CONTEXT = {
    'sensational_language': ['SHOCKING', 'BREAKING', 'EXCLUSIVE', '!', 'MUST WATCH', 'ज़रूर देखें', 'सनसनीखेज'],
    'urgency_claims': ['SHARE NOW', 'FORWARDED AS RECEIVED', 'URGENT', 'तुरंत शेयर करें'],
    'unverified_sources': ['sources say', 'according to a WhatsApp forward', 'सूत्रों के अनुसार'],
    'emotional_appeals': ['danger', 'threat', 'conspiracy', 'साजिश', 'खतरा'],
    'political_bias_extreme': ['anti-national', 'pro-[party_A]', 'anti-[party_B]'], # Placeholder
    'health_misinfo_cues': ['miracle cure', 'government hiding this', 'आयुर्वेदिक रामबाण']
}

KNOWN_FAKE_NEWS_DOMAINS_PATTERNS = [
    r'fakenewssite\.com',
    r'satireofficial\.net', # Example, can be expanded
    r'whatsapp-forward-only\.info'
]

NEWS_SOURCE_CREDIBILITY = { # Simplified, can be expanded or dynamically updated
    'reliable_source_1.com': 0.9,
    'reliable_source_2.org': 0.85,
    'local_news_reliable.in': 0.75,
    'questionable_source.net': 0.3,
    'known_bias_source.com': 0.4,
    'unverified_blog.wordpress.com': 0.2
}

logger = logging.getLogger(__name__)

# Constants for Misinformation Detection (Indian Context)
MISINFO_INDICATORS_INDIAN_CONTEXT = {
    'sensational_language': ['SHOCKING', 'BREAKING', 'EXCLUSIVE', '!', 'MUST WATCH', 'ज़रूर देखें', 'सनसनीखेज'],
    'urgency_claims': ['SHARE NOW', 'FORWARDED AS RECEIVED', 'URGENT', 'तुरंत शेयर करें'],
    'unverified_sources': ['sources say', 'according to a WhatsApp forward', 'सूत्रों के अनुसार'],
    'emotional_appeals': ['danger', 'threat', 'conspiracy', 'साजिश', 'खतरा'],
    'political_bias_extreme': ['anti-national', 'pro-[party_A]', 'anti-[party_B]'], # Placeholder
    'health_misinfo_cues': ['miracle cure', 'government hiding this', 'आयुर्वेदिक रामबाण']
}

KNOWN_FAKE_NEWS_DOMAINS_PATTERNS = [
    r'fakenewssite\.com',
    r'satireofficial\.net', # Example, can be expanded
    r'whatsapp-forward-only\.info'
]

NEWS_SOURCE_CREDIBILITY = { # Simplified, can be expanded or dynamically updated
    'reliable_source_1.com': 0.9,
    'reliable_source_2.org': 0.85,
    'local_news_reliable.in': 0.75,
    'questionable_source.net': 0.3,
    'known_bias_source.com': 0.4,
    'unverified_blog.wordpress.com': 0.2
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

    def _calculate_misinfo_score(self, text: str, source: Optional[str] = None) -> float:
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

    def _generate_counter_narrative_stub(self, narrative: str, use_llm: bool = False) -> str:
        """Generates a basic counter-narrative suggestion. Enhanced with LLM."""
        if use_llm and self.llm_client:
            prompt = f"""
            A potentially problematic narrative circulating is: "{narrative}"
            This narrative is relevant to an Indian context.
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
        alerts: List[MisinformationAlert] = []
        now = datetime.now()

        # Convert lists of dicts to DataFrames
        social_media_df = pd.DataFrame(social_media_data if social_media_data else [])
        news_articles_df = pd.DataFrame(news_articles_data if news_articles_data else [])

        # Analyze news credibility first
        news_articles_df = self._analyze_news_credibility(news_articles_df) # Pass DataFrame

        # Process social media data
        if not social_media_df.empty:
            for index, post in social_media_df.iterrows():
                text = post.get('text', '')
                if not text:  # Skip if text is empty
                    continue
                source_platform = post.get('source_platform', 'social_media')
                timestamp = pd.to_datetime(post.get('timestamp', now))
                location = post.get('location_approx', 'Unknown')

                extracted_narratives = self._extract_narratives_from_text(text, use_llm=use_llm_for_narratives)
                misinfo_score = self._calculate_misinfo_score(text, post.get('source_url'))

                for single_narrative in extracted_narratives:
                    if not single_narrative: # Skip empty narratives if any
                        continue

                    tracker_entry = self.narrative_tracker[single_narrative]
                    tracker_entry['mentions'] += 1
                    
                    if tracker_entry['first_seen'] is None:
                        tracker_entry['first_seen'] = timestamp
                    
                    if tracker_entry['first_seen']:
                         time_delta_hours = (timestamp - tracker_entry['first_seen']).total_seconds() / 3600 + 1e-6
                         if time_delta_hours > 0:
                            tracker_entry['spread_rate'] = tracker_entry['mentions'] / time_delta_hours
                         else: 
                            tracker_entry['spread_rate'] = float(tracker_entry['mentions'])
                    else: 
                        tracker_entry['spread_rate'] = 0.0

                    tracker_entry['last_seen'] = timestamp
                    tracker_entry['locations'].add(location)

                    news_credibility_for_narrative = 0.5 
                    if not news_articles_df.empty and 'title' in news_articles_df.columns:
                        for _, news_row in news_articles_df.iterrows():
                            news_title = news_row.get('title', '')
                            if single_narrative.replace('_', ' ') in news_title.lower():
                                news_credibility_for_narrative = news_row.get('credibility_score', 0.3)
                                break

                    current_spread_rate = tracker_entry.get('spread_rate', 0)
                    if misinfo_score > 0.5 or (current_spread_rate > 10 and news_credibility_for_narrative < 0.4):
                        severity = 'low'
                        if misinfo_score > 0.7 or current_spread_rate > 20: severity = 'medium'
                        if misinfo_score > 0.85 or current_spread_rate > 50: severity = 'high'

                        alerts.append(MisinformationAlert(
                            narrative=single_narrative,
                            source_platform=source_platform,
                            timestamp=timestamp,
                            severity=severity,
                            spread_velocity=current_spread_rate,
                            potential_impact="Could mislead public, incite unrest if unchecked.",
                            confidence=misinfo_score,
                            affected_locations=list(tracker_entry['locations'])[:3],
                            counter_narrative_suggestions=self._generate_counter_narrative_stub(single_narrative, use_llm=use_llm_for_counter_narratives),
                            credibility_score=news_credibility_for_narrative
                        ))

        # Prune old narratives from tracker (optional)
        # for narrative, data in list(self.narrative_tracker.items()):
        #     if (now - data['last_seen']) > timedelta(days=7):
        #         del self.narrative_tracker[narrative]

        return alerts