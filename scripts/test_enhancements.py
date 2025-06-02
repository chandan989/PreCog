import sys
import os
import pandas as pd
from datetime import datetime
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add project root to Python path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Import components
from src.precog.analysis.sentiment_analyzer import HyperlocalSentimentAnalyzer
from src.precog.analysis.misinformation_detector import MisinformationDetector
from src.precog.core.system import HyperlocalIntelligenceSystem

def test_sentiment_analyzer():
    """Test the enhanced sentiment analyzer."""
    logger.info("Testing enhanced sentiment analyzer...")
    
    # Initialize analyzer
    analyzer = HyperlocalSentimentAnalyzer()
    
    # Test texts in different languages and contexts
    test_texts = [
        # English texts
        "I'm very happy with the new road construction in our area.",
        "The water supply has been terrible for weeks now. This is unacceptable!",
        "There are rumors that people from the other community are planning to attack our neighborhood.",
        
        # Hindi texts
        "हमारे क्षेत्र में नई सड़क का निर्माण बहुत अच्छा है।",
        "पिछले कई हफ्तों से पानी की आपूर्ति बहुत खराब है। यह अस्वीकार्य है!",
        "अफवाहें हैं कि दूसरे समुदाय के लोग हमारे मोहल्ले पर हमला करने की योजना बना रहे हैं।",
        
        # Mixed language (Hinglish)
        "Road construction bahut slow hai, government ko speed up karna chahiye.",
        "Water supply nahi hai aur municipality kuch kar nahi rahi hai.",
        "Dusre community ke log hamari area mein tension create kar rahe hain."
    ]
    
    # Analyze each text
    for text in test_texts:
        logger.info(f"\nAnalyzing text: {text}")
        
        # Test with rule-based analysis
        result = analyzer.analyze_single_text(text, use_llm=False)
        logger.info(f"Rule-based analysis result:")
        logger.info(f"  Sentiment score: {result['sentiment_score']:.2f}")
        logger.info(f"  Sentiment label: {result['sentiment_label']}")
        logger.info(f"  Category: {result['category']}")
        logger.info(f"  Regional context: {result['regional_context']}")
        logger.info(f"  Civic issues: {result['civic_issues']}")
        logger.info(f"  Keywords: {result['keywords'][:5]}")
        
        # Test batch analysis
        batch_result = analyzer.analyze_text_batch([text], ["Test Location"])
        logger.info(f"Batch analysis result:")
        logger.info(f"  Alert level: {batch_result['Test Location']['alert_level']}")
        logger.info(f"  Top issues: {batch_result['Test Location']['top_issues'][:2]}")
    
    logger.info("Sentiment analyzer testing complete.")

def test_misinformation_detector():
    """Test the enhanced misinformation detector."""
    logger.info("\nTesting enhanced misinformation detector...")
    
    # Initialize detector
    detector = MisinformationDetector()
    
    # Test texts with varying levels of misinformation indicators
    test_posts = [
        {
            'text': "BREAKING NEWS! Government hiding the truth about water contamination. SHARE BEFORE DELETED!",
            'source_platform': 'social_media',
            'timestamp': datetime.now(),
            'location_approx': 'Delhi',
            'location_lat': 28.6139,
            'location_lon': 77.2090
        },
        {
            'text': "According to reliable sources, the new infrastructure project is actually a conspiracy to displace local residents.",
            'source_platform': 'social_media',
            'timestamp': datetime.now(),
            'location_approx': 'Mumbai',
            'location_lat': 19.0760,
            'location_lon': 72.8777
        },
        {
            'text': "The water supply will be disrupted tomorrow from 10 AM to 2 PM due to maintenance work.",
            'source_platform': 'social_media',
            'timestamp': datetime.now(),
            'location_approx': 'Bangalore',
            'location_lat': 12.9716,
            'location_lon': 77.5946
        }
    ]
    
    # Test news articles
    test_news = [
        {
            'title': "Water Contamination Reports Being Investigated",
            'content': "Local authorities are investigating reports of water contamination in Delhi.",
            'source': 'thehindu.com',
            'url': 'https://www.thehindu.com/news/article123.html',
            'timestamp': datetime.now()
        },
        {
            'title': "Infrastructure Project Aims to Improve Living Conditions",
            'content': "The new infrastructure project in Mumbai aims to improve living conditions for residents.",
            'source': 'timesofindia.indiatimes.com',
            'url': 'https://timesofindia.indiatimes.com/news/article456.html',
            'timestamp': datetime.now()
        }
    ]
    
    # Test misinformation score calculation
    for post in test_posts:
        logger.info(f"\nAnalyzing text for misinformation: {post['text']}")
        score_result = detector._calculate_misinfo_score(post['text'])
        logger.info(f"Misinformation score: {score_result['score']:.2f}")
        logger.info(f"Explanation: {score_result['explanation']}")
        logger.info(f"Indicators found: {score_result['indicators']}")
    
    # Test full misinformation detection
    alerts = detector.detect_misinformation(test_posts, test_news)
    logger.info(f"\nDetected {len(alerts)} misinformation alerts:")
    for i, alert in enumerate(alerts):
        logger.info(f"Alert {i+1}:")
        logger.info(f"  Narrative: {alert.narrative}")
        logger.info(f"  Severity: {alert.severity}")
        logger.info(f"  Confidence: {alert.confidence:.2f}")
        logger.info(f"  Potential impact: {alert.potential_impact}")
        logger.info(f"  Counter-narrative: {alert.counter_narrative_suggestion}")
    
    logger.info("Misinformation detector testing complete.")

def test_system_integration():
    """Test the integration of enhanced components in the system."""
    logger.info("\nTesting system integration...")
    
    # Initialize system
    system = HyperlocalIntelligenceSystem()
    
    # Create test data
    social_df = pd.DataFrame([
        {
            'id': 1,
            'text': "The water quality in our area has been terrible for weeks. The authorities are not responding!",
            'location_approx': 'Delhi',
            'timestamp': datetime.now(),
            'source_platform': 'Twitter'
        },
        {
            'id': 2,
            'text': "URGENT! There are rumors of violence between communities in the eastern part of the city. SHARE NOW!",
            'location_approx': 'Mumbai',
            'timestamp': datetime.now(),
            'source_platform': 'Facebook'
        },
        {
            'id': 3,
            'text': "The new road construction is progressing well and should be completed ahead of schedule.",
            'location_approx': 'Bangalore',
            'timestamp': datetime.now(),
            'source_platform': 'Twitter'
        }
    ])
    
    news_df = pd.DataFrame([
        {
            'title': "Water Quality Concerns Raised by Residents",
            'content': "Residents in Delhi have raised concerns about water quality in several neighborhoods.",
            'source': 'thehindu.com',
            'url': 'https://www.thehindu.com/news/article123.html',
            'timestamp': datetime.now()
        },
        {
            'title': "Authorities Deny Community Tension Reports",
            'content': "Officials have denied reports of community tensions in Mumbai, calling them baseless rumors.",
            'source': 'timesofindia.indiatimes.com',
            'url': 'https://timesofindia.indiatimes.com/news/article456.html',
            'timestamp': datetime.now()
        }
    ])
    
    # Run analysis
    logger.info("Running system analysis...")
    analysis_results = system.analyze_current_situation(social_df, news_df)
    
    # Check results
    logger.info(f"Analysis results:")
    logger.info(f"  Sentiment alerts: {len(analysis_results.get('anomaly_alerts', []))}")
    logger.info(f"  Misinformation alerts: {len(analysis_results.get('misinformation_alerts', []))}")
    logger.info(f"  Friction risks: {len(analysis_results.get('friction_risks', []))}")
    logger.info(f"  Intervention recommendations: {len(analysis_results.get('intervention_recommendations', []))}")
    
    # Check sentiment analysis results
    sentiment_df = analysis_results.get('sentiment_analysis', pd.DataFrame())
    if not sentiment_df.empty:
        logger.info(f"  Sentiment analysis results:")
        for _, row in sentiment_df.iterrows():
            logger.info(f"    Text: {row['text'][:50]}...")
            logger.info(f"    Score: {row.get('sentiment_score', 'N/A')}")
            logger.info(f"    Category: {row.get('category', 'N/A')}")
    
    logger.info("System integration testing complete.")

if __name__ == "__main__":
    logger.info("Starting enhancement tests...")
    
    # Run tests
    test_sentiment_analyzer()
    test_misinformation_detector()
    test_system_integration()
    
    logger.info("All tests completed successfully.")