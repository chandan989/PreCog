import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
from typing import Optional

# Sample data for generation
SAMPLE_LOCATIONS_INDIA = [
    'Mumbai, Maharashtra', 'Delhi, Delhi', 'Bangalore, Karnataka', 'Hyderabad, Telangana',
    'Chennai, Tamil Nadu', 'Kolkata, West Bengal', 'Pune, Maharashtra', 'Ahmedabad, Gujarat',
    'Jaipur, Rajasthan', 'Lucknow, Uttar Pradesh', 'Kanpur, Uttar Pradesh', 'Nagpur, Maharashtra',
    'Indore, Madhya Pradesh', 'Thane, Maharashtra', 'Bhopal, Madhya Pradesh', 'Patna, Bihar'
]

SAMPLE_TEXT_TEMPLATES = {
    'positive': [
        "Loving the new park in {location}! #community #goodgovernance",
        "Great initiative by local authorities for {topic}. #positivechange",
        "Feeling hopeful about developments in {location}.",
        "Thank you for addressing the {issue} issue promptly!",
        "बहुत अच्छा काम हो रहा है {location} में। #विकास"
    ],
    'negative': [
        "Terrible traffic jam again in {location}. When will this be fixed? #civicissue",
        "Power outage for hours in {location}! This is unacceptable. @ কর্তৃপক্ষ",
        "Concerned about the rising {problem} in our area. #safety",
        "{location} में {समस्या} से परेशान हूँ। कोई सुनवाई नहीं।",
        "This new policy is a disaster for people in {location}."
    ],
    'neutral': [
        "Attended a meeting about {topic} in {location} today.",
        "Weather in {location} is quite {weather_condition} today.",
        "{location} market seems busy as usual.",
        "Just observing the situation regarding {issue} in {location}."
    ],
    'grievance': [
        "No water supply in {location} for 3 days! #watercrisis @localMLA",
        "The roads in {ward_name}, {location} are in pathetic condition. #badroads",
        "Garbage not collected in {area_name} for a week. #swachhbharat?",
        "Endless wait times at the {government_office} in {location}. #inefficiency",
        "{location} में बिजली की कटौती से जनता परेशान। #powercut"
    ],
    'misinformation_rumor': [
        "Heard a rumor that {rumor_topic} is happening in {location}. Can anyone confirm? #fakenewsalert",
        "Forwarded as received: {sensational_claim} in {location}! Is this true?",
        "Warning! Unverified news about {event_description} spreading in {location}.",
        "They are saying {conspiracy_theory_snippet} about the {group_or_entity} in {location}."
    ]
}

SAMPLE_TOPICS = ['infrastructure', 'public safety', 'health services', 'education', 'local events', 'environment']
SAMPLE_ISSUES = ['traffic', 'water shortage', 'power cuts', 'waste management', 'crime', 'pollution']
SAMPLE_PROBLEMS = ['theft', 'noise pollution', 'stray animals', 'illegal construction']
SAMPLE_RUMOR_TOPICS = ['a new lockdown', 'a protest march', 'a celebrity visit', 'a natural disaster warning']
SAMPLE_SENSATIONAL_CLAIMS = ['all schools closing tomorrow', 'free money distribution', 'a dangerous chemical spill']

class SyntheticDataGenerator:
    def __init__(self, seed: Optional[int] = None):
        self.seed = seed
        if self.seed is not None:
            random.seed(self.seed)
            np.random.seed(self.seed)

    def _generate_timestamp(self, days_past=30) -> datetime:
        return datetime.now() - timedelta(days=np.random.randint(0, days_past), 
                                         hours=np.random.randint(0, 24),
                                         minutes=np.random.randint(0, 60))

    def generate_social_media_data(self, num_records: int = 1000) -> pd.DataFrame:
        data = []
        for i in range(num_records):
            location = random.choice(SAMPLE_LOCATIONS_INDIA)
            post_type = random.choices(list(SAMPLE_TEXT_TEMPLATES.keys()), weights=[0.25, 0.3, 0.2, 0.15, 0.1], k=1)[0]
            template = random.choice(SAMPLE_TEXT_TEMPLATES[post_type])
            
            context = {
                'location': location.split(',')[0], # City name
                'topic': random.choice(SAMPLE_TOPICS),
                'issue': random.choice(SAMPLE_ISSUES),
                'problem': random.choice(SAMPLE_PROBLEMS),
                'समस्या': random.choice(SAMPLE_PROBLEMS), # Add Hindi key for problem
                'weather_condition': random.choice(['sunny', 'cloudy', 'rainy', 'hot']),
                'ward_name': f"Ward {random.randint(1,20)}",
                'area_name': f"{random.choice(['North', 'South', 'East', 'West'])} {location.split(',')[0]}",
                'government_office': random.choice(['Municipal Office', 'Police Station', 'Collectorate']),
                'rumor_topic': random.choice(SAMPLE_RUMOR_TOPICS),
                'sensational_claim': random.choice(SAMPLE_SENSATIONAL_CLAIMS),
                'event_description': 'a local festival', # simplified
                'conspiracy_theory_snippet': 'a secret plan is underway',
                'group_or_entity': random.choice(['the government', 'a large corporation', 'outsiders'])
            }
            text = template.format(**context)
            
            data.append({
                'id': f'post_{i:04d}',
                'text': text,
                'timestamp': self._generate_timestamp(),
                'user_id': f'user_{np.random.randint(1000, 9999)}',
                'location_approx': location,
                'source_platform': random.choice(['Twitter', 'Facebook', 'Instagram', 'LocalForum']),
                'likes': np.random.randint(0, 500),
                'shares': np.random.randint(0, 100),
                'language': random.choices(['en', 'hi', 'hinglish'], weights=[0.5, 0.2, 0.3], k=1)[0]
            })
        return pd.DataFrame(data)

    def generate_news_articles_data(self, num_records: int = 200) -> pd.DataFrame:
        data = []
        sources = ['Times of City', 'Local Chronicle', 'State News Network', 'Independent Reporter', 'Community Voice Portal']
        for i in range(num_records):
            location_city = random.choice(SAMPLE_LOCATIONS_INDIA).split(',')[0]
            topic = random.choice(SAMPLE_TOPICS)
            headline_verb = random.choice(['discusses', 'reports on', 'investigates', 'highlights', 'questions'])
            title = f"{location_city} {headline_verb.capitalize()} {topic.capitalize()} {random.choice(['Issues', 'Developments', 'Concerns'])}"
            text_snippet = f"This article covers recent {topic} related activities in {location_city}. Various stakeholders have expressed their views. The situation is evolving."
            
            data.append({
                'id': f'news_{i:04d}',
                'title': title,
                'text': text_snippet + " " + "Lorem ipsum dolor sit amet, consectetur adipiscing elit." * np.random.randint(1,4), # Longer text
                'source': random.choice(sources),
                'publish_date': self._generate_timestamp(days_past=60),
                'url': f'http://fakenewsdomain.example.com/{topic.replace(" ", "-")}/{location_city.lower()}/article{i}',
                'category': topic,
                'location_mentioned': location_city
            })
        return pd.DataFrame(data)

    def generate_all_synthetic_data(self, num_social=1000, num_news=200) -> dict[str, pd.DataFrame]:
        return {
            'social_media': self.generate_social_media_data(num_social),
            'news_articles': self.generate_news_articles_data(num_news)
        }

# Example usage:
if __name__ == '__main__':
    generator = SyntheticDataGenerator(random_seed=42)
    
    social_df = generator.generate_social_media_data(5)
    print("--- Synthetic Social Media Data ---")
    print(social_df[['text', 'location_approx', 'timestamp']])
    
    news_df = generator.generate_news_articles_data(2)
    print("\n--- Synthetic News Articles Data ---")
    print(news_df[['title', 'source', 'publish_date']])

    all_data = generator.generate_all_synthetic_data(num_social=10, num_news=3)
    print(f"\nGenerated {len(all_data['social_media'])} social posts and {len(all_data['news_articles'])} news articles.")