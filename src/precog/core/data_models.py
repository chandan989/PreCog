from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple, Optional
from datetime import datetime

@dataclass
class SentimentAlert:
    location_name: str # Changed from 'location' for clarity
    timestamp: datetime
    severity: str  # e.g., 'low', 'medium', 'high', 'critical'
    message: str # Renamed from 'description' for consistency with other alerts if needed, or keep as description
    sentiment_score: float
    contributing_factors: List[str] = field(default_factory=list)
    location_lat: Optional[float] = None
    location_lon: Optional[float] = None

@dataclass
class FrictionRisk:
    location_name: str # Changed from 'location' for clarity
    risk_level: float  # Renamed from risk_score for consistency with app.py
    predicted_timeline: str  # e.g., 'imminent', 'short-term', 'medium-term'
    primary_contributing_factor: str # Simplified from risk_factors for dashboard display
    explanation: str # To hold more detailed explanation
    confidence: float
    location_lat: Optional[float] = None
    location_lon: Optional[float] = None
    # risk_factors: List[str] # Kept for internal use, but primary_contributing_factor for display

@dataclass
class MisinformationAlert:
    narrative: str
    source_type: str # Renamed from source_platform for consistency with app.py
    timestamp: datetime
    severity: str
    spread_velocity: float  # e.g., posts per hour
    potential_impact: str
    confidence: float
    counter_narrative_suggestion: str # Simplified from list for dashboard display
    affected_locations: List[str] = field(default_factory=list) # These are location names
    credibility_score: Optional[float] = None
    # For map, we might need a primary location for the alert origin if applicable
    origin_location_name: Optional[str] = None
    origin_location_lat: Optional[float] = None
    origin_location_lon: Optional[float] = None

@dataclass
class InterventionAction:
    action_id: str
    description: str
    type: str # 'immediate', 'short-term', 'long-term'
    target_issue: str # 'misinformation', 'civic_grievance', 'social_tension'
    resources_needed: List[str]
    estimated_impact: str
    success_metrics: List[str]

@dataclass
class NewsArticle:
    id: str
    title: str
    text: str
    source: str
    publish_date: datetime
    url: str
    credibility_score: Optional[float] = None
    topics: List[str] = field(default_factory=list)