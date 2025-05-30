# src/ai_models/intervention_recommender.py
from typing import List, Dict, Any

# Define a more structured recommendation format
class InterventionRecommendation:
    def __init__(self, issue_trigger: Dict[str, Any], priority: str, category: str, 
                 short_term_actions: List[str], medium_term_actions: List[str], 
                 long_term_actions: List[str], responsible_actors: List[str], 
                 monitoring_indicators: List[str]):
        self.issue_trigger = issue_trigger
        self.priority = priority # e.g., "High", "Medium", "Low"
        self.category = category # e.g., "Public Communication", "Community Engagement", "Resource Allocation", "Policy Review"
        self.short_term_actions = short_term_actions
        self.medium_term_actions = medium_term_actions
        self.long_term_actions = long_term_actions
        self.responsible_actors = responsible_actors # e.g., ["Local Police", "Municipal Corp", "NGOs"]
        self.monitoring_indicators = monitoring_indicators # e.g., ["Reduction in rumor spread", "Improved sentiment scores"]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "issue_trigger_summary": self.issue_trigger.get('description', 'N/A'),
            "issue_details": self.issue_trigger,
            "priority": self.priority,
            "category": self.category,
            "short_term_actions": self.short_term_actions,
            "medium_term_actions": self.medium_term_actions,
            "long_term_actions": self.long_term_actions,
            "responsible_actors": self.responsible_actors,
            "monitoring_indicators": self.monitoring_indicators
        }

def recommend_interventions(detected_issues: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Recommends interventions based on detected issues.
    `detected_issues` is a list of dictionaries, where each dictionary represents an anomaly or concern.
    Each issue dict is expected to have at least 'type' and 'description'. 
    Other fields like 'severity', 'location', 'keywords' can be used for more nuanced recommendations.
    """
    recommendations = []
    if not detected_issues:
        return recommendations

    for issue in detected_issues:
        issue_type = issue.get('type', 'Unknown').lower()
        issue_description = issue.get('description', 'No specific description provided.')
        # severity = issue.get('severity', 'medium') # Example: could be 'low', 'medium', 'high'
        # location = issue.get('location_info', None) # Example: GPS coords, district name

        # Default values
        priority = "Medium"
        category = "General Investigation"
        short_term = ["Verify the issue details and scope through multiple channels."]
        medium_term = ["Engage with relevant community leaders/stakeholders for perspective."]
        long_term = ["Review existing protocols for handling similar issues."]
        actors = ["Cross-departmental Task Force"]
        indicators = ["Clarity on issue specifics", "Stakeholder feedback report"]

        if "misinformation" in issue_type or "rumor" in issue_type or "keyword velocity spike" in issue_type:
            priority = "High"
            category = "Public Communication & Counter-Information"
            short_term = [
                "Rapidly deploy verified information through official channels (social media, press releases, local announcements).",
                "Identify and engage key influencers to disseminate correct information.",
                "Monitor online platforms for the spread and mutation of the misinformation."
            ]
            medium_term = [
                "Conduct targeted awareness campaigns on media literacy and identifying fake news.",
                "Collaborate with fact-checking organizations."
            ]
            long_term = ["Develop a proactive communication strategy for potential crisis scenarios."]
            actors = ["Press Information Bureau (PIB)", "Local Administration", "Cyber Cell", "Community Volunteers"]
            indicators = ["Reduction in shares/mentions of misinformation", "Increase in engagement with official clarifications", "Sentiment shift on related topics"]
        
        elif "sentiment spike" in issue_type and "negative" in issue_description.lower():
            priority = "High"
            category = "Community Engagement & Grievance Redressal"
            short_term = [
                "Deploy community liaison officers or social workers to the affected area for on-ground assessment and dialogue.",
                "Acknowledge the negative sentiment publicly and express intent to understand/address.",
                "Identify specific grievances contributing to the negative sentiment."
            ]
            medium_term = [
                "Organize community meetings or forums to discuss concerns and gather feedback.",
                "If specific service failures are identified, initiate corrective actions and communicate them."
            ]
            long_term = ["Establish a permanent feedback mechanism for the community.", "Review and improve service delivery based on findings."]
            actors = ["Local Administration", "Social Welfare Department", "Elected Representatives", "NGOs"]
            indicators = ["Improvement in sentiment scores in the affected area/topic", "Reduction in complaints", "Increased participation in community dialogues"]

        elif "civic grievance" in issue_type or any(kw in issue_description.lower() for kw in ["water supply", "electricity", "sanitation", "road repair"]):
            priority = "Medium" if "High" not in priority else "High" # Can be escalated by sentiment
            category = "Service Delivery & Infrastructure"
            short_term = [
                "Acknowledge receipt of the grievance and provide a timeline for initial assessment.",
                "Dispatch a technical team to assess the specific civic issue (e.g., water leak, power outage zone).",
                "Provide interim relief if possible (e.g., water tankers, temporary power restoration updates)."
            ]
            medium_term = [
                "Implement necessary repairs or service restoration.",
                "Communicate progress and expected completion timelines to affected citizens regularly."
            ]
            long_term = ["Review maintenance schedules and infrastructure capacity to prevent recurrence.", "Allocate budget for necessary upgrades."]
            actors = ["Municipal Corporation", "Public Works Department", "Utility Companies", "Ward Officers"]
            indicators = ["Resolution of the specific grievance", "Restoration of service", "Citizen satisfaction feedback"]
        
        else: # Default for unknown or less defined issues
            priority = "Low" if "High" not in priority and "Medium" not in priority else priority
            category = "Exploratory Investigation"
            short_term.append("Conduct deeper analysis of the data associated with the issue (e.g., related posts, user networks).")
            medium_term.append("Consult subject matter experts if the issue domain is unclear.")
            long_term.append("If patterns emerge, develop specific protocols for this new issue type.")
            actors.append("Data Analysis Team")
            indicators.append("Comprehensive report on the nature and potential impact of the issue.")

        recommendation_obj = InterventionRecommendation(
            issue_trigger=issue,
            priority=priority,
            category=category,
            short_term_actions=short_term,
            medium_term_actions=medium_term,
            long_term_actions=long_term,
            responsible_actors=actors,
            monitoring_indicators=indicators
        )
        recommendations.append(recommendation_obj.to_dict())
        # print(f"Generated recommendation for issue type: '{issue_type}' -> Priority: {priority}")

    return recommendations

if __name__ == '__main__':
    sample_issues_from_detector = [
        {
            "type": "Keyword Velocity Spike",
            "keyword": "protestmarch",
            "count_in_window": 15,
            "time_window_minutes": 60,
            "start_time_of_window": "2023-04-01T10:00:00Z",
            "end_time_of_window": "2023-04-01T11:00:00Z",
            "description": "Keyword 'protestmarch' appeared 15 times in 60 mins ending at 2023-04-01T11:00:00Z.",
            "triggering_data_indices": [10, 12, 15, 22, 23, 24, 25, 28, 30, 31, 32, 33, 34, 35, 36],
            "location_info": "City Center Plaza"
        },
        {
            "type": "Sentiment Spike",
            "index_start": 50,
            "index_end": 58,
            "description": "Potential sentiment anomaly: 7/9 items highly negative (score < -0.6) in window ending at index 58, related to 'hospital services'.",
            "triggering_data_indices": [50, 51, 52, 54, 55, 57, 58],
            "keywords": ["hospital", "negligence", "poor service"]
        },
        {
            "type": "Civic Grievance", 
            "description": "Multiple reports of 'no water supply' in 'Sector 15' for '3 days'.",
            "source": "Crowdsourced Reports",
            "urgency_score": 0.8 # Example additional field
        },
        {
            "type": "Unknown Anomaly", 
            "description": "Unusual coordinated messaging pattern detected from a cluster of new accounts discussing 'election dates'.",
            "pattern_id": "COORDMSG_003",
            "confidence": 0.75
        }
    ]

    recommended_actions = recommend_interventions(sample_issues_from_detector)

    if recommended_actions:
        print(f"\nTotal recommendations generated: {len(recommended_actions)}")
        for i, rec in enumerate(recommended_actions):
            print(f"\n--- Recommendation {i+1} ---")
            print(f"Trigger: {rec['issue_trigger_summary']}")
            print(f"  Priority: {rec['priority']}")
            print(f"  Category: {rec['category']}")
            print(f"  Responsible: {', '.join(rec['responsible_actors'])}")
            print(f"  Short-term Actions:")
            for action in rec['short_term_actions']:
                print(f"    - {action}")
            if rec['medium_term_actions']:
                print(f"  Medium-term Actions:")
                for action in rec['medium_term_actions']:
                    print(f"    - {action}")
            print(f"  Monitoring Indicators: {', '.join(rec['monitoring_indicators'])}")
            # print(f"  Full Trigger Details: {rec['issue_details']}") # For verbosity
    else:
        print("\nNo interventions recommended for the sample issues.")