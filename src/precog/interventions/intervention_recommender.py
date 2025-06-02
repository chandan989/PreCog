from typing import List, Dict, Any, Tuple, Optional
from collections import defaultdict

from ..core.data_models import SentimentAlert, FrictionRisk, MisinformationAlert, InterventionAction # Adjusted import path
from ..core.llm_clients import BaseLLMClient # New import for LLM
import logging # Add logging

logger = logging.getLogger(__name__) # Initialize logger

# Intervention Templates (Simplified)
INTERVENTION_TEMPLATES = {
    'misinformation': {
        'immediate': [
            "Issue pre-bunking/debunking message on official channels regarding '{narrative}'.",
            "Amplify verified information from trusted sources on social media.",
            "Monitor spread of '{narrative}' closely and report violating content to platforms."
        ],
        'short_term': [
            "Launch targeted awareness campaign about identifying misinformation related to '{topic}'.",
            "Collaborate with local influencers to spread factual information.",
            "Fact-check and publicly refute key false claims associated with '{narrative}'."
        ],
        'long_term': [
            "Develop media literacy programs for vulnerable communities.",
            "Strengthen partnerships with fact-checking organizations."
        ]
    },
    'civic_grievance': {
        'immediate': [
            "Acknowledge receipt of grievance regarding '{issue}' publicly.",
            "Assign relevant department to investigate '{issue}' and provide timeline."
        ],
        'short_term': [
            "Provide regular updates on the status of resolving '{issue}'.",
            "Organize a local meeting (if feasible) to discuss '{issue}' with affected citizens."
        ],
        'long_term': [
            "Implement systemic changes to address root cause of '{issue}'.",
            "Improve feedback mechanisms for civic issues."
        ]
    },
    'social_tension': {
        'immediate': [
            "Deploy peace-keeping forces to '{location}' if credible threat of violence.",
            "Issue calming messages from community leaders and authorities.",
            "Open lines of communication with representatives of involved groups."
        ],
        'short_term': [
            "Facilitate dialogue and mediation between conflicting groups.",
            "Address underlying grievances contributing to tension."
        ],
        'long_term': [
            "Promote inter-community harmony programs.",
            "Strengthen local governance and dispute resolution mechanisms."
        ]
    }
}

RESOURCE_REQUIREMENTS = {
    'communication_team': ['Issue statements', 'Run campaigns', 'Manage social media'],
    'field_officers': ['Local investigation', 'Community meeting', 'Peace-keeping presence'],
    'data_analysts': ['Monitor spread', 'Fact-check claims'],
    'legal_team': ['Report content', 'Address incitement'],
    'external_partners': ['Fact-checking orgs', 'Influencers', 'Community leaders']
}

class InterventionRecommender:
    def __init__(self, llm_client: Optional[BaseLLMClient] = None):
        self.llm_client = llm_client
        # Could load more sophisticated rules or ML models for recommendation here
        pass

    def _get_issue_type_from_alert(self, alert: Any) -> Optional[str]:
        if isinstance(alert, MisinformationAlert):
            return 'misinformation'
        if isinstance(alert, SentimentAlert):
            # Inferring civic grievance from highly negative sentiment about specific keywords
            # This is a simplification; real system would need more context.
            if alert.severity in ['high', 'critical'] and any(kw in alert.message.lower() for kw in ['water', 'electricity', 'road', 'corruption', 'garbage']):
                return 'civic_grievance'
        if isinstance(alert, FrictionRisk):
            if alert.risk_level > 0.7: # High friction risk could indicate social tension
                return 'social_tension'
        return None

    def _get_context_from_alert(self, alert: Any) -> Dict[str, str]:
        context = {'location': 'Unknown', 'narrative': 'general concern', 'issue': 'unspecified issue', 'topic': 'various topics'}
        if hasattr(alert, 'location_name') and alert.location_name: context['location'] = alert.location_name
        elif hasattr(alert, 'location') and alert.location: context['location'] = alert.location
        if hasattr(alert, 'narrative') and alert.narrative: context['narrative'] = alert.narrative
        if hasattr(alert, 'message') and alert.message: context['issue'] = alert.message[:50] + '...'
        if hasattr(alert, 'description') and alert.description: context['issue'] = alert.description[:50] + '...'
        if hasattr(alert, 'risk_factors') and alert.risk_factors: context['topic'] = ', '.join(alert.risk_factors)
        return context

    def _generate_llm_intervention_suggestion(self, issue_type: str, term: str, context: Dict[str, Any]) -> Optional[str]:
        """Generates intervention suggestion using LLM."""
        if not self.llm_client:
            return None

        # Construct a more detailed prompt for the LLM
        prompt = f"""
        Context: An issue of type '{issue_type}' has been identified.
        Details: {context.get('issue', 'N/A')}
        Location: {context.get('location', 'N/A')}
        Narrative (if applicable): {context.get('narrative', 'N/A')}
        Topic (if applicable): {context.get('topic', 'N/A')}

        Task: Suggest a specific and actionable '{term}' intervention for local authorities or community leaders in an Indian context.
        The suggestion should be concise (1-2 sentences) and practical.
        Avoid generic advice. Focus on the provided context.

        Example for 'misinformation' and 'immediate' term:
        "Immediately issue a clarification on official channels regarding the rumor about '{context.get('narrative', 'specific rumor')}' affecting '{context.get('location', 'area')}'."

        Example for 'civic_grievance' and 'short_term' term:
        "Organize a town hall meeting in '{context.get('location', 'area')}' within the next two weeks to discuss the '{context.get('issue', 'civic issue')}' and outline planned actions."

        Based on the above, provide your suggestion:
        """
        try:
            response = self.llm_client.generate_text(prompt, max_tokens=150)
            if response and response.strip():
                # Clean up response, remove potential conversational fluff
                cleaned_response = response.strip()
                # Remove common conversational prefixes if LLM adds them despite instructions
                prefixes_to_remove = ["Certainly, here's a suggestion:", "Here's a suggested intervention:", "A possible intervention could be:"]
                for prefix in prefixes_to_remove:
                    if cleaned_response.lower().startswith(prefix.lower()):
                        cleaned_response = cleaned_response[len(prefix):].strip()
                return cleaned_response
            logger.warning(f"LLM returned empty response for intervention suggestion: {issue_type}, {term}")
            return None
        except Exception as e:
            logger.error(f"Error generating LLM intervention suggestion: {e}")
            return None

    def recommend_interventions(
        self, 
        anomaly_alerts: List[SentimentAlert],
        friction_risks: List[FrictionRisk],
        misinfo_alerts: List[MisinformationAlert],
        use_llm_for_suggestions: bool = False # Flag to control LLM usage
    ) -> Dict[str, Any]:

        recommendations = {
            'immediate_actions': [],
            'short_term_actions': [],
            'long_term_actions': [],
            'priority_locations': set(),
            'resource_allocation': defaultdict(set)
        }

        all_alerts = anomaly_alerts + friction_risks + misinfo_alerts
        # Sort alerts by severity/risk score to prioritize
        # Custom sort key: Misinfo > Friction > Sentiment; then by severity/score
        def sort_key(alert):
            if isinstance(alert, MisinformationAlert):
                base_priority = 0
                severity_map = {'high': 0, 'critical':0, 'medium': 1, 'low': 2}
                return (base_priority, severity_map.get(alert.severity, 3), -getattr(alert, 'spread_velocity', 0))
            elif isinstance(alert, FrictionRisk):
                base_priority = 1
                return (base_priority, -alert.risk_level) 
            elif isinstance(alert, SentimentAlert):
                base_priority = 2
                severity_map = {'critical': 0, 'high': 1, 'medium': 2, 'low': 3}
                return (base_priority, severity_map.get(alert.severity, 4))
            return (3, 0) # Default for unknown types

        sorted_alerts = sorted(all_alerts, key=sort_key)

        for alert in sorted_alerts:
            issue_type = self._get_issue_type_from_alert(alert)
            if not issue_type: continue

            context = self._get_context_from_alert(alert)
            if hasattr(alert, 'location_name') and alert.location_name:
                recommendations['priority_locations'].add(alert.location_name)
            elif hasattr(alert, 'location') and alert.location:
                recommendations['priority_locations'].add(alert.location)

            # Add actions based on templates or LLM
            for term in ['immediate', 'short_term', 'long_term']:
                action_text = None
                if use_llm_for_suggestions and self.llm_client:
                    action_text = self._generate_llm_intervention_suggestion(issue_type, term, context)
                    if action_text:
                        logger.info(f"LLM generated suggestion for {issue_type} ({term}): {action_text}")
                    else:
                        logger.info(f"LLM failed to generate suggestion for {issue_type} ({term}), falling back to template.")

                # Fallback to template if LLM is not used, fails, or returns nothing
                if not action_text and issue_type in INTERVENTION_TEMPLATES and term in INTERVENTION_TEMPLATES[issue_type]:
                    # Pick one template randomly or the first one if LLM failed
                    # For simplicity, let's take the first template if LLM fails for a specific term
                    template_to_use = INTERVENTION_TEMPLATES[issue_type][term][0] 
                    action_text = template_to_use.format(**context) # Fill placeholders

                if action_text:
                    # Avoid duplicate actions for similar alerts if using templates primarily
                    # If LLM is used, it might generate unique suggestions, so check might be less critical or different
                    is_duplicate = False
                    if not (use_llm_for_suggestions and self.llm_client): # Only check for duplicates if primarily using templates
                        if action_text in recommendations[f'{term}_actions']:
                            is_duplicate = True

                    if not is_duplicate:
                        recommendations[f'{term}_actions'].append(action_text)
                        # Basic resource mapping
                        for res_type, keywords in RESOURCE_REQUIREMENTS.items():
                            if any(kw.lower() in action_text.lower() for kw in keywords):
                                recommendations['resource_allocation'][res_type].add(action_text.split('.')[0][:30] + '...') 

        # Convert sets to lists for JSON serializability
        recommendations['priority_locations'] = list(recommendations['priority_locations'])
        for res_type in recommendations['resource_allocation']:
            recommendations['resource_allocation'][res_type] = list(recommendations['resource_allocation'][res_type])

        # Limit number of recommendations per category for brevity
        recommendations['immediate_actions'] = recommendations['immediate_actions'][:5]
        recommendations['short_term_actions'] = recommendations['short_term_actions'][:3]
        recommendations['long_term_actions'] = recommendations['long_term_actions'][:2]

        return recommendations

    def _calculate_resource_needs(self, actions: List[InterventionAction]) -> Dict[str, int]:
        """Placeholder for a more detailed resource calculation."""
        needs = defaultdict(int)
        for action in actions:
            for resource in action.resources_needed:
                needs[resource] += 1 # Simple count, could be more complex (e.g., man-hours)
        return needs

    def _define_success_metrics(self, issue_type: str, narrative: Optional[str] = None) -> List[str]:
        """Define success metrics based on issue type."""
        metrics = [
            "Reduction in negative sentiment related to the issue.",
            "Decrease in spread velocity of misinformation (if applicable).",
            "Increased public trust and satisfaction (survey-based)."
        ]
        if issue_type == 'misinformation' and narrative:
            metrics.append(f"Reduced engagement with content related to '{narrative}'.")
        if issue_type == 'civic_grievance':
            metrics.append("Timely resolution of reported grievances.")
        return metrics
