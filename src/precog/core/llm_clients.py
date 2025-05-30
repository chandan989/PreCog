import os
import logging
from typing import Optional, Dict, Any

# Attempt to import Azure OpenAI and Vertex AI SDKs
try:
    from openai import AzureOpenAI
except ImportError:
    AzureOpenAI = None
    logging.warning("Azure OpenAI SDK not found. AzureLLMClient will not be available.")

try:
    import vertexai
    from vertexai.generative_models import GenerativeModel
except ImportError:
    vertexai = None
    GenerativeModel = None
    logging.warning("Google Vertex AI SDK not found. VertexAILLMClient will not be available.")

logger = logging.getLogger(__name__)

class BaseLLMClient:
    def __init__(self, config: Dict[str, Any]):
        self.config = config

    def generate_text(self, prompt: str, max_tokens: int = 150) -> Optional[str]:
        raise NotImplementedError("Subclasses must implement this method.")

class AzureLLMClient(BaseLLMClient):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        if not AzureOpenAI:
            logger.error("Azure OpenAI SDK is not installed. Cannot initialize AzureLLMClient.")
            self.client = None
            return

        self.api_key = config.get("AZURE_OPENAI_API_KEY")
        self.endpoint = config.get("AZURE_OPENAI_ENDPOINT")
        self.api_version = config.get("AZURE_OPENAI_API_VERSION")
        self.deployment_name = config.get("AZURE_OPENAI_DEPLOYMENT_NAME")

        if not all([self.api_key, self.endpoint, self.deployment_name]):
            logger.error("Azure OpenAI API key, endpoint, or deployment name not configured.")
            self.client = None
            return
        
        if self.api_key == "YOUR_AZURE_OPENAI_API_KEY_HERE":
            logger.warning("Using placeholder Azure OpenAI API Key. AzureLLMClient may not function.")
            self.client = None # Or allow initialization but expect failures
            return

        try:
            self.client = AzureOpenAI(
                api_key=self.api_key,
                api_version=self.api_version,
                azure_endpoint=self.endpoint
            )
        except Exception as e:
            logger.error(f"Failed to initialize Azure OpenAI client: {e}")
            self.client = None

    def generate_text(self, prompt: str, max_tokens: int = 500) -> Optional[str]:
        if not self.client or not self.deployment_name:
            logger.error("AzureLLMClient not initialized or deployment name missing.")
            return None
        try:
            response = self.client.chat.completions.create(
                model=self.deployment_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error generating text with Azure OpenAI: {e}")
            return None

class VertexAILLMClient(BaseLLMClient):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        if not vertexai or not GenerativeModel:
            logger.error("Vertex AI SDK is not installed. Cannot initialize VertexAILLMClient.")
            self.model = None
            return

        self.project_id = config.get("VERTEX_AI_PROJECT_ID")
        self.location = config.get("VERTEX_AI_LOCATION")
        self.model_name = config.get("VERTEX_AI_MODEL_NAME")

        if not all([self.project_id, self.location, self.model_name]):
            logger.error("Vertex AI project ID, location, or model name not configured.")
            self.model = None
            return

        if self.project_id == "YOUR_VERTEX_AI_PROJECT_ID_HERE":
            logger.warning("Using placeholder Vertex AI Project ID. VertexAILLMClient may not function.")
            self.model = None
            return
        
        try:
            vertexai.init(project=self.project_id, location=self.location)
            self.model = GenerativeModel(self.model_name)
        except Exception as e:
            logger.error(f"Failed to initialize Vertex AI client or model: {e}")
            self.model = None

    def generate_text(self, prompt: str, max_tokens: int = 500) -> Optional[str]: # max_tokens for Vertex is controlled differently
        if not self.model:
            logger.error("VertexAILLMClient not initialized.")
            return None
        try:
            # For Gemini, max_output_tokens is part of generation_config
            generation_config = {"max_output_tokens": max_tokens}
            response = self.model.generate_content(prompt, generation_config=generation_config)
            return response.text
        except Exception as e:
            logger.error(f"Error generating text with Vertex AI: {e}")
            return None

def get_llm_client(config: Dict[str, Any], preferred_client: str = 'azure') -> Optional[BaseLLMClient]:
    """Factory function to get an LLM client based on configuration and preference."""
    if preferred_client == 'azure':
        if AzureOpenAI and config.get("AZURE_OPENAI_API_KEY") != "YOUR_AZURE_OPENAI_API_KEY_HERE":
            logger.info("Attempting to initialize AzureLLMClient.")
            client = AzureLLMClient(config)
            if client.client: return client
        if vertexai and config.get("VERTEX_AI_PROJECT_ID") != "YOUR_VERTEX_AI_PROJECT_ID_HERE": # Fallback to Vertex
            logger.info("Azure client failed or not preferred, attempting VertexAILLMClient as fallback.")
            client = VertexAILLMClient(config)
            if client.model: return client
    elif preferred_client == 'vertex':
        if vertexai and config.get("VERTEX_AI_PROJECT_ID") != "YOUR_VERTEX_AI_PROJECT_ID_HERE":
            logger.info("Attempting to initialize VertexAILLMClient.")
            client = VertexAILLMClient(config)
            if client.model: return client
        if AzureOpenAI and config.get("AZURE_OPENAI_API_KEY") != "YOUR_AZURE_OPENAI_API_KEY_HERE": # Fallback to Azure
            logger.info("Vertex client failed or not preferred, attempting AzureLLMClient as fallback.")
            client = AzureLLMClient(config)
            if client.client: return client
            
    logger.warning(f"Could not initialize any LLM client based on preference '{preferred_client}' and configuration.")
    return None