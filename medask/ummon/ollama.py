"""
Ollama LLM provider for open-source models (Llama, Qwen, DeepSeek R1, Mistral, etc.)
"""
import json
from typing import Dict, List, Optional

from medask.models.comms.models import CMessage
from medask.models.orm.models import Role
from medask.util.client import post
from medask.util.decorator import timeit
from medask.util.log import get_logger
from medask.ummon.base import BaseUmmon

logger = get_logger("ummon.ollama")

# Default Ollama URL
DEFAULT_OLLAMA_URL = "http://localhost:11434"


class UmmonOllama(BaseUmmon):
    """
    Ollama provider for open-source LLMs.
    
    Supports models like: llama3.1, qwen2.5, deepseek-r1, mistral, etc.
    Requires Ollama to be installed and running locally.
    """
    
    def __init__(self, model: str, url: str = DEFAULT_OLLAMA_URL) -> None:
        """
        Initialize Ollama client.
        
        Args:
            model: Model name (e.g., "llama3.1", "qwen2.5", "deepseek-r1")
            url: Ollama server URL (default: http://localhost:11434)
        """
        self._model = model
        self._url = url.rstrip("/")  # Remove trailing slash
        
    def _converse_raw(self, history: List[Dict[str, str]]) -> str:
        """
        Send messages to Ollama API and return response.
        
        Args:
            history: List of message dicts with "role" and "content" keys
            
        Returns:
            Response text from the model
            
        Raises:
            RuntimeError: If API request fails or model not found
        """
        # Prepare request body
        body = json.dumps({
            "model": self._model,
            "messages": history,
            "stream": False,
        })
        
        try:
            # Ollama API endpoint: /api/chat
            resp = post("api/chat", body=body, url=self._url)
            
            # Ollama response format: {"message": {"content": "..."}}
            if "message" in resp and "content" in resp["message"]:
                return resp["message"]["content"]
            else:
                raise RuntimeError(f"Unexpected Ollama response format: {resp}")
                
        except RuntimeError as e:
            error_msg = str(e)
            # Check for common errors and provide helpful messages
            if "404" in error_msg or "model" in error_msg.lower():
                raise RuntimeError(
                    f"Model '{self._model}' not found in Ollama. "
                    f"Please run: ollama pull {self._model}"
                ) from e
            elif "connection" in error_msg.lower() or "refused" in error_msg.lower():
                raise RuntimeError(
                    f"Cannot connect to Ollama at {self._url}. "
                    f"Make sure Ollama is running: ollama serve"
                ) from e
            else:
                raise RuntimeError(f"Ollama API error: {error_msg}") from e
    
    def _raw_to_out(self, user_id: int, chat_id: int, raw: str) -> CMessage:
        """Convert raw response to CMessage."""
        return CMessage(
            user_id=user_id,
            chat_id=chat_id,
            role=Role.ASSISTANT,
            body=raw,
        )
    
    @timeit(logger, log_kwargs=False)
    def inquire(self, prompt: CMessage) -> CMessage:
        """Single-turn inquiry."""
        prompt_raw = prompt.to_openai()
        retort: str = self._converse_raw([prompt_raw])
        return self._raw_to_out(prompt.user_id, prompt.chat_id, retort)
    
    @timeit(logger, log_kwargs=False)
    def converse(self, history: List[CMessage]) -> CMessage:
        """Multi-turn conversation."""
        history_raw = [msg.to_openai() for msg in history]
        retort: str = self._converse_raw(history_raw)
        
        msg = history[-1]
        return self._raw_to_out(msg.user_id, msg.chat_id, retort)

