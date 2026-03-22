# recommender/client.py
import os
from typing import List

from dotenv import load_dotenv
from openai import AsyncOpenAI

load_dotenv()


class LocalVLLMClient:
    """
    Wrapper around OpenAI-compatible vLLM endpoints for generation and embeddings.
    """

    def __init__(
        self,
        generator_model: str = "",
        rec_model: str = "",
        embedding_model: str = "",
        chat_base_url: str = "",
        embedding_base_url: str = "",
        api_key: str = "",
        timeout: float = 360.0,
    ):
        self.generator_model = generator_model or os.getenv("VLLM_GENERATOR_MODEL", "qwen-local")
        self.rec_model = rec_model or os.getenv("VLLM_REC_MODEL", "qwen-local")
        self.embedding_model = embedding_model or os.getenv("VLLM_EMBEDDING_MODEL", "bge-local")

        resolved_api_key = api_key or os.getenv("VLLM_API_KEY", "EMPTY")
        resolved_chat_base_url = chat_base_url or os.getenv("VLLM_CHAT_BASE_URL", "http://127.0.0.1:8000/v1")
        resolved_embedding_base_url = embedding_base_url or os.getenv(
            "VLLM_EMBEDDING_BASE_URL",
            resolved_chat_base_url,
        )

        self.chat_client = AsyncOpenAI(
            api_key=resolved_api_key,
            base_url=resolved_chat_base_url,
            timeout=timeout,
        )
        self.embedding_client = AsyncOpenAI(
            api_key=resolved_api_key,
            base_url=resolved_embedding_base_url,
            timeout=timeout,
        )

    async def generate_text(self, prompt: str, model: str, max_tokens: int = 100000) -> str:
        """
        Generate text using a model served by vLLM's OpenAI-compatible API.
        """
        resp = await self.chat_client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=0.0,
        )
        print(f"Debug: Received response from vLLM: {resp.choices[0].message.content}")
        return resp.choices[0].message.content or ""

    async def generate_embedding(self, text: str) -> List[float]:
        """
        Generate embeddings using a model served by vLLM's embeddings endpoint.
        """
        resp = await self.embedding_client.embeddings.create(
            model=self.embedding_model,
            input=text,
        )
        return resp.data[0].embedding


# Backward compatibility for old imports while migrating from Ollama.
# LocalOllamaClient = LocalVLLMClient


