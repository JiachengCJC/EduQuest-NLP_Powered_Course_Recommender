# recommender/client.py
import asyncio
import ollama
from typing import List

class LocalOllamaClient:
    """
    Wrapper around local Ollama models for text generation and embeddings.
    """

    def __init__(
        self,
        generator_model: str = "mistral",
        rec_model: str = "qwen2.5:7b-instruct",
        embedding_model: str = "nomic-embed-text",
    ):
        self.generator_model = generator_model
        self.rec_model = rec_model
        self.embedding_model = embedding_model

    async def generate_text(self, prompt: str, model: str, max_tokens: int = 8000) -> str:
        """
        Generate text using a local Ollama model.
        """
        loop = asyncio.get_event_loop()

        def _call():
            return ollama.chat(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                options={"num_predict": max_tokens, "temperature": 0.0},
            )

        resp = await loop.run_in_executor(None, _call)
        return resp["message"]["content"]

    async def generate_embedding(self, text: str) -> List[float]:
        """
        Generate embeddings for a given text using Ollama embedding model.
        """
        loop = asyncio.get_event_loop()

        def _call():
            return ollama.embeddings(model=self.embedding_model, prompt=text)

        resp = await loop.run_in_executor(None, _call)
        return resp["embedding"]
