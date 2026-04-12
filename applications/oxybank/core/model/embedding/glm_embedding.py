# -*- coding: utf-8 -*-
"""
Zhipu GLM Embedding implementation for LlamaIndex.

This module implements GLM (Zhipu AI) embedding models following
LlamaIndex BaseEmbedding interface.

GLM API: https://open.bigmodel.cn/api/paas/v4/embeddings
"""
from typing import List, Optional, Any
import requests
import orjson as json

from llama_index.core.base.embeddings.base import BaseEmbedding, Embedding
from llama_index.core.bridge.pydantic import Field


class GLMEmbedding(BaseEmbedding):
    """
    Zhipu GLM Embedding for LlamaIndex.

    Supports GLM embedding models:
    - embedding-2: 1024 dimensions

    Args:
        model_name: GLM model name (e.g., embedding-2, embedding-3)
        api_key: Zhipu API key
        api_base: API base URL (default: https://open.bigmodel.cn/api/paas/v4/embeddings)
        embed_batch_size: Batch size for embedding requests (default: 10)
        max_retries: Maximum number of retries (default: 3)
        timeout: Timeout for API requests in seconds (default: 60)
        normalization: Whether to normalize embeddings (default: True)
    """

    api_key: str = Field(description="The Zhipu GLM API key.")
    api_base: str = Field(
        default="https://open.bigmodel.cn/api/paas/v4/embeddings",
        description="The base URL for GLM API."
    )
    embed_batch_size: int = Field(
        default=10,
        description="The batch size for embedding calls.",
        gt=0,
        le=2048,
    )
    max_retries: int = Field(
        default=3,
        description="Maximum number of retries.",
        ge=0,
    )
    timeout: float = Field(
        default=60.0,
        description="Timeout for each request in seconds.",
        ge=0,
    )
    normalization: bool = Field(
        default=True,
        description="Whether to normalize embeddings.",
    )

    def __init__(
        self,
        model_name: str = "embedding-2",
        embed_batch_size: int = 10,
        max_retries: int = 3,
        timeout: float = 60.0,
        normalization: bool = True,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        # Handle api_key and api_base defaults
        if api_key is None:
            api_key = kwargs.get("api_key", "")
        if api_base is None:
            api_base = kwargs.get("api_base", "https://open.bigmodel.cn/api/paas/v4/embeddings")
        
        # Build kwargs for parent class
        parent_kwargs = {
            "model_name": model_name,
            "embed_batch_size": embed_batch_size,
            "max_retries": max_retries,
            "timeout": timeout,
            "normalization": normalization,
            "api_key": api_key,
            "api_base": api_base,
        }
        # Add any additional kwargs
        parent_kwargs.update(kwargs)
        
        # Initialize parent class
        super().__init__(**parent_kwargs)

    def _get_query_embedding(self, query: str) -> Embedding:
        """
        Embed input query synchronously.
        """
        return self._get_text_embedding(query)

    async def _aget_query_embedding(self, query: str) -> Embedding:
        """
        Embed input query asynchronously.
        """
        return self._get_text_embedding(query)

    def _get_text_embedding(self, text: str) -> Embedding:
        """
        Embed input text synchronously.

        Returns:
            Embedding result as a list of floats.
        """
        return self._get_text_embeddings([text])[0]

    async def _aget_text_embedding(self, text: str) -> Embedding:
        """
        Embed input text asynchronously.

        Returns:
            Embedding result as a list of floats.
        """
        return self._get_text_embedding(text)

    def _get_text_embeddings(self, texts: List[str]) -> List[Embedding]:
        """
        Embed input sequence of text synchronously.

        Supports batch processing for efficiency.

        Returns:
            List of embeddings.
        """
        all_embeddings = []
        for i in range(0, len(texts), self.embed_batch_size):
            batch = texts[i: i + self.embed_batch_size]
            batch_embeddings = self._request_embedding(batch)
            all_embeddings.extend(batch_embeddings)
        return all_embeddings

    async def _aget_text_embeddings(self, texts: List[str]) -> List[Embedding]:
        """
        Embed input sequence of text asynchronously.

        Returns:
            List of embeddings.
        """
        return self._get_text_embeddings(texts)

    def _request_embedding(self, texts: List[str]) -> List[List[float]]:
        """
        Request GLM API for embeddings with retry logic.

        Args:
            texts: List of texts to embed

        Returns:
            List of embeddings (each embedding is a list of floats)

        Raises:
            Exception: If API request fails after all retries.
        """
        payload = {
            "model": self.model_name,
            "input": texts
        }

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        last_error = None
        for attempt in range(self.max_retries):
            try:
                response = requests.post(
                    url=self.api_base,
                    data=json.dumps(payload),
                    headers=headers,
                    timeout=self.timeout
                )

                if response.status_code != 200:
                    error_msg = f"GLM API error: {response.status_code} - {response.text}"
                    if attempt < self.max_retries - 1:
                        continue
                    else:
                        raise Exception(error_msg)

                result = response.json()

                if "data" not in result:
                    error_msg = result.get("error", {}).get("message", "Unknown error")
                    raise Exception(f"GLM API error: {error_msg}")

                embeddings = [item["embedding"] for item in result["data"]]
                return embeddings

            except requests.exceptions.Timeout as e:
                last_error = f"Request timeout: {e}"
                if attempt < self.max_retries - 1:
                    continue
                else:
                    raise Exception(f"GLM API timeout after {self.max_retries} attempts: {e}")

            except requests.exceptions.RequestException as e:
                last_error = f"Request error: {e}"
                if attempt < self.max_retries - 1:
                    continue
                else:
                    raise Exception(f"GLM API request failed after {self.max_retries} attempts: {e}")

            except (json.JSONDecodeError, KeyError) as e:
                last_error = f"Response parsing error: {e}"
                if attempt < self.max_retries - 1:
                    continue
                else:
                    raise Exception(f"GLM API response parsing error: {e}")

        # If all retries failed
        raise Exception(f"GLM API request failed after {self.max_retries} attempts. Last error: {last_error}")

    @classmethod
    def class_name(cls) -> str:
        """Return the class name."""
        return "GLMEmbedding"
