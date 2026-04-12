"""
Embedding Factory Implementation

Contains EmbeddingType enum and EmbeddingFactory class
for creating GLM embedding backends.

Features:
- Auto-loads API keys from environment variables
- Supports GLM embedding models (embedding-2)
- Unified interface with customizable parameters
"""

import os
from enum import Enum
from typing import Dict, Any, Optional, Union


class EmbeddingType(Enum):
    """Supported embedding backend types"""
    GLM = "glm"


class EmbeddingFactory:
    """
    Embedding Factory Class

    Provides unified interface for creating GLM embedding backends.
    Automatically loads API keys from environment variables when needed.
    """

    # Predefined configurations
    EMBEDDING_CONFIGS = {
        EmbeddingType.GLM: {
            "description": "Zhipu GLM Embedding (embedding-2)",
            "default_params": {
                "model": "embedding-2",
                "api_base": "https://open.bigmodel.cn/api/paas/v4/embeddings"
            },
            "env_var": "EMBEDDING_API_KEY",
            "required_packages": [],
        }
    }

    @classmethod
    def _normalize_embedding_type(
        cls, embedding_type: Union[EmbeddingType, str]
    ) -> EmbeddingType:
        """
        Normalize embedding type input to EmbeddingType enum

        Args:
            embedding_type: Embedding type (enum or string)

        Returns:
            EmbeddingType enum value

        Raises:
            ValueError: If embedding_type is invalid string or unsupported type
        """
        if isinstance(embedding_type, EmbeddingType):
            return embedding_type
        elif isinstance(embedding_type, str):
            # Try to find enum by value
            embedding_type_lower = embedding_type.lower()
            for enum_member in EmbeddingType:
                if enum_member.value.lower() == embedding_type_lower:
                    return enum_member
            # If not found, raise error with helpful message
            valid_values = [e.value for e in EmbeddingType]
            raise ValueError(
                f"Invalid embedding type string: '{embedding_type}'. "
                f"Valid values are: {valid_values} or use EmbeddingType enum."
            )
        else:
            raise ValueError(
                f"embedding_type must be EmbeddingType enum or string, got {type(embedding_type).__name__}"
            )

    @classmethod
    def create_embedding(
        cls,
        embedding_type: Union[EmbeddingType, str] = EmbeddingType.GLM,
        api_key: Optional[str] = None,
        **kwargs,
    ):
        """
        Create embedding of specified type

        Args:
            embedding_type: Embedding type (enum or string like "glm", default: GLM)
            api_key: Optional API key (auto-loads from env if not provided)
            **kwargs: Custom configuration parameters
                - model: Model name (e.g., embedding-2)
                - Other GLM embedding parameters

        Returns:
            Configured embedding instance

        Raises:
            ValueError: Unsupported embedding type or missing API key
            ImportError: Missing required dependencies
        """
        # Normalize embedding_type to enum
        embedding_type = cls._normalize_embedding_type(embedding_type)
        config = cls.EMBEDDING_CONFIGS[embedding_type]

        # Load API key from environment if not provided
        if api_key is None and config.get("env_var"):
            api_key = os.getenv(config["env_var"])
            if api_key is None:
                raise ValueError(
                    f"API key not provided and {config['env_var']} not set in environment. "
                    f"Please set {config['env_var']} or pass api_key parameter."
                )

        # Merge default params with user params (user params override defaults)
        embedding_params = {**config["default_params"], **kwargs, "api_key": api_key}

        # Create instance based on type
        if embedding_type == EmbeddingType.GLM:
            return cls._create_glm_embedding(**embedding_params)
        else:
            raise ValueError(f"Unsupported embedding type: {embedding_type}")

    @staticmethod
    def _create_glm_embedding(model: str, api_key: str, **kwargs):
        """
        Create Zhipu GLM Embedding

        Args:
            model: GLM model name (embedding-2)
            api_key: Zhipu API key
            **kwargs: Other parameters

        Returns:
            GLM embedding instance

        Raises:
            ImportError: GLM embedding module not found
        """
        try:
            from core.model.embedding.glm_embedding import GLMEmbedding
        except ImportError as e:
            raise ImportError(
                "GLM embedding not installed. Please check if glm_embedding.py exists in core/model/embedding/"
            ) from e
        
        return GLMEmbedding(model=model, api_key=api_key, **kwargs)

    @classmethod
    def get_embedding_info(
        cls, embedding_type: Union[EmbeddingType, str] = EmbeddingType.GLM
    ) -> Dict[str, Any]:
        """
        Get embedding type information

        Args:
            embedding_type: Embedding type

        Returns:
            Dictionary containing type information:
            - type: Type name
            - description: Description
            - default_params: Default parameters
            - env_var: Environment variable name for API key
            - required_packages: Required dependencies
        """
        # Normalize embedding_type to enum
        embedding_type = cls._normalize_embedding_type(embedding_type)
        config = cls.EMBEDDING_CONFIGS[embedding_type]
        return {
            "type": embedding_type.value,
            "description": config["description"],
            "default_params": config["default_params"],
            "env_var": config["env_var"],
            "required_packages": config["required_packages"],
        }

    @classmethod
    def list_embedding_types(cls) -> list[Dict[str, Any]]:
        """
        List all available embedding types

        Returns:
            List containing all type information
        """
        return [
            cls.get_embedding_info(embedding_type) for embedding_type in EmbeddingType
        ]
