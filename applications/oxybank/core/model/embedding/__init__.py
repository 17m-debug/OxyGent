"""
Embedding Factory Pattern Implementation

Provides unified interface for creating different embedding backends.

Architecture:
- factory.py: Contains EmbeddingType enum and EmbeddingFactory class
- __init__.py: Exports public interface

Features:
- Auto-loads API keys from environment variables (for GLM)
- Supports GLM embedding models (embedding-2)
- Unified configuration interface
"""

from .factory import EmbeddingFactory, EmbeddingType

# Export public interface
__all__ = [
    'EmbeddingFactory',
    'EmbeddingType',
]
