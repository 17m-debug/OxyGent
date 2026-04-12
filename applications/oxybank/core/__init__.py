"""Core components for knowledge base platform."""

from .parser.factory import ParserFactory
from .model.embedding.factory import EmbeddingFactory

__all__ = [
    'ParserFactory',
    'EmbeddingFactory',
]
