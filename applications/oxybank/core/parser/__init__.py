"""
Parser Factory Pattern Implementation

Provides unified interface for creating different LlamaIndex node parsers.

Architecture:
- factory.py: Contains ParserType enum and ParserFactory class
- __init__.py: Exports public interface

Features:
- Supports SentenceSplitter, SemanticSplitterNodeParser, MarkdownNodeParser
- Unified configuration interface
- Auto-dependency checking with helpful error messages
"""

from .factory import ParserFactory

# Export public interface
__all__ = [
    'ParserFactory',
]
