"""Parser factory for knowledge base platform."""

from typing import Optional, Union

from llama_index.core.node_parser import NodeParser
from loguru import logger

from .text_parser import (
    TokenTextSplitterParser,
    SentenceSplitterParser,
    MarkdownNodeParserWrapper,
    HTMLNodeParserWrapper,
    JSONNodeParserWrapper,
    ExtensibleSplitterParser,
)


class ParserFactory:
    """Factory for creating parsers based on configuration."""
    
    _parsers = {
        "token": TokenTextSplitterParser,
        "sentence": SentenceSplitterParser,
        "markdown": MarkdownNodeParserWrapper,
        "html": HTMLNodeParserWrapper,
        "json": JSONNodeParserWrapper,
        "extensible": ExtensibleSplitterParser,
        "smart": SentenceSplitterParser
    }
    
    @classmethod
    def create_parser(cls, parser_type: str = "smart", **kwargs) -> Union[NodeParser, ExtensibleSplitterParser]:
        """Create a parser instance.
        
        Args:
            parser_type: Type of parser to create
            **kwargs: Configuration parameters for the parser
            
        Returns:
            Parser instance
            
        Raises:
            ValueError: If parser_type is not supported
        """
        if parser_type not in cls._parsers:
            raise ValueError(f"Unsupported parser type: {parser_type}. "
                           f"Available types: {list(cls._parsers.keys())}")
        
        parser_class = cls._parsers[parser_type]
        logger.info(f"Creating {parser_type} parser: {parser_class.__name__}")
        
        try:
            return parser_class(**kwargs)
        except Exception as e:
            logger.error(f"Failed to create {parser_type} parser: {e}")
            raise
    
    @classmethod
    def register_parser(cls, parser_type: str, parser_class: type) -> None:
        """Register a custom parser type.
        
        Args:
            parser_type: Type name for the parser
            parser_class: Parser class to register
        """
        cls._parsers[parser_type] = parser_class
        logger.info(f"Registered parser type: {parser_type}")
    
    @classmethod
    def get_supported_types(cls) -> list:
        """Get list of supported parser types."""
        return list(cls._parsers.keys())
    
    @classmethod
    def create_auto_parser(cls, documents: list, **kwargs) -> Optional[Union[NodeParser, ExtensibleSplitterParser]]:
        """Automatically select appropriate parser based on documents.
        
        Args:
            documents: List of documents to analyze
            **kwargs: Configuration parameters
            
        Returns:
            Appropriate parser instance or None if not supported
        """
        if not documents:
            logger.warning("No documents provided for auto parser selection")
            return None
        
        # Analyze document types
        file_types = set()
        for doc in documents:
            file_type = doc.metadata.get("file_type", "").lower()
            if file_type:
                file_types.add(file_type)
        
        # Select parser based on file types
        if not file_types:
            logger.info("No file type metadata found, using token parser")
            return cls.create_parser("sentence", **kwargs)
        
        # If all documents are of the same type, use specific parser
        if len(file_types) == 1:
            file_type = list(file_types)[0]
            
            type_mapping = {
                '.md': 'markdown',
                '.markdown': 'markdown',
                '.html': 'html',
                '.htm': 'html',
                '.json': 'json',
                '.csv': 'token',  # Use token splitter for CSV
                '.xlsx': 'token',  # Use token splitter for Excel
                '.xls': 'token',   # Use token splitter for Excel
            }
            
            parser_type = type_mapping.get(file_type)
            if parser_type:
                logger.info(f"Selected {parser_type} parser for file type {file_type}")
                return cls.create_parser(parser_type, **kwargs)
        
        # Mixed types or unknown types, use token parser as default
        logger.info(f"Mixed file types {file_types}, using token parser")
        return cls.create_parser("token", **kwargs)
