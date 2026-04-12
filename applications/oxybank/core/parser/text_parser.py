"""Text parser implementations using llama_index node parsers."""

from typing import Any, Dict, List

from llama_index.core.node_parser import (
    TokenTextSplitter,
    SentenceSplitter,
    MarkdownNodeParser,
    HTMLNodeParser,
    JSONNodeParser,
)
from llama_index.core.schema import BaseNode, Document


class TokenTextSplitterParser(TokenTextSplitter):
    """Token-based text splitter parser."""
    
    def __init__(
        self,
        chunk_size: int = 1024,
        chunk_overlap: int = 20,
        separator: str = " ",
        **kwargs: Any,
    ):
        super().__init__(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separator=separator,
            **kwargs
        )
    
    def supports_document_type(self, doc: Document) -> bool:
        """Check if parser supports the document type."""
        return True  # Token splitter supports any text document
    
    def get_parser_metadata(self) -> Dict[str, Any]:
        """Get parser metadata."""
        return {
            "parser_type": self.__class__.__name__,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "separator": self.separator
        }


class SentenceSplitterParser(SentenceSplitter):
    """Sentence-based text splitter parser."""
    
    def __init__(
        self,
        chunk_size: int = 1024,
        chunk_overlap: int = 200,
        separator: str = " ",
        **kwargs: Any,
    ):
        super().__init__(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separator=separator,
            **kwargs
        )
    
    def supports_document_type(self, doc: Document) -> bool:
        """Check if parser supports the document type."""
        return True  # Sentence splitter supports any text document
    
    def get_parser_metadata(self) -> Dict[str, Any]:
        """Get parser metadata."""
        return {
            "parser_type": self.__class__.__name__,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "separator": self.separator
        }


class MarkdownNodeParserWrapper(MarkdownNodeParser):
    """Markdown node parser wrapper."""
    
    def __init__(
        self,
        include_metadata: bool = True,
        include_prev_next_rel: bool = True,
        **kwargs: Any,
    ):
        super().__init__(
            include_metadata=include_metadata,
            include_prev_next_rel=include_prev_next_rel,
            **kwargs
        )
    
    def supports_document_type(self, doc: Document) -> bool:
        """Check if parser supports the document type."""
        return doc.mime_type == "text/markdown" if hasattr(doc, 'mime_type') else False
    
    def get_parser_metadata(self) -> Dict[str, Any]:
        """Get parser metadata."""
        return {
            "parser_type": self.__class__.__name__,
            "include_metadata": self.include_metadata,
            "include_prev_next_rel": self.include_prev_next_rel
        }


class HTMLNodeParserWrapper(HTMLNodeParser):
    """HTML node parser wrapper."""
    
    def __init__(
        self,
        include_metadata: bool = True,
        include_prev_next_rel: bool = True,
        **kwargs: Any,
    ):
        super().__init__(
            include_metadata=include_metadata,
            include_prev_next_rel=include_prev_next_rel,
            **kwargs
        )
    
    def supports_document_type(self, doc: Document) -> bool:
        """Check if parser supports the document type."""
        return doc.mime_type == "text/html" if hasattr(doc, 'mime_type') else False
    
    def get_parser_metadata(self) -> Dict[str, Any]:
        """Get parser metadata."""
        return {
            "parser_type": self.__class__.__name__,
            "include_metadata": self.include_metadata,
            "include_prev_next_rel": self.include_prev_next_rel
        }


class JSONNodeParserWrapper(JSONNodeParser):
    """JSON node parser wrapper."""
    
    def __init__(
        self,
        include_metadata: bool = True,
        include_prev_next_rel: bool = True,
        **kwargs: Any,
    ):
        super().__init__(
            include_metadata=include_metadata,
            include_prev_next_rel=include_prev_next_rel,
            **kwargs
        )
    
    def supports_document_type(self, doc: Document) -> bool:
        """Check if parser supports the document type."""
        # Support both JSON files and text that looks like JSON
        if hasattr(doc, 'mime_type'):
            return doc.mime_type == "application/json"
        
        # Check if file extension is .json
        if hasattr(doc, 'metadata') and 'file_type' in doc.metadata:
            return doc.metadata['file_type'] == ".json"
        
        # Check if text content looks like JSON (starts with { or [)
        if hasattr(doc, 'text') and doc.text:
            text = doc.text.strip()
            return text.startswith('{') or text.startswith('[')
        
        return False
    
    def get_parser_metadata(self) -> Dict[str, Any]:
        """Get parser metadata."""
        return {
            "parser_type": self.__class__.__name__,
            "include_metadata": self.include_metadata,
            "include_prev_next_rel": self.include_prev_next_rel
        }


class ExtensibleSplitterParser:
    """Extensible splitter parser for future extensions."""
    
    def __init__(self, splitter_type: str, **kwargs: Any):
        """
        Initialize extensible splitter parser.
        
        Args:
            splitter_type: Type of splitter to use
            **kwargs: Additional arguments for the splitter
        """
        self.splitter_type = splitter_type
        self.kwargs = kwargs
        self._splitter = None
        
        # Initialize the appropriate splitter based on type
        if splitter_type == "token":
            self._splitter = TokenTextSplitter(**kwargs)
        elif splitter_type == "sentence":
            self._splitter = SentenceSplitter(**kwargs)
        elif splitter_type == "markdown":
            self._splitter = MarkdownNodeParser(**kwargs)
        elif splitter_type == "html":
            self._splitter = HTMLNodeParser(**kwargs)
        elif splitter_type == "json":
            self._splitter = JSONNodeParser(**kwargs)
        else:
            raise ValueError(f"Unsupported splitter type: {splitter_type}")
    
    def get_nodes_from_documents(
        self,
        documents: List[Document],
        show_progress: bool = False,
    ) -> List[BaseNode]:
        """Parse documents into nodes."""
        return self._splitter.get_nodes_from_documents(documents, show_progress=show_progress)
    
    def supports_document_type(self, doc: Document) -> bool:
        """Check if parser supports the document type."""
        if self.splitter_type == "markdown":
            return doc.mime_type == "text/markdown" if hasattr(doc, 'mime_type') else False
        elif self.splitter_type == "html":
            return doc.mime_type == "text/html" if hasattr(doc, 'mime_type') else False
        elif self.splitter_type == "json":
            return doc.mime_type == "application/json" if hasattr(doc, 'mime_type') else False
        else:
            return True  # Token, sentence, and code splitters support any text document
    
    def get_parser_metadata(self) -> Dict[str, Any]:
        """Get parser metadata."""
        return {
            "parser_type": self.__class__.__name__,
            "splitter_type": self.splitter_type,
            "config": self.kwargs
        }
