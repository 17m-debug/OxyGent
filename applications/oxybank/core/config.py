"""Configuration management for knowledge base platform."""
from pathlib import Path

from elasticsearch import Elasticsearch
from pydantic import field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import List, Union

from vearch.config import Config
from vearch.core.vearch import Vearch

from utils.url_util import ensure_url_protocol

# Get the path to the project root directory
ROOT_DIR = Path(__file__).parent.parent


class ElasticsearchConfig(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="ELASTICSEARCH_",
        env_file=ROOT_DIR / ".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )

    hosts: Union[List[str], str] = ["localhost:9200"]
    username: str = ""
    password: str = ""

    @field_validator("hosts", mode="before")
    @classmethod
    def parse_hosts(cls, v):
        if isinstance(v, str):
            return [host.strip() for host in v.split(",") if host.strip()]
        return v

class VearchConfig(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="VEARCH_",
        env_file=ROOT_DIR / ".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )

    host: str = ""
    token: str | None = None
    db_name: str = ""
    vector_dimension: int = 1024


class EmbeddingConfig(BaseSettings):
    """Embedding model configuration class."""
    model_config = SettingsConfigDict(
        env_prefix="EMBEDDING_",
        env_file=ROOT_DIR / ".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )

    type: str = "glm"  # Default type: glm
    api_key: str = ""

    # GLM configuration
    glm_model_name: str = "embedding-2"
    glm_api_base: str = "https://open.bigmodel.cn/api/paas/v4/embeddings"


class AnnotationConfig(BaseSettings):
    """
    Annotation platform configuration class

    The annotation platform is responsible for collecting, annotating, approving, and injecting QA data into knowledge bases.

    Knowledge base structured field description:
    ===========================================
    When annotation data is injected into a knowledge base, a knowledge base chunk containing the following structured fields will be created:

    All fields (defined by knowledge base schema):
    - question: str              # Question/input content
    - answer: str                # Answer/output content
    - caller: str                # Caller (e.g., user, agent_name)
    - callee: str                # Callee (e.g., agent_name, tool_name)
    - score: float               # Quality score (0-1)
    - remark: str                # Remarks
    - source_trace_id: str       # Original trace_id (for trace tracking)
    - source_request_id: str     # Original request_id (for request tracking)
    - data_type: str             # Data type (e2e/agent/llm/tool/custom)
    - priority: int              # Priority (0-4, P0=0)
    - category: str              # Data category

    Notes:
    1. The target knowledge base must support these structured fields
    2. If the knowledge base schema does not support certain fields, injection will fail
    3. Please ensure the target knowledge base schema includes all the above fields
    """

    model_config = SettingsConfigDict(
        env_prefix="ANNOTATION_",
        env_file=ROOT_DIR / ".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )

    # ============= ES Index Configuration =============
    es_index_prefix: str = "qa_annotation"

    # ============= KB Injection Configuration =============
    kb_enabled: bool = True  # Whether to enable KB injection functionality
    kb_id: str = ""  # Target knowledge base ID (required when kb_enabled=True)

    # KB injection auto-trigger configuration
    kb_auto_ingest: bool = False  # Automatically inject to KB after approval
    kb_timeout: int = 30
    kb_retry_times: int = 3
    kb_retry_interval: int = 5

    # ============= Data Validation Configuration =============
    batch_size: int = 100

    # ============= Data Type Inference Configuration =============
    # Default data type when both caller and callee are empty
    default_data_type: str = "custom"

    # Default priority when caller is empty
    default_priority: int = 4

    @field_validator("batch_size")
    @classmethod
    def validate_positive_int(cls, v):
        """Validate positive integer"""
        if v < 0:
            raise ValueError("Must be a positive integer")
        return v

    @field_validator("kb_timeout", "kb_retry_times", "kb_retry_interval")
    @classmethod
    def validate_kb_config(cls, v):
        """Validate KB configuration"""
        if v < 0:
            raise ValueError("Must be a positive integer")
        return v

    @field_validator("default_data_type")
    @classmethod
    def validate_data_type(cls, v):
        """Validate data type"""
        valid_types = ["e2e", "agent", "llm", "tool", "custom"]
        if v not in valid_types:
            raise ValueError(f"Invalid data type, must be one of: {', '.join(valid_types)}")
        return v

    @field_validator("default_priority")
    @classmethod
    def validate_priority(cls, v):
        """Validate priority"""
        if not 0 <= v <= 4:
            raise ValueError("Priority must be between 0 and 4")
        return v

    @model_validator(mode='after')
    def validate_kb_config_after(self):
        """Post-model validation check: if KB injection is enabled, kb_id cannot be empty"""
        if self.kb_enabled and (not self.kb_id or not self.kb_id.strip()):
            raise ValueError(
                "KB injection functionality is enabled (ANNOTATION_KB_ENABLED=true), "
                "but knowledge base ID is not configured.\n"
                "Please configure in .env file: ANNOTATION_KB_ID=<your-knowledge-base-id>\n"
                "Note: The target knowledge base schema must support all the following structured fields:\n"
                "  - question (Question)\n"
                "  - answer (Answer)\n"
                "  - caller (Caller)\n"
                "  - callee (Callee)\n"
                "  - score (Score)\n"
                "  - remark (Remarks)\n"
                "  - source_trace_id (Trace ID)\n"
                "  - source_request_id (Request ID)\n"
                "  - data_type (Data Type)\n"
                "  - priority (Priority)\n"
                "  - category (Category)"
            )
        return self


class ServiceConfig(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=ROOT_DIR / ".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )

    app_name: str = "OxyGentKnowledgeBank"
    host: str = "0.0.0.0"
    port: int = 8000
    _api_base_url: str = None  # Private field for api_base_url
    es_config: ElasticsearchConfig = ElasticsearchConfig()
    vearch_config: VearchConfig = VearchConfig()
    embedding_config: EmbeddingConfig = EmbeddingConfig()
    annotation_config: AnnotationConfig = AnnotationConfig()

    @property
    def es_client(self) -> Elasticsearch:
        return Elasticsearch(
            hosts=self.es_config.hosts,
            basic_auth=(self.es_config.username, self.es_config.password)
        )

    @property
    def vearch_client(self) -> Vearch:
        config = Config(
            host=ensure_url_protocol(self.vearch_config.host),
            token=self.vearch_config.token
        )
        return Vearch(config)

    @property
    def api_base_url(self) -> str:
        """Get API base URL, compute from host:port if not set."""
        if hasattr(self, "_api_base_url") and self._api_base_url:
            return self._api_base_url.rstrip("/")
        # Default to localhost if api_base_url not configured
        return f"http://{self.host}:{self.port}"

    @api_base_url.setter
    def api_base_url(self, value):
        """Allow setting api_base_url from env config."""
        self._api_base_url = value

    @property
    def embedding_model(self):
        """Get the configured embedding model (lazy loading)."""
        from core.model.embedding import EmbeddingFactory

        embedding_type = self.embedding_config.type.lower()

        if embedding_type == "glm":
            return EmbeddingFactory.create_embedding(
                embedding_type="glm",
                api_key=self.embedding_config.api_key,
                model_name=self.embedding_config.glm_model_name,
                api_base=self.embedding_config.glm_api_base
            )
        else:
            raise ValueError(
                f"Unsupported embedding type: {embedding_type}. "
                f"Supported type: glm"
            )


settings = ServiceConfig()
