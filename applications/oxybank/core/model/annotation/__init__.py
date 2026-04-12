"""Annotation platform API models

Export all annotation platform related Pydantic models for use in API endpoints
"""

# ============= Common response models =============
from .common import APIResponse, DataListResponse

# ============= Core data models =============
from .qa_data import QADataItem, QADataSummary

# ============= Deposit request/response models =============
from .deposit import (
    DepositRequest,
    DepositBatchRequest,
    DepositResponse,
    DepositBatchResponse,
)

# ============= Annotation and approval models =============
from .annotation import (
    AnnotationUpdateRequest,
    ApprovalRequest,
    RejectionRequest,
)

# ============= Statistics analysis models =============
from .stats import (
    OverallStatsResponse,
    PendingP0Item,
    PendingP0Response,
    TypeStatsItem,
    TypeStatsResponse,
    GroupSummaryItem,
    GroupsSummaryResponse,
)

# ============= Query filter models =============
from .query import (
    DataListQueryParams,
    TraceQueryParams,
    GroupQueryParams,
    get_data_list_query_params,
)

__all__ = [
    # Common responses
    "APIResponse",
    "DataListResponse",

    # Core data
    "QADataItem",
    "QADataSummary",

    # Deposit
    "DepositRequest",
    "DepositBatchRequest",
    "DepositResponse",
    "DepositBatchResponse",

    # Annotation and approval
    "AnnotationUpdateRequest",
    "ApprovalRequest",
    "RejectionRequest",

    # Statistics
    "OverallStatsResponse",
    "PendingP0Item",
    "PendingP0Response",
    "TypeStatsItem",
    "TypeStatsResponse",
    "GroupSummaryItem",
    "GroupsSummaryResponse",

    # Query
    "DataListQueryParams",
    "TraceQueryParams",
    "GroupQueryParams",
    "get_data_list_query_params",
]
