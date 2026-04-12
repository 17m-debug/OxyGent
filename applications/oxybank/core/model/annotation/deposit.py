"""Data deposit request and response models"""
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


class DepositRequest(BaseModel):
    """Single data deposit request"""

    # QA content
    question: str = Field(..., min_length=2, description="Question content")
    answer: str = Field(..., description="Answer content")

    # Source tracking (optional, auto-generated if not provided)
    source_trace_id: Optional[str] = Field(None, description="Original trace_id")
    source_request_id: Optional[str] = Field(None, description="Original request_id")
    source_group_id: Optional[str] = Field(None, description="Session group ID")

    # Call chain info
    caller: str = Field(default="", description="Caller")
    callee: str = Field(default="", description="Callee")
    caller_type: Optional[str] = Field(None, description="Caller type")
    callee_type: Optional[str] = Field(None, description="Callee type")

    # Data attributes
    data_type: Optional[str] = Field(None, description="Data type (auto-inferred if not provided)")
    priority: Optional[int] = Field(None, ge=0, le=4, description="Priority (auto-set if not provided)")
    category: Optional[str] = Field(None, description="Data category")
    tags: List[str] = Field(default_factory=list, description="Data tags")

    # Extra info
    extra: Dict[str, Any] = Field(default_factory=dict, description="Extra data")


class DepositBatchRequest(BaseModel):
    """Batch data deposit request"""
    data_list: List[DepositRequest] = Field(..., min_items=1, max_items=1000, description="Data list")
    batch_id: Optional[str] = Field(None, description="Batch ID (auto-generated if not provided)")


class DepositResponse(BaseModel):
    """Data deposit response"""
    data_id: str = Field(..., description="Data ID")
    data_hash: str = Field(..., description="Data hash")
    status: str = Field(..., description="Data status")
    is_duplicate: bool = Field(default=False, description="Is duplicate data")
    message: str = Field(default="success", description="Response message")


class DepositBatchResponse(BaseModel):
    """Batch data deposit response"""
    batch_id: str = Field(..., description="Batch ID")
    total: int = Field(..., description="Total count")
    success_count: int = Field(..., description="Success count")
    duplicate_count: int = Field(..., description="Duplicate count")
    failed_count: int = Field(..., description="Failed count")
    results: List[DepositResponse] = Field(..., description="Detailed results")
