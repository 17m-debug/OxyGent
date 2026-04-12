"""QA data core models"""
from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


class QADataItem(BaseModel):
    """QA data item model

    Used to display and return complete QA data information
    """

    # Unique identifier
    data_id: str = Field(..., description="Unique data ID", example="550e8400-e29b-41d4-a716-446655440000")
    data_hash: str = Field(..., description="Data hash (for deduplication)", example="a1b2c3d4e5f6")

    # QA content
    question: str = Field(..., description="Question content", example="What is AI?")
    answer: str = Field(..., description="Answer content", example="AI is a branch of computer science...")

    # Source tracking info
    source_trace_id: str = Field(..., description="Original trace_id", example="trace_001")
    source_request_id: str = Field(..., description="Original request_id", example="request_001")
    source_group_id: str = Field(..., description="Session group ID", example="group_001")

    # Call chain info
    caller: str = Field(..., description="Caller", example="agent_A")
    callee: str = Field(..., description="Callee", example="agent_B")
    caller_type: Optional[str] = Field(None, description="Caller type (reserved)", example="agent")
    callee_type: Optional[str] = Field(None, description="Callee type (reserved)", example="tool")

    # Data attributes
    data_type: str = Field(..., description="Data type", example="custom")
    priority: int = Field(..., description="Priority (0-4, P0=0)", ge=0, le=4)
    category: Optional[str] = Field(None, description="Data category", example="Technical question")
    tags: List[str] = Field(default_factory=list, description="Data tags", example=["AI", "Technology"])

    # Status management
    status: str = Field(..., description="Data status", example="pending")
    annotation: Dict[str, Any] = Field(default_factory=dict, description="Annotation results")
    scores: Dict[str, Any] = Field(default_factory=dict, description="Score info")
    reject_reason: Optional[str] = Field(None, description="Rejection reason")

    # KB info
    kb_status: str = Field(default="pending", description="KB ingestion status", example="pending")
    kb_ingested_at: Optional[str] = Field(None, description="KB ingestion time", example="2025-01-12 10:00:00")
    kb_error_message: Optional[str] = Field(None, description="KB ingestion error message")
    kb_extra: Dict[str, Any] = Field(default_factory=dict, description="KB extra info")

    # Batch info
    batch_id: Optional[str] = Field(None, description="Batch ID", example="batch_001")

    # Timestamps
    created_at: str = Field(..., description="Creation time", example="2025-01-12 10:00:00")
    updated_at: str = Field(..., description="Update time", example="2025-01-12 10:00:00")

    # Extended data
    extra: Dict[str, Any] = Field(default_factory=dict, description="Extra data")

    @field_validator("data_type")
    @classmethod
    def validate_data_type(cls, v):
        """Validate data type"""
        valid_types = ["e2e", "agent", "llm", "tool", "custom"]
        if v not in valid_types:
            raise ValueError(f"Invalid data type, must be one of: {', '.join(valid_types)}")
        return v

    @field_validator("status")
    @classmethod
    def validate_status(cls, v):
        """Validate status"""
        valid_statuses = [
            "pending",           # Pending
            "annotated",         # Annotated
            "approved",          # Approved
            "rejected",          # Rejected
            "kb_ingested",       # KB ingested
            "kb_failed"          # KB ingestion failed
        ]
        if v not in valid_statuses:
            raise ValueError(f"Invalid status, must be one of: {', '.join(valid_statuses)}")
        return v

    @field_validator("kb_status")
    @classmethod
    def validate_kb_status(cls, v):
        """Validate KB status"""
        valid_kb_statuses = ["pending", "ingested", "failed"]
        if v not in valid_kb_statuses:
            raise ValueError(f"Invalid KB status, must be one of: {', '.join(valid_kb_statuses)}")
        return v


class QADataSummary(BaseModel):
    """QA data summary model

    Used for list display, only contains key fields
    """
    data_id: str = Field(..., description="Data ID")
    question: str = Field(..., description="Question")
    answer: str = Field(..., description="Answer")
    data_type: str = Field(..., description="Data type")
    priority: int = Field(..., description="Priority")
    status: str = Field(..., description="Status")
    caller: str = Field(..., description="Caller")
    callee: str = Field(..., description="Callee")
    created_at: str = Field(..., description="Creation time")
