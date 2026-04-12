"""Statistics analysis models"""
from typing import Dict, List, Optional


from pydantic import BaseModel, Field


class OverallStatsResponse(BaseModel):
    """Overall statistics response"""

    # Overall statistics
    total_count: int = Field(..., description="Total data count")
    pending_count: int = Field(..., description="Pending count")
    annotated_count: int = Field(..., description="Annotated count")
    approved_count: int = Field(..., description="Approved count")
    rejected_count: int = Field(..., description="Rejected count")

    # KB statistics
    kb_ingested_count: int = Field(default=0, description="KB ingested count")
    kb_pending_count: int = Field(default=0, description="KB pending count")
    kb_failed_count: int = Field(default=0, description="KB failed count")

    # Statistics by priority
    priority_stats: Dict[str, int] = Field(
        default_factory=dict,
        description="Statistics by priority {priority: count}"
    )

    # Statistics by type
    type_stats: Dict[str, int] = Field(
        default_factory=dict,
        description="Statistics by data type {type: count}"
    )


class PendingP0Item(BaseModel):
    """Pending P0 data item"""
    data_id: str = Field(..., description="Data ID")
    question: str = Field(..., description="Question")
    answer: str = Field(..., description="Answer")
    caller: str = Field(..., description="Caller")
    callee: str = Field(..., description="Callee")
    source_trace_id: str = Field(..., description="trace_id")
    created_at: str = Field(..., description="Creation time")


class PendingP0Response(BaseModel):
    """Pending P0 data response"""
    total: int = Field(..., description="Total count")
    items: List[PendingP0Item] = Field(..., description="Data list")


class TypeStatsItem(BaseModel):
    """Type statistics item"""
    data_type: str = Field(..., description="Data type")
    total_count: int = Field(..., description="Total count")
    pending_count: int = Field(..., description="Pending count")
    approved_count: int = Field(..., description="Approved count")
    rejected_count: int = Field(..., description="Rejected count")


class TypeStatsResponse(BaseModel):
    """Statistics by type response"""
    items: List[TypeStatsItem] = Field(..., description="Type statistics list")


class GroupSummaryItem(BaseModel):
    """Group summary item"""
    group_id: str = Field(..., description="Group ID")
    total_count: int = Field(..., description="Total count")
    status_counts: Dict[str, int] = Field(..., description="Count by status")


class GroupsSummaryResponse(BaseModel):
    """Group summary response"""
    items: List[GroupSummaryItem] = Field(..., description="Group summary list")
