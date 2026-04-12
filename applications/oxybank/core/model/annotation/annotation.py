"""Annotation and approval request models"""
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class AnnotationUpdateRequest(BaseModel):
    """Annotation update request"""

    # Annotation content - main annotation object containing content, question, score, comment
    annotation: Optional[Dict[str, Any]] = Field(None, description="Main annotation data (content, question, score, comment)")

    # Legacy fields for backward compatibility
    category: Optional[str] = Field(None, description="Data category")
    tags: Optional[List[str]] = Field(None, description="Data tags")
    scores: Optional[Dict[str, Any]] = Field(None, description="Score info")

    # Extended annotation fields
    comment: Optional[str] = Field(None, description="Annotation comment")
    remark: Optional[str] = Field(None, description="Remark info")
    annotation_data: Optional[Dict[str, Any]] = Field(None, description="Other annotation data")


class ApprovalRequest(BaseModel):
    """Approval request"""
    action: str = Field(..., description="Approval action: approve-pass, reject-reject", pattern="^(approve|reject)$")
    reason: Optional[str] = Field(None, description="Approval reason/rejection reason")


class RejectionRequest(BaseModel):
    """Rejection request"""
    reason: str = Field(..., min_length=1, description="Rejection reason")
    reject_category: Optional[str] = Field(None, description="Rejection category")
