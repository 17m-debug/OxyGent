"""Annotation platform common response models"""
from typing import Generic, TypeVar, Optional

from pydantic import BaseModel, Field

T = TypeVar("T")


class APIResponse(BaseModel, Generic[T]):
    """Unified API response format

    Used for response wrapper of all annotation platform APIs
    """
    code: int = Field(default=200, description="Response status code")
    msg: str = Field(default="success", description="Response message")
    data: Optional[T] = Field(default=None, description="Response data")


class DataListResponse(BaseModel, Generic[T]):
    """Paginated list response

    Used for all APIs that return list data
    """
    total: int = Field(..., description="Total record count")
    page: int = Field(..., description="Current page number")
    page_size: int = Field(..., description="Items per page")
    items: list[T] = Field(..., description="Data list")
