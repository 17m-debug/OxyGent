"""Query filter models"""
from datetime import datetime
from typing import List, Optional

from fastapi import Query
from pydantic import BaseModel, Field, field_validator


class DataListQueryParams(BaseModel):
    """Data list query parameters"""

    # Basic filters
    status: Optional[str] = Field(None, description="Data status")
    priority: Optional[int] = Field(None, ge=0, le=4, description="Priority")
    data_type: Optional[str] = Field(None, description="Data type")
    caller: Optional[str] = Field(None, description="Caller")
    callee: Optional[str] = Field(None, description="Callee")
    category: Optional[str] = Field(None, description="Data category")

    # Tag filters (support multiple)
    tags: Optional[List[str]] = Field(None, description="Data tag list")

    # Date range filters
    created_after: Optional[str] = Field(None, description="Creation time start (YYYY-MM-DD HH:MM:SS)")
    created_before: Optional[str] = Field(None, description="Creation time end (YYYY-MM-DD HH:MM:SS)")

    # Text search
    search_text: Optional[str] = Field(None, description="Full-text search (search question or answer)")

    # Source tracking filters
    trace_id: Optional[str] = Field(None, description="Source trace ID (fuzzy match)")
    group_id: Optional[str] = Field(None, description="Source group ID (fuzzy match)")

    # Pagination params
    page: int = Field(default=1, ge=1, description="Page number")
    page_size: int = Field(default=10, ge=1, le=200, description="Items per page")

    # Sorting params
    sort_by: str = Field(default="created_at", description="Sort field")
    sort_order: str = Field(default="desc", pattern=r"^(asc|desc)$", description="Sort direction")

    @field_validator('created_after', 'created_before', mode='before')
    @classmethod
    def convert_datetime_format(cls, v: Optional[str]) -> Optional[str]:
        """
        Convert ISO 8601 datetime format to ES-compatible format

        Converts:
        - 2026-01-10T22:17:00 -> 2026-01-10 22:17:00
        - 2026-01-10 22:17:00 -> 2026-01-10 22:17:00 (no change)
        """
        if v is None:
            return v

        # If contains 'T', replace with space
        if 'T' in v:
            return v.replace('T', ' ')

        return v


class TraceQueryParams(BaseModel):
    """Trace query parameters"""
    limit: int = Field(default=100, ge=1, le=500, description="Result count limit")


class GroupQueryParams(BaseModel):
    """Group query parameters"""
    limit: int = Field(default=100, ge=1, le=500, description="Result count limit")


# FastAPI dependency function
def get_data_list_query_params(
    status: Optional[str] = Query(None, description="Data status"),
    priority: Optional[int] = Query(None, ge=0, le=4, description="Priority"),
    data_type: Optional[str] = Query(None, description="Data type"),
    caller: Optional[str] = Query(None, description="Caller"),
    callee: Optional[str] = Query(None, description="Callee"),
    category: Optional[str] = Query(None, description="Data category"),
    tags: Optional[str] = Query(None, description="Data tags (comma separated)"),
    created_after: Optional[str] = Query(None, description="Creation time start", alias="start_time"),
    created_before: Optional[str] = Query(None, description="Creation time end", alias="end_time"),
    search: Optional[str] = Query(None, description="Full-text search (alias for search_text)", alias="search"),
    search_text: Optional[str] = Query(None, description="Full-text search"),
    trace_id: Optional[str] = Query(None, description="Source trace ID"),
    group_id: Optional[str] = Query(None, description="Source group ID"),
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(10, ge=1, le=200, description="Items per page"),
    sort_by: str = Query("created_at", description="Sort field"),
    sort_order: str = Query("desc", regex="^(asc|desc)$", description="Sort direction")
) -> DataListQueryParams:
    """Get data list query parameters"""
    # Process tags: convert from comma-separated string to list
    tags_list = None
    if tags:
        tags_list = [t.strip() for t in tags.split(",") if t.strip()]

    # Use 'search' parameter as alias for 'search_text' (prefer 'search' if both provided)
    final_search_text = search or search_text

    return DataListQueryParams(
        status=status,
        priority=priority,
        data_type=data_type,
        caller=caller,
        callee=callee,
        category=category,
        tags=tags_list,
        created_after=created_after,
        created_before=created_before,
        search_text=final_search_text,
        trace_id=trace_id,
        group_id=group_id,
        page=page,
        page_size=page_size,
        sort_by=sort_by,
        sort_order=sort_order
    )
