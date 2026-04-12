"""API Response Model Definitions"""
from typing import Dict, Any, Optional, Generic, TypeVar, List

from fastapi import Query
from pydantic import BaseModel, Field

T = TypeVar("T")


class APIResponse(BaseModel, Generic[T]):
    code: int = Field(default=200, description="Response status code")
    msg: str = Field(default="success", description="Response message")
    data: Optional[T] = Field(default=None, description="Response data")


class PaginationParams(BaseModel):
    """Pagination parameter model"""
    page: int = Field(default=1, ge=1, description="Page number, starting from 1")
    size: int = Field(default=10, description="Page size, options: 10/20/50/100/200")


def get_pagination_params(
        page: int = Query(1, ge=1, description="Page number, starting from 1"),
        size: int = Query(10, description="Page size, options: 10/20/50/100/200")
):
    """Dependency function to get pagination parameters"""
    allowed_sizes = [10, 20, 50, 100, 200]
    if size not in allowed_sizes:
        raise ValueError(f"size must be one of the following values: {allowed_sizes}")

    return PaginationParams(page=page, size=size)


class PaginatedResponse(BaseModel, Generic[T]):
    """Paginated response model"""
    items: list[T] = Field(..., description="Data list")
    total: int = Field(..., description="Total number of records")
    page: int = Field(..., description="Current page number")
    size: int = Field(..., description="Page size")
    pages: int = Field(..., description="Total number of pages")


class KnowledgeBaseItem(BaseModel):
    """Knowledge base item model"""
    kb_id: str = Field("", description="Knowledge base ID", example="d1686c75272e7ab78643367eb438751c")
    kb_name: str = Field(..., description="Knowledge base name", example="kb_test1")
    kb_type: str = Field(..., description="Knowledge base type", example="unstructured")
    kb_description: str = Field("", description="Knowledge base description", example="this is kb kb_test1, file_path is xxx, kb_id is c1caa264-ce70-4544-8ffa-108dac1d6f64")
    kb_status: str = Field("active", description="Knowledge base status", example="active")
    create_time: str = Field("", description="Knowledge base creation time", example="2025-01-01 10:00:00")
    update_time: str = Field("", description="Knowledge base update time", example="2025-01-01 10:00:00")
    kb_create_user: str = Field("", description="Knowledge base creator", example="bob")
    kb_update_user: str = Field("", description="Knowledge base updater", example="bob")
    kb_store_type: list[str] = Field(default_factory=list, description="Knowledge base storage type", example="[elasticsearch]")
    kb_extra_info: dict[str, Any] = Field(default_factory=dict, description="Knowledge base extra information")
    kb_schema: dict[str, Any] = Field(default_factory=dict, description="Knowledge base schema information")
    auto_bind_query: bool = Field(default=True, description="Whether to automatically bind query interfaces for retrieval strategies on restart")


class KnowledgeFileItem(BaseModel):
    """Knowledge base file item model"""
    ori_file_id: str = Field(..., description="File ID", example="file_123456")
    kb_id: str = Field(..., description="Knowledge base ID", example="d1686c75272e7ab78643367eb438751c")
    document_md5: str = Field(..., description="File content MD5 value", example="e10adc3949ba59abbe56e057f20f883e")
    ori_file_type: str = Field(..., description="File type", example="pdf")
    file_name: str = Field(..., description="File name", example="document.pdf")
    file_store_mode: str = Field(..., description="File storage mode", example="unstructured")
    file_extra_info: Dict[str, Any] = Field(default_factory=dict, description="File extra information")
    language: str = Field(..., description="File language", example="zh")
    create_time: str = Field("", description="Knowledge base creation time", example="2025-01-01 10:00:00")
    update_time: str = Field("", description="Knowledge base update time", example="2025-01-01 10:00:00")


class KnowledgeChunkItem(BaseModel):
    """Knowledge base document chunk item model"""
    kb_id: str = Field(..., description="Knowledge base ID", example="d1686c75272e7ab78643367eb438751c")
    ori_file_id: str = Field(..., description="Original file ID", example="file_123456")
    chunk_id: str = Field(..., description="Document chunk ID", example="chunk_789abc")
    chunk_text: str = Field(..., description="Document chunk text content", example="This is a segment of the document...")
    chunk_extra_data: Dict[str, Any] = Field(default_factory=dict, description="Document chunk extra data")
    language: str = Field(..., description="Document chunk language", example="zh")
    create_time: str = Field("", description="Knowledge base creation time", example="2025-01-01 10:00:00")
    update_time: str = Field("", description="Knowledge base update time", example="2025-01-01 10:00:00")


class FileUploadInfo(BaseModel):
    """File upload response model"""
    file_id: str = Field("", description="File ID")
    file_name: str = Field(..., description="File name")
    file_type: str = Field(..., description="File type")
    file_size: int = Field("", description="File size (bytes)")
    file_path: str = Field(..., description="File storage path")
    md5: str = Field("", description="File MD5 value")
    upload_time: str = Field("", description="Upload time")


