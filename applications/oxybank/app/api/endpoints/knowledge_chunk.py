from typing import Dict, Any

from fastapi import APIRouter, Path, Depends, HTTPException
from loguru import logger

from app.api.models import KnowledgeChunkItem, APIResponse, PaginatedResponse, PaginationParams, get_pagination_params
from core.config import settings
from core.storer.doc_manager.es_index_manager import ElasticsearchIndexManager
from core.storer.doc_manager.es_kb_base_manager import ElasticsearchKbBaseManager
from core.storer.doc_manager.es_kb_chunk_manager import ElasticsearchKbChunkManager

router = APIRouter(prefix="/kb_base/{kb_id}", tags=["Knowledge Base Document Chunk Management"])

# Initialize instances
kb_chunk_client = ElasticsearchKbChunkManager(settings.es_client)
kb_base_client = ElasticsearchKbBaseManager(settings.es_client)
kb_index_client = ElasticsearchIndexManager(settings.es_client)


@router.get(
    "/file/{file_id}/chunks",
    response_model=APIResponse[PaginatedResponse[Dict[str, Any]]],
    summary="Get all document chunks for a specified file in the knowledge base",
    description="Return list of all document chunks for a specific file in the specified knowledge base",
    response_description="List of document chunks, including chunk ID, text content, language, etc.",
)
def get_kb_file_chunks(
        kb_id: str = Path(..., description="Knowledge base ID"),
        file_id: str = Path(..., description="File ID"),
        pagination: PaginationParams = Depends(get_pagination_params)
) -> APIResponse[PaginatedResponse[Dict[str, Any]]]:
    """
    Get all document chunks for a specified file in the knowledge base

    Return all document chunk information for a specific file in the specified knowledge base, including:
    - kb_id: Knowledge base unique identifier
    - ori_file_id: Original file ID
    - chunk_id: Document chunk ID
    - chunk_text: Document chunk text content
    - chunk_extra_data: Document chunk extra data
    - language: Document chunk language

    Args:
        kb_id: Knowledge base ID
        file_id: File ID
        pagination: Pagination parameters

    Returns:
        APIResponse: Contains paginated chunk list

    Raises:
        HTTPException 400: Knowledge base ID or file ID is empty
        HTTPException 500: Query failed
    """
    # Validate input
    if not kb_id or not kb_id.strip():
        raise HTTPException(
            status_code=400,
            detail="Knowledge base ID cannot be empty"
        )

    if not file_id or not file_id.strip():
        raise HTTPException(
            status_code=400,
            detail="File ID cannot be empty"
        )

    try:
        # 1. Get kb_name by kb_id
        kb_name = kb_base_client.get_kb_name_by_id(kb_id)
        if not kb_name:
            raise HTTPException(
                status_code=404,
                detail=f"Knowledge base with ID '{kb_id}' does not exist"
            )

        # 2. Check if ES index exists
        if not kb_index_client.index_exists(kb_name):
            raise HTTPException(
                status_code=404,
                detail=f"ES index for knowledge base '{kb_name}' does not exist, please create knowledge base Schema and generate index first"
            )

        # 3. Query all documents in ES index (paginated)
        from_value = (pagination.page - 1) * pagination.size

        query = {
            "query": {"term": {"ori_file_id": file_id}},  # Query all documents
            "from": from_value,
            "size": pagination.size
        }

        resp = settings.es_client.search(
            index=kb_name,
            body=query
        )

        # 4. Parse results
        hits = resp['hits']['hits']
        total = resp['hits']['total']['value']

        # Extract document data (_source field)
        items = [doc["_source"] for doc in hits]

        # Calculate total pages
        pages = (total + pagination.size - 1) // pagination.size if total > 0 else 0

        paginated_response = PaginatedResponse[Dict[str, Any]](
            items=items,
            total=total,
            page=pagination.page,
            size=pagination.size,
            pages=pages
        )

        return APIResponse(
            code=200,
            msg="Successfully retrieved document chunks",
            data=paginated_response
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get chunks for file {file_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get file document chunk list: {str(e)}"
        )


@router.get(
    "/chunks",
    response_model=APIResponse[PaginatedResponse[Dict[str, Any]]],
    summary="Get document data for structured knowledge base",
    description="Query all document data in the ES index of a structured knowledge base (knowledge base with Schema)",
    response_description="Document data list, including all field information",
)
def get_kb_chunks(
        kb_id: str = Path(..., description="Knowledge base ID"),
        pagination: PaginationParams = Depends(get_pagination_params)
) -> APIResponse[PaginatedResponse[Dict[str, Any]]]:
    """
    Get document data for structured knowledge base

    Query the corresponding kb_name based on kb_id, then check if the ES index for that kb_name exists.
    If the index does not exist, return an error; if it exists, query and return all document data according to pagination rules.

    Args:
        kb_id: Knowledge base ID
        pagination: Pagination parameters

    Returns:
        APIResponse: Contains paginated document data list

    Raises:
        HTTPException 400: Knowledge base ID is empty
        HTTPException 404: Knowledge base does not exist or ES index does not exist
        HTTPException 500: Query failed
    """
    # Validate input
    if not kb_id or not kb_id.strip():
        raise HTTPException(
            status_code=400,
            detail="Knowledge base ID cannot be empty"
        )

    try:
        # 1. Get kb_name by kb_id
        kb_name = kb_base_client.get_kb_name_by_id(kb_id)
        if not kb_name:
            raise HTTPException(
                status_code=404,
                detail=f"Knowledge base with ID '{kb_id}' does not exist"
            )

        # 2. Check if ES index exists
        if not kb_index_client.index_exists(kb_name):
            raise HTTPException(
                status_code=404,
                detail=f"ES index for knowledge base '{kb_name}' does not exist, please create knowledge base Schema and generate index first"
            )

        # 3. Query all documents in ES index (paginated)
        from_value = (pagination.page - 1) * pagination.size

        query = {
            "query": {"match_all": {}},  # Query all documents
            "from": from_value,
            "size": pagination.size
        }

        resp = settings.es_client.search(
            index=kb_name,
            body=query
        )

        # 4. Parse results
        hits = resp['hits']['hits']
        total = resp['hits']['total']['value']

        # Extract document data (_source field)
        items = [doc["_source"] for doc in hits]

        # Calculate total pages
        pages = (total + pagination.size - 1) // pagination.size if total > 0 else 0

        paginated_response = PaginatedResponse[Dict[str, Any]](
            items=items,
            total=total,
            page=pagination.page,
            size=pagination.size,
            pages=pages
        )

        return APIResponse(
            code=200,
            msg="Successfully retrieved structured knowledge base document data",
            data=paginated_response
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get structured knowledge base {kb_id} document data: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get document data: {str(e)}"
        )

