"""KB ingestion API endpoints"""
from typing import Optional

from fastapi import APIRouter, HTTPException, Depends, Query
from loguru import logger

from core.model.annotation import APIResponse
from core.services import get_annotation_service, AnnotationService


router = APIRouter(prefix="/kb", tags=["Annotation KB Ingestion"])


@router.post(
    "/data/{data_id}/ingest",
    response_model=APIResponse[dict],
    summary="Ingest QA data to knowledge base",
    description="Manually ingest annotation data to knowledge base"
)
async def ingest_to_kb(
    data_id: str,
    score: Optional[float] = Query(None, ge=0, le=1, description="Quality score (0-1)"),
    remark: Optional[str] = Query(None, description="Remark info"),
    service: AnnotationService = Depends(get_annotation_service)
) -> APIResponse[dict]:
    """
    Ingest data to knowledge base

    Features:
    - Ingest approved data to knowledge base
    - Support manual triggering (even if auto-ingest not enabled)

    Prerequisites:
    - KB ingestion must be enabled (ANNOTATION_KB_ENABLED=true)
    - Data must exist

    Returns:
    - kb_doc_id: KB document ID
    - Other ingestion result info
    """
    try:
        result = await service.ingest_to_kb(data_id, score, remark)

        return APIResponse(
            code=200,
            msg="KB ingestion successful",
            data=result
        )

    except ValueError as e:
        logger.warning(f"KB ingestion failed (invalid params): {str(e)}")
        raise HTTPException(
            status_code=400,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"KB ingestion failed: {data_id}, error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"KB ingestion failed: {str(e)}"
        )


@router.get(
    "/status",
    response_model=APIResponse[dict],
    summary="Get KB ingestion status",
    description="Get knowledge base ingestion feature configuration status"
)
async def get_kb_status(
    service: AnnotationService = Depends(get_annotation_service)
) -> APIResponse[dict]:
    """
    Get KB ingestion status

    Returns:
    - enabled: Is KB ingestion enabled
    - kb_id: Target knowledge base ID
    - auto_ingest: Auto ingest enabled
    - Other config info
    """
    try:
        config = service.config

        status = {
            "enabled": config.kb_enabled,
            "kb_id": config.kb_id if config.kb_enabled else None,
            "auto_ingest": config.kb_auto_ingest,
            "timeout": config.kb_timeout,
            "retry_times": config.kb_retry_times,
            "retry_interval": config.kb_retry_interval
        }

        return APIResponse(
            code=200,
            msg="Query successful",
            data=status
        )

    except Exception as e:
        logger.error(f"Get KB status failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Query failed: {str(e)}"
        )
