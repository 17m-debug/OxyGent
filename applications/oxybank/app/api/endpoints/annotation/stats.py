"""Statistics analysis API endpoints"""
from typing import Optional

from fastapi import APIRouter, HTTPException, Depends, Query
from loguru import logger

from core.model.annotation import (
    APIResponse,
    OverallStatsResponse,
    PendingP0Response,
    PendingP0Item,
    TypeStatsResponse,
    TypeStatsItem,
    DataListQueryParams,
    get_data_list_query_params,
)
from core.services import get_annotation_service, AnnotationService


router = APIRouter(prefix="/stats", tags=["Annotation Statistics"])


@router.get(
    "",
    response_model=APIResponse[OverallStatsResponse],
    summary="Get overall QA statistics",
    description="Get overall statistics data for annotation platform"
)
async def get_overall_stats(
    filters: Optional[DataListQueryParams] = Depends(get_data_list_query_params),
    service: AnnotationService = Depends(get_annotation_service)
) -> APIResponse[OverallStatsResponse]:
    """
    Get overall statistics

    Supports the same filters as data query:
    - status, priority, data_type, caller, callee, category, tags
    - created_after/before: Time range
    - search: Full-text search
    - trace_id, group_id: Source tracking filters

    Returns:
    - total_count: Total data count
    - pending_count: Pending count
    - annotated_count: Annotated count
    - approved_count: Approved count
    - rejected_count: Rejected count
    - kb_ingested_count: KB ingested count
    - kb_pending_count: KB pending count
    - kb_failed_count: KB failed count
    - priority_stats: Stats by priority
    - type_stats: Stats by type
    """
    try:
        stats = await service.get_overall_stats(filters)

        return APIResponse(
            code=200,
            msg="Query successful",
            data=stats
        )

    except Exception as e:
        logger.error(f"Get overall stats failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Query failed: {str(e)}"
        )


@router.get(
    "/pending-p0",
    response_model=APIResponse[PendingP0Response],
    summary="Get pending P0 data",
    description="Get all pending P0 priority data"
)
async def get_pending_p0_stats(
    limit: int = Query(100, ge=1, le=500, description="Result count limit"),
    service: AnnotationService = Depends(get_annotation_service)
) -> APIResponse[PendingP0Response]:
    """
    Get pending P0 data

    P0 data definition:
    - Data with priority 0
    - Usually e2e (end-to-end) type data
    - Highest priority, needs priority processing

    Returns:
    - total: Total P0 pending data count
    - items: P0 data list
    """
    try:
        result = await service.get_pending_p0_stats(limit)

        return APIResponse(
            code=200,
            msg="Query successful",
            data=result
        )

    except Exception as e:
        logger.error(f"Get P0 stats failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Query failed: {str(e)}"
        )


@router.get(
    "/by-type",
    response_model=APIResponse[TypeStatsResponse],
    summary="Get statistics by type",
    description="Get statistics by data type (e2e/agent/llm/tool/custom)"
)
async def get_stats_by_type(
    filters: Optional[DataListQueryParams] = Depends(get_data_list_query_params),
    service: AnnotationService = Depends(get_annotation_service)
) -> APIResponse[TypeStatsResponse]:
    """
    Get statistics by type

    Supports the same filters as data query:
    - status, priority, data_type, caller, callee, category, tags
    - created_after/before: Time range
    - search: Full-text search
    - trace_id, group_id: Source tracking filters

    Returns statistics for each data type:
    - data_type: Data type
    - total_count: Total count
    - pending_count: Pending count
    - approved_count: Approved count
    - rejected_count: Rejected count
    """
    try:
        result = await service.get_stats_by_type(filters)

        return APIResponse(
            code=200,
            msg="Query successful",
            data=result
        )

    except Exception as e:
        logger.error(f"Get type stats failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Query failed: {str(e)}"
        )
