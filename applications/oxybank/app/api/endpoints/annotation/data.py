"""Data management API endpoints"""
from typing import Optional

from fastapi import APIRouter, HTTPException, Depends, Query
from loguru import logger

from core.model.annotation import (
    APIResponse,
    DataListResponse,
    QADataItem,
    AnnotationUpdateRequest,
    ApprovalRequest,
    get_data_list_query_params,
    DataListQueryParams,
)
from core.services import get_annotation_service, AnnotationService


router = APIRouter(prefix="/data", tags=["Annotation Data Management"])


@router.get(
    "",
    response_model=APIResponse[DataListResponse[QADataItem]],
    summary="Get QA data list",
    description="Query annotation data list with filtering, pagination and sorting"
)
async def get_data_list(
    params: DataListQueryParams = Depends(get_data_list_query_params),
    service: AnnotationService = Depends(get_annotation_service)
) -> APIResponse[DataListResponse[QADataItem]]:
    """
    Get data list

    Supported filters:
    - status: Data status (pending/annotated/approved/rejected, etc.)
    - priority: Priority (0-4)
    - data_type: Data type (e2e/agent/llm/tool/custom)
    - caller: Caller
    - callee: Callee
    - category: Data category
    - tags: Tags (comma separated)
    - created_after/before: Time range
    - search: Full-text search (search question or answer)
    - trace_id: Source trace ID (fuzzy match)
    - group_id: Source group ID (fuzzy match)

    Supported sorting:
    - created_at, updated_at, priority, etc.
    - asc (ascending) or desc (descending)
    """
    try:
        # Execute query
        result = await service.get_data_list(params)

        # Convert to QADataItem
        items = [QADataItem(**item) for item in result["items"]]

        response = DataListResponse[QADataItem](
            total=result["total"],
            page=params.page,
            page_size=params.page_size,
            items=items
        )

        return APIResponse(
            code=200,
            msg="Query successful",
            data=response
        )

    except Exception as e:
        logger.error(f"Data list query failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Query failed: {str(e)}"
        )


@router.get(
    "/{data_id}",
    response_model=APIResponse[QADataItem],
    summary="Get data details",
    description="Get complete QA data information by data ID"
)
async def get_data_by_id(
    data_id: str,
    service: AnnotationService = Depends(get_annotation_service)
) -> APIResponse[QADataItem]:
    """
    Get data details

    Returns complete QA data for specified data_id, including:
    - Question, answer
    - Source tracking info
    - Annotation results
    - KB ingestion status
    - etc.
    """
    try:
        data = await service.get_data_by_id(data_id)

        if not data:
            raise HTTPException(
                status_code=404,
                detail=f"Data not found: {data_id}"
            )

        return APIResponse(
            code=200,
            msg="Query successful",
            data=QADataItem(**data)
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get data details failed: {data_id}, error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Query failed: {str(e)}"
        )


@router.put(
    "/{data_id}/annotate",
    response_model=APIResponse[dict],
    summary="Update annotation",
    description="Update annotation info for specified data, including category, tags, scores, etc."
)
async def update_annotation(
    data_id: str,
    request: AnnotationUpdateRequest,
    service: AnnotationService = Depends(get_annotation_service)
) -> APIResponse[dict]:
    """
    Update annotation

    Updatable fields:
    - category: Data category
    - tags: Data tag list
    - scores: Score info (dict)
    - comment: Annotation comment
    - remark: Note info
    - annotation_data: Other custom annotation data

    Note:
    - After updating annotation, data status will automatically become annotated
    """
    try:
        success = await service.update_annotation(data_id, request)

        if not success:
            raise HTTPException(
                status_code=500,
                detail="Annotation update failed"
            )

        return APIResponse(
            code=200,
            msg="Annotation updated successfully",
            data={"data_id": data_id}
        )

    except ValueError as e:
        logger.warning(f"Annotation update failed (invalid params): {str(e)}")
        raise HTTPException(
            status_code=400,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Annotation update failed: {data_id}, error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Update failed: {str(e)}"
        )


@router.post(
    "/{data_id}/approve",
    response_model=APIResponse[dict],
    summary="Approve data",
    description="Mark specified data as approved. If auto-ingest is enabled, will auto-inject to KB"
)
async def approve_data(
    data_id: str,
    action: str = Query("approve", regex="^(approve)$", description="Approval action"),
    reason: Optional[str] = Query(None, description="Approval reason"),
    service: AnnotationService = Depends(get_annotation_service)
) -> APIResponse[dict]:
    """
    Approve data

    Features:
    - Update data status to approved
    - Record approval reason and time
    - If kb_auto_ingest enabled, auto-inject to KB

    Status flow:
    - annotated/approved → approved → (optional) kb_ingested
    """
    try:
        success = await service.approve_data(data_id, reason)

        if not success:
            raise HTTPException(
                status_code=500,
                detail="Approval failed"
            )

        return APIResponse(
            code=200,
            msg="Data approved",
            data={"data_id": data_id, "status": "approved"}
        )

    except ValueError as e:
        logger.warning(f"Approval failed (invalid params): {str(e)}")
        raise HTTPException(
            status_code=400,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Approval failed: {data_id}, error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Approval failed: {str(e)}"
        )


@router.post(
    "/{data_id}/reject",
    response_model=APIResponse[dict],
    summary="Reject data",
    description="Mark specified data as rejected, requires rejection reason"
)
async def reject_data(
    data_id: str,
    reason: str = Query(..., min_length=1, description="Rejection reason"),
    reject_category: Optional[str] = Query(None, description="Rejection category"),
    service: AnnotationService = Depends(get_annotation_service)
) -> APIResponse[dict]:
    """
    Reject data

    Features:
    - Update data status to rejected
    - Record rejection reason and category
    - Record rejection time

    Status flow:
    - annotated/approved → rejected
    """
    try:
        success = await service.reject_data(data_id, reason, reject_category)

        if not success:
            raise HTTPException(
                status_code=500,
                detail="Rejection failed"
            )

        return APIResponse(
            code=200,
            msg="Data rejected",
            data={
                "data_id": data_id,
                "status": "rejected",
                "reason": reason
            }
        )

    except ValueError as e:
        logger.warning(f"Rejection failed (invalid params): {str(e)}")
        raise HTTPException(
            status_code=400,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Rejection failed: {data_id}, error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Rejection failed: {str(e)}"
        )


@router.get(
    "/trace/{trace_id}",
    response_model=APIResponse[list[QADataItem]],
    summary="Query by trace_id",
    description="Query all data related to specified trace_id"
)
async def get_by_trace_id(
    trace_id: str,
    service: AnnotationService = Depends(get_annotation_service)
) -> APIResponse[list[QADataItem]]:
    """
    Query by trace_id

    Returns all QA data related to specified trace_id
    """
    try:
        data_list = await service.get_by_trace_id(trace_id)

        items = [QADataItem(**item) for item in data_list]

        return APIResponse(
            code=200,
            msg=f"Query successful, {len(items)} items in total",
            data=items
        )

    except Exception as e:
        logger.error(f"trace_id query failed: {trace_id}, error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Query failed: {str(e)}"
        )


@router.get(
    "/group/{group_id}",
    response_model=APIResponse[list[QADataItem]],
    summary="Query by group_id",
    description="Query all data related to specified group_id"
)
async def get_by_group_id(
    group_id: str,
    service: AnnotationService = Depends(get_annotation_service)
) -> APIResponse[list[QADataItem]]:
    """
    Query by group_id

    Returns all QA data related to specified group_id
    """
    try:
        data_list = await service.get_by_group_id(group_id)

        items = [QADataItem(**item) for item in data_list]

        return APIResponse(
            code=200,
            msg=f"Query successful, {len(items)} items in total",
            data=items
        )

    except Exception as e:
        logger.error(f"group_id query failed: {group_id}, error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Query failed: {str(e)}"
        )


@router.get(
    "/groups/summary",
    response_model=APIResponse[list],
    summary="Group summary statistics",
    description="Get summary statistics for all groups"
)
async def get_groups_summary(
    service: AnnotationService = Depends(get_annotation_service)
) -> APIResponse[list]:
    """
    Group summary statistics

    Returns statistics for each group, including:
    - group_id: Group ID
    - total_count: Total count
    - status_counts: Count by status
    """
    try:
        summary = await service.get_groups_summary()

        return APIResponse(
            code=200,
            msg="Query successful",
            data=summary
        )

    except Exception as e:
        logger.error(f"Group summary query failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Query failed: {str(e)}"
        )
