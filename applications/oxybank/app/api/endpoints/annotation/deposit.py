"""Data deposit API endpoints"""
from fastapi import APIRouter, HTTPException, Depends
from fastapi.params import Body
from loguru import logger

from core.model.annotation import (
    APIResponse,
    DepositRequest,
    DepositBatchRequest,
    DepositResponse,
    DepositBatchResponse,
)
from core.services import get_annotation_service, AnnotationService


router = APIRouter(prefix="/deposit", tags=["Annotation Data Deposit"])


@router.post(
    "",
    response_model=APIResponse[DepositResponse],
    summary="Deposit single QA pair",
    description="Store QA pair to annotation platform with type inference"
)
async def deposit_data(
    request: DepositRequest,
    service: AnnotationService = Depends(get_annotation_service)
) -> APIResponse[DepositResponse]:
    """
    Deposit single QA pair

    Features:
    - Auto hash-based deduplication
    - Auto type inference (e2e/agent/llm/tool/custom)
    - Auto priority inference (0-4)
    - Generate unique data_id
    - Save to Elasticsearch

    Deduplication:
    - Hash based on question + answer
    - Check cache first, then ES
    - Return existing data_id if duplicate
    """
    try:
        result = await service.deposit_data(request)

        return APIResponse(
            code=200,
            msg="Data deposited successfully",
            data=result
        )

    except ValueError as e:
        logger.warning(f"Deposit failed (invalid params): {str(e)}")
        raise HTTPException(
            status_code=400,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Deposit failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Deposit failed: {str(e)}"
        )


@router.post(
    "/batch",
    response_model=APIResponse[DepositBatchResponse],
    summary="Batch deposit",
    description="Deposit multiple QA pairs (1-1000 items)"
)
async def deposit_batch(
    request: DepositBatchRequest,
    service: AnnotationService = Depends(get_annotation_service)
) -> APIResponse[DepositBatchResponse]:
    """
    Batch deposit QA pairs

    Features:
    - Auto generate batch_id
    - Process each item
    - Count success/duplicate/failed
    - Return detailed results

    Notes:
    - Continue processing even if some items fail
    - Duplicates are marked in results, not treated as errors
    """
    try:
        # Validate data count
        data_count = len(request.data_list)
        if data_count == 0:
            raise HTTPException(
                status_code=400,
                detail="Data list cannot be empty"
            )

        if data_count > 1000:
            raise HTTPException(
                status_code=400,
                detail=f"Batch deposit max 1000 items, got {data_count}"
            )

        result = await service.deposit_batch(request)

        return APIResponse(
            code=200,
            msg=f"Batch completed: {result.success_count} success, {result.duplicate_count} duplicate, {result.failed_count} failed",
            data=result
        )

    except HTTPException:
        raise
    except ValueError as e:
        logger.warning(f"Batch deposit failed (invalid params): {str(e)}")
        raise HTTPException(
            status_code=400,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Batch deposit failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Batch deposit failed: {str(e)}"
        )
