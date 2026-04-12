"""
Log Configuration Endpoint: For dynamically modifying log levels
"""
from fastapi import APIRouter, HTTPException
from loguru import logger
from typing import Literal

from app.api.models import APIResponse

router = APIRouter(prefix="/log", tags=["Log Configuration"])

# Store current log level and handler ID
_log_level: str = "DEBUG"
_handler_id: int | None = None


def get_log_level() -> str:
    """Get current log level"""
    return _log_level


def set_log_level(level: str) -> None:
    """Dynamically set log level

    Args:
        level: Log level, options: TRACE, DEBUG, INFO, SUCCESS, WARNING, ERROR, CRITICAL
    """
    global _log_level, _handler_id
    
    valid_levels = ["TRACE", "DEBUG", "INFO", "SUCCESS", "WARNING", "ERROR", "CRITICAL"]
    level_upper = level.upper()
    
    if level_upper not in valid_levels:
        raise ValueError(f"Invalid log level: {level}, valid values: {valid_levels}")

    # If handler exists, remove it first
    if _handler_id is not None:
        try:
            logger.remove(_handler_id)
        except ValueError:
            pass  # Handler may have been removed

    # Add new handler with new log level
    import sys
    _handler_id = logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level=level_upper,
        colorize=True
    )
    
    _log_level = level_upper
    logger.info(f"Log level updated to: {level_upper}")


@router.get("/level", summary="Get current log level")
async def get_current_log_level() -> APIResponse[str]:
    """Get current log level"""
    return APIResponse(
        code=200,
        msg="Successfully retrieved log level",
        data=_log_level
    )


@router.post("/level", summary="Set log level")
async def set_log_level_endpoint(
    level: Literal["TRACE", "DEBUG", "INFO", "SUCCESS", "WARNING", "ERROR", "CRITICAL"]
) -> APIResponse[str]:
    """Dynamically set log level

    Allows modifying log level at runtime without restarting the service.

    Args:
        level: Log level
            - TRACE: Most detailed logging
            - DEBUG: Debug information
            - INFO: General information (default)
            - SUCCESS: Success information
            - WARNING: Warning information
            - ERROR: Error information
            - CRITICAL: Critical error

    Returns:
        APIResponse: Contains the updated log level
    """
    try:
        set_log_level(level)
        return APIResponse(
            code=200,
            msg="success",
            data=f"Log level updated to: {level}"
        )
    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Failed to set log level: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to set log level: {str(e)}"
        )

