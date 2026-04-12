"""Service Factory - Service factory class

Used to create and manage service instances, using singleton pattern
"""
from typing import Optional

from loguru import logger

from core.config import settings
from core.services.annotation_service import AnnotationService
from core.storer.doc_manager.annotation_manager import AnnotationManager

# Singleton instances
_annotation_service: Optional[AnnotationService] = None
_annotation_manager: Optional[AnnotationManager] = None


def get_annotation_manager() -> AnnotationManager:
    """
    Get annotation data manager (singleton)

    Returns:
        AnnotationManager instance
    """
    global _annotation_manager

    if _annotation_manager is None:
        logger.info("Initializing AnnotationManager...")
        _annotation_manager = AnnotationManager(
            es_client=settings.es_client,
            index_prefix=settings.annotation_config.es_index_prefix
        )

        # Initialize index
        _annotation_manager.initialize()
        logger.info(f"AnnotationManager initialized, index: {_annotation_manager.index_name}")

    return _annotation_manager


def get_annotation_service() -> AnnotationService:
    """
    Get annotation service (singleton)

    Returns:
        AnnotationService instance
    """
    global _annotation_service

    if _annotation_service is None:
        logger.info("Initializing AnnotationService...")

        # Get dependencies
        manager = get_annotation_manager()
        config = settings.annotation_config

        # Create service instance
        _annotation_service = AnnotationService(
            annotation_manager=manager,
            config=config
        )

        logger.info("AnnotationService initialized")

    return _annotation_service


def reset_annotation_services():
    """
    Reset all services (mainly for testing)

    Note: Use with caution in production
    """
    global _annotation_service, _annotation_manager

    _annotation_service = None
    _annotation_manager = None

    logger.info("Annotation services reset")
