import datetime

from fastapi import APIRouter, HTTPException, Path, Request
from fastapi.params import Depends
from loguru import logger

from core.interface.endpoint_registry import EndpointRegistry
from app.api.models import (
    APIResponse,
    KnowledgeBaseItem,
    PaginationParams,
    PaginatedResponse,
    get_pagination_params,
)
from core.config import settings
from core.storer.doc_manager.es_index_manager import ElasticsearchIndexManager
from core.storer.doc_manager.es_kb_base_manager import ElasticsearchKbBaseManager
from core.storer.doc_manager.es_kb_chunk_manager import ElasticsearchKbChunkManager
from core.storer.doc_manager.es_kb_file_manager import ElasticsearchKbFileManager
from core.storer.doc_manager.knowledge_index import KBSchema, infer_mapping_from_schema, infer_vearch_space_schema
from core.storer.doc_manager.rule_query_infer import DynamicEndpointGenerator
from core.storer.vector_manager.vearch_manager import VearchManager
from utils.hash_util import str_to_md5

router = APIRouter(prefix="/kb_base", tags=["Knowledge Base Management"])

# Initialize instances
kb_index_client = ElasticsearchIndexManager(settings.es_client)
kb_base_client = ElasticsearchKbBaseManager(settings.es_client)
kb_chunk_client = ElasticsearchKbChunkManager(settings.es_client)
kb_file_client = ElasticsearchKbFileManager(settings.es_client)
vearch_manager = VearchManager(vearch_client=settings.vearch_client)


@router.get(
    "",
    response_model=APIResponse[PaginatedResponse[KnowledgeBaseItem]],
    summary="Get all knowledge bases",
    description="Returns a list of all knowledge bases in the system",
    response_description="Knowledge base list containing ID, name, description, type, path, and other information",
)
def get_all_knowledge_base(
        pagination: PaginationParams = Depends(get_pagination_params)
) -> APIResponse[PaginatedResponse[KnowledgeBaseItem]]:
    """
    Get all knowledge bases

    Returns information about all created knowledge bases, including:
    - kb_id: Unique identifier of the knowledge base
    - kb_name: Name of the knowledge base
    - kb_description: Description of the knowledge base
    - kb_type: Type of the knowledge base (e.g., unstructured)
    - kb_extra_info: Additional information (dictionary format)
    """
    try:
        result = kb_base_client.kb_list(
            page=pagination.page,
            size=pagination.size,
        )
        
        # Validate return result
        if not result or "items" not in result:
            logger.warning("Knowledge base list query returned empty result")
            return APIResponse(
                code=200,
                msg="Successfully retrieved knowledge base list",
                data=PaginatedResponse[KnowledgeBaseItem](
                    items=[],
                    total=0,
                    page=pagination.page,
                    size=pagination.size,
                    pages=0
                )
            )

        kb_items = [KnowledgeBaseItem(**kb_item) for kb_item in result["items"]]

        paginated_response = PaginatedResponse[KnowledgeBaseItem](
            items=kb_items,
            total=result["total"],
            page=result["page"],
            size=result["size"],
            pages=result["pages"],
        )

        return APIResponse(
            code=200,
            msg="Successfully retrieved knowledge base list",
            data=paginated_response
        )
    except Exception as e:
        logger.error(f"Failed to get knowledge base list: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get knowledge base list: {str(e)}"
        )


@router.post(
    "",
    response_model=APIResponse[str],
    summary="Create an empty knowledge base",
    description="Create an empty knowledge base"
)
def create_knowledge_base(
        kb_item: KnowledgeBaseItem,
) -> APIResponse[str]:
    """
    Create a new knowledge base

    Args:
        kb_item: Knowledge base information

    Returns:
        APIResponse: Contains the ID of the newly created knowledge base

    Raises:
        HTTPException 409: Knowledge base name already exists
        HTTPException 400: Knowledge base name is empty or format is incorrect
        HTTPException 500: Creation failed
    """
    # Validate knowledge base name
    if not kb_item.kb_name or not kb_item.kb_name.strip():
        raise HTTPException(
            status_code=400,
            detail="Knowledge base name cannot be empty"
        )

    # Check if knowledge base already exists
    try:
        if kb_base_client.kb_exists(kb_item.kb_name):
            raise HTTPException(
                status_code=409,
                detail=f"Knowledge base '{kb_item.kb_name}' already exists"
            )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to check knowledge base existence: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to check knowledge base existence: {str(e)}"
        )

    # Generate knowledge base ID and timestamp
    kb_id = str_to_md5(kb_item.kb_name)
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    kb_item.kb_id = kb_id
    kb_item.create_time = current_time
    kb_item.update_time = current_time

    try:
        # Create knowledge base
        if not kb_base_client.kb_add(kb_item.model_dump()):
            logger.error(f"Knowledge base {kb_item.kb_name} creation failed")
            raise HTTPException(
                status_code=500,
                detail="Knowledge base creation failed, please check system status"
            )

        logger.info(f"Knowledge base {kb_item.kb_name} (ID: {kb_id}) created successfully")
        return APIResponse(
            code=200,
            msg="Knowledge base created successfully",
            data=kb_id
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Exception while creating knowledge base: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to create knowledge base: {str(e)}"
        )


@router.delete(
    "/{kb_name}",
    response_model=APIResponse[str],
    summary="Delete knowledge base",
    description="Delete the specified knowledge base and its indexes and vector space"
)
def delete_knowledge_base(
        kb_name: str = Path(..., description="Knowledge base name"),
) -> APIResponse[str]:
    """
    Delete knowledge base

    Args:
        kb_name: Knowledge base name

    Returns:
        APIResponse: Deletion result

    Raises:
        HTTPException 400: Knowledge base name is empty
        HTTPException 404: Knowledge base does not exist
        HTTPException 500: Deletion failed
    """
    # Validate input
    if not kb_name or not kb_name.strip():
        raise HTTPException(
            status_code=400,
            detail="Knowledge base name cannot be empty"
        )

    # 1. Query knowledge base information
    try:
        kb_info_list = kb_base_client.kb_info_search_name(kb_name=kb_name)
        if not kb_info_list:
            raise HTTPException(
                status_code=404,
                detail=f"Knowledge base '{kb_name}' does not exist"
            )
        kb_id = kb_info_list[0].get("kb_id")
        if not kb_id:
            raise HTTPException(
                status_code=500,
                detail="Knowledge base information is missing kb_id"
            )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to query knowledge base: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to query knowledge base: {str(e)}"
        )

    # 2. Unbind dynamic endpoints
    try:
        endpoint_registry = EndpointRegistry.get_instance()
        if endpoint_registry and endpoint_registry.is_kb_registered(kb_name):
            endpoint_registry.unbind_kb_endpoints(kb_name)
    except Exception as e:
        logger.error(f"Failed to unbind knowledge base endpoints: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to unbind knowledge base endpoints: {str(e)}"
        )

    # 3. Delete knowledge base file/chunk information
    try:
        file_info_list = kb_file_client.kb_file_search(kb_id=kb_id) or []
        file_ids = [
            file_info.get("ori_file_id")
            for file_info in file_info_list
            if file_info.get("ori_file_id")
        ]
        if file_ids:
            if not kb_chunk_client.kb_delete_chunk(kb_id, file_ids):
                raise HTTPException(
                    status_code=500,
                    detail="Failed to delete knowledge base document chunks"
                )
            if not kb_file_client.kb_delete_file(kb_id, file_ids):
                raise HTTPException(
                    status_code=500,
                    detail="Failed to delete knowledge base file records"
                )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete knowledge base file information: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to delete knowledge base file information: {str(e)}"
        )

    # 4. Delete ES index
    try:
        if kb_index_client.index_exists(kb_name):
            if not kb_index_client.delete_index(kb_name):
                raise HTTPException(
                    status_code=500,
                    detail="Failed to delete ES index"
                )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete ES index: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to delete ES index: {str(e)}"
        )

    # 5. Delete Vearch vector space
    try:
        if not vearch_manager.delete_space(settings.vearch_config.db_name, kb_name):
            raise HTTPException(
                status_code=500,
                detail="Failed to delete Vearch vector space"
            )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete Vearch vector space: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to delete Vearch vector space: {str(e)}"
        )

    # 6. Delete knowledge base basic information
    try:
        if not kb_base_client.kb_delete(kb_name):
            raise HTTPException(
                status_code=500,
                detail="Failed to delete knowledge base information"
            )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete knowledge base information: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to delete knowledge base information: {str(e)}"
        )

    logger.info(f"Knowledge base {kb_name} (ID: {kb_id}) deleted successfully")
    return APIResponse(
        code=200,
        msg="Knowledge base deleted successfully",
        data=kb_name
    )


# This endpoint is used to update the knowledge base schema
@router.post(
    "/{kb_name}/schema",
    response_model=APIResponse[str],
    summary="Update knowledge base Schema",
    description="Update the Schema configuration of the specified knowledge base"
)
async def update_kb_schema(
        request: Request,
        kb_schema: KBSchema,
        kb_name: str = Path(..., description="Knowledge base name")
) -> APIResponse[str]:
    """
    Update knowledge base Schema configuration

    Args:
        request: FastAPI Request object, used to get app instance
        kb_name: Knowledge base name
        kb_schema: New Schema configuration, including field definitions and matching policies

    Returns:
        APIResponse: Update result

    Raises:
        HTTPException 400: Schema configuration is invalid or missing matching policies
        HTTPException 404: Knowledge base does not exist
        HTTPException 500: Update failed or index creation failed
    """
    # Validate input
    if not kb_name or not kb_name.strip():
        raise HTTPException(
            status_code=400,
            detail="Knowledge base name cannot be empty"
        )

    # Validate Schema validity
    if not kb_schema.fields or len(kb_schema.fields) == 0:
        raise HTTPException(
            status_code=400,
            detail="Schema must contain at least one field definition"
        )

    # 1. Query whether knowledge base exists
    try:
        kb_info_list = kb_base_client.kb_info_search_name(kb_name=kb_name)
        if not kb_info_list or len(kb_info_list) == 0:
            raise HTTPException(
                status_code=404,
                detail=f"Knowledge base '{kb_name}' does not exist"
            )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to query knowledge base: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to query knowledge base: {str(e)}"
        )

    # 2. Update knowledge base schema fields and update time
    update_fields = {
        "kb_schema": kb_schema.model_dump(),
        "update_time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    try:
        # 3. Call update method
        if not kb_base_client.kb_update(kb_name=kb_name, update_fields=update_fields):
            raise HTTPException(
                status_code=500,
                detail="Failed to update knowledge base Schema"
            )
        logger.info(f"Knowledge base {kb_name} Schema updated successfully")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Schema update exception: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to update Schema: {str(e)}"
        )

    # 4. Infer and create ES index
    es_index_mapping = None
    vearch_space_schema = None

    try:
        es_index_mapping = infer_mapping_from_schema(schema=kb_schema)
        if es_index_mapping:
            index_create_result = kb_index_client.create_index(kb_name, es_index_mapping)
            if not index_create_result:
                logger.error(f"Failed to create ES index {kb_name}")
                raise HTTPException(
                    status_code=500,
                    detail="Failed to create ES index"
                )
            logger.info(f"ES index {kb_name} created successfully")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"ES index creation exception: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to create ES index: {str(e)}"
        )

    # 5. Infer and create Vearch space
    try:
        vearch_space_schema = infer_vearch_space_schema(schema=kb_schema, kb_name=kb_name)
        if vearch_space_schema:
            # Check if space already exists, if so delete it first
            if vearch_manager.space_exists(settings.vearch_config.db_name, kb_name):
                logger.warning(f"Vearch space {kb_name} already exists, will delete and recreate")
                delete_result = vearch_manager.delete_space(settings.vearch_config.db_name, kb_name)
                if not delete_result:
                    logger.error(f"Failed to delete Vearch space {kb_name}")
                    raise HTTPException(
                        status_code=500,
                        detail="Failed to delete old Vearch vector space"
                    )
                logger.info(f"Vearch space {kb_name} deleted")

            # Create new Vearch space
            space_create_result = vearch_manager.create_space_with_schema(settings.vearch_config.db_name, vearch_space_schema)
            if not space_create_result:
                logger.error(f"Failed to create Vearch space {kb_name}")
                raise HTTPException(
                    status_code=500,
                    detail="Failed to create Vearch vector space"
                )
            logger.info(f"Vearch space {kb_name} created successfully")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Vearch space creation exception: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to create Vearch space: {str(e)}"
        )

    # If there is no retrieval strategy, raise an error
    if not vearch_space_schema and not es_index_mapping:
        raise HTTPException(
            status_code=400,
            detail="Schema must include at least one retrieval strategy (ES full-text search or Vearch vector search)"
        )

    # 6. Check if endpoints are already bound
    try:
        endpoint_registry = EndpointRegistry.get_instance()
        if endpoint_registry and endpoint_registry.is_kb_registered(kb_name):
            logger.info(f"Knowledge base '{kb_name}' endpoints are already bound, this schema update operation requires unbinding and rebinding endpoints")
            endpoint_registry.unbind_kb_endpoints(kb_name)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to check endpoint binding status: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to check endpoint binding status: {str(e)}"
        )

    # 7. Create dynamic endpoint generator
    try:
        generator = DynamicEndpointGenerator(kb_name=kb_name, kb_schema=kb_schema)
        dynamic_router = generator.generate_all_endpoints()
    except Exception as e:
        logger.error(f"Failed to create dynamic endpoint generator: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to create retrieval endpoint: {str(e)}"
        )

    # 8. Get FastAPI app instance and register routes
    try:
        app = request.app
        app.include_router(dynamic_router)

        # Generate endpoint list
        endpoints = []
        if kb_schema.match_rules:
            for rule_idx in range(len(kb_schema.match_rules)):
                endpoints.append(f"POST /kb/{kb_name}/search/rule_{rule_idx}")

        logger.info(f"✅ Knowledge base '{kb_name}' retrieval endpoints created successfully, total {len(endpoints)} endpoints")

        return APIResponse(
            code=200,
            msg="success",
            data=f"Knowledge base '{kb_name}' schema updated successfully and retrieval endpoints created successfully"
        )
    except Exception as e:
        logger.error(f"Failed to register retrieval endpoint to FastAPI: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to register retrieval endpoint: {str(e)}"
        )


@router.get(
    "/{kb_name}/schema/exists",
    response_model=APIResponse[bool],
    summary="Check if knowledge base Schema exists",
    description="Check if the specified knowledge base has Schema configured"
)
def get_kb_schema(
        kb_name: str = Path(..., description="Knowledge base name"),
) -> APIResponse[bool]:
    """
    Check if knowledge base Schema exists

    Args:
        kb_name: Knowledge base name

    Returns:
        APIResponse[bool]: Whether there is a valid Schema configuration

    Raises:
        HTTPException 400: Knowledge base name is empty
        HTTPException 404: Knowledge base does not exist
        HTTPException 500: Query failed
    """
    # Validate input
    if not kb_name or not kb_name.strip():
        raise HTTPException(
            status_code=400,
            detail="Knowledge base name cannot be empty"
        )

    try:
        result = kb_base_client.kb_info_search_name(kb_name=kb_name)
        if not result or len(result) == 0:
            raise HTTPException(
                status_code=404,
                detail=f"Knowledge base '{kb_name}' does not exist"
            )

        kb_info = result[0]

        # Determine if schema exists and is valid
        has_schema = (
            kb_info.get("kb_schema") is not None and
            kb_info.get("kb_schema").get("fields") is not None and
            len(kb_info.get("kb_schema").get("fields")) > 0 and
            kb_info.get("kb_schema").get("match_rules") is not None and
            len(kb_info.get("kb_schema").get("match_rules")) > 0
        )

        return APIResponse(
            code=200,
            msg="Query successful",
            data=has_schema
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to check Schema existence: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to check Schema: {str(e)}"
        )
