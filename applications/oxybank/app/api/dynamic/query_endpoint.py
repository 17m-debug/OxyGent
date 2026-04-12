import types

from fastapi import APIRouter, HTTPException, Request, Path
from loguru import logger

from app.api.models import APIResponse
from core.config import settings
from core.interface.endpoint_registry import EndpointRegistry
from core.interface.endpoint_show import get_query_api_info
from core.storer.doc_manager.es_kb_base_manager import ElasticsearchKbBaseManager
from core.storer.doc_manager.knowledge_index import KBSchema, check_kb_schema
from core.storer.doc_manager.rule_query_infer import DynamicEndpointGenerator

router = APIRouter(prefix="/query_interface", tags=["Knowledge Base Retrieval Method Dynamic Management"])

# Initialize knowledge base management client
kb_base_client = ElasticsearchKbBaseManager(settings.es_client)

# Get endpoint registry singleton instance
def get_endpoint_registry() -> EndpointRegistry:
    """Get endpoint registry singleton instance"""
    return EndpointRegistry.get_instance()


@router.post(
    "/{kb_name}",
    response_model=APIResponse[dict],
    summary="Create knowledge base retrieval endpoint based on kb_name's kb_schema and bind it to the current service"
)
async def create_kb_query_interface(
        kb_name: str,
        request: Request
):
    """
    Create knowledge base retrieval endpoint based on kb_schema corresponding to kb_name

    Process:
    1. Query knowledge base information based on kb_name
    2. Check if knowledge base exists
    3. Get and validate kb_schema
    4. Use DynamicEndpointGenerator to create retrieval endpoint
    5. Dynamically register to FastAPI application
    6. Persist registration information to configuration file

    Args:
        kb_name: Knowledge base name
        request: FastAPI Request object, used to get app instance

    Returns:
        APIResponse: Contains generated endpoint information

    Raises:
        HTTPException 404: Knowledge base does not exist
        HTTPException 400: Schema is invalid or does not meet requirements
        HTTPException 409: Knowledge base endpoint already registered
        HTTPException 500: Failed to create endpoint
    """
    # 1. Check if knowledge base is already registered
    endpoint_registry = get_endpoint_registry()
    if endpoint_registry and endpoint_registry.is_kb_registered(kb_name):
        raise HTTPException(
            status_code=409,
            detail=f"Retrieval endpoint for knowledge base '{kb_name}' has already been created, please restart the service to recreate"
        )

    # 2. Query knowledge base information
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
        logger.error(f"Failed to query knowledge base '{kb_name}': {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to query knowledge base: {str(e)}"
        )

    # 3. Get kb_schema
    kb_info = kb_info_list[0]
    kb_schema_dict = kb_info.get("kb_schema")

    if not kb_schema_dict:
        raise HTTPException(
            status_code=400,
            detail=f"Knowledge base '{kb_name}' has not configured Schema, please configure Schema via POST /api/v1/kb_base/{kb_name}/schema first"
        )

    # 4. Convert dictionary to KBSchema object
    try:
        kb_schema = KBSchema(**kb_schema_dict)
    except Exception as e:
        logger.error(f"Failed to parse knowledge base '{kb_name}' Schema: {e}")
        raise HTTPException(
            status_code=400,
            detail=f"Invalid Schema format: {str(e)}"
        )

    # 5. Validate Schema meets requirements
    try:
        if not check_kb_schema(kb_schema):
            raise HTTPException(
                status_code=400,
                detail="Schema validation failed: missing match_rules configuration"
            )
    except ValueError as e:
        logger.error(f"Knowledge base '{kb_name}' Schema validation failed: {e}")
        raise HTTPException(
            status_code=400,
            detail=f"Schema validation failed: {str(e)}"
        )

    # 6. Create dynamic endpoint generator
    try:
        generator = DynamicEndpointGenerator(kb_name=kb_name, kb_schema=kb_schema)
        dynamic_router = generator.generate_all_endpoints()
    except Exception as e:
        logger.error(f"Failed to create dynamic endpoint generator: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to create retrieval endpoint: {str(e)}"
        )

    # 7. Get FastAPI app instance and register routes
    try:
        app = request.app
        app.include_router(dynamic_router)

        # Generate endpoint list
        endpoints = []
        if kb_schema.match_rules:
            for rule_idx in range(len(kb_schema.match_rules)):
                endpoints.append(f"POST /kb/{kb_name}/search/rule_{rule_idx}")

        logger.info(f"✅ Knowledge base '{kb_name}' retrieval endpoint created successfully, total {len(endpoints)} endpoints")

        return APIResponse(
            code=200,
            msg=f"Knowledge base '{kb_name}' retrieval endpoint created successfully",
            data={
                "kb_name": kb_name,
                "total_rules": len(kb_schema.match_rules or []),
                "endpoints": endpoints,
                "router_prefix": f"/kb/{kb_name}"
            }
        )
    except Exception as e:
        logger.error(f"Failed to register retrieval endpoint to FastAPI: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to register retrieval endpoint: {str(e)}"
        )


@router.get(
    "/{kb_name}",
    response_model=APIResponse[list],
    summary="Get bound knowledge base retrieval endpoint information based on kb_name"
)
def get_kb_query_interface(
        request: Request,
        kb_name: str = Path(...),
):
    try:
        routes = []

        app = request.app
        base_url = str(request.base_url)
        for route in app.routes:
            if hasattr(route, "path") and route.path.startswith(f"/kb/{kb_name}/search"):
                routes.append(route)

        api_infos = get_query_api_info(routes, base_url)
        logger.info(f"Successfully queried knowledge base {kb_name} retrieval endpoint information")

        return APIResponse(
            code=200,
            msg=f"Successfully queried knowledge base {kb_name} endpoint information",
            data=api_infos
        )
    except Exception as e:
        logger.error(f"Failed to query knowledge base {kb_name} endpoint information: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to query knowledge base {kb_name} endpoint information: {str(e)}"
        )
