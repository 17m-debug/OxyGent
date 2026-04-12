from fastapi import APIRouter

from app.api.endpoints import knowledge_base, knowledge_file, knowledge_chunk
from app.api.endpoints.annotation import deposit, data, stats, kb
from app.api.log import log_config
from app.api.dynamic import query_endpoint

api_router = APIRouter()

# Knowledge base management routes
api_router.include_router(knowledge_base.router)
api_router.include_router(knowledge_file.router)
api_router.include_router(knowledge_chunk.router)
api_router.include_router(query_endpoint.router)

# Annotation platform routes
api_router.include_router(deposit.router)
api_router.include_router(data.router)
api_router.include_router(kb.router)
api_router.include_router(stats.router)

# Log routes
api_router.include_router(log_config.router)
