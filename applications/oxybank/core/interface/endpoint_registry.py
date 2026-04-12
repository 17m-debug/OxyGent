"""
Dynamic Endpoint Registration Management Module

Responsible for persistence and recovery of knowledge base retrieval endpoints
- Save registered knowledge base Schema to configuration file
- Restore all registered endpoints from configuration file on application startup
- Automatically persist when registering endpoints dynamically at runtime
"""
from typing import Optional

from fastapi import FastAPI
from loguru import logger

from core.config import settings
from core.storer.doc_manager.es_kb_base_manager import ElasticsearchKbBaseManager
from core.storer.doc_manager.knowledge_index import KBSchema
from core.storer.doc_manager.rule_query_infer import DynamicEndpointGenerator


class EndpointRegistry:
    """Dynamic Endpoint Registration Manager (Singleton Pattern)

    Responsible for managing registration, persistence, and recovery of knowledge base retrieval endpoints
    Uses singleton pattern to ensure only one instance in the entire application
    """

    _instance: Optional['EndpointRegistry'] = None
    _initialized: bool = False

    def __new__(cls, app: FastAPI = None):
        """Singleton pattern implementation

        Args:
            app: FastAPI application instance (only needed on first creation)

        Returns:
            EndpointRegistry singleton instance
        """
        if cls._instance is None:
            if app is None:
                raise ValueError("Must provide app parameter when creating EndpointRegistry for the first time")
            cls._instance = super(EndpointRegistry, cls).__new__(cls)
        return cls._instance

    def __init__(self, app: FastAPI = None):
        """Initialize registry

        Args:
            app: FastAPI application instance (only needed on first creation)
        """
        # Avoid duplicate initialization
        if self._initialized:
            return

        if app is None:
            raise ValueError("Must provide app parameter when creating EndpointRegistry for the first time")

        self.app = app
        # ES client (for querying knowledge bases)
        self.kb_base_client = ElasticsearchKbBaseManager(settings.es_client)
        # Mark as initialized
        EndpointRegistry._initialized = True

    @classmethod
    def get_instance(cls) -> Optional['EndpointRegistry']:
        """Get singleton instance

        Returns:
            EndpointRegistry instance, or None if not initialized
        """
        return cls._instance


    def is_kb_registered(self, kb_name: str) -> bool:
        """Check if knowledge base is registered (by checking FastAPI routes)

        Args:
            kb_name: Knowledge base name

        Returns:
            Whether registered
        """
        # Check if the knowledge base route is registered in FastAPI application
        prefix = f"/kb/{kb_name}"
        for route in self.app.routes:
            if hasattr(route, 'path') and route.path.startswith(prefix):
                return True
        return False
    
    def unbind_kb_endpoints(self, kb_name: str) -> bool:
        """Unbind all endpoints for the specified knowledge base

        Remove all routes related to this knowledge base from the FastAPI application
        Supports route identification via both path prefix and route name

        Args:
            kb_name: Knowledge base name

        Returns:
            Whether unbinding was successful (returns False if knowledge base is not registered)
        """
        prefix = f"/kb/{kb_name}"
        routes_to_remove = []

        # Find all routes that need to be removed
        # Method 1: Match by path prefix
        # Method 2: Match by route name (DynamicEndpointGenerator generates route names in format: {kb_name}_search_rule_{rule_idx})
        for route in self.app.routes:
            should_remove = False

            # Check path prefix
            if hasattr(route, 'path') and route.path.startswith(prefix):
                should_remove = True

            # Check route name (as backup method)
            if not should_remove and hasattr(route, 'name') and route.name:
                if route.name.startswith(f"{kb_name}_search_rule_"):
                    should_remove = True

            if should_remove:
                routes_to_remove.append(route)

        if not routes_to_remove:
            logger.warning(f"⚠️  Knowledge base '{kb_name}' endpoints not registered, no need to unbind")
            return False

        # Remove these routes from app.routes
        removed_count = 0
        for route in routes_to_remove:
            try:
                self.app.routes.remove(route)
                route_path = getattr(route, 'path', 'unknown')
                route_name = getattr(route, 'name', 'unknown')
                logger.debug(f"🗑️  Removed route: {route_path} (name: {route_name})")
                removed_count += 1
            except ValueError:
                # Route may have already been removed, ignore
                route_path = getattr(route, 'path', 'unknown')
                logger.debug(f"⚠️  Route {route_path} does not exist in routes list")

        logger.info(f"✅ Knowledge base '{kb_name}' {removed_count} endpoints unbound")
        return True
    
    def restore_all_endpoints(self):
        """Restore all knowledge bases that need endpoint binding

        Query all knowledge bases from ES and check the auto_bind_query field:
        - If auto_bind_query=True, bind the endpoints
        - If auto_bind_query=False and endpoints are bound, unbind the endpoints
        """
        try:
            # Query all knowledge bases from ES
            all_kbs = self.kb_base_client.kb_list_all()

            if not all_kbs:
                logger.info("ℹ️  No knowledge bases to process")
                return

            logger.info(f"🚀 Starting to check auto_bind_query configuration for {len(all_kbs)} knowledge bases...")

            bound_count = 0
            skipped_count = 0
            unbound_count = 0

            # Iterate through each knowledge base
            for kb_info in all_kbs:
                kb_name = kb_info.get("kb_name")
                auto_bind_query = kb_info.get("auto_bind_query", True)  # Default is True
                kb_schema_dict = kb_info.get("kb_schema")

                if not kb_name:
                    logger.warning(f"⚠️  Knowledge base information missing kb_name, skipping")
                    continue

                # If auto_bind_query is False, check if unbinding is needed
                if not auto_bind_query:
                    if self.is_kb_registered(kb_name):
                        # Endpoints are bound but configured as False, need to unbind
                        if self.unbind_kb_endpoints(kb_name):
                            unbound_count += 1
                    else:
                        logger.debug(f"ℹ️  Knowledge base '{kb_name}' auto_bind_query is False and endpoints not bound, skipping")
                        skipped_count += 1
                    continue

                # auto_bind_query=True, need to bind endpoints
                # Check if already bound
                if self.is_kb_registered(kb_name):
                    logger.info(f"ℹ️  Knowledge base '{kb_name}' endpoints already bound, skipping")
                    skipped_count += 1
                    continue

                # Check if schema exists
                if not kb_schema_dict:
                    logger.warning(f"⚠️  Knowledge base '{kb_name}' Schema information is empty, skipping binding")
                    skipped_count += 1
                    continue

                try:
                    # Convert dictionary to KBSchema object
                    kb_schema = KBSchema(**kb_schema_dict)

                    # Use DynamicEndpointGenerator to generate endpoints
                    generator = DynamicEndpointGenerator(kb_name=kb_name, kb_schema=kb_schema)
                    dynamic_router = generator.generate_all_endpoints()

                    # Register to FastAPI application
                    self.app.include_router(dynamic_router)

                    # Log bound endpoint information
                    rules_count = len(kb_schema.match_rules or [])
                    logger.info(f"✅ Knowledge base '{kb_name}' {rules_count} retrieval endpoints bound")
                    bound_count += 1

                except Exception as e:
                    logger.error(f"❌ Failed to bind knowledge base '{kb_name}' endpoints: {e}")
                    continue

            logger.info(f"✅ Endpoint processing complete: {bound_count} bound, {unbound_count} unbound, {skipped_count} skipped")

        except Exception as e:
            logger.error(f"❌ Error occurred while restoring registered endpoints: {e}")

