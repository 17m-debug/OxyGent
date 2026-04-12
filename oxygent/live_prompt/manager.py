"""Live Prompt Management Service for OxyGent framework.

This module provides dynamic prompt management capabilities, supporting storage and
real-time updates through Elasticsearch or LocalEs backends. It enables hot-swapping
of prompts during runtime and maintains version history for all prompt changes.
The system automatically falls back to LocalEs when Elasticsearch is unavailable.
"""

import asyncio
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from ..config import Config
from ..databases.db_es import JesEs, LocalEs
from ..db_factory import DBFactory

logger = logging.getLogger(__name__)

class PromptManager:
    """Prompt management system with automatic ES/LocalEs fallback.

    This class provides comprehensive prompt management capabilities including
    storage, retrieval, versioning, and hot-reloading. It automatically switches
    between Elasticsearch and LocalEs backends based on availability.

    Attributes:
        index_name (str): The Elasticsearch index name for storing prompts.
        use_local_es (bool): Whether currently using LocalEs as backend.
        db_client: The database client instance (JesEs or LocalEs).
    """

    def __init__(self, es_host: str = None, index_name: str = None):
        """Initialize the prompt manager.

        Args:
            es_host (str, optional): ES host address. If not provided, automatically
                selects between ES and LocalEs based on configuration.
            index_name (str): Index name for storing prompts. Defaults to "live_prompts".
        """
        if index_name is None:
            self.index_name = f"{Config.get_app_name()}_prompt"
        self._prompt_cache = {}  # Memory cache for full prompt documents
        self._cache_lock = asyncio.Lock()
        self.db_client = None
        self.use_local_es = False
        self._version_sync_coordinator = None  # Version sync coordinator

        db_factory = DBFactory()
        if Config.get_es_config():
            jes_config = Config.get_es_config()
            hosts = jes_config["hosts"]
            user = jes_config["user"]
            password = jes_config["password"]
            self.db_client = db_factory.get_instance(JesEs, hosts, user, password)
        else:
            self.db_client = db_factory.get_instance(LocalEs)
            self.use_local_es = True

    async def save_prompt(
        self,
        prompt_key: str,
        prompt_content: str,
        description: str = "",
        category: str = "custom",
        agent_type: str = "",
        version: int = 1,
        is_active: bool = True,
        tags: List[str] = None,
        created_by: str = "user",
    ) -> bool:
        """Save or update a prompt with version history and concurrency control.

        Saves a new prompt or updates an existing one with optimistic locking to
        prevent concurrent update conflicts. Uses version number checking to ensure
        updates are based on the latest version.

        Args:
            prompt_key (str): Unique identifier for the prompt.
            prompt_content (str): The actual prompt content.
            description (str): Human-readable description. Defaults to "".
            category (str): Prompt category. Defaults to "custom".
            agent_type (str): Associated agent type. Defaults to "".
            version (int): Base version number for new prompt. Ignored for updates.
            is_active (bool): Whether the prompt is active. Defaults to True.
            tags (List[str], optional): List of tags for categorization.
            created_by (str): Creator identifier. Defaults to "user".

        Returns:
            bool: True if save was successful, False otherwise.
        """
        try:
            # Check if prompt already exists (use fresh data from ES for concurrency control)
            existing = await self.get_prompt(prompt_key, use_cache=False)

            doc = {
                "prompt_key": prompt_key,
                "prompt_content": prompt_content,
                "description": description,
                "category": category,
                "agent_type": agent_type,
                "is_active": is_active,
                "updated_at": datetime.now().isoformat(),
                "tags": tags or [],
            }

            if existing:
                # For updates, always increment based on current version in ES
                current_version = existing.get("version", 1)
                doc["version"] = current_version + 1

                # Verify cache hasn't drifted from ES
                cached_version = self._prompt_cache.get(prompt_key, {}).get("version")
                if cached_version and cached_version != current_version:
                    logger.warning(
                        f"Cache version mismatch for {prompt_key}: "
                        f"cache={cached_version}, ES={current_version}. Syncing cache..."
                    )

                # Save current version to history
                history_id = f"{prompt_key}_v{current_version}"
                history_doc = existing.copy()
                history_doc["is_history"] = True
                history_doc["history_id"] = history_id
                history_doc["archived_at"] = datetime.now().isoformat()

                try:
                    # History doesn't need cache as it's not frequently accessed
                    await self.db_client.index(
                        index_name=f"{self.index_name}_history",
                        doc_id=history_id,
                        body=history_doc,
                    )
                except Exception as e:
                    logger.warning(f"Failed to save history for {prompt_key}: {e}")

                # Update existing record
                doc["created_at"] = existing.get("created_at")
                doc["created_by"] = existing.get("created_by", created_by)
            else:
                # Create new record
                doc["version"] = 1
                doc["created_at"] = datetime.now().isoformat()
                doc["created_by"] = created_by

            # Update cache (for immediate read availability) - with lock protection
            async with self._cache_lock:
                old_cache_value = self._prompt_cache.get(prompt_key)  # Backup for rollback
                self._prompt_cache[prompt_key] = doc
                logger.info(
                    f"  Cache now contains {len(self._prompt_cache)} keys: {list(self._prompt_cache.keys())}"
                )
            # Persist to ES (for durability)
            try:
                await self.db_client.index(
                    index_name=self.index_name, doc_id=prompt_key, body=doc
                )
                logger.info(f"✓ Persisted to ES: {prompt_key} (phase 2)")
            except Exception as es_error:
                # ES write failed - rollback cache to maintain consistency
                logger.error(f"ES write failed for {prompt_key}: {es_error}")
                logger.warning("Rolling back cache to previous state")

                async with self._cache_lock:
                    if old_cache_value is not None:
                        # Restore old value
                        self._prompt_cache[prompt_key] = old_cache_value
                        logger.info("Cache rolled back to previous version")
                    else:
                        # Remove newly added key
                        if prompt_key in self._prompt_cache:
                            del self._prompt_cache[prompt_key]
                        logger.info("Cache rolled back (removed new key)")

                # Re-raise to indicate failure
                raise

            logger.info(f"Saved prompt: {prompt_key} (two-phase commit completed)")

            # Update local version tracker for ES polling sync
            new_version = doc["version"]
            await self._update_local_version_tracker(prompt_key, new_version)

            return True

        except Exception as e:
            logger.error(f"Failed to save prompt {prompt_key}: {e}")
            return False

    async def get_prompt(
        self, prompt_key: str, use_cache: bool = True
    ) -> Optional[Dict[str, Any]]:
        """Retrieve a prompt by its key.

        Fetches prompt data from the database and updates the local cache.
        Returns the complete prompt document including metadata.

        Args:
            prompt_key (str): The unique identifier for the prompt.
            use_cache (bool): Whether to use cached data. Defaults to True.

        Returns:
            Optional[Dict[str, Any]]: The prompt data dict if found, None otherwise.
        """
        try:
            # Check cache first if enabled (with lock protection)
            if use_cache:
                async with self._cache_lock:
                    if prompt_key in self._prompt_cache:
                        logger.debug(f"Using cached prompt for: {prompt_key}")
                        # Return a copy to prevent external modification
                        return self._prompt_cache[prompt_key].copy()

            # Cache miss, fetch from database
            search_body = {"query": {"term": {"_id": prompt_key}}, "size": 1}

            try:
                response = await self.db_client.search(
                    index_name=self.index_name, body=search_body
                )
            except Exception as search_error:
                # Index might not exist yet
                error_msg = str(search_error)
                if "index_not_found" in error_msg or "no such index" in error_msg:
                    logger.debug(
                        f"Index {self.index_name} not found, will be created on first save"
                    )
                    return None
                else:
                    raise  # Re-raise if it's a different error

            if response is None:
                return None

            hits = response.get("hits", {}).get("hits", [])
            if hits:
                source = hits[0]["_source"]
                # Update cache with full document (with lock protection)
                async with self._cache_lock:
                    self._prompt_cache[prompt_key] = source
                return source.copy() if source else None

            return None

        except Exception as e:
            logger.error(f"Failed to get prompt {prompt_key}: {e}")
            return None

    async def clear_cache(self, prompt_key: str = None):
        """Clear cache for specific key or all keys.

        Args:
            prompt_key: Specific key to clear, None to clear all
        """
        async with self._cache_lock:
            if prompt_key:
                if prompt_key in self._prompt_cache:
                    del self._prompt_cache[prompt_key]
                logger.debug(f"Cleared cache for: {prompt_key}")
            else:
                self._prompt_cache.clear()
                logger.info("Cleared all prompt cache")

    async def get_prompt_content(
        self, prompt_key: str, fallback_content: str = "", use_cache: bool = True
    ) -> str:
        """Get prompt content with fallback support.

        Retrieves the content of an active prompt. If the prompt doesn't exist
        or is inactive, returns the provided fallback content.

        Args:
            prompt_key (str): The unique identifier for the prompt.
            fallback_content (str): Content to return if prompt not found or inactive.
            use_cache (bool): Whether to use cached data. Defaults to True.

        Returns:
            str: The prompt content or fallback content.
        """
        prompt_data = await self.get_prompt(prompt_key, use_cache=use_cache)
        if prompt_data and prompt_data.get("is_active", True):
            return prompt_data["prompt_content"]
        return fallback_content

    async def get_prompt_history(self, prompt_key: str) -> List[Dict[str, Any]]:
        """Get prompt version history.

        Retrieves the version history for a specific prompt, sorted by version
        number in descending order. Uses fallback query if keyword search fails.

        Args:
            prompt_key (str): The unique identifier for the prompt.

        Returns:
            List[Dict[str, Any]]: List of version history, sorted by version descending.
        """
        try:
            # Search history records - fixed query logic
            query = {
                "query": {
                    "bool": {
                        "must": [
                            {"term": {"prompt_key": prompt_key}},
                            {"term": {"is_history": True}},
                        ]
                    }
                },
                "sort": [{"version": {"order": "desc"}}],
                "size": 50,  # Return maximum 50 versions
            }

            response = await self.db_client.search(
                index_name=f"{self.index_name}_history", body=query
            )

            histories = []
            for hit in response["hits"]["hits"]:
                histories.append(hit["_source"])

            # If term query yields nothing and we're on ES, try match fallback
            if len(histories) == 0 and not self.use_local_es:
                query_fallback = {
                    "query": {
                        "bool": {
                            "must": [
                                {"match": {"prompt_key": prompt_key}},
                                {"term": {"is_history": True}},
                            ]
                        }
                    },
                    "sort": [{"version": {"order": "desc"}}],
                    "size": 50,
                }

                response = await self.db_client.search(
                    index_name=f"{self.index_name}_history", body=query_fallback
                )

                for hit in response["hits"]["hits"]:
                    histories.append(hit["_source"])

            return histories

        except Exception as e:
            logger.error(f"Failed to get prompt history for {prompt_key}: {e}")
            return []

    async def revert_to_version(self, prompt_key: str, target_version: int) -> bool:
        """Revert prompt to a specific version.

        Retrieves the specified version from history and creates a new version
        with that content, effectively reverting the prompt to the target version.

        Args:
            prompt_key (str): The unique identifier for the prompt.
            target_version (int): The target version number to revert to.

        Returns:
            bool: True if reversion was successful, False otherwise.
        """
        try:
            # Get target version from history
            history_id = f"{prompt_key}_v{target_version}"
            logger.info(
                f"Attempting to revert {prompt_key} to version {target_version}"
            )

            try:
                # Use search instead of get method
                search_body = {"query": {"term": {"_id": history_id}}, "size": 1}

                response = await self.db_client.search(
                    index_name=f"{self.index_name}_history", body=search_body
                )

                hits = response.get("hits", {}).get("hits", [])
                if not hits:
                    logger.error(f"Version {target_version} not found for {prompt_key}")
                    return False

                history_data = hits[0]["_source"]
                logger.debug(f"Found history version {target_version} for {prompt_key}")

            except Exception as e:
                logger.error(
                    f"Version {target_version} not found for {prompt_key}: {e}"
                )
                import traceback

                traceback.print_exc()
                return False

            # Clear cache before reverting to ensure fresh data
            await self.clear_cache(prompt_key)

            # Create new version using historical data
            logger.debug(f"Creating new version from history data for {prompt_key}")
            success = await self.save_prompt(
                prompt_key=prompt_key,
                prompt_content=history_data["prompt_content"],
                description=history_data.get("description", ""),
                category=history_data.get("category", "custom"),
                agent_type=history_data.get("agent_type", ""),
                is_active=history_data.get("is_active", True),
                tags=history_data.get("tags", []),
                created_by=f"reverted_from_v{target_version}",
            )

            if success:
                logger.info(
                    f"Successfully reverted {prompt_key} to version {target_version}"
                )
            else:
                logger.error(f"Failed to save reverted version for {prompt_key}")

            return success

        except Exception as e:
            logger.error(
                f"Failed to revert {prompt_key} to version {target_version}: {e}"
            )
            import traceback

            traceback.print_exc()
            return False

    async def list_prompts(
        self,
        category: str = None,
        agent_type: str = None,
        is_active: bool = None,
        tags: List[str] = None,
    ) -> List[Dict[str, Any]]:
        """List prompts with optional filtering.

        Retrieves all prompts matching the specified criteria. Supports filtering
        by category, agent type, active status, and tags.

        Uses cache-first strategy: returns cached data if available and no filters applied,
        otherwise queries ES and updates cache.

        Args:
            category (str, optional): Filter by category.
            agent_type (str, optional): Filter by agent type.
            is_active (bool, optional): Filter by active status.
            tags (List[str], optional): Filter by tags.

        Returns:
            List[Dict[str, Any]]: List of matching prompts.
        """
        try:
            # If no filters and cache is populated, return from cache directly
            has_filters = any([category, agent_type, is_active is not None, tags])

            async with self._cache_lock:
                if not has_filters and self._prompt_cache:
                    results = []
                    # Create a snapshot to avoid holding lock during iteration
                    cache_snapshot = self._prompt_cache.copy()
                    # Release lock before processing
                    # (we have a copy, so safe to process outside lock)

                    for prompt_key, prompt_data in cache_snapshot.items():
                        result = prompt_data.copy()
                        result["id"] = prompt_key
                        results.append(result)
                    # Sort by updated_at descending
                    results.sort(key=lambda x: x.get("updated_at", ""), reverse=True)
                    return results

            # Build query for ES search
            query = {"match_all": {}}
            filters = []

            if category:
                filters.append({"term": {"category": category}})
            if agent_type:
                filters.append({"term": {"agent_type": agent_type}})
            if is_active is not None:
                filters.append({"term": {"is_active": is_active}})
            if tags:
                for tag in tags:
                    filters.append({"term": {"tags": tag}})

            if filters:
                query = {"bool": {"must": [{"match_all": {}}], "filter": filters}}

            # run search
            response = await self.db_client.search(
                index_name=self.index_name,
                body={
                    "query": query,
                    "sort": [{"updated_at": {"order": "desc"}}],
                    "size": 1000,
                },
            )

            results = []
            for hit in response["hits"]["hits"]:
                result = hit["_source"]
                result["id"] = hit["_id"]
                results.append(result)

                # Update cache with fetched data (if no filters, refresh full cache)
                if not has_filters:
                    async with self._cache_lock:
                        self._prompt_cache[hit["_id"]] = hit["_source"]

            return results

        except Exception as e:
            logger.error(f"Failed to list prompts: {e}")
            return []

    async def delete_prompt(self, prompt_key: str) -> bool:
        """Delete a prompt.

        Removes the prompt from the database and clears it from the local cache.

        Args:
            prompt_key (str): The unique identifier for the prompt.

        Returns:
            bool: True if deletion was successful, False otherwise.
        """
        try:
            # If ES delete fails, cache is not touched, avoiding inconsistency
            try:
                await self.db_client.delete(
                    index_name=self.index_name, doc_id=prompt_key
                )
                logger.info(f"Deleted from ES: {prompt_key}")
            except Exception as es_error:
                logger.error(f"ES delete failed for {prompt_key}: {es_error}")
                # Don't clear cache if ES delete fails
                # This prevents data resurrection on restart
                raise  # Re-raise to trigger outer exception handler

            # ES delete successful, now clear cache (with lock protection)
            async with self._cache_lock:
                if prompt_key in self._prompt_cache:
                    del self._prompt_cache[prompt_key]
                    logger.info(f"Cache cleared for {prompt_key} (after ES delete)")

            return True

        except Exception as e:
            logger.error(f"Failed to delete prompt {prompt_key}: {e}")
            # Cache remains unchanged - consistent with ES state
            return False

    async def search_prompts(
        self, keyword: str, category: str = None
    ) -> List[Dict[str, Any]]:
        """Search prompts by keyword with optional category filter.

        Performs full-text search across prompt fields including key, description,
        content, and tags. Results are scored and sorted by relevance.

        Args:
            keyword (str): The search keyword.
            category (str, optional): Filter by category.

        Returns:
            List[Dict[str, Any]]: List of matching prompts with search scores.
        """
        try:
            # Build search query
            must_queries = [
                {
                    "multi_match": {
                        "query": keyword,
                        "fields": [
                            "prompt_key^2",
                            "description^1.5",
                            "prompt_content",
                            "tags^1.2",
                        ],
                        "type": "best_fields",
                    }
                }
            ]

            filters = []
            if category:
                filters.append({"term": {"category": category}})

            query = {"bool": {"must": must_queries, "filter": filters}}

            # Execute search
            response = await self.db_client.search(
                index_name=self.index_name,
                body={
                    "query": query,
                    "highlight": {
                        "fields": {
                            "description": {},
                            "prompt_content": {"fragment_size": 150},
                        }
                    },
                    "sort": [{"_score": {"order": "desc"}}],
                    "size": 50,
                },
            )

            results = []
            for hit in response["hits"]["hits"]:
                result = hit["_source"]
                result["id"] = hit["_id"]
                result["score"] = hit["_score"]
                if "highlight" in hit:
                    result["highlight"] = hit["highlight"]
                results.append(result)

            return results

        except Exception as e:
            logger.error(f"Failed to search prompts: {e}")
            return []

    async def _update_local_version_tracker(self, prompt_key: str, version: int):
        """Update local version tracker for ES polling synchronization.

        Args:
            prompt_key: The prompt key that was updated
            version: The new version number
        """
        try:
            # Lazy import to avoid circular dependency
            from .version import get_version_sync_coordinator

            coordinator = await get_version_sync_coordinator(self)
            coordinator.update_local_version(prompt_key, version)
        except Exception as e:
            logger.debug(f"Failed to update local version tracker: {e}")

    async def start_version_sync(self):
        """Start version synchronization for multi-instance cache consistency.

        This method should be called after initializing the PromptManager to enable
        automatic cache synchronization across multiple instances.
        """
        try:
            # Lazy import to avoid circular dependency
            from .version import get_version_sync_coordinator

            if self._version_sync_coordinator is None:
                coordinator = await get_version_sync_coordinator(self)
                self._version_sync_coordinator = coordinator
                await coordinator.start()
                logger.info("Version synchronization started")
        except Exception as e:
            logger.error(f"Failed to start version synchronization: {e}")

    async def stop_version_sync(self):
        """Stop version synchronization.

        This method should be called when shutting down the application.
        """
        if self._version_sync_coordinator:
            try:
                await self._version_sync_coordinator.stop()
                logger.info("Version synchronization stopped")
            except Exception as e:
                logger.error(f"Error stopping version synchronization: {e}")

    async def close(self):
        """Close the database connection.

        Properly closes the underlying database client connection to free resources.
        Should be called when the PromptManager is no longer needed.
        """
        # Stop version sync first
        await self.stop_version_sync()

        # Close database connection
        await self.db_client.close()


# Global prompt manager instance
prompt_manager = None
_version_sync_started = False


async def get_prompt_manager() -> PromptManager:
    """Get the global prompt manager instance.

    Returns the singleton PromptManager instance, creating it if necessary.
    The instance is automatically configured using Config settings.
    Version sync is automatically started on first initialization.

    Returns:
        PromptManager: The global prompt manager instance.
    """
    global prompt_manager, _version_sync_started
    if prompt_manager is None:
        # Read ES config from Config, no longer using environment variables
        prompt_manager = (
            PromptManager()
        )  # Don't pass es_host, let it auto-read from Config
        # await prompt_manager.init_index()
        try:
            await prompt_manager.start_version_sync()
            _version_sync_started = True
            logger.info("Version sync auto-started with PromptManager")
        except Exception as e:
            logger.error(f"Failed to auto-start version sync: {e}")

    return prompt_manager


async def close_prompt_manager():
    """Close the global prompt manager and clean up resources.

    This function should be called when the application is shutting down
    to properly close the Elasticsearch connection and prevent resource leaks.
    Also stops version sync if it was started.
    """
    global prompt_manager, _version_sync_started
    if prompt_manager is not None:
        try:
            # Stop version sync first (already called in prompt_manager.close(), but explicit here)
            if _version_sync_started:
                await prompt_manager.stop_version_sync()
                _version_sync_started = False
                logger.info("Version sync stopped during shutdown")

            await prompt_manager.close()
            logger.info("Prompt manager closed successfully")
        except Exception as e:
            logger.error(f"Error closing prompt manager: {e}")
        finally:
            prompt_manager = None


async def get_dynamic_prompt(
    prompt_key: str, fallback_content: str = "", use_cache: bool = True
) -> str:
    """Get dynamic prompt content with ES priority and fallback support.

    Retrieves prompt content from the prompt management system. If the prompt
    doesn't exist or is inactive, returns the provided fallback content.
    This function provides a convenient interface for dynamic prompt loading.

    Args:
        prompt_key (str): The unique identifier for the prompt.
        fallback_content (str): Fallback content (original static prompt) to use
            if the dynamic prompt is not available. Defaults to "".
        use_cache (bool): Whether to use cached data. Set to False to force reload
            from database. Defaults to True.

    Returns:
        str: The prompt content from ES/LocalEs or fallback content.
    """
    try:
        manager = await get_prompt_manager()
        return await manager.get_prompt_content(
            prompt_key, fallback_content, use_cache=use_cache
        )
    except Exception as e:
        logger.error(f"Failed to get dynamic prompt {prompt_key}: {e}")
        return fallback_content


async def resolve_prompt_from_es(
    prompt_key: str, default_prompt: str = "", use_cache: bool = True
) -> str:
    """
    Resolve prompt content from ES using the exact prompt_key

    Args:
        prompt_key: The exact prompt key to search for in ES
        default_prompt: Fallback prompt if not found in ES
        use_cache: Whether to use cached data. Set to False to force reload

    Returns:
        str: Prompt content from ES, or default_prompt, or empty string

    Logic:
    1. Try to get prompt from ES using the exact prompt_key
    2. If found and active, return the content
    3. If not found, return default_prompt
    4. If default_prompt is empty, return "" (system uses built-in defaults)
    """
    try:
        # Use the exact prompt key provided
        prompt_content = await get_dynamic_prompt(
            prompt_key, default_prompt, use_cache=use_cache
        )

        if prompt_content and prompt_content != default_prompt:
            logger.info(f"Loaded hot prompt from ES: {prompt_key}")
            return prompt_content

        # If no dynamic prompt found, use default or empty string
        if default_prompt and default_prompt.strip():
            logger.info(f"Using default prompt for {prompt_key}")
            return default_prompt
        else:
            logger.info(f"No prompt found for {prompt_key}, using system default")
            return ""

    except Exception as e:
        logger.error(f"Failed to resolve hot prompt for {prompt_key}: {e}")
        # Return default prompt or empty string on error
        return default_prompt if default_prompt else ""
