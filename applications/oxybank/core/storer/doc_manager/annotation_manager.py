"""Annotation data manager for Elasticsearch."""
from datetime import datetime
from typing import Any, Dict, List, Optional

from elasticsearch import Elasticsearch
from loguru import logger
from pydantic import BaseModel

from core.storer.doc_manager.es_index_manager import ElasticsearchIndexManager


class QADataModel(BaseModel):
    """QA data model"""

    # Unique identifiers
    data_id: str
    data_hash: str

    # QA content
    question: str
    answer: str

    # Source tracking
    source_trace_id: str
    source_request_id: str
    source_group_id: str

    # Call chain info
    caller: str
    callee: str
    caller_type: Optional[str] = None
    callee_type: Optional[str] = None

    # Data attributes
    data_type: str  # e2e/agent/llm/tool/custom
    priority: int  # 0-4, P0=end-to-end
    category: Optional[str] = None
    tags: List[str] = []

    # Status management
    status: str  # pending/annotated/approved/rejected/kb_pending/kb_ingested/kb_failed
    annotation: Dict[str, Any] = {}
    scores: Dict[str, Any] = {}
    reject_reason: Optional[str] = None

    # Knowledge base info
    kb_status: str = "pending"  # pending/ingested/failed
    kb_ingested_at: Optional[datetime] = None
    kb_error_message: Optional[str] = None
    kb_extra: Dict[str, Any] = {}

    # Batch info
    batch_id: Optional[str] = None

    # Timestamps
    created_at: datetime
    updated_at: datetime

    # Extra data
    extra: Dict[str, Any] = {}


class AnnotationManager:
    """Annotation data manager - handles ES index operations"""

    # QA data index mapping definition
    INDEX_MAPPING = {
        "properties": {
            "data_id": {"type": "keyword"},
            "data_hash": {"type": "keyword"},

            "question": {
                "type": "text",
                "fields": {
                    "keyword": {"type": "keyword", "ignore_above": 256}
                }
            },
            "answer": {
                "type": "text",
                "fields": {
                    "keyword": {"type": "keyword", "ignore_above": 256}
                }
            },

            "source_trace_id": {"type": "keyword"},
            "source_request_id": {"type": "keyword"},
            "source_group_id": {"type": "keyword"},

            "caller": {"type": "keyword"},
            "callee": {"type": "keyword"},
            "caller_type": {"type": "keyword"},
            "callee_type": {"type": "keyword"},

            "data_type": {"type": "keyword"},
            "priority": {"type": "integer"},
            "category": {"type": "keyword"},
            "tags": {"type": "keyword"},

            "status": {"type": "keyword"},
            "annotation": {"type": "object", "enabled": True},
            "scores": {"type": "object", "enabled": True},
            "reject_reason": {"type": "text"},

            "kb_status": {"type": "keyword"},
            "kb_ingested_at": {"type": "date", "format": "yyyy-MM-dd HH:mm:ss.SSSSSS||yyyy-MM-dd HH:mm:ss||epoch_millis"},
            "kb_error_message": {"type": "text"},
            "kb_extra": {"type": "object", "enabled": True},

            "batch_id": {"type": "keyword"},

            "created_at": {"type": "date", "format": "yyyy-MM-dd HH:mm:ss.SSSSSS||yyyy-MM-dd HH:mm:ss||epoch_millis"},
            "updated_at": {"type": "date", "format": "yyyy-MM-dd HH:mm:ss.SSSSSS||yyyy-MM-dd HH:mm:ss||epoch_millis"},

            "extra": {"type": "object", "enabled": True}
        }
    }

    # Index settings
    INDEX_SETTINGS = {
        "number_of_shards": 3,
        "number_of_replicas": 1,
        "refresh_interval": "1s"
    }

    def __init__(self, es_client: Elasticsearch, index_prefix: str = "qa_annotation"):
        """
        Initialize annotation data manager

        Args:
            es_client: Elasticsearch client
            index_prefix: Index name prefix
        """
        self.es_client = es_client
        self.index_name = f"{index_prefix}_data"
        self.index_manager = ElasticsearchIndexManager(es_client)

    def initialize(self) -> bool:
        """
        Initialize index (ensure index exists and structure is correct)

        Returns:
            bool: Success or not
        """
        logger.info(f"Initializing annotation data index: {self.index_name}")
        return self.index_manager.ensure_index(
            index_name=self.index_name,
            expected_mapping=self.INDEX_MAPPING,
            settings=self.INDEX_SETTINGS
        )

    def health_check(self) -> bool:
        """
        Health check

        Returns:
            bool: Whether index is healthy
        """
        return self.index_manager.index_exists(self.index_name)

    def delete_index(self) -> bool:
        """
        Delete index

        Returns:
            bool: Success or not
        """
        return self.index_manager.delete_index(self.index_name)

    # ============= CRUD operations =============

    async def get_by_id(self, data_id: str) -> Optional[Dict[str, Any]]:
        """
        Get data by ID

        Args:
            data_id: Data ID

        Returns:
            Data dict, None if not found
        """
        try:
            response = self.es_client.get(index=self.index_name, id=data_id)
            return response["_source"]
        except Exception as e:
            if "index_not_found" in str(e) or "not_found" in str(e):
                return None
            logger.error(f"Get data failed: {data_id}, error: {str(e)}")
            raise

    async def create(self, data: QADataModel) -> bool:
        """
        Create data

        Args:
            data: QA data model

        Returns:
            bool: Success or not
        """
        try:
            # Convert to dict
            data_dict = data.model_dump()

            # Handle datetime fields
            if isinstance(data_dict.get("created_at"), datetime):
                data_dict["created_at"] = data_dict["created_at"].strftime("%Y-%m-%d %H:%M:%S.%f")
            if isinstance(data_dict.get("updated_at"), datetime):
                data_dict["updated_at"] = data_dict["updated_at"].strftime("%Y-%m-%d %H:%M:%S.%f")
            if isinstance(data_dict.get("kb_ingested_at"), datetime):
                data_dict["kb_ingested_at"] = data_dict["kb_ingested_at"].strftime("%Y-%m-%d %H:%M:%S.%f")

            # Use data_id as document ID, force refresh to ensure immediate searchability
            self.es_client.index(
                index=self.index_name,
                id=data.data_id,
                body=data_dict,
                refresh=True  # Force index refresh to ensure data is immediately searchable
            )
            logger.debug(f"Data created successfully: {data.data_id}")
            return True
        except Exception as e:
            logger.error(f"Create data failed: {data.data_id}, error: {str(e)}")
            raise

    async def update(self, data_id: str, **kwargs) -> bool:
        """
        Update data

        Args:
            data_id: Data ID
            **kwargs: Fields to update

        Returns:
            bool: Success or not
        """
        try:
            # Update timestamp
            kwargs["updated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")

            self.es_client.update(
                index=self.index_name,
                id=data_id,
                body={"doc": kwargs},
                refresh=True  # Force index refresh to ensure data is immediately searchable
            )
            logger.debug(f"Data updated successfully: {data_id}")
            return True
        except Exception as e:
            logger.error(f"Update data failed: {data_id}, error: {str(e)}")
            raise

    async def delete(self, data_id: str) -> bool:
        """
        Delete data

        Args:
            data_id: Data ID

        Returns:
            bool: Success or not
        """
        try:
            self.es_client.delete(
                index=self.index_name,
                id=data_id,
                refresh=True  # Force index refresh to ensure deletion is immediately searchable
            )
            logger.debug(f"Data deleted successfully: {data_id}")
            return True
        except Exception as e:
            logger.error(f"Delete data failed: {data_id}, error: {str(e)}")
            raise

    async def exists_by_hash(self, data_hash: str) -> bool:
        """
        Check if hash exists (deduplication)

        Args:
            data_hash: Data hash

        Returns:
            bool: Whether exists
        """
        try:
            query = {
                "query": {
                    "term": {
                        "data_hash": data_hash
                    }
                },
                "size": 0  # No need to return results, only count
            }
            response = self.es_client.search(index=self.index_name, body=query)
            return response["hits"]["total"]["value"] > 0
        except Exception as e:
            logger.error(f"Check hash existence failed: {data_hash}, error: {str(e)}")
            return False

    # ============= Query operations =============

    async def list_query(
        self,
        filters: Optional[Dict[str, Any]] = None,
        pagination: Optional[Dict[str, Any]] = None,
        sorting: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        List query

        Args:
            filters: Filter conditions
            pagination: Pagination params {page, page_size}
            sorting: Sorting params [{field, order}]

        Returns:
            Query result {total, items}
        """
        try:
            # Build query
            query = {"query": {"bool": {"must": []}}}
            must = query["query"]["bool"]["must"]

            # Add filter conditions
            if filters:
                if "status" in filters:
                    must.append({"term": {"status": filters["status"]}})
                if "priority" in filters:
                    must.append({"term": {"priority": filters["priority"]}})
                if "data_type" in filters:
                    must.append({"term": {"data_type": filters["data_type"]}})
                if "caller" in filters:
                    must.append({
                        "wildcard": {
                            "caller": {
                                "value": f"*{filters['caller']}*"
                            }
                        }
                    })
                if "callee" in filters:
                    must.append({
                        "wildcard": {
                            "callee": {
                                "value": f"*{filters['callee']}*"
                            }
                        }
                    })
                if "category" in filters:
                    must.append({"term": {"category": filters["category"]}})
                if "tags" in filters:
                    must.append({"terms": {"tags": filters["tags"]}})

                # Date range filter
                if "created_after" in filters:
                    must.append({
                        "range": {
                            "created_at": {
                                "gte": filters["created_after"]
                            }
                        }
                    })
                if "created_before" in filters:
                    must.append({
                        "range": {
                            "created_at": {
                                "lte": filters["created_before"]
                            }
                        }
                    })

                # Full-text search (search in question and answer fields)
                if "search_text" in filters:
                    must.append({
                        "multi_match": {
                            "query": filters["search_text"],
                            "fields": ["question", "answer"],
                            "type": "best_fields"
                        }
                    })

                # Trace ID filter (wildcard for fuzzy match)
                if "trace_id" in filters:
                    must.append({
                        "wildcard": {
                            "source_trace_id": {
                                "value": f"*{filters['trace_id']}*"
                            }
                        }
                    })

                # Group ID filter (wildcard for fuzzy match)
                if "group_id" in filters:
                    must.append({
                        "wildcard": {
                            "source_group_id": {
                                "value": f"*{filters['group_id']}*"
                            }
                        }
                    })

            # Use match_all if no filter conditions
            if not must:
                query["query"] = {"match_all": {}}

            # Add sorting
            if sorting:
                query["sort"] = []
                for sort_item in sorting:
                    field = sort_item.get("field", "created_at")
                    order = sort_item.get("order", "desc")
                    query["sort"].append({field: {"order": order}})
            else:
                # Default sort by creation time descending
                query["sort"] = [{"created_at": {"order": "desc"}}]

            # Add pagination
            if pagination:
                page = pagination.get("page", 1)
                page_size = pagination.get("page_size", 10)
                query["from"] = (page - 1) * page_size
                query["size"] = page_size
            else:
                query["from"] = 0
                query["size"] = 10

            # Execute query
            response = self.es_client.search(index=self.index_name, body=query)

            # Extract results
            items = [hit["_source"] for hit in response["hits"]["hits"]]
            total = response["hits"]["total"]["value"]

            return {
                "total": total,
                "items": items
            }
        except Exception as e:
            logger.error(f"List query failed: {str(e)}")
            raise

    async def get_by_trace_id(self, trace_id: str) -> List[Dict[str, Any]]:
        """
        Get data by trace_id

        Args:
            trace_id: trace_id

        Returns:
            Data list
        """
        try:
            query = {
                "query": {
                    "term": {"source_trace_id": trace_id}
                },
                "sort": [{"created_at": {"order": "desc"}}],
                "size": 100
            }
            response = self.es_client.search(index=self.index_name, body=query)
            return [hit["_source"] for hit in response["hits"]["hits"]]
        except Exception as e:
            logger.error(f"Query by trace_id failed: {trace_id}, error: {str(e)}")
            raise

    async def get_by_group_id(self, group_id: str) -> List[Dict[str, Any]]:
        """
        Get data by group_id

        Args:
            group_id: group_id

        Returns:
            Data list
        """
        try:
            query = {
                "query": {
                    "term": {"source_group_id": group_id}
                },
                "sort": [{"created_at": {"order": "desc"}}],
                "size": 100
            }
            response = self.es_client.search(index=self.index_name, body=query)
            return [hit["_source"] for hit in response["hits"]["hits"]]
        except Exception as e:
            logger.error(f"Query by group_id failed: {group_id}, error: {str(e)}")
            raise

    async def get_groups_summary(self) -> List[Dict[str, Any]]:
        """
        Get groups summary

        Returns:
            Group statistics list
        """
        try:
            query = {
                "size": 0,
                "aggs": {
                    "group_by_source_group_id": {
                        "terms": {
                            "field": "source_group_id",
                            "size": 1000
                        },
                        "aggs": {
                            "by_status": {
                                "terms": {
                                    "field": "status",
                                    "size": 10
                                }
                            }
                        }
                    }
                }
            }

            response = self.es_client.search(index=self.index_name, body=query)

            # Extract aggregation results
            buckets = response["aggregations"]["group_by_source_group_id"]["buckets"]

            summary = []
            for bucket in buckets:
                group_id = bucket["key"]
                total_count = bucket["doc_count"]

                # Count status quantities
                status_counts = {}
                for status_bucket in bucket["by_status"]["buckets"]:
                    status_counts[status_bucket["key"]] = status_bucket["doc_count"]

                summary.append({
                    "group_id": group_id,
                    "total_count": total_count,
                    "status_counts": status_counts
                })

            return summary
        except Exception as e:
            logger.error(f"Get groups summary failed: {str(e)}")
            raise

    # ============= KB ingestion status update =============

    async def update_kb_status(
        self,
        data_id: str,
        kb_status: str,
        kb_ingested_at: Optional[datetime] = None,
        kb_error_message: Optional[str] = None,
        kb_extra: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Update KB ingestion status

        Updates both main status (status) and KB status (kb_status) to maintain state machine consistency.

        Args:
            data_id: Data ID
            kb_status: KB status (pending/ingested/failed)
            kb_ingested_at: Ingestion time
            kb_error_message: Error message
            kb_extra: Extra info

        Returns:
            bool: Success or not
        """
        # Update both main status and KB status to maintain state machine consistency
        update_data = {
            "status": f"kb_{kb_status}",  # Sync main status
            "kb_status": kb_status
        }

        if kb_ingested_at:
            update_data["kb_ingested_at"] = kb_ingested_at.strftime("%Y-%m-%d %H:%M:%S.%f")

        if kb_error_message:
            update_data["kb_error_message"] = kb_error_message

        if kb_extra:
            update_data["kb_extra"] = kb_extra

        return await self.update(data_id, **update_data)

    # ============= Batch operations =============

    async def bulk_create(self, data_list: List[QADataModel]) -> Dict[str, Any]:
        """
        Batch create data

        Args:
            data_list: QA data model list

        Returns:
            Batch operation result
        """
        try:
            from elasticsearch.helpers import bulk

            actions = []
            for data in data_list:
                data_dict = data.model_dump()

                # Handle datetime fields
                if isinstance(data_dict.get("created_at"), datetime):
                    data_dict["created_at"] = data_dict["created_at"].strftime("%Y-%m-%d %H:%M:%S.%f")
                if isinstance(data_dict.get("updated_at"), datetime):
                    data_dict["updated_at"] = data_dict["updated_at"].strftime("%Y-%m-%d %H:%M:%S.%f")
                if isinstance(data_dict.get("kb_ingested_at"), datetime):
                    data_dict["kb_ingested_at"] = data_dict["kb_ingested_at"].strftime("%Y-%m-%d %H:%M:%S.%f")

                action = {
                    "_index": self.index_name,
                    "_id": data.data_id,
                    "_source": data_dict
                }
                actions.append(action)

            # Execute batch operation
            success, failed = bulk(self.es_client, actions, raise_on_error=False)

            logger.info(f"Batch create completed: success={success}, failed={failed}")

            return {
                "success": success,
                "failed": failed
            }
        except Exception as e:
            logger.error(f"Batch create failed: {str(e)}")
            raise
