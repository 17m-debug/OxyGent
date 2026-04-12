"""Annotation Service - Core business logic layer"""
import hashlib
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional
import httpx

from loguru import logger

from core.config import AnnotationConfig, settings
from core.storer.doc_manager.annotation_manager import AnnotationManager, QADataModel
from core.model.annotation import (
    DepositRequest,
    DepositResponse,
    DepositBatchRequest,
    DepositBatchResponse,
    AnnotationUpdateRequest,
    ApprovalRequest,
    DataListQueryParams,
    QADataItem,
    OverallStatsResponse,
    TypeStatsItem,
    TypeStatsResponse,
)


class AnnotationService:
    """Annotation service - Core business logic

    Handles all annotation platform business logic:
    - Data deposit and deduplication
    - Data query and filtering
    - Annotation management
    - KB ingestion (method calls)
    - Statistics and analysis
    """

    def __init__(
        self,
        annotation_manager: AnnotationManager,
        config: AnnotationConfig
    ):
        """
        Initialize annotation service

        Args:
            annotation_manager: ES data manager
            config: Annotation platform config
        """
        self.annotation_manager = annotation_manager
        self.config = config

        # Current batch ID
        self._current_batch_id: Optional[str] = None

    # ============= Helper methods =============

    def _generate_data_id(self) -> str:
        """Generate data ID"""
        return str(uuid.uuid4())

    def _generate_batch_id(self) -> str:
        """Generate batch ID"""
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S%f")
        return hashlib.md5(timestamp.encode()).hexdigest()[:16]

    def _compute_data_hash(self, question: str, answer: str) -> str:
        """Compute data hash for deduplication"""
        content = f"{question.strip()}|||{answer.strip()}"
        return hashlib.md5(content.encode("utf-8")).hexdigest()

    def _build_es_query_from_filters(self, filters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Build Elasticsearch query from filter dict

        Args:
            filters: Filter dictionary

        Returns:
            Elasticsearch query
        """
        query = {"bool": {"must": []}}
        must = query["bool"]["must"]

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
        if "search_text" in filters:
            must.append({
                "multi_match": {
                    "query": filters["search_text"],
                    "fields": ["question", "answer"],
                    "type": "best_fields"
                }
            })
        if "trace_id" in filters:
            must.append({
                "wildcard": {
                    "source_trace_id": {
                        "value": f"*{filters['trace_id']}*"
                    }
                }
            })
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
            return {"match_all": {}}

        return query

    def _infer_data_type(
        self,
        caller: str,
        callee: str,
        caller_type: Optional[str] = None,
        callee_type: Optional[str] = None
    ) -> str:
        """
        Infer data type

        Rules:
        - If caller and callee are both empty → custom
        - If caller is empty, callee is not → tool
        - If caller is not empty, callee is empty → agent
        - If both have values → agent or llm (based on type fields)
        - If caller == "user" → e2e
        """
        if not caller and not callee:
            return self.config.default_data_type

        if caller == "user":
            return "e2e"

        if not caller and callee:
            return "tool"

        if caller and not callee:
            return "agent"

        # If both have values → agent or llm (based on type field)
        if callee_type == "llm" or caller_type == "llm":
            return "llm"

        return "agent"

    def _infer_priority(
        self,
        caller: str,
        data_type: str
    ) -> int:
        """
        Infer priority

        Rules:
        - e2e type → 0 (P0, highest priority)
        - If caller is empty → use configured default priority
        - Other types → set based on business rules
        """
        if data_type == "e2e":
            return 0

        if not caller:
            return self.config.default_priority

        # Priority for agent/llm/tool can be adjusted as needed
        return 4

    # ============= Data deposit features =============

    async def deposit_data(self, request: DepositRequest) -> DepositResponse:
        """
        Deposit single data

        Args:
            request: Deposit request

        Returns:
            Deposit response
        """
        try:
            # 1. Compute data hash
            data_hash = self._compute_data_hash(request.question, request.answer)

            # 2. Deduplication check (ES)
            is_duplicate = await self.annotation_manager.exists_by_hash(data_hash)
            if is_duplicate:
                existing = await self._get_existing_by_hash(data_hash)
                if existing:
                    logger.info(f"Duplicate found in ES: hash={data_hash[:16]}...")
                    return DepositResponse(
                        data_id=existing["data_id"],
                        data_hash=data_hash,
                        status="pending",
                        is_duplicate=True,
                        message="Data already exists (ES)"
                    )

            # 3. Infer data type and priority
            data_type = request.data_type or self._infer_data_type(
                caller=request.caller,
                callee=request.callee,
                caller_type=request.caller_type,
                callee_type=request.callee_type
            )

            priority = request.priority or self._infer_priority(
                caller=request.caller,
                data_type=data_type
            )

            # 4. Generate batch ID (if needed)
            if not self._current_batch_id:
                self._current_batch_id = self._generate_batch_id()

            # 5. Create QA data model
            now = datetime.now()
            qa_data = QADataModel(
                data_id=self._generate_data_id(),
                data_hash=data_hash,
                question=request.question,
                answer=request.answer,
                source_trace_id=request.source_trace_id or self._generate_data_id(),
                source_request_id=request.source_request_id or self._generate_data_id(),
                source_group_id=request.source_group_id or self._generate_batch_id(),
                caller=request.caller or "unknown",
                callee=request.callee or "unknown",
                caller_type=request.caller_type,
                callee_type=request.callee_type,
                data_type=data_type,
                priority=priority,
                category=request.category,
                tags=request.tags or [],
                status="pending",
                annotation={},
                scores={},
                batch_id=self._current_batch_id,
                created_at=now,
                updated_at=now,
                extra=request.extra or {}
            )

            # 6. Save to ES
            await self.annotation_manager.create(qa_data)

            logger.info(
                f"Data deposited: data_id={qa_data.data_id}, "
                f"trace_id={qa_data.source_trace_id}, "
                f"priority={qa_data.priority}, "
                f"type={qa_data.data_type}, "
                f"caller={qa_data.caller}, "
                f"callee={qa_data.callee}"
            )

            return DepositResponse(
                data_id=qa_data.data_id,
                data_hash=data_hash,
                status="pending",
                is_duplicate=False,
                message="Deposit successful"
            )

        except Exception as e:
            logger.error(f"Deposit failed: {str(e)}")
            raise

    async def deposit_batch(self, request: DepositBatchRequest) -> DepositBatchResponse:
        """
        Batch deposit data

        Args:
            request: Batch deposit request

        Returns:
            Batch deposit response
        """
        try:
            # Generate batch ID
            batch_id = request.batch_id or self._generate_batch_id()
            self._current_batch_id = batch_id

            results = []
            success_count = 0
            duplicate_count = 0
            failed_count = 0

            for deposit_request in request.data_list:
                try:
                    response = await self.deposit_data(deposit_request)
                    results.append(response)

                    if response.is_duplicate:
                        duplicate_count += 1
                    else:
                        success_count += 1

                except Exception as e:
                    logger.error(f"Batch deposit item failed: {str(e)}")
                    failed_count += 1
                    # Continue processing next item
                    continue

            logger.info(
                f"Batch deposit completed: batch_id={batch_id}, "
                f"total={len(request.data_list)}, "
                f"success={success_count}, "
                f"duplicate={duplicate_count}, "
                f"failed={failed_count}"
            )

            return DepositBatchResponse(
                batch_id=batch_id,
                total=len(request.data_list),
                success_count=success_count,
                duplicate_count=duplicate_count,
                failed_count=failed_count,
                results=results
            )

        except Exception as e:
            logger.error(f"Batch deposit failed: {str(e)}")
            raise

    async def _get_existing_by_hash(self, data_hash: str) -> Optional[Dict[str, Any]]:
        """Get existing data by hash"""
        try:
            # Query ES
            query = {
                "query": {
                    "term": {"data_hash": data_hash}
                },
                "size": 1
            }
            response = self.annotation_manager.es_client.search(
                index=self.annotation_manager.index_name,
                body=query
            )

            hits = response.get("hits", {}).get("hits", [])
            if hits:
                return hits[0]["_source"]
            return None

        except Exception as e:
            logger.error(f"Query existing data failed: {str(e)}")
            return None

    # ============= Data query features =============

    async def get_data_list(self, params: DataListQueryParams) -> Dict[str, Any]:
        """
        Get data list

        Args:
            params: Query parameters

        Returns:
            Data list
        """
        try:
            # Build filters
            filters = {}
            if params.status:
                filters["status"] = params.status
            if params.priority is not None:
                filters["priority"] = params.priority
            if params.data_type:
                filters["data_type"] = params.data_type
            if params.caller:
                filters["caller"] = params.caller
            if params.callee:
                filters["callee"] = params.callee
            if params.category:
                filters["category"] = params.category
            if params.tags:
                filters["tags"] = params.tags
            if params.created_after:
                filters["created_after"] = params.created_after
            if params.created_before:
                filters["created_before"] = params.created_before
            if params.search_text:
                filters["search_text"] = params.search_text
            if params.trace_id:
                filters["trace_id"] = params.trace_id
            if params.group_id:
                filters["group_id"] = params.group_id

            # Build pagination params
            pagination = {
                "page": params.page,
                "page_size": params.page_size
            }

            # Build sorting params - ensure default is created_at descending
            sorting = [{
                "field": params.sort_by or "created_at",
                "order": params.sort_order or "desc"
            }]

            # Execute query
            result = await self.annotation_manager.list_query(
                filters=filters,
                pagination=pagination,
                sorting=sorting
            )

            return result

        except Exception as e:
            logger.error(f"Data list query failed: {str(e)}")
            raise

    async def get_data_by_id(self, data_id: str) -> Optional[Dict[str, Any]]:
        """Get data by ID"""
        try:
            return await self.annotation_manager.get_by_id(data_id)
        except Exception as e:
            logger.error(f"Get data failed: {data_id}, error: {str(e)}")
            raise

    async def get_by_trace_id(self, trace_id: str) -> List[Dict[str, Any]]:
        """Get data by trace_id"""
        try:
            return await self.annotation_manager.get_by_trace_id(trace_id)
        except Exception as e:
            logger.error(f"Query by trace_id failed: {trace_id}, error: {str(e)}")
            raise

    async def get_by_group_id(self, group_id: str) -> List[Dict[str, Any]]:
        """Get data by group_id"""
        try:
            return await self.annotation_manager.get_by_group_id(group_id)
        except Exception as e:
            logger.error(f"Query by group_id failed: {group_id}, error: {str(e)}")
            raise

    async def get_groups_summary(self) -> List[Dict[str, Any]]:
        """Get groups summary"""
        try:
            return await self.annotation_manager.get_groups_summary()
        except Exception as e:
            logger.error(f"Get groups summary failed: {str(e)}")
            raise

    # ============= Annotation management features =============

    async def update_annotation(
        self,
        data_id: str,
        request: AnnotationUpdateRequest
    ) -> bool:
        """
        Update annotation

        Args:
            data_id: Data ID
            request: Annotation update request

        Returns:
            Success or not
        """
        try:
            # Build update data
            update_data = {}

            # Handle main annotation field (contains content, question, score, comment)
            if request.annotation is not None:
                update_data["annotation"] = request.annotation

            # Legacy fields for backward compatibility
            if request.category is not None:
                update_data["category"] = request.category

            if request.tags is not None:
                update_data["tags"] = request.tags

            if request.scores is not None:
                update_data["scores"] = request.scores

            # Merge legacy annotation fields if annotation field is not provided
            annotation_updates = {}
            if request.comment:
                annotation_updates["comment"] = request.comment
            if request.remark:
                annotation_updates["remark"] = request.remark
            if request.annotation_data:
                annotation_updates.update(request.annotation_data)

            # Only merge legacy annotation fields if main annotation field is not provided
            if annotation_updates and request.annotation is None:
                if "annotation" not in update_data:
                    update_data["annotation"] = {}
                update_data["annotation"].update(annotation_updates)

            # Status flow: pending -> annotated
            update_data["status"] = "annotated"

            # Execute update
            await self.annotation_manager.update(data_id, **update_data)

            logger.info(f"Annotation updated: data_id={data_id}")
            return True

        except Exception as e:
            logger.error(f"Annotation update failed: {data_id}, error: {str(e)}")
            raise

    async def approve_data(
        self,
        data_id: str,
        reason: Optional[str] = None
    ) -> bool:
        """
        Approve data

        Args:
            data_id: Data ID
            reason: Approval reason

        Returns:
            Success or not
        """
        try:
            # Get data
            data = await self.annotation_manager.get_by_id(data_id)
            if not data:
                raise ValueError(f"Data not found: {data_id}")

            # Status flow: annotated/approved -> approved
            update_data = {
                "status": "approved",
                "annotation.approval_reason": reason or "Approved",
                "annotation.approved_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
            }

            await self.annotation_manager.update(data_id, **update_data)

            logger.info(f"Data approved: data_id={data_id}, reason={reason}")

            # Auto ingest to KB if enabled
            if self.config.kb_auto_ingest:
                logger.info(f"Auto ingest to KB: data_id={data_id}")
                await self.ingest_to_kb(data_id)

            return True

        except Exception as e:
            logger.error(f"Approval failed: {data_id}, error: {str(e)}")
            raise

    async def reject_data(
        self,
        data_id: str,
        reason: str,
        reject_category: Optional[str] = None
    ) -> bool:
        """
        Reject data

        Args:
            data_id: Data ID
            reason: Rejection reason
            reject_category: Rejection category

        Returns:
            Success or not
        """
        try:
            # Status flow: annotated/approved -> rejected
            update_data = {
                "status": "rejected",
                "reject_reason": reason,
                "annotation.rejection_category": reject_category or "other",
                "annotation.rejected_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
            }

            await self.annotation_manager.update(data_id, **update_data)

            logger.info(f"Data rejected: data_id={data_id}, reason={reason}")
            return True

        except Exception as e:
            logger.error(f"Rejection failed: {data_id}, error: {str(e)}")
            raise

    # ============= KB ingestion features =============

    async def ingest_to_kb(
        self,
        data_id: str,
        score: Optional[float] = None,
        remark: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Ingest annotation data to knowledge base

        Calls KB HTTP API endpoint (knowledge_file.py:ingest_kb_data).
        Only passes structured fields defined in original implementation to ensure KB schema compatibility.

        KB structured fields (defined by AnnotationConfig):
        - question: Question/input content
        - answer: Answer/output content
        - caller: Caller
        - callee: Callee
        - score: Quality score (0-1)
        - remark: Note
        - source_trace_id: Original trace_id
        - source_request_id: Original request_id
        - data_type: Data type
        - priority: Priority
        - category: Data category

        Args:
            data_id: Data ID
            score: Quality score (0-1)
            remark: Note

        Returns:
            Ingest result
        """
        try:
            # Check if KB ingestion is enabled
            if not self.config.kb_enabled:
                raise ValueError("KB ingestion not enabled, check ANNOTATION_KB_ENABLED config")

            # Get data
            data = await self.annotation_manager.get_by_id(data_id)
            if not data:
                raise ValueError(f"Data not found: {data_id}")

            # Validate data status (only approved or annotated can be ingested)
            if data.get("status") not in ["approved", "annotated", "kb_failed"]:
                raise ValueError(f"Only approved, annotated or kb_failed data can be ingested to KB, current status: {data.get('status')}")

            # Build KB ingestion request (only include fields defined in original implementation)
            kb_data = {
                "question": data.get("question", ""),
                "answer": data.get("answer", ""),
                "caller": data.get("caller", ""),
                "callee": data.get("callee", ""),
                "score": score,
                "remark": remark,
                "source_trace_id": data.get("source_trace_id"),
                "source_request_id": data.get("source_request_id"),
                "data_type": data.get("data_type"),
                "priority": data.get("priority"),
                "category": data.get("category"),
            }

            # Remove None values
            kb_data = {k: v for k, v in kb_data.items() if v is not None}

            # Build API URL
            api_url = f"{settings.api_base_url}/api/v1/kb_base/{self.config.kb_id}/ingest_data"

            # Use retry mechanism with HTTP API call
            last_error = None
            for attempt in range(self.config.kb_retry_times):
                try:
                    # Call HTTP API asynchronously
                    async with httpx.AsyncClient(timeout=60.0) as client:
                        response = await client.post(
                            api_url,
                            json=kb_data
                        )

                    if response.status_code == 200:
                        result = response.json()
                        logger.info(
                            f"KB ingestion successful via HTTP API: data_id={data_id}, "
                            f"kb_id={self.config.kb_id}, "
                            f"response={result}"
                        )

                        # Update KB ingestion status in local ES
                        await self.annotation_manager.update_kb_status(
                            data_id=data_id,
                            kb_status="ingested",
                            kb_ingested_at=datetime.now(),
                            kb_extra={
                                "ingest_method": "http_api",
                                "api_response": result,
                                "quality_score": score,
                                "remark": remark
                            }
                        )

                        return {
                            "success": True,
                            "kb_id": self.config.kb_id,
                            "data_id": data_id,
                            "message": "KB ingestion successful",
                            "kb_result": result
                        }
                    else:
                        error_text = response.text
                        last_error = f"API error {response.status_code}: {error_text}"
                        logger.warning(f"KB ingestion API returned {response.status_code}: {error_text}")

                except httpx.HTTPError as e:
                    last_error = str(e)
                    logger.warning(f"KB ingestion HTTP request failed (attempt {attempt + 1}/{self.config.kb_retry_times}): {e}")

                # Don't wait on last attempt
                if attempt < self.config.kb_retry_times - 1:
                    import asyncio
                    await asyncio.sleep(self.config.kb_retry_interval)

            # All attempts failed
            error_msg = f"KB ingestion failed, retried {self.config.kb_retry_times} times: {last_error}"
            logger.error(error_msg)

            # Update failed status
            await self.annotation_manager.update_kb_status(
                data_id=data_id,
                kb_status="failed",
                kb_error_message=error_msg
            )

            raise Exception(error_msg)

        except ValueError as e:
            logger.error(f"KB ingestion failed (invalid params): {data_id}, error: {str(e)}")

            # Update failed status
            await self.annotation_manager.update_kb_status(
                data_id=data_id,
                kb_status="failed",
                kb_error_message=str(e)
            )

            raise

        except Exception as e:
            logger.error(f"KB ingestion failed: {data_id}, error: {str(e)}")

            # Update failed status
            await self.annotation_manager.update_kb_status(
                data_id=data_id,
                kb_status="failed",
                kb_error_message=str(e)
            )

            raise

    # ============= Statistics features =============

    async def get_overall_stats(self, filters: Optional[DataListQueryParams] = None) -> OverallStatsResponse:
        """
        Get overall statistics

        Args:
            filters: Optional query filters to apply before aggregation

        Returns:
            Overall statistics response
        """
        try:
            # Build base query from filters if provided
            base_query = None
            if filters:
                # Build filters dict
                filter_dict = {}
                if filters.status:
                    filter_dict["status"] = filters.status
                if filters.priority is not None:
                    filter_dict["priority"] = filters.priority
                if filters.data_type:
                    filter_dict["data_type"] = filters.data_type
                if filters.caller:
                    filter_dict["caller"] = filters.caller
                if filters.callee:
                    filter_dict["callee"] = filters.callee
                if filters.category:
                    filter_dict["category"] = filters.category
                if filters.tags:
                    filter_dict["tags"] = filters.tags
                if filters.created_after:
                    filter_dict["created_after"] = filters.created_after
                if filters.created_before:
                    filter_dict["created_before"] = filters.created_before
                if filters.search_text:
                    filter_dict["search_text"] = filters.search_text
                if filters.trace_id:
                    filter_dict["trace_id"] = filters.trace_id
                if filters.group_id:
                    filter_dict["group_id"] = filters.group_id

                # Build ES query from filters
                if filter_dict:
                    base_query = self._build_es_query_from_filters(filter_dict)

            # Build aggregation query
            query = {
                "size": 0,
                "aggs": {
                    # Aggregate by status
                    "by_status": {
                        "terms": {
                            "field": "status",
                            "size": 20
                        }
                    },
                    # Aggregate by priority
                    "by_priority": {
                        "terms": {
                            "field": "priority",
                            "size": 5
                        }
                    },
                    # Aggregate by type
                    "by_type": {
                        "terms": {
                            "field": "data_type",
                            "size": 10
                        }
                    },
                    # Aggregate by KB status
                    "by_kb_status": {
                        "terms": {
                            "field": "kb_status",
                            "size": 10
                        }
                    }
                }
            }

            # Add base query filter if provided
            if base_query:
                query["query"] = base_query

            # Execute query
            response = self.annotation_manager.es_client.search(
                index=self.annotation_manager.index_name,
                body=query
            )

            # Extract aggregation results (ES client returns 'aggregations', not 'aggs')
            aggregations = response.get("aggregations", response.get("aggs", {}))

            # Status stats
            status_counts = {}
            for bucket in aggregations.get("by_status", {}).get("buckets", []):
                status_counts[bucket["key"]] = bucket["doc_count"]

            # Priority stats (convert key to string for JSON compatibility)
            priority_stats = {}
            for bucket in aggregations.get("by_priority", {}).get("buckets", []):
                priority_stats[str(bucket["key"])] = bucket["doc_count"]

            # Type stats
            type_stats = {}
            for bucket in aggregations.get("by_type", {}).get("buckets", []):
                type_stats[bucket["key"]] = bucket["doc_count"]

            # KB status stats
            kb_status_counts = {}
            for bucket in aggregations.get("by_kb_status", {}).get("buckets", []):
                kb_status_counts[bucket["key"]] = bucket["doc_count"]

            # Get total count
            total_count = response["hits"]["total"]["value"]

            # Count KB-related statuses based on status field (not kb_status to avoid double counting)
            kb_ingested_count = status_counts.get("kb_ingested", 0)
            kb_failed_count = status_counts.get("kb_failed", 0)

            return OverallStatsResponse(
                total_count=total_count,
                pending_count=status_counts.get("pending", 0),
                annotated_count=status_counts.get("annotated", 0),
                approved_count=status_counts.get("approved", 0),
                rejected_count=status_counts.get("rejected", 0),
                kb_ingested_count=kb_ingested_count,
                kb_pending_count=kb_status_counts.get("pending", 0),
                kb_failed_count=kb_failed_count,
                priority_stats=priority_stats,
                type_stats=type_stats
            )

        except Exception as e:
            logger.error(f"Get overall stats failed: {str(e)}")
            raise

    async def get_pending_p0_stats(self, limit: int = 100) -> Dict[str, Any]:
        """Get pending P0 data"""
        try:
            # P0 data definition: priority=0 and status=pending
            query = {
                "query": {
                    "bool": {
                        "must": [
                            {"term": {"priority": 0}},
                            {"term": {"status": "pending"}}
                        ]
                    }
                },
                "sort": [{"created_at": {"order": "asc"}}],
                "size": limit
            }

            # Execute query
            response = self.annotation_manager.es_client.search(
                index=self.annotation_manager.index_name,
                body=query
            )

            # Extract results
            items = [hit["_source"] for hit in response["hits"]["hits"]]
            total = response["hits"]["total"]["value"]

            return {
                "total": total,
                "items": items[:limit]
            }

        except Exception as e:
            logger.error(f"Get P0 stats failed: {str(e)}")
            raise

    async def get_stats_by_type(self, filters: Optional[DataListQueryParams] = None) -> TypeStatsResponse:
        """
        Get statistics by type

        Args:
            filters: Optional query filters to apply before aggregation

        Returns:
            Type statistics response
        """
        try:
            # Build base query from filters if provided
            base_query = None
            if filters:
                # Build filters dict
                filter_dict = {}
                if filters.status:
                    filter_dict["status"] = filters.status
                if filters.priority is not None:
                    filter_dict["priority"] = filters.priority
                if filters.data_type:
                    filter_dict["data_type"] = filters.data_type
                if filters.caller:
                    filter_dict["caller"] = filters.caller
                if filters.callee:
                    filter_dict["callee"] = filters.callee
                if filters.category:
                    filter_dict["category"] = filters.category
                if filters.tags:
                    filter_dict["tags"] = filters.tags
                if filters.created_after:
                    filter_dict["created_after"] = filters.created_after
                if filters.created_before:
                    filter_dict["created_before"] = filters.created_before
                if filters.search_text:
                    filter_dict["search_text"] = filters.search_text
                if filters.trace_id:
                    filter_dict["trace_id"] = filters.trace_id
                if filters.group_id:
                    filter_dict["group_id"] = filters.group_id

                # Build ES query from filters
                if filter_dict:
                    base_query = self._build_es_query_from_filters(filter_dict)

            # Build aggregation query
            query = {
                "size": 0,
                "aggs": {
                    "by_type": {
                        "terms": {
                            "field": "data_type",
                            "size": 20
                        },
                        "aggs": {
                            # Sub-aggregation by status
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

            # Add base query filter if provided
            if base_query:
                query["query"] = base_query

            # Execute query
            response = self.annotation_manager.es_client.search(
                index=self.annotation_manager.index_name,
                body=query
            )

            # Extract aggregation results (ES client returns 'aggregations', not 'aggs')
            aggregations = response.get("aggregations", response.get("aggs", {}))
            buckets = aggregations["by_type"]["buckets"]

            items = []
            for bucket in buckets:
                data_type = bucket["key"]
                total_count = bucket["doc_count"]

                # Extract sub-aggregation (by status)
                status_counts = {}
                for status_bucket in bucket["by_status"]["buckets"]:
                    status_counts[status_bucket["key"]] = status_bucket["doc_count"]

                items.append(TypeStatsItem(
                    data_type=data_type,
                    total_count=total_count,
                    pending_count=status_counts.get("pending", 0),
                    approved_count=status_counts.get("approved", 0),
                    rejected_count=status_counts.get("rejected", 0)
                ))

            return TypeStatsResponse(items=items)

        except Exception as e:
            logger.error(f"Get type stats failed: {str(e)}")
            raise
