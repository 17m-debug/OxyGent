"""
Dynamic endpoint generator.

Dynamically generates retrieval endpoints based on match_rules in KBSchema.

Rule constraints:
1. Each retrieval rule can only have one advanced search strategy (es_text or vearch_vector) as the main query
2. Precise search strategies (precise) can appear multiple times for filtering conditions

Query template design:
- Pre-build query template structure when strategy is determined
- Only fill parameter values at runtime, avoiding repeated DSL construction
- Unified query interface supporting multiple search engines (ES, Vearch, etc.)
"""
import datetime
import uuid
from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any, Type

import pandas as pd
from fastapi import APIRouter, HTTPException, Body
from fastapi.routing import APIRoute
from loguru import logger
from pydantic import Field, create_model, BaseModel
from pydantic.fields import FieldInfo

from core.config import settings
from core.interface.endpoint_show import get_field_type
from core.storer.doc_manager.es_kb_base_manager import ElasticsearchKbBaseManager
from core.storer.doc_manager.es_kb_file_manager import ElasticsearchKbFileManager
from core.storer.doc_manager.knowledge_index import (
    KBSchema, MatchRule, MatchPolicy,
    PreciseMatchPolicy, check_kb_schema, VearchVectorMatchPolicy
)
from core.storer.doc_manager.schema_utils import convert_dataframe_types_by_schema
from core.storer.vector_manager.vearch_manager import VearchManager


class SearchResult(BaseModel):
    """Generic search result"""
    items: List[dict]
    total: int
    took_ms: float


# ============================================================================
# Query Template Abstract Base Class
# ============================================================================

class BaseQueryTemplate(ABC):
    """Query template abstract base class

    Provides a unified query interface supporting multiple search engines (ES, Vearch, etc.)
    Subclasses must implement the execute method
    """

    def __init__(
        self,
        advanced_search_fields: List[str],
        precise_filter_fields: List[str],
        output_fields: List[str]
    ):
        """
        Initialize query template

        Args:
            advanced_search_fields: List of advanced search fields
            precise_filter_fields: List of precise filter fields
            output_fields: List of output fields
        """
        self.advanced_search_fields = advanced_search_fields
        self.precise_filter_fields = precise_filter_fields
        self.output_fields = output_fields

    @abstractmethod
    async def execute(
        self,
        request_data: Dict[str, Any],
        top_k: int,
        **kwargs
    ) -> SearchResult:
        """
        Execute query (abstract method, must be implemented by subclasses)

        Args:
            request_data: Request parameter dictionary
            top_k: Number of results to return
            **kwargs: Other parameters (e.g., kb_name, etc.)

        Returns:
            SearchResult: Search results
        """
        pass


class ESQueryTemplate(BaseQueryTemplate):
    """ES query template

    Encapsulates Elasticsearch full-text search logic
    """

    def __init__(
        self,
        advanced_search_fields: List[str],
        precise_filter_fields: List[str],
        output_fields: List[str],
        es_client
    ):
        """
        Initialize ES query template

        Args:
            advanced_search_fields: List of advanced search fields
            precise_filter_fields: List of precise filter fields
            output_fields: List of output fields
            es_client: Elasticsearch client instance
        """
        super().__init__(
            advanced_search_fields=advanced_search_fields,
            precise_filter_fields=precise_filter_fields,
            output_fields=output_fields
        )
        self.es_client = es_client

    def build_query(self, request_data: Dict[str, Any], top_k: int) -> Dict[str, Any]:
        """
        Build ES query DSL

        Args:
            request_data: Request parameter dictionary
            top_k: Number of results to return

        Returns:
            ES query DSL
        """
        # Build basic query structure
        query = {
            "size": top_k,
            "_source": self.output_fields,
            "query": {
                "bool": {}
            }
        }

        bool_query = query["query"]["bool"]

        # Build must clause (advanced search)
        must_clause = self._build_must_clause(request_data)
        if must_clause:
            bool_query["must"] = [must_clause]

        # Build filter clause (precise filtering)
        filter_clauses = self._build_filter_clauses(request_data)
        if filter_clauses:
            bool_query["filter"] = filter_clauses

        # If bool query is empty, use match_all
        if not bool_query:
            query["query"] = {"match_all": {}}

        return query

    def _build_must_clause(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Build must clause (advanced search)

        Args:
            request_data: Request parameter dictionary

        Returns:
            must clause dictionary
        """
        # Only one field, get value directly
        value = request_data.get(self.advanced_search_fields[0])

        if value is None or value == "":
            raise ValueError("Advanced search strategy field cannot be empty when executing query strategy")

        # Use match query
        return {
            "match": {
                self.advanced_search_fields[0]: str(value)
            }
        }

    def _build_filter_clauses(self, request_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Build filter clause (precise filtering)

        Args:
            request_data: Request parameter dictionary

        Returns:
            filter clause list
        """
        clauses = []

        for field in self.precise_filter_fields:
            filter_field_name = f"filter_{field}"
            value = request_data.get(filter_field_name)

            if value is None:
                continue

            # Use term query uniformly (single-value exact match)
            clauses.append({
                "term": {
                    field: str(value)
                }
            })

        return clauses

    async def execute(
        self,
        request_data: Dict[str, Any],
        top_k: int,
        kb_name: str,
        **kwargs
    ) -> SearchResult:
        """
        Execute ES query

        Args:
            request_data: Request parameter dictionary
            top_k: Number of results to return
            kb_name: Knowledge base name
            **kwargs: Other parameters

        Returns:
            SearchResult: Search results
        """
        # Build query
        es_query = self.build_query(request_data, top_k)

        # Use json.dumps in log output to ensure Chinese characters display correctly (instead of Unicode escape)
        import json
        logger.debug(f"ES Query: {json.dumps(es_query, ensure_ascii=False, indent=2)}")

        # Execute query
        # Elasticsearch Python client automatically handles Unicode strings
        # Ensure all strings in the query dictionary passed to ES are Unicode strings
        response = self.es_client.search(
            index=kb_name,
            body=es_query
        )

        # Parse results
        hits = response.get("hits", {})
        total = hits.get("total", {}).get("value", 0)
        took_ms = response.get("took", 0)

        items = []
        for hit in hits.get("hits", []):
            # _source is already limited by _source field in query DSL, only contains fields specified by output_fields
            item = hit.get("_source", {}).copy()
            # Add ES relevance score
            item["_score"] = hit.get("_score", 0.0)
            items.append(item)

        logger.info(f"✅ ES search completed, returned {len(items)} results, total {total}")

        return SearchResult(
            items=items,
            total=total,
            took_ms=took_ms
        )


class VearchQueryTemplate(BaseQueryTemplate):
    """Vearch query template

    Encapsulates Vearch vector search logic
    """

    def __init__(
        self,
        advanced_search_fields: List[str],
        precise_filter_fields: List[str],
        output_fields: List[str],
        vearch_client,
        embedding_model
    ):
        """
        Initialize Vearch query template

        Args:
            advanced_search_fields: List of advanced search fields
            precise_filter_fields: List of precise filter fields
            output_fields: List of output fields
            vearch_client: Vearch client instance
            embedding_model: Embedding model instance
        """
        super().__init__(
            advanced_search_fields=advanced_search_fields,
            precise_filter_fields=precise_filter_fields,
            output_fields=output_fields
        )
        self.vearch_client = vearch_client
        self.embedding_model = embedding_model

    def build_filter(self, request_data: Dict[str, Any]) -> Optional:
        """
        Build Vearch filter conditions

        Args:
            request_data: Request parameter dictionary

        Returns:
            Vearch Filter object or None
        """
        from vearch.filter import Condition, FieldValue, Filter

        conditions = []

        for field in self.precise_filter_fields:
            filter_field_name = f"filter_{field}"
            value = request_data.get(filter_field_name)

            if value is None:
                continue

            if isinstance(value, (int, float)):
                field_condition = Condition(operator="=", fv=FieldValue(field=field, value=value))
            elif isinstance(value, str):
                field_condition = Condition(operator="IN", fv=FieldValue(field=field, value=[value]))
            else:
                raise ValueError("Value type passed to query interface does not meet requirements, only string, integer, float are supported.")

            # Add exact match condition for current field
            conditions.append(field_condition)

        # If there are filter conditions, create Filter object
        if conditions:
            return Filter(operator="AND", conditions=conditions)

        return None

    async def execute(
        self,
        request_data: Dict[str, Any],
        top_k: int,
        kb_name: str,
        **kwargs
    ) -> SearchResult:
        """
        Execute Vearch vector search

        Args:
            request_data: Request parameter dictionary
            top_k: Number of results to return
            kb_name: Knowledge base name
            **kwargs: Other parameters

        Returns:
            SearchResult: Search results
        """
        import numpy as np
        from vearch.utils import VectorInfo

        # Get value of advanced search field
        query_text = request_data.get(self.advanced_search_fields[0])

        if not query_text or query_text == "":
            raise ValueError("Advanced search strategy field cannot be empty when executing query strategy")

        logger.info(f"📝 Query text: {query_text}")

        # Use embedding model to convert text to vector
        try:
            query_vector = self.embedding_model.get_text_embedding(query_text)
            logger.debug(f"✅ Embedding generated successfully, vector dimension: {len(query_vector)}")
        except Exception as e:
            logger.error(f"❌ Embedding generation failed: {e}")
            raise HTTPException(status_code=500, detail=f"Embedding service exception: {str(e)}")

        # Vector normalization
        query_vector_norm = np.array(query_vector)
        query_vector_norm = query_vector_norm / np.linalg.norm(query_vector_norm)

        # Build VectorInfo object
        # Use dynamically calculated vector field name: field_name + "_vector"
        # For example: if advanced_search_fields = ["name"], then vector_field_name = "name_vector"
        vector_field_name = f"{self.advanced_search_fields[0]}_vector"
        vector_info = VectorInfo(
            field_name=vector_field_name,
            feature=query_vector_norm.tolist()
        )

        # Build filter conditions
        vearch_filter = self.build_filter(request_data)

        # Execute Vearch vector search
        try:
            search_result = self.vearch_client.search(
                database_name=settings.vearch_config.db_name,
                space_name=kb_name,
                vector_infos=[vector_info],
                fields=self.output_fields,
                limit=top_k,
                filter=vearch_filter
            )

            if not search_result.is_success():
                logger.error(f"Vearch search failed: {search_result.msg}")
                raise HTTPException(status_code=500, detail=f"Vearch retrieval failed: {search_result.msg}")

            # Parse results
            items = []
            total = 0
            if search_result.documents and len(search_result.documents) > 0:
                total = len(search_result.documents[0])
                for hit in search_result.documents[0]:
                    item = {}
                    for field in self.output_fields:
                        item[field] = hit.get(field, "")
                    # Add Vearch similarity score
                    item["_score"] = hit.get("_score", 0.0)
                    items.append(item)

            logger.info(f"✅ Vearch search completed, returned {len(items)} results, total {total}")

            return SearchResult(
                items=items,
                total=total,
                took_ms=0  # Vearch does not return took information
            )

        except Exception as e:
            logger.error(f"Vearch retrieval exception: {e}")
            raise HTTPException(status_code=500, detail=f"Vearch retrieval service exception: {str(e)}")


# ============================================================================
# Dynamic Endpoint Generator
# ============================================================================

class DynamicEndpointGenerator:
    """Dynamically generate and register retrieval endpoints"""

    def __init__(self, kb_name: str, kb_schema: KBSchema):
        self.kb_name = kb_name
        self.kb_schema = kb_schema
        self.router = APIRouter(prefix=f"/kb/{kb_name}", tags=[f"KB:{kb_name}"])
        # Use passed es_client or get from settings
        self.es_client = settings.es_client
        # Get vearch_client
        self.vearch_client = settings.vearch_client
        # Get embedding model
        self.embedding_model = settings.embedding_model
        # Build field type mapping
        self.field_types = {f.field_name: f.field_type for f in self.kb_schema.fields}
        # Build field description mapping
        self.field_descriptions = {f.field_name: f.field_desc for f in self.kb_schema.fields}
        
        # Query knowledge base description
        self.kb_description = None
        try:
            kb_base_client = ElasticsearchKbBaseManager(self.es_client)
            kb_info_list = kb_base_client.kb_info_search_name(kb_name=kb_name)
            if kb_info_list and len(kb_info_list) > 0:
                kb_info = kb_info_list[0]
                kb_desc = kb_info.get("kb_description", "").strip()
                if kb_desc:
                    self.kb_description = kb_desc
                    logger.info(f"Knowledge base '{kb_name}' description loaded: {kb_desc[:50]}...")
        except Exception as e:
            logger.warning(f"Failed to query knowledge base '{kb_name}' description: {e}, will use default description generation logic")

    def generate_all_endpoints(self) -> APIRouter:
        """Generate corresponding endpoint for each match_rule"""
        # 1. Generate retrieval interfaces based on match_rules
        for idx, match_rule in enumerate(self.kb_schema.match_rules):
            self._generate_search_endpoint(idx, match_rule)
        
        # 2. Automatically add deposit interface (data write interface)
        self._generate_deposit_endpoint()
        
        # 3. Automatically add list_bank interface (list all data in knowledge base)
        @self.router.get(
            "/list_banks",
            summary="List all bank interface information",
            description=f"Return all interface information in bank {self.kb_name}",
            tags=[f"KB:{self.kb_name}"]
        )
        async def list_banks():
            """List all interface information in bank
            
            Returns:
                retrieve and deposit interface information in bank
            """
            from typing import cast
            bank_apis = []

            for route in self.router.routes:
                if route.endpoint.__name__ == "list_banks":
                    continue

                route = cast(APIRoute, route)
                description = route.description
                input_schema = {"type": "object", "properties": {}, "required": []}

                # Extract type from route.path (full path format: /kb/{kb_name}/{type}/...)
                # Example: /kb/new_structure_1/search/rule_0 -> type is "search" (3rd position, index 3)
                # Example: /kb/new_structure_1/deposit -> type is "deposit" (3rd position, index 3)
                path_parts = route.path.split("/")
                # path_parts: ['', 'kb', '{kb_name}', '{type}', ...]
                # We want the 3rd position (index 3) which is the type
                endpoint_type = path_parts[3] if len(path_parts) > 3 else "unknown"
                
                # Convert "search" to "retrieve", keep others unchanged
                if endpoint_type == "search":
                    endpoint_type = "retrieve"

                # Only process routes with body_field (POST/PUT requests, etc.)
                if hasattr(route, "body_field") and route.body_field is not None:
                    request_type = route.body_field.type_

                    for field_name, field_type_info in request_type.model_fields.items():
                        field_type_info = cast(FieldInfo, field_type_info)

                        field_type = get_field_type(field_type_info)

                        if field_type is int:
                            t = "integer"
                        elif field_type is float:
                            t = "float"
                        else:
                            t = "string"

                        input_schema["properties"][field_name] = {
                            "type": t,
                            "description": field_type_info.description or "",
                        }

                        if field_type_info.is_required():
                            input_schema["required"].append(field_name)

                bank_apis.append(
                    {
                        "name": route.endpoint.__name__,
                        "endpoint": route.path,
                        "type": endpoint_type,
                        "methods": route.methods,
                        "description": description,
                        "inputSchema": input_schema,
                    }
                )
            return bank_apis

        return self.router

    def _generate_search_endpoint(self, rule_idx: int, match_rule: MatchRule):
        """Generate corresponding endpoint for a single match_rule"""
        # 1. Analyze strategies in match_rule, separate advanced search and precise search
        advanced_policy, precise_policies = self._separate_policies(match_rule)

        # 2. Dynamically generate request model (distinguish advanced search fields and filter fields)
        request_model = self._create_request_model(advanced_policy, precise_policies)

        # 3. Dynamically generate response model
        response_model = self._create_response_model(match_rule)

        # 4. Generate query function
        search_func = self._create_search_function(
            rule_idx=rule_idx,
            match_rule=match_rule,
            advanced_policy=advanced_policy,
            precise_policies=precise_policies,
            request_model=request_model
        )

        # 5. Register to router
        endpoint_path = f"/search/rule_{rule_idx}"
        self.router.add_api_route(
            path=endpoint_path,
            endpoint=search_func,
            methods=["POST"],
            response_model=SearchResult,
            summary=f"Retrieval interface - Rule {rule_idx}",
            description=self._generate_description(advanced_policy, precise_policies, match_rule),
            tags=[f"KB:{self.kb_name}"],
            name=f"{self.kb_name}_search_rule_{rule_idx}",
            # Specify request body model to let FastAPI correctly parse request
            dependencies=[]  # Automatically inferred from function signature
        )

        logger.info(f"✅ Generated endpoint: POST {self.router.prefix}{endpoint_path}")

    def _separate_policies(self, match_rule: MatchRule) -> tuple[MatchPolicy, list[PreciseMatchPolicy]]:
        """Separate advanced search strategies and precise search strategies

        Returns:
            tuple: (advanced search strategy, list of precise search strategies)
        """
        advanced_policy = None
        precise_policies = []

        for policy in match_rule.match_policies:
            if policy.mode in ["es_text", "vearch_vector"]:
                advanced_policy = policy
            elif policy.mode == "precise":
                precise_policies.append(policy)

        return advanced_policy, precise_policies

    def _create_request_model(
            self,
            advanced_policy: MatchPolicy,
            precise_policies: list[PreciseMatchPolicy]
    ):
        """Dynamically create request model

        Field naming rules:
        - Advanced search fields: keep original field name, e.g., name, description
        - Precise search fields: add filter_ prefix, e.g., filter_name, filter_category

        This allows fields with the same name to have different purposes, e.g.:
        - name: for full-text search
        - filter_name: for precise filtering
        """
        endpoint_input_fields = {}

        # 1. Add advanced search fields
        for field_name in advanced_policy.input_fields:
            python_type = self._get_python_type(field_name)
            # Use field description defined in Schema, if not available use default description
            field_desc = self.field_descriptions.get(field_name, field_name)
            endpoint_input_fields[field_name] = (
                python_type,
                Field(..., description=field_desc)
            )

        # 2. Add precise search fields (add filter_ prefix)
        for policy in precise_policies:
            for field_name in policy.input_fields:
                filter_field_name = f"filter_{field_name}"
                python_type = self._get_python_type(field_name)
                # Use field description defined in Schema, if not available use default description
                field_desc = self.field_descriptions.get(field_name, field_name)
                endpoint_input_fields[filter_field_name] = (
                    python_type,
                    Field(..., description=field_desc)
                )

        endpoint_input_fields["top_k"] = (int, Field(..., description="Query top k"))

        # Dynamically create model
        model_name = f"SearchRequest_{self.kb_name}_Rule{id(advanced_policy)}"
        return create_model(model_name, **endpoint_input_fields)

    def _create_response_model(self, match_rule: MatchRule):
        """Dynamically create response model"""
        endpoint_output_fields = {}

        for field_name in match_rule.output_fields:
            python_type = self._get_python_type(field_name)
            # Use field description defined in Schema, if not available use field name
            field_desc = self.field_descriptions.get(field_name, field_name)
            endpoint_output_fields[field_name] = (
                Optional[python_type],
                Field(None, description=field_desc)
            )

        model_name = f"SearchResponse_{self.kb_name}_Rule{id(match_rule)}"
        return create_model(model_name, **endpoint_output_fields)

    def _get_python_type(self, field_name: str) -> type:
        """Get corresponding Python type based on field name"""
        field_type = self.field_types.get(field_name, "string")
        if field_type == "integer":
            return int
        elif field_type == "float":
            return float
        else:
            return str

    def _create_deposit_request_model(self):
        """Dynamically create deposit request model based on schema fields
        
        Creates a Pydantic model that includes all fields defined in the knowledge base schema.
        The model supports both single item and list of items.
        
        Returns:
            A Pydantic model class for deposit data items
        """
        deposit_fields = {}
        
        # Add all fields from schema
        for field_info in self.kb_schema.fields:
            field_name = field_info.field_name
            python_type = self._get_python_type(field_name)
            field_desc = field_info.field_desc or field_name
            
            # All fields are required
            deposit_fields[field_name] = (
                python_type,
                Field(..., description=field_desc)
            )
        
        # Dynamically create model
        model_name = f"DepositDataItem_{self.kb_name}"
        return create_model(model_name, **deposit_fields)

    def _create_search_function(
            self,
            rule_idx: int,
            match_rule: MatchRule,
            advanced_policy: MatchPolicy,
            precise_policies: list[PreciseMatchPolicy],
            request_model: Type[BaseModel]
    ):
        """Create actual query function

        Pre-build query template when strategy is determined, only fill values at runtime
        """

        # Use closure to capture required parameters
        kb_name = self.kb_name

        # Collect field information
        advanced_search_fields = advanced_policy.input_fields
        precise_filter_fields = []
        for policy in precise_policies:
            precise_filter_fields.extend(policy.input_fields)
        output_fields = match_rule.output_fields

        # Determine search mode and create corresponding query template
        search_mode = advanced_policy.mode  # "es_text" or "vearch_vector"

        # Create corresponding query template based on search mode
        if search_mode == "es_text":
            # Create ES query template
            query_template = ESQueryTemplate(
                advanced_search_fields=advanced_search_fields,
                precise_filter_fields=precise_filter_fields,
                output_fields=output_fields,
                es_client=self.es_client
            )
        elif search_mode == "vearch_vector":
            # Create Vearch query template
            query_template = VearchQueryTemplate(
                advanced_search_fields=advanced_search_fields,
                precise_filter_fields=precise_filter_fields,
                output_fields=output_fields,
                vearch_client=self.vearch_client,
                embedding_model=self.embedding_model
            )
        else:
            raise ValueError(f"Unsupported search mode: {search_mode}")

        # Use type annotation: request_model is a dynamically created Pydantic model class
        # Capture request_model through closure to let FastAPI correctly identify the type
        async def search_endpoint(request: request_model) -> SearchResult:
            """Dynamically generated retrieval interface

            Only fill values in template at runtime, no need to rebuild query structure

            Args:
                request: Dynamically created request model instance (inherits from BaseModel)
            """
            logger.info(f"🔍 Executing search - KB: {kb_name}, Rule: {rule_idx}, Mode: {search_mode}")

            try:
                request_data = request.model_dump()  # Pydantic v2
            except Exception as e:
                logger.warning(f"Failed to get request parameters: {e}")

            try:
                # Extract parameters from request (already obtained above)
                top_k = request_data.get("top_k", 10)

                # Use template uniformly to execute query (both ES and Vearch use the same interface!)
                if search_mode == "es_text":
                    result = await query_template.execute(
                        request_data=request_data,
                        top_k=top_k,
                        kb_name=kb_name
                    )
                elif search_mode == "vearch_vector":
                    result = await query_template.execute(
                        request_data=request_data,
                        top_k=top_k,
                        kb_name=kb_name
                    )

                return result

            except Exception as e:
                logger.error(f"Search failed: {e}")
                raise HTTPException(status_code=500, detail=f"Search service exception: {str(e)}")

        # Set function name
        search_endpoint.__name__ = f"search_rule_{rule_idx}"

        return search_endpoint

    def _generate_description(
            self,
            advanced_policy: MatchPolicy,
            precise_policies: list[PreciseMatchPolicy],
            match_rule: MatchRule
    ) -> str:
        """Generate endpoint description
        
        Priority:
        1. If knowledge base has kb_description, use it directly
        2. Otherwise use default field detail generation logic
        """
        # If knowledge base description exists, use it preferentially
        if self.kb_description:
            return self.kb_description
        
        # Otherwise use default generation logic
        desc_parts = []

        # Advanced search description (with field descriptions)
        if advanced_policy:
            if advanced_policy.mode == "es_text":
                desc_parts.append(f"**Main Query Strategy**: ES full-text search")
                # Add field detailed descriptions
                field_details = []
                for field_name in advanced_policy.input_fields:
                    field_desc = self.field_descriptions.get(field_name, field_name)
                    field_details.append(f"{field_name} ({field_desc})")
                desc_parts.append(f"**Search Fields**: {', '.join(field_details)}")
            elif advanced_policy.mode == "vearch_vector":
                desc_parts.append(f"**Main Query Strategy**: Vearch vector search")
                # Add field detailed descriptions
                field_details = []
                for field_name in advanced_policy.input_fields:
                    field_desc = self.field_descriptions.get(field_name, field_name)
                    field_details.append(f"{field_name} ({field_desc})")
                desc_parts.append(f"**Search Fields**: {', '.join(field_details)}")

        # Filter condition description (with field descriptions)
        if precise_policies:
            filter_field_details = []
            for policy in precise_policies:
                for field_name in policy.input_fields:
                    field_desc = self.field_descriptions.get(field_name, field_name)
                    filter_field_details.append(f"{field_name} ({field_desc})")
            desc_parts.append(f"**Filter Conditions**: {', '.join(set(filter_field_details))} (exact match)")

        # Output fields (with field descriptions)
        output_field_details = []
        for field_name in match_rule.output_fields:
            field_desc = self.field_descriptions.get(field_name, field_name)
            output_field_details.append(f"{field_name} ({field_desc})")
        desc_parts.append(f"**Output Fields**: {', '.join(output_field_details)}")

        return "\n\n".join(desc_parts)

    def _generate_deposit_endpoint(self):
        """Generate deposit interface (data write interface)"""
        # Create dynamic deposit request model based on schema
        deposit_item_model = self._create_deposit_request_model()
        
        # Use closure to capture the model type
        kb_name = self.kb_name
        kb_schema = self.kb_schema
        
        @self.router.post(
            "/deposit",
            summary="Data write interface",
            description=f"Write data to knowledge base {self.kb_name}. Accepts a single data item matching the knowledge base schema.",
            tags=[f"KB:{self.kb_name}"]
        )
        async def deposit_data(
            data: deposit_item_model = Body(
                ..., 
                description=f"Data to write (single item). Fields: {', '.join([f.field_name for f in kb_schema.fields])}"
            )
        ):
            """Data write interface
            
            Functions:
            - Receive a single data item matching the knowledge base schema
            - Perform data type conversion and validation according to knowledge base schema
            - Write to Elasticsearch and Vearch (if vectorization is needed)
            
            Args:
                data: Single data item to write, matching the schema
                
            Returns:
                Write result information
            """
            # Initialize clients
            kb_base_client = ElasticsearchKbBaseManager(self.es_client)
            kb_file_client = ElasticsearchKbFileManager(self.es_client)
            
            # 1. Query knowledge base information, get kb_id and kb_type
            kb_info_list = kb_base_client.kb_info_search_name(kb_name=kb_name)
            if not kb_info_list or len(kb_info_list) == 0:
                raise HTTPException(
                    status_code=404,
                    detail=f"Knowledge base '{kb_name}' does not exist"
                )
            
            kb_info = kb_info_list[0]
            kb_id = kb_info.get("kb_id")
            kb_type = kb_info.get("kb_type", "structured")
            
            if not kb_id:
                raise HTTPException(
                    status_code=400,
                    detail=f"Failed to get kb_id for knowledge base '{kb_name}'"
                )
            
            # 2. Validate schema
            if not check_kb_schema(kb_schema):
                raise HTTPException(
                    status_code=400,
                    detail="Current knowledge base kb schema validation failed, please check kb schema!"
                )
            
            # 3. Convert Pydantic model to dictionary
            data_dict = data.model_dump(exclude_none=False)
            
            # 4. Convert to DataFrame (wrap single dict in list for DataFrame creation)
            try:
                df = pd.DataFrame([data_dict])
            except Exception as e:
                logger.error(f"Failed to convert data to DataFrame: {e}")
                raise HTTPException(
                    status_code=400,
                    detail=f"Failed to convert data to DataFrame: {str(e)}"
                )
            
            # 6. Convert DataFrame column types according to schema
            df = convert_dataframe_types_by_schema(df, kb_schema)
            
            # 7. Add fixed columns
            df['kb_id'] = kb_id
            mock_file_id = f"mock_file_{kb_id}"
            df['ori_file_id'] = mock_file_id
            df['chunk_id'] = [str(uuid.uuid4().hex[:16]) for _ in range(len(df))]
            
            # 8. Write to ES
            es_add_result = kb_file_client.kb_add_df(kb_name=kb_name, df=df)
            if not es_add_result:
                logger.error("add file data into es failed")
                raise HTTPException(
                    status_code=500,
                    detail="add file data into es failed"
                )
            logger.info(f"add file data into es success")
            
            # 9. If vectorization is needed, write to Vearch
            if kb_schema.match_rules:
                # Collect all vector matching strategies
                vec_matches = [
                    policy
                    for match_rule in kb_schema.match_rules
                    for policy in match_rule.match_policies
                    if isinstance(policy, VearchVectorMatchPolicy)
                ]
                
                if vec_matches:
                    vearch_manager = VearchManager(vearch_client=self.vearch_client)
                    
                    try:
                        for vec_match in vec_matches:
                            field_name = vec_match.input_fields[0]
                            
                            # Check if field exists
                            if field_name not in df.columns:
                                logger.warning(f"Field {field_name} does not exist in DataFrame, skipping vectorization")
                                continue
                            
                            # Check if field has data
                            if df[field_name].isnull().all():
                                logger.warning(f"Field {field_name} is all empty values, skipping vectorization")
                                continue
                            
                            texts = df[field_name].astype(str).tolist()
                            embeddings = self.embedding_model.get_text_embedding_batch(texts, show_progress=True)
                            
                            vector_field_name = f"{field_name}_vector"
                            df[vector_field_name] = embeddings
                            logger.info(f"Successfully generated vector field {vector_field_name} for field {field_name}")
                        
                        # Write to Vearch
                        vearch_add_result = vearch_manager.add_df(
                            database_name=settings.vearch_config.db_name,
                            space_name=kb_name,
                            df=df
                        )
                        
                        if not vearch_add_result:
                            logger.error("Failed to write vector data to Vearch")
                            raise HTTPException(
                                status_code=500,
                                detail="Failed to write vector data to Vearch, please check Vearch service status"
                            )
                        logger.info(f"Vector data successfully written to Vearch space: {kb_name}")
                    
                    except HTTPException:
                        raise
                    except Exception as e:
                        logger.error(f"Vectorization processing failed: {e}")
                        raise HTTPException(
                            status_code=500,
                            detail=f"Vectorization processing failed: {str(e)}"
                        )
            
            # 10. Update or create file information
            result = kb_file_client.get_kb_file_info(kb_id=kb_id, ori_file_id=mock_file_id)
            ori_file_info = result["_source"] if result else None
            current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            if not ori_file_info:
                # If current knowledge base doesn't have this file yet, write file information to kb_file
                mock_file_type = "md" if kb_type == "unstructured" else "csv"
                kb_file_item = {
                    "kb_id": kb_id,
                    "ori_file_id": mock_file_id,
                    "ori_file_type": mock_file_type,
                    "file_name": f"deposit_data.{mock_file_type}",
                    "file_path": f"/{mock_file_id}.{mock_file_type}",
                    "document_md5": "",
                    "file_store_mode": "",
                    "file_extra_info": {},
                    "language": "zh",
                    "create_time": current_time,
                    "update_time": current_time,
                }
                if not kb_file_client.kb_add_file(kb_file_item):
                    logger.error("add kb file info into es failed")
                    raise HTTPException(
                        status_code=500,
                        detail="add kb file info into es failed"
                    )
            else:
                ori_file_info["update_time"] = current_time
                if not kb_file_client.kb_update_file_info(ori_file_info):
                    logger.error("update kb file info into es failed")
                    raise HTTPException(
                        status_code=500,
                        detail="update kb file info into es failed"
                    )
            
            return {
                "code": 200,
                "msg": "success",
                "data": "Data inserted successfully"
            }
        
        logger.info(f"✅ Generated deposit interface: POST {self.router.prefix}/deposit")

