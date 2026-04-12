#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Vearch Space Manager
Use official pyvearch SDK to check, validate, delete and create spaces
"""
from typing import Dict, Any, List

import numpy as np
import pandas as pd
from loguru import logger
from vearch.config import Config
from vearch.core.vearch import Vearch
from vearch.filter import Filter, Condition, FieldValue
from vearch.schema.field import Field
from vearch.schema.index import FlatIndex, ScalarIndex
from vearch.schema.space import SpaceSchema
from vearch.utils import DataType, MetricType, VectorInfo

from utils.url_util import ensure_url_protocol


class VearchManager:
    """Vearch Space Manager"""

    def __init__(
            self,
            host: str = None,
            token: str = None,
            vearch_client: Vearch = None
    ):
        """
        Initialize Vearch client

        Args:
            host: Vearch host address
            token: Authentication token
            vearch_client: Vearch client instance
        """
        # If client exists, directly pass client to create manager
        if vearch_client is not None:
            self.client = vearch_client
        elif host is not None:
            # Build configuration
            config = Config(host=ensure_url_protocol(host), token=token)

            # Initialize client
            try:
                self.client = Vearch(config)
            except Exception as e:
                raise ConnectionError(f"Failed to connect to Vearch server: {str(e)}")
        else:
            raise ValueError("Must provide either vearch_client or host parameter")

        try:
            # Test connection
            self._check_connection()
            logger.info(f"Successfully connected to Vearch server {host}")
        except Exception as e:
            raise ConnectionError(f"Unable to connect to Vearch server: {str(e)}")


    def _check_connection(self):
        """Check Vearch connection"""
        try:
            # Try to get database list to verify connection
            self.client.list_databases()
        except Exception as e:
            raise ConnectionError(f"Unable to connect to Vearch server: {str(e)}")

    def database_exists(self, db_name: str) -> bool:
        """
        Check if database exists

        Args:
            db_name: Database name

        Returns:
            bool: Whether database exists
        """
        try:
            databases = self.client.list_databases()
            for db in databases:
                if db.name == db_name:
                    return True
            return False
        except Exception as e:
            logger.error(f"Error checking if database exists: {str(e)}")
            return False

    def space_exists(self, db_name: str, space_name: str) -> bool:
        """
        Check if space exists

        Args:
            db_name: Database name
            space_name: Space name

        Returns:
            bool: Whether space exists
        """
        try:
            spaces = self.client.list_spaces(db_name)
            for space in spaces:
                if space.name == space_name:
                    return True
            return False
        except Exception as e:
            logger.error(f"Error checking if space exists: {str(e)}")
            return False

    def get_space_structure(self, db_name: str, space_name: str) -> Dict[str, Any]:
        """
        Get space structure

        Args:
            db_name: Database name
            space_name: Space name

        Returns:
            Dict: Space structure
        """
        try:
            # Use more reliable method to get space structure
            space_info = self.client.database(db_name).space(space_name).describe()
            return space_info.to_dict()
        except Exception as e:
            try:
                # Backup method
                space_schema = self.client.database(db_name).space(space_name).exist()[1]
                return space_schema.to_dict()
            except Exception as e2:
                raise ValueError(f"Failed to get space structure: {str(e)}, backup method also failed: {str(e2)}")

    def validate_space_structure(self, db_name: str, space_name: str, expected_structure: Dict[str, Any]) -> bool:
        """
        Validate if space structure matches expectations

        Args:
            db_name: Database name
            space_name: Space name
            expected_structure: Expected space structure

        Returns:
            bool: Whether structure matches
        """
        if not self.space_exists(db_name, space_name):
            logger.warning(f"Space {space_name} does not exist")
            return False

        try:
            current_structure = self.get_space_structure(db_name, space_name)
            logger.info(f"Current space {space_name} structure: {current_structure}")

            # Extract current fields - handle different return formats
            current_fields = []
            if "fields" in current_structure:
                current_fields = current_structure["fields"]
            elif "field" in current_structure:
                current_fields = current_structure["field"]
            else:
                logger.error(f"Unable to extract field information from space structure: {current_structure}")
                return False

            expected_fields = expected_structure.get("fields", [])

            # Create field name to field mapping
            current_field_map = {}
            for field in current_fields:
                if isinstance(field, dict):
                    field_name = field.get("name") or field.get("field_name") or field.get("field")
                    if field_name:
                        current_field_map[field_name] = field
                else:
                    # Handle field object
                    if hasattr(field, 'name'):
                        current_field_map[field.name] = field

            expected_field_map = {field["name"]: field for field in expected_fields}

            logger.info(f"Current fields: {list(current_field_map.keys())}")
            logger.info(f"Expected fields: {list(expected_field_map.keys())}")

            # Check if all expected fields exist in current space
            missing_fields = []
            for field_name, expected_field in expected_field_map.items():
                if field_name not in current_field_map:
                    missing_fields.append(field_name)
                    logger.error(f"Field {field_name} does not exist in current space")

            if missing_fields:
                logger.error(f"Missing fields: {missing_fields}")
                return False

            # Check field types and attributes
            for field_name, expected_field in expected_field_map.items():
                current_field = current_field_map[field_name]

                # Get field type information
                current_type = None
                if isinstance(current_field, dict):
                    current_type = current_field.get("type") or current_field.get("data_type") or current_field.get(
                        "dtype")
                elif hasattr(current_field, 'type'):
                    current_type = str(current_field.type)
                elif hasattr(current_field, 'data_type'):
                    current_type = str(current_field.data_type)

                expected_type = expected_field.get("type")

                # Map type names
                type_mapping = {
                    "STRING": "string",
                    "INTEGER": "integer",
                    "FLOAT": "float",
                    "VECTOR": "vector"
                }

                if current_type in type_mapping:
                    current_type = type_mapping[current_type]
                elif hasattr(current_type, 'name'):
                    current_type = current_type.name.lower()

                if current_type != expected_type:
                    logger.error(f"Field {field_name} type mismatch: expected {expected_type}, actual {current_type}")
                    return False

                # For vector fields, check dimension
                if expected_type == "vector":
                    expected_dim = expected_field.get("dimension", expected_structure.get("dimension", 1024))
                    current_dim = None

                    if isinstance(current_field, dict):
                        current_dim = current_field.get("dimension") or current_field.get("dim")
                    elif hasattr(current_field, 'dimension'):
                        current_dim = current_field.dimension

                    if current_dim != expected_dim:
                        logger.error(f"Vector field {field_name} dimension mismatch: expected {expected_dim}, actual {current_dim}")
                        return False

            logger.info(f"Space {space_name} structure validation passed")
            return True
        except Exception as e:
            logger.error(f"Error validating space structure: {str(e)}")
            import traceback
            logger.error(f"Detailed error information: {traceback.format_exc()}")
            return False

    def create_database(self, db_name: str) -> bool:
        """
        Create database

        Args:
            db_name: Database name

        Returns:
            bool: Whether creation was successful
        """
        try:
            if not self.database_exists(db_name):
                ret = self.client.create_database(db_name)
                if ret.is_success():
                    logger.info(f"Database {db_name} created successfully")
                    return True
                else:
                    logger.error(f"Failed to create database {db_name}: {ret.msg}")
                    return False
            else:
                logger.info(f"Database {db_name} already exists")
                return True
        except Exception as e:
            logger.error(f"Failed to create database {db_name}: {str(e)}")
            return False

    def delete_space(self, db_name: str, space_name: str) -> bool:
        """
        Delete space

        Args:
            db_name: Database name
            space_name: Space name

        Returns:
            bool: Whether deletion was successful
        """
        try:
            if self.space_exists(db_name, space_name):
                ret = self.client.drop_space(db_name, space_name)
                if ret.is_success():
                    logger.info(f"Space {space_name} deleted")
                    return True
                else:
                    logger.error(f"Failed to delete space {space_name}: {ret.msg}")
                    return False
            else:
                logger.info(f"Space {space_name} does not exist, no need to delete")
                return True
        except Exception as e:
            logger.error(f"Failed to delete space {space_name}: {str(e)}")
            return False

    def create_space(self, db_name: str, space_name: str, structure: Dict[str, Any]) -> bool:
        """
        Create space

        Args:
            db_name: Database name
            space_name: Space name
            structure: Space structure

        Returns:
            bool: Whether creation was successful
        """
        try:
            # Ensure database exists
            if not self.create_database(db_name):
                return False

            # If space already exists, delete it first
            if self.space_exists(db_name, space_name):
                logger.info(f"Space {space_name} already exists, will delete first")
                if not self.delete_space(db_name, space_name):
                    return False

            # Build field list
            fields = []
            for field_config in structure.get("fields", []):
                field_name = field_config.get("name")
                field_type = field_config.get("type")
                desc = field_config.get("desc", "")
                index = field_config.get("index", True)

                if field_type == "string":
                    field = Field(
                        name=field_name,
                        data_type=DataType.STRING,
                        desc=desc,
                        index=ScalarIndex(f"{field_name}_idx") if index else None
                    )
                elif field_type == "integer":
                    field = Field(
                        name=field_name,
                        data_type=DataType.INTEGER,
                        desc=desc,
                        index=ScalarIndex(f"{field_name}_idx") if index else None
                    )
                elif field_type == "float":
                    field = Field(
                        name=field_name,
                        data_type=DataType.FLOAT,
                        desc=desc,
                        index=ScalarIndex(f"{field_name}_idx") if index else None
                    )
                elif field_type == "vector":
                    dimension = field_config.get("dimension", structure.get("dimension", 128))
                    metric_type = MetricType.Inner_product
                    if structure.get("metric_type") == "L2":
                        metric_type = MetricType.L2

                    field = Field(
                        name=field_name,
                        data_type=DataType.VECTOR,
                        desc=desc,
                        index=FlatIndex(f"{field_name}_idx", metric_type),
                        dimension=dimension
                    )
                else:
                    logger.warning(f"Unsupported field type: {field_type}")
                    continue

                fields.append(field)

            # Create space schema
            space_schema = SpaceSchema(space_name, fields=fields)

            # Create space
            ret = self.client.create_space(db_name, space_schema)
            if ret.is_success():
                logger.info(f"Space {space_name} created successfully")
                return True
            else:
                logger.error(f"Failed to create space {space_name}: {ret.msg}")
                return False

        except Exception as e:
            logger.error(f"Failed to create space {space_name}: {str(e)}")
            return False

    def create_space_with_schema(self, db_name: str, schema: SpaceSchema) -> bool:
        try:
            ret = self.client.create_space(db_name, schema)
            if ret.is_success():
                logger.info(f"Space {schema.name} created successfully")
                return True
            else:
                error_msg = f"Failed to create space {schema.name}: {ret.msg}"
                logger.error(error_msg)
                raise RuntimeError(error_msg)
        except RuntimeError:
            raise  # Re-raise RuntimeError
        except Exception as e:
            error_msg = f"Failed to create space {schema.name}: {str(e)}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e  # Wrap exception, preserve original exception info


    def ensure_space(self, db_name: str, space_name: str, expected_structure: Dict[str, Any],
                     force_recreate: bool = False) -> bool:
        """
        Ensure space exists and structure matches expectations

        Args:
            db_name: Database name
            space_name: Space name
            expected_structure: Expected space structure
            force_recreate: Whether to force recreate space (even if structure validation passes)

        Returns:
            bool: Whether operation was successful
        """
        # Check if database exists
        if not self.database_exists(db_name):
            logger.info(f"Database {db_name} does not exist, will create new database")
            # Create database
            if not self.create_database(db_name):
                logger.error(f"Failed to create database {db_name}")
                return False

        # Check if space exists
        if not self.space_exists(db_name, space_name):
            logger.info(f"Space {space_name} does not exist, will create new space")
            return self.create_space(db_name, space_name, expected_structure)

        # If force recreate, delete space first
        if force_recreate:
            logger.warning(f"Force recreate mode: delete space {space_name}")
            if not self.delete_space(db_name, space_name):
                logger.error(f"Failed to delete space {space_name}")
                return False
            return self.create_space(db_name, space_name, expected_structure)

        # Validate space structure
        if self.validate_space_structure(db_name, space_name, expected_structure):
            logger.info(f"Space {space_name} exists and structure matches expectations")
            return True
        else:
            logger.warning(f"Space {space_name} structure does not match expectations, will delete and recreate")
            return self.create_space(db_name, space_name, expected_structure)

    def delete_vector(self, db_name: str, space_name: str, kb_id: str, file_ids: List[str]) -> bool:
        try:
            if not file_ids:
                logger.info('No file IDs to delete')
                return True

            # Build Vearch query conditions
            # Vearch uses filter for exact match queries
            filter_expr = {
                "operator": "AND",
                "conditions": [
                    {
                        "field": "kb_id",
                        "value": [kb_id],
                        "operator": "IN"
                    },
                    {
                        "field": "ori_file_id",
                        "value": file_ids,  # Vearch supports passing list directly
                        "operator": "IN"
                    }
                ]
            }

            condition_list = []
            for key in filter_expr.get("conditions", []):
                condition = Condition(operator=key.get("operator"), fv=FieldValue(field=key.get("field"), value=key.get("value")))
                condition_list.append(condition)

            # conditions = Conditions(operator=filter_expr.get("operator"), conditions=condition_list)
            filters = Filter(operator=filter_expr.get("operator"), conditions=condition_list)

            # Execute delete operation
            response = self.client.delete(
                database_name=db_name,
                space_name=space_name,
                filter=filters
            )

            # Check response
            if response.is_success():
                # Debug: print complete response structure
                print(f"Delete response: {response}")
                print(f"Response data: {response.document_ids}")

                print(f"Successfully deleted {response.total} records")
                return True
            else:
                print(f"Delete failed: {response.msg}")
                return False

        except Exception as e:
            print(f"Error batch deleting data: {str(e)}")
            return False

    def add_df(self, database_name: str, space_name: str, df: pd.DataFrame) -> bool:
        documents = []
        for index, row in df.iterrows():
            doc = row.to_dict()

            # Handle special values, processing order is important
            for key, value in doc.items():
                if isinstance(value, list):
                    doc[key] = value
                elif pd.isna(value):
                    doc[key] = None

            documents.append(doc)

        if not documents:
            logger.error("No valid documents to write")
            return False

        # Batch insert data
        batch_size = 100  # Process 100 records per batch
        success_count = 0
        total_count = len(documents)

        for i in range(0, total_count, batch_size):
            batch_docs = documents[i:i + batch_size]

            try:
                # Execute batch insert
                response = self.client.upsert(
                    database_name=database_name,
                    space_name=space_name,
                    data=batch_docs
                )

                if response.is_success():
                    batch_success = len(batch_docs)
                    success_count += batch_success
                    logger.info(f"Successfully inserted batch {i // batch_size + 1}, total {batch_success} records")
                else:
                    logger.error(f"Failed to insert batch {i // batch_size + 1}: {response.msg}")
                    return False

            except Exception as batch_error:
                logger.error(f"Error processing batch {i // batch_size + 1}: {str(batch_error)}")
                return False

        logger.info(f"Batch insert completed, successfully inserted {success_count}/{total_count} records")
        return True

    def add_vector(self, database_name: str, space_name: str, documents: List[dict]) -> bool:
        """
        Batch add vector data to Vearch space

        Args:
            database_name: Database name
            space_name: Space name
            documents: List of documents to add, each document is a dict containing all required fields

        Returns:
            bool: Whether addition was successful
        """
        try:
            if not documents:
                logger.info("No data to add")
                return True

            # Verify space exists
            if not self.space_exists(database_name, space_name):
                logger.error(f"Space {space_name} does not exist in database {database_name}")
                return False

            # Verify all fields exist in space
            try:
                # Get space field information
                space_structure = self.get_space_structure(database_name, space_name)
                current_fields = []

                if "fields" in space_structure:
                    current_fields = space_structure["fields"]
                elif "field" in space_structure:
                    current_fields = space_structure["field"]

                # Create field name mapping
                existing_fields = set()
                for field in current_fields:
                    if isinstance(field, dict):
                        field_name = field.get("name") or field.get("field_name") or field.get("field")
                        if field_name:
                            existing_fields.add(field_name)
                    elif hasattr(field, 'name'):
                        existing_fields.add(field.name)

                # Check if all document fields exist
                all_document_fields = set()
                for doc in documents:
                    all_document_fields.update(doc.keys())

                missing_fields = all_document_fields - existing_fields
                if missing_fields:
                    logger.error(f"Documents contain fields not in space: {missing_fields}")
                    logger.error(f"Existing space fields: {existing_fields}")
                    logger.error(f"Fields used by documents: {all_document_fields}")
                    return False

                logger.info(f"Field validation passed, all fields exist in space")

            except Exception as e:
                logger.error(f"Error validating space fields: {str(e)}")
                # If field validation fails but space exists, try to continue insertion
                logger.warning("Field validation failed, but attempting to continue data insertion")

            # Validate data format
            required_fields = ["kb_id", "ori_file_id", "chunk_id", "chunk_vector"]
            for i, doc in enumerate(documents):
                for field in required_fields:
                    if field not in doc:
                        logger.error(f"Document {i + 1} is missing required field: {field}")
                        return False

                # Validate vector dimension
                vector = doc["chunk_vector"]
                if not isinstance(vector, list) or len(vector) != 1024:
                    logger.error(
                        f"Document {i + 1} has incorrect vector dimension, expected 1024, actual {len(vector) if isinstance(vector, list) else 'N/A'}")
                    return False

            # Batch insert data
            batch_size = 100  # Process 100 records per batch
            success_count = 0
            total_count = len(documents)

            for i in range(0, total_count, batch_size):
                batch_docs = documents[i:i + batch_size]

                try:
                    # Execute batch insert
                    response = self.client.upsert(
                        database_name=database_name,
                        space_name=space_name,
                        data=batch_docs
                    )

                    if response.is_success():
                        batch_success = len(batch_docs)
                        success_count += batch_success
                        logger.info(f"Successfully inserted batch {i // batch_size + 1}, total {batch_success} records")
                    else:
                        logger.error(f"Failed to insert batch {i // batch_size + 1}: {response.msg}")
                        return False

                except Exception as batch_error:
                    logger.error(f"Error processing batch {i // batch_size + 1}: {str(batch_error)}")
                    return False

            logger.info(f"Batch insert completed, successfully inserted {success_count}/{total_count} records")
            return success_count == total_count

        except Exception as e:
            logger.error(f"Error batch inserting data into Vearch: {str(e)}")
            return False

    def add_nodes(self, database_name: str, space_name: str, nodes: List, emb_version: str = "2.0") -> bool:
        """
        Convert LlamaIndex nodes to Vearch format and insert

        Args:
            database_name: Database name
            space_name: Space name
            nodes: List of LlamaIndex BaseNodes
            emb_version: Embedding version identifier

        Returns:
            bool: Whether addition was successful
        """
        try:
            if not nodes:
                logger.info("No nodes to add")
                return True

            # Convert nodes to Vearch document format
            documents = []
            for node in nodes:
                metadata = node.metadata or {}

                # Build Vearch document
                doc = {
                    "kb_id": metadata.get("kb_id", ""),
                    "ori_file_id": metadata.get("ori_file_id", ""),
                    "chunk_id": metadata.get("chunk_id", f"chunk_{hash(node.text) % 1000000}"),
                    "chunk_text": node.text,
                    "chunk_extra_info": str(metadata.get("chunk_extra_info", {})),
                    "return_text": metadata.get("return_text", ""),
                    "chunk_vector": node.embedding if hasattr(node, 'embedding') and node.embedding else [],
                    "emb_version": emb_version,
                    "language": metadata.get("language", "")
                }

                # Validate required fields
                if not doc["kb_id"] or not doc["ori_file_id"] or not doc["chunk_vector"]:
                    logger.warning(f"Node missing required fields, skipping: chunk_id={doc['chunk_id']}")
                    continue

                documents.append(doc)

            if not documents:
                logger.error("No valid documents to insert")
                return False

            # Call add_vector method to insert data
            return self.add_vector(database_name, space_name, documents)

        except Exception as e:
            logger.error(f"Error converting nodes and inserting into Vearch: {str(e)}")
            return False

    def diagnose_space(self, db_name: str, space_name: str, expected_structure: Dict[str, Any]) -> Dict[str, Any]:
        """
        Diagnose space structure issues

        Args:
            db_name: Database name
            space_name: Space name
            expected_structure: Expected space structure

        Returns:
            Dict: Diagnosis result
        """
        diagnosis = {
            "database_exists": False,
            "space_exists": False,
            "structure_match": False,
            "current_fields": [],
            "expected_fields": [],
            "missing_fields": [],
            "type_mismatches": [],
            "errors": []
        }

        try:
            # Check database
            diagnosis["database_exists"] = self.database_exists(db_name)
            if not diagnosis["database_exists"]:
                diagnosis["errors"].append(f"Database {db_name} does not exist")
                return diagnosis

            # Check space
            diagnosis["space_exists"] = self.space_exists(db_name, space_name)
            if not diagnosis["space_exists"]:
                diagnosis["errors"].append(f"Space {space_name} does not exist")
                return diagnosis

            # Get current structure
            try:
                current_structure = self.get_space_structure(db_name, space_name)
                logger.info(f"Current space structure: {current_structure}")

                # Extract field information
                current_fields = []
                if "fields" in current_structure:
                    current_fields = current_structure["fields"]
                elif "field" in current_structure:
                    current_fields = current_structure["field"]

                # Create field mapping
                current_field_map = {}
                for field in current_fields:
                    if isinstance(field, dict):
                        field_name = field.get("name") or field.get("field_name") or field.get("field")
                        if field_name:
                            current_field_map[field_name] = field
                    elif hasattr(field, 'name'):
                        current_field_map[field.name] = field

                diagnosis["current_fields"] = list(current_field_map.keys())
                diagnosis["expected_fields"] = [field["name"] for field in expected_structure.get("fields", [])]

                # Check missing fields
                expected_field_map = {field["name"]: field for field in expected_structure.get("fields", [])}
                for field_name in expected_field_map.keys():
                    if field_name not in current_field_map:
                        diagnosis["missing_fields"].append(field_name)

                # Check type mismatches
                for field_name, expected_field in expected_field_map.items():
                    if field_name in current_field_map:
                        current_field = current_field_map[field_name]
                        expected_type = expected_field.get("type")

                        # Get current type
                        current_type = None
                        if isinstance(current_field, dict):
                            current_type = current_field.get("type") or current_field.get("data_type")
                        elif hasattr(current_field, 'type'):
                            current_type = str(current_field.type)
                        elif hasattr(current_field, 'data_type'):
                            current_type = str(current_field.data_type)

                        # Type mapping
                        type_mapping = {
                            "STRING": "string",
                            "INTEGER": "integer",
                            "FLOAT": "float",
                            "VECTOR": "vector"
                        }

                        if current_type in type_mapping:
                            current_type = type_mapping[current_type]
                        elif hasattr(current_type, 'name'):
                            current_type = current_type.name.lower()

                        if current_type != expected_type:
                            diagnosis["type_mismatches"].append({
                                "field": field_name,
                                "expected": expected_type,
                                "current": current_type
                            })

                diagnosis["structure_match"] = (
                        len(diagnosis["missing_fields"]) == 0 and
                        len(diagnosis["type_mismatches"]) == 0
                )

            except Exception as e:
                diagnosis["errors"].append(f"Failed to get space structure: {str(e)}")

        except Exception as e:
            diagnosis["errors"].append(f"Error during diagnosis: {str(e)}")

        return diagnosis

    def force_recreate_space(self, db_name: str, space_name: str, expected_structure: Dict[str, Any]) -> bool:
        """
        Force recreate space (to resolve structure mismatch issues)

        Args:
            db_name: Database name
            space_name: Space name
            expected_structure: Expected space structure

        Returns:
            bool: Whether recreation was successful
        """
        logger.warning(f"Force recreate space {space_name}, this will delete all existing data")

        try:
            # First diagnose current state
            diagnosis = self.diagnose_space(db_name, space_name, expected_structure)
            logger.info(f"Pre-recreation diagnosis result: {diagnosis}")

            # If space exists, delete it first
            if diagnosis["space_exists"]:
                logger.info(f"Deleting existing space {space_name}")
                if not self.delete_space(db_name, space_name):
                    logger.error(f"Failed to delete space {space_name}")
                    return False
                logger.info(f"Space {space_name} successfully deleted")

            # Create new space
            logger.info(f"Creating new space {space_name}")
            if self.create_space(db_name, space_name, expected_structure):
                logger.info(f"Space {space_name} recreated successfully")

                # Verify recreation result
                new_diagnosis = self.diagnose_space(db_name, space_name, expected_structure)
                if new_diagnosis["structure_match"]:
                    logger.info(f"Space {space_name} passed verification after recreation")
                    return True
                else:
                    logger.error(f"Space {space_name} still has issues after recreation: {new_diagnosis}")
                    return False
            else:
                logger.error(f"Failed to create new space {space_name}")
                return False

        except Exception as e:
            logger.error(f"Error force recreating space {space_name}: {str(e)}")
            import traceback
            logger.error(f"Detailed error information: {traceback.format_exc()}")
            return False

    def reset_space_if_needed(self, db_name: str, space_name: str, expected_structure: Dict[str, Any]) -> bool:
        """
        Automatically reset space if there are structure issues

        Args:
            db_name: Database name
            space_name: Space name
            expected_structure: Expected space structure

        Returns:
            bool: Whether operation was successful
        """
        try:
            # Diagnose current state
            diagnosis = self.diagnose_space(db_name, space_name, expected_structure)

            # If space does not exist or structure does not match, recreate
            if not diagnosis["space_exists"] or not diagnosis["structure_match"]:
                logger.warning(f"Space {space_name} needs reset, diagnosis result: {diagnosis}")
                return self.force_recreate_space(db_name, space_name, expected_structure)
            else:
                logger.info(f"Space {space_name} is in normal state, no reset needed")
                return True

        except Exception as e:
            logger.error(f"Error checking and resetting space {space_name}: {str(e)}")
            return False

    def search_vectors(self, query_vector, top_k=100, db_name="knowledge_base",
                       space_name="kb_chunks"):
        """
        Perform similarity search based on vectors

        Args:
            query_vector: Query vector (1024 dimensions)
            top_k: Number of results to return
            db_name: Database name
            space_name: Space name

        Returns:
            list: List of search results
        """
        try:
            # Vector normalization
            query_vector_norm = np.array(query_vector)
            query_vector_norm = query_vector_norm / np.linalg.norm(query_vector_norm)

            # Create VectorInfo object
            vector_info = VectorInfo(
                field_name="chunk_vector",
                feature=query_vector_norm.tolist()
            )

            # Execute vector search
            search_result = self.client.search(
                database_name=db_name,
                space_name=space_name,
                vector_infos=[vector_info],  # Corrected parameter name
                fields=["chunk_text", "kb_id", "ori_file_id", "chunk_id",
                        "emb_version", "language"],  # Fields to return
                limit=top_k
            )

            # Process results
            results = []
            if search_result.is_success() and search_result.documents:
                for hit in search_result.documents[0]:  # First query result
                    result = {
                        "score": hit.get("_score", 0.0),
                        "chunk_text": hit.get("chunk_text", ""),
                        "kb_id": hit.get("kb_id", ""),
                        "ori_file_id": hit.get("ori_file_id", ""),
                        "chunk_id": hit.get("chunk_id", ""),
                        "emb_version": hit.get("emb_version", ""),
                        "language": hit.get("language", "")
                    }
                    results.append(result)

            return results

        except Exception as e:
            print(f"Search error: {e}")
            return []
