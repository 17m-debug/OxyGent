#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Elasticsearch Index Management
Demonstrates how to use ES Python client to check, validate, delete, and create indexes
"""

from elasticsearch import Elasticsearch
from elasticsearch.exceptions import NotFoundError, RequestError
from typing import Dict, Any, Optional

from loguru import logger


class ElasticsearchIndexManager:
    """Elasticsearch Index Manager"""

    def __init__(self, es_client: Elasticsearch):
        self.client = es_client

        # Check connection
        try:
            if not self.client.ping():
                raise ConnectionError("Elasticsearch client error")
        except Exception as e:
            raise ConnectionError(f"Elasticsearch client error: {str(e)}")

    def index_exists(self, index_name: str) -> bool:
        """
        Check if index exists

        Args:
            index_name: Index name

        Returns:
            bool: Whether index exists
        """
        return self.client.indices.exists(index=index_name)

    def get_index_mapping(self, index_name: str) -> Dict[str, Any]:
        """
        Get index mapping structure

        Args:
            index_name: Index name

        Returns:
            Dict: Index mapping structure
        """
        try:
            return self.client.indices.get_mapping(index=index_name)
        except NotFoundError:
            raise ValueError(f"Index {index_name} does not exist")

    def validate_index_structure(self, index_name: str, expected_mapping: Dict[str, Any]) -> bool:
        """
        Validate if index structure matches expectations

        Args:
            index_name: Index name
            expected_mapping: Expected mapping structure

        Returns:
            bool: Whether structure matches
        """
        if not self.index_exists(index_name):
            return False

        current_mapping = self.get_index_mapping(index_name)

        # Extract actual mapping part (remove index name level)
        current_properties = current_mapping[index_name]["mappings"].get("properties", {})
        expected_properties = expected_mapping.get("properties", {})

        # Simple comparison: check if all expected fields exist in current mapping and types match
        for field, field_config in expected_properties.items():
            if field not in current_properties:
                logger.info(f"Field {field} does not exist in current index")
                return False

            current_type = current_properties[field].get("type")
            expected_type = field_config.get("type")

            # For object type, also accept when ES returns no type (version compatibility)
            if expected_type == "object" and current_type is None:
                continue

            if current_type != expected_type:
                logger.error(f"Field {field} type mismatch: expected {expected_type}, actual {current_type}")
                return False

        return True

    def delete_index(self, index_name: str) -> bool:
        """
        Delete index

        Args:
            index_name: Index name

        Returns:
            bool: Whether deletion was successful
        """
        try:
            if self.index_exists(index_name):
                self.client.indices.delete(index=index_name)
                logger.info(f"Index {index_name} deleted")
                return True
            else:
                logger.info(f"Index {index_name} does not exist, no need to delete")
                return False
        except Exception as e:
            logger.error(f"Failed to delete index {index_name}: {str(e)}")
            return False

    def create_index(self, index_name: str, mapping: Dict[str, Any], settings: Optional[Dict[str, Any]] = None) -> bool:
        """
        Create index

        Args:
            index_name: Index name
            mapping: Index mapping structure
            settings: Index settings

        Returns:
            bool: Whether creation was successful
        """
        try:
            # If index exists, delete it first
            if self.index_exists(index_name):
                logger.info(f"Index {index_name} already exists, will delete first")
                self.delete_index(index_name)

            # Build request body for index creation
            body = {
                "mappings": mapping
            }

            if settings:
                body["settings"] = settings

            # Create index
            self.client.indices.create(index=index_name, body=body)
            logger.info(f"Index {index_name} created successfully")
            return True
        except RequestError as e:
            logger.error(f"Failed to create index {index_name}: {str(e)}")
            return False

    def ensure_index(self, index_name: str, expected_mapping: Dict[str, Any],
                     settings: Optional[Dict[str, Any]] = None) -> bool:
        """
        Ensure index exists and structure matches expectations

        Args:
            index_name: Index name
            expected_mapping: Expected mapping structure
            settings: Index settings

        Returns:
            bool: Whether operation was successful
        """
        logger.info(f"Checking index {index_name}...")

        # Check if index exists
        if not self.index_exists(index_name):
            logger.warning(f"Index {index_name} does not exist, will create new index")
            return self.create_index(index_name, expected_mapping, settings)

        # Validate index structure
        if self.validate_index_structure(index_name, expected_mapping):
            logger.info(f"Index {index_name} exists and structure matches expectations")
            return True
        else:
            logger.error(f"Index {index_name} structure does not match expectations, will delete and recreate")
            return self.create_index(index_name, expected_mapping, settings)

