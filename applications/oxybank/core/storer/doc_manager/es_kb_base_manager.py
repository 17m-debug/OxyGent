from typing import Optional, Dict, Any, List

from elasticsearch import Elasticsearch
from loguru import logger


class ElasticsearchKbBaseManager:

    def __init__(
            self,
            es_client: Elasticsearch,
            index_name: str = 'knowledge_base_info'
    ):
        self.client = es_client
        self.index_name = index_name

        # Check connection
        try:
            if not self.client.ping():
                raise ConnectionError("Elasticsearch client error")
        except Exception as e:
            raise ConnectionError(f"Elasticsearch client error: {str(e)}")

    def kb_exists(self, kb_name: str) -> bool:
        """Check if knowledge base exists for the given kb_name"""
        try:
            hits = self.kb_info_search_name(kb_name=kb_name)
            return len(hits) > 0
        except Exception as e:
            print(f"Query index {self.index_name} failed: {str(e)}")
            return False

    def kb_info_search_name(self, kb_name: str) -> Optional[List[Dict[str, Any]]]:
        """Search knowledge base information for kb_name
        Sample output:
        {
            "kb_id": "ac741bbe-2603-4d97-8712-a4e8c63b5e85",
            "kb_name": "kb_1",
            "kb_description": "this is the first knowledge base",
            "kb_type": "unstructured",
            "kb_extra_info": {}
        }
        """
        try:
            query = {
                "query": {
                    "term": {
                        "kb_name": kb_name  # Use term query for exact match
                    }
                },
                "_source": True
            }

            resp = self.client.search(
                index=self.index_name,
                body=query,
            )
            return [hit["_source"] for hit in resp['hits']['hits']]
        except Exception as e:
            print(f"Query index {self.index_name} failed: {str(e)}")
            return None


    def kb_info_search_id(self, kb_id: str) -> Optional[List[Dict[str, Any]]]:
        """Search knowledge base information for kb_id
        """
        try:
            query = {
                "query": {
                    "term": {
                        "kb_id": kb_id  # Use term query for exact match
                    }
                },
                "_source": True
            }

            resp = self.client.search(
                index=self.index_name,
                body=query,
            )
            return [hit["_source"] for hit in resp['hits']['hits']]
        except Exception as e:
            print(f"Query index {self.index_name} failed: {str(e)}")
            return None

    def get_kb_schema_by_id(self, kb_id: str) -> Dict[str, Any] | None:
        kb_info = self.kb_info_search_id(kb_id=kb_id)
        # Knowledge base information not found for kb_id
        if not kb_info:
            return None
        kb_schema = kb_info[0].get("kb_schema")
        return kb_schema


    def kb_add(self, kb_info: Dict[str, Any]) -> bool:
        """Add knowledge base information for kb_name"""
        try:
            # Write to ES index
            response = self.client.index(
                index=self.index_name,  # Index name
                body=kb_info,  # Document content
                refresh=True  # Refresh immediately to make document searchable
            )

            # Check response result
            if response.get("result") == "created":
                logger.info(f"Knowledge base {kb_info['kb_id']} created successfully")
                return True
            else:
                logger.error(f"Failed to create knowledge base information: {response}")
                return False

        except Exception as e:
            logger.error(f"Exception creating knowledge base information: {str(e)}")
            return False

    def kb_list(
            self,
            page: int = 1,
            size: int = 10
    ) -> Dict[str, Any]:
        """Return all current knowledge base information"""
        try:
            # Calculate from value (starts from 0)
            from_value = (page - 1) * size

            query = {
                "query": {"match_all": {}},
                "from": from_value,
                "size": size,
                "sort": [
                    {
                        "create_time": {
                            "order": "desc"
                        }
                    }
                ]
            }

            resp = self.client.search(
                index=self.index_name,
                body=query,
            )

            hits = resp['hits']['hits']
            total = resp['hits']['total']['value']
            result = [doc["_source"] for doc in hits]
            return {
                "items": result,
                "total": total,
                "page": page,
                "size": size,
                "pages": (total + size - 1) // size
            }
        except Exception as e:
            logger.error(f"search all knowledge base error: {e}")
            return {
                "items": [],
                "total": 0,
                "page": page,
                "size": size,
                "pages": 0
            }

    def kb_list_all(self) -> List[Dict[str, Any]]:
        """Get all knowledge base information (no pagination)

        Returns:
            List of all knowledge base information
        """
        try:
            # Use scroll API to get all data
            query = {
                "query": {"match_all": {}}
            }

            # First query
            resp = self.client.search(
                index=self.index_name,
                body=query,
                scroll='2m',
                size=1000
            )

            all_kbs = []
            scroll_id = resp.get('_scroll_id')
            hits = resp['hits']['hits']

            # Add first batch of results
            all_kbs.extend([doc["_source"] for doc in hits])

            # Continue scrolling to get remaining data
            while len(hits) > 0:
                resp = self.client.scroll(
                    scroll_id=scroll_id,
                    scroll='2m'
                )
                hits = resp['hits']['hits']
                all_kbs.extend([doc["_source"] for doc in hits])

            # Clear scroll context
            if scroll_id:
                try:
                    self.client.clear_scroll(scroll_id=scroll_id)
                except:
                    pass

            logger.info(f"Successfully retrieved {len(all_kbs)} knowledge base information")
            return all_kbs
        except Exception as e:
            logger.error(f"Failed to get all knowledge base information: {e}")
            return []

    def kb_update(self, kb_name: str, update_fields: Dict[str, Any]) -> bool:
        """Update knowledge base information

        Args:
            kb_name: Knowledge base name
            update_fields: Dictionary of fields to update

        Returns:
            bool: True if update successful, False otherwise
        """
        try:
            # First query to get document ID
            query = {
                "query": {
                    "term": {
                        "kb_name": kb_name
                    }
                }
            }

            resp = self.client.search(
                index=self.index_name,
                body=query,
            )

            hits = resp['hits']['hits']
            if not hits:
                logger.warning(f"Knowledge base {kb_name} does not exist")
                return False

            # Get document ID (take first matching document)
            doc_id = hits[0]['_id']

            # If update_fields contains kb_schema, need to delete old value first then set new value
            # Because Elasticsearch object type fields merge by default rather than replace
            if "kb_schema" in update_fields:
                # Delete kb_schema field first to ensure complete replacement
                delete_body = {
                    "script": {
                        "source": "ctx._source.remove('kb_schema')",
                        "lang": "painless"
                    }
                }
                try:
                    self.client.update(
                        index=self.index_name,
                        id=doc_id,
                        body=delete_body,
                        refresh=False  # Don't refresh yet, wait for final update
                    )
                except Exception as e:
                    logger.warning(f"Exception deleting old kb_schema field (field may not exist): {e}")

            # Use update API to update document
            update_body = {
                "doc": update_fields
            }

            update_resp = self.client.update(
                index=self.index_name,
                id=doc_id,
                body=update_body,
                refresh=True  # Refresh immediately to make update visible
            )

            if update_resp.get("result") in ["updated", "noop"]:
                logger.info(f"Knowledge base {kb_name} updated successfully")
                return True
            else:
                logger.error(f"Failed to update knowledge base information: {update_resp}")
                return False

        except Exception as e:
            logger.error(f"Exception updating knowledge base information: {str(e)}")
            return False

    def kb_delete(self, kb_name: str) -> bool:
        """Delete knowledge base information

        Args:
            kb_name: Knowledge base name

        Returns:
            bool: True if deletion successful, False otherwise
        """
        try:
            # First query to get document ID
            query = {
                "query": {
                    "term": {
                        "kb_name": kb_name
                    }
                }
            }

            resp = self.client.search(
                index=self.index_name,
                body=query,
            )

            hits = resp['hits']['hits']
            if not hits:
                logger.warning(f"Knowledge base {kb_name} does not exist")
                return False

            # Get document ID (take first matching document)
            doc_id = hits[0]['_id']

            delete_resp = self.client.delete(
                index=self.index_name,
                id=doc_id,
                refresh=True
            )

            if delete_resp.get("result") == "deleted":
                logger.info(f"Knowledge base {kb_name} deleted successfully")
                return True
            else:
                logger.error(f"Failed to delete knowledge base information: {delete_resp}")
                return False

        except Exception as e:
            logger.error(f"Exception deleting knowledge base information: {str(e)}")
            return False

    def get_kb_name_by_id(self, kb_id: str) -> str | None:
        """Get kb_name by kb_id"""
        kb_info = self.kb_info_search_id(kb_id=kb_id)
        if kb_info:
            return kb_info[0]["kb_name"]
        return None
