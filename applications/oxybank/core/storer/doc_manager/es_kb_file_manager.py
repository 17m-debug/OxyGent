from typing import Optional, Dict, Any, List

import pandas as pd
from elasticsearch import Elasticsearch, helpers
from elasticsearch.exceptions import NotFoundError
from loguru import logger


class ElasticsearchKbFileManager:

    def __init__(
            self,
            es_client: Elasticsearch,
            index_name: str = 'knowledge_file_info'
    ):
        self.client = es_client
        self.index_name = index_name

        # Check connection
        try:
            if not self.client.ping():
                raise ConnectionError("Elasticsearch client error")
        except Exception as e:
            raise ConnectionError(f"Elasticsearch client error: {str(e)}")

    def kb_file_search(self, kb_id: str) -> Optional[List[Dict[str, Any]]]:
        """Search for all file IDs in the knowledge base corresponding to kb_name
        """
        try:
            query = {
                "query": {
                    "term": {
                        "kb_id": kb_id  # Use term query for exact matching
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
            print(f"Failed to query index {self.index_name}: {str(e)}")
            return None

    def kb_delete_file(self, kb_id: str, file_ids: List[str]) -> bool:
        """
        Delete corresponding file information based on knowledge base ID and file ID list

        Args:
            kb_id: Knowledge base ID
            file_ids: List of file IDs to delete

        Returns:
            bool: Whether deletion was successful
        """
        try:
            if not file_ids:
                logger.info("No file IDs to delete")
                return True
            if not self.client.indices.exists(index=self.index_name):
                logger.info(f"Index {self.index_name} does not exist, no need to delete file records")
                return True

            # Build delete query: match both kb_id and file_id
            query = {
                "query": {
                    "bool": {
                        "must": [
                            {"term": {"kb_id": kb_id}},
                            {"terms": {"ori_file_id": file_ids}}
                        ]
                    }
                }
            }

            # Execute delete operation
            response = self.client.delete_by_query(
                index=self.index_name,
                body=query,
                refresh=True  # Refresh immediately to make deletion effective
            )

            deleted_count = response.get("deleted", 0)
            logger.info(f"Successfully deleted {deleted_count} file records (kb_id: {kb_id}, file_ids: {file_ids})")

            return True

        except NotFoundError:
            logger.info(f"Index {self.index_name} does not exist, no need to delete file records")
            return True
        except Exception as e:
            logger.error(f"Failed to delete file records (kb_id: {kb_id}, file_ids: {file_ids}): {str(e)}")
            return False


    def kb_add_batch_files(self, kb_info_list: List[Dict[str, Any]]) -> bool:
        """Batch add knowledge base information corresponding to kb_name"""
        try:
            if not kb_info_list:
                logger.info("No knowledge file information to add")
                return True

            actions = []
            for kb_file in kb_info_list:
                action = {
                    "_index": self.index_name,
                    "_source": kb_file
                }
                actions.append(action)
            # Execute batch operation
            success, failed = helpers.bulk(self.client, actions)

            print(f"Successfully indexed {success} documents, failed {failed} documents")
            if failed != 0:
                return False
            return True
        except Exception as e:
            logger.error(f"Error adding knowledge base file information: {str(e)}")
            return False


    def kb_add_file(self, kb_info: Dict[str, Any]) -> bool:
        """Add document information passed to knowledge base"""
        try:
            # Write to ES index
            response = self.client.index(
                index=self.index_name,  # Index name
                body=kb_info,  # Document content
                refresh=True  # Refresh immediately to make document searchable
            )

            # Check response result
            if response.get("result") in ["created", "updated"]:
                logger.info(f"Knowledge base file {kb_info['kb_id']} created/updated successfully")
                return True
            else:
                logger.error(f"Knowledge base file information creation/update failed: {response}")
                return False

        except Exception as e:
            logger.error(f"Knowledge base file information creation/update error: {str(e)}")
            return False

    def kb_update_file_info(self, kb_info: Dict[str, Any]) -> bool:
        """Update information of documents passed to knowledge base, such as update time. The kb_info in the parameter is the content to be replaced"""
        kb_id = kb_info["kb_id"]
        ori_file_id = kb_info["ori_file_id"]

        # Based on the passed document information, get the original document content of es doc, including _source content and _id which is the es document id
        result = self.get_kb_file_info(kb_id=kb_id, ori_file_id=ori_file_id)
        es_doc_id = result["_id"]

        if not result:
            return False

        try:
            response = self.client.index(
                index=self.index_name,
                id=es_doc_id,
                body=kb_info,
                refresh=True
            )

            if response.get("result") in ["updated", "noop"]:
                logger.info(f"Successfully updated document (es_doc_id: {es_doc_id}, kb_id: {kb_id}, ori_file_id: {ori_file_id})")
                return True
            else:
                logger.error(f"Failed to update document: {response}")
                return False

        except Exception as e:
            logger.error(f"Error updating document (es_doc_id: {es_doc_id}): {str(e)}")
            return False


    def get_kb_file_info(self, kb_id: str, ori_file_id: str) -> Dict[str, Any] | None:
        """Get document storage information through knowledge base id and file id, including es document id"""
        try:
            query = {
                "query": {
                    "bool": {
                        "must": [
                            {"term": {"kb_id": kb_id}},
                            {"term": {"ori_file_id": ori_file_id}}
                        ]
                    }
                },
                "_source": True
            }

            resp = self.client.search(
                index=self.index_name,
                body=query,
                size=1
            )

            hits = resp["hits"]["hits"]
            if hits:
                return {
                    "_id": hits[0]["_id"],
                    "_source": hits[0]["_source"]
                }
            else:
                logger.info(f"File information not found (kb_id: {kb_id}, ori_file_id: {ori_file_id})")
                return None

        except Exception as e:
            logger.error(f"Failed to query file information (kb_id: {kb_id}, ori_file_id: {ori_file_id}): {str(e)}")
            return None


    def get_kb_files(
            self,
            kb_id: str,
            page: int = 1,
            size: int = 10,
    ) -> Dict[str, Any]:
        try:
            # Calculate from value (starts from 0)
            from_value = (page - 1) * size

            query = {
                "query": {
                    "term": {
                        "kb_id": kb_id
                    }
                },
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
            logger.error(f"search all knowledge base file of [kb_id:{kb_id}] error: {e}")
            return {
                "items": [],
                "total": 0,
                "page": page,
                "size": size,
                "pages": 0
            }


    def check_file_exists(
            self,
            kb_id: str,
            ori_file_id: str,
    ) -> bool:
        """Check if file ID already exists in knowledge base file list"""
        try:
            query = {
                "query": {
                    "bool": {
                        "must": [
                            {"term": {"kb_id": kb_id}},
                            {"term": {"ori_file_id": ori_file_id}}
                        ]
                    }
                }
            }

            resp = self.client.count(
                index=self.index_name,
                body=query,
            )

            count = resp.get("count", 0)
            return count > 0

        except Exception as e:
            logger.error(f"Failed to check if file exists (kb_id: {kb_id}, ori_file_id: {ori_file_id}): {str(e)}")
            return False


    def kb_add_df(self, kb_name: str, df: pd.DataFrame) -> bool:
        actions = []
        batch_size = 100

        # Batch write data to corresponding es database
        for index, row in df.iterrows():
            try:
                # Handle NaN values
                doc = row.to_dict()
                for key, value in doc.items():
                    if pd.isna(value):
                        doc[key] = None

                action = {
                    "_index": kb_name,
                    "_source": doc
                }
                actions.append(action)

                # Write data in batches
                if len(actions) >= batch_size:
                    success, failed = helpers.bulk(self.client, actions)
                    logger.info(f"bulk add data into {kb_name}. {success} success, {failed} failed")
                    actions = []

            except Exception as e:
                logger.error(f"process data line {index + 1} exception: {e}")
                continue

        # Process remaining documents
        if actions:
            try:
                success, failed = helpers.bulk(self.client, actions)
                logger.info(f"bulk add data into {kb_name}. {success} success, {failed} failed")
                return len(failed) == 0
            except Exception as e:
                logger.error(f"batch add data into es index {kb_name} exception: {e}")
                return False

        return True
