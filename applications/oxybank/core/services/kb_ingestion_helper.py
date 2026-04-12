"""Knowledge base ingestion helper module

Provides methods to directly call KB underlying services, avoiding in-process HTTP calls
"""
import uuid
from typing import Dict, List, Any

import pandas as pd
from loguru import logger

from core.config import settings
from core.storer.doc_manager.es_kb_base_manager import ElasticsearchKbBaseManager
from core.storer.doc_manager.es_kb_file_manager import ElasticsearchKbFileManager
from core.storer.doc_manager.knowledge_index import KBSchema, check_kb_schema, VearchVectorMatchPolicy
from core.storer.vector_manager.vearch_manager import VearchManager


class KBIngestionHelper:
    """Knowledge base ingestion helper class

    Encapsulates KB ingestion underlying logic for direct call by annotation platform
    """

    def __init__(self):
        """Initialize KB ingestion helper"""
        self.kb_base_client = ElasticsearchKbBaseManager(settings.es_client)
        self.kb_file_client = ElasticsearchKbFileManager(settings.es_client)

    async def ingest_data(
        self,
        kb_id: str,
        data: Dict[str, Any] | List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Ingest data into knowledge base

        This is the underlying implementation of ingest_kb_data in knowledge_file.py,
        directly calling service layer methods to avoid HTTP calls.

        Args:
            kb_id: Knowledge base ID
            data: Data to ingest (dict or list[dict])

        Returns:
            Ingestion result

        Raises:
            ValueError: Invalid params or KB config issue
            Exception: Ingestion failed
        """
        # 1. Get schema by kb_id
        kb_schema_dict = self.kb_base_client.get_kb_schema_by_id(kb_id)
        if not kb_schema_dict:
            raise ValueError(f"kb_id: [{kb_id}] has no KB schema")

        # 2. Convert Dict to KBSchema object
        try:
            kb_schema = KBSchema(**kb_schema_dict)
            if not check_kb_schema(kb_schema):
                raise ValueError("KB schema validation failed, please check KB schema!")
        except Exception as e:
            logger.error(f"Failed to parse kb_schema: {e}")
            raise ValueError(f"KB schema parse/validation failed: {str(e)}")

        # 3. Query KB info
        search_result = self.kb_base_client.kb_info_search_id(kb_id=kb_id)
        if not search_result:
            raise ValueError("Failed to query KB info by KB ID")

        # 4. Get KB name
        kb_name = self.kb_base_client.get_kb_name_by_id(kb_id=kb_id)
        if not kb_name:
            raise ValueError("Failed to query KB name by KB ID")

        # 5. Convert data to DataFrame
        try:
            df = pd.DataFrame(data)
        except ValueError:
            # If scalar dict, wrap into list
            df = pd.DataFrame([data])

        # 6. Add fixed three columns
        df['kb_id'] = kb_id
        mock_file_id = f"mock_file_{kb_id}"
        df['ori_file_id'] = mock_file_id
        df['chunk_id'] = [str(uuid.uuid4().hex[:16]) for _ in range(len(df))]

        # 7. Write to ES index
        es_add_result = self.kb_file_client.kb_add_df(kb_name=kb_name, df=df)
        if not es_add_result:
            logger.error("add file data into es failed")
            raise Exception("add file data into es failed")

        logger.info(f"add file data into es success")

        # 8. Process vector embeddings and write to Vearch
        logger.info(f"KB schema match_rules: {kb_schema.match_rules}")
        logger.info(f"DataFrame columns: {list(df.columns)}")

        if kb_schema.match_rules:
            # Collect all vector match policies
            vec_matches = [
                policy
                for match_rule in kb_schema.match_rules
                for policy in match_rule.match_policies
                if isinstance(policy, VearchVectorMatchPolicy)
            ]

            logger.info(f"Found {len(vec_matches)} VearchVectorMatchPolicy")

            if vec_matches:
                # Create embedding model outside loop to avoid repeated initialization
                embed_model = settings.embedding_model

                try:
                    for vec_match in vec_matches:
                        field_name = vec_match.input_fields[0]

                        # Check if field exists
                        if field_name not in df.columns:
                            logger.warning(f"Field {field_name} not in DataFrame, skip vectorization")
                            continue

                        # Check if field has data
                        if df[field_name].isnull().all():
                            logger.warning(f"Field {field_name} is all null, skip vectorization")
                            continue

                        # Generate vector embeddings
                        texts = df[field_name].astype(str).tolist()
                        embeddings = embed_model.get_text_embedding_batch(texts, show_progress=True)

                        vector_field_name = f"{field_name}_vector"
                        df[vector_field_name] = embeddings

                        logger.info(f"Field {field_name} vectorization complete, {len(embeddings)} items")

                    # Write to Vearch
                    vearch_manager = VearchManager(vearch_client=settings.vearch_client)
                    vearch_add_result = vearch_manager.add_df(
                        database_name=settings.vearch_config.db_name,
                        space_name=kb_name,
                        df=df
                    )

                    if not vearch_add_result:
                        logger.error("Failed to write vector data to Vearch")
                        raise Exception("Failed to write vector data to Vearch, please check Vearch service status")

                    logger.info(f"Vector data successfully written to Vearch space: {kb_name}")

                except Exception as e:
                    logger.error(f"Vectorization failed: {e}")
                    raise Exception(f"Vectorization failed: {str(e)}")
            else:
                logger.warning("vec_matches is empty, no vector match policies found")
        else:
            logger.warning("No match_rules configured or no vector match policies found, skipping Vearch write")

        # 9. Return success result
        return {
            "success": True,
            "kb_id": kb_id,
            "kb_name": kb_name,
            "chunk_count": len(df),
            "message": "Data ingestion successful"
        }


# Global singleton
_kb_ingestion_helper = None


def get_kb_ingestion_helper() -> KBIngestionHelper:
    """Get KB ingestion helper (singleton)"""
    global _kb_ingestion_helper
    if _kb_ingestion_helper is None:
        _kb_ingestion_helper = KBIngestionHelper()
    return _kb_ingestion_helper
