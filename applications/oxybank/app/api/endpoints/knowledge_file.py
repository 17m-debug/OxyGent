import datetime
import os
import pathlib
import shutil
import tempfile
import uuid
from typing import Union

import numpy as np
import pandas as pd
from fastapi import APIRouter, Path, Depends, HTTPException, Query, UploadFile, File, Body
from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import MarkdownNodeParser, NodeParser
from loguru import logger

from app.api.models import KnowledgeFileItem, APIResponse, get_pagination_params, PaginationParams, PaginatedResponse, \
    FileUploadInfo
from core.config import settings
from core.parser.factory import ParserFactory
from core.parser.text_parser import ExtensibleSplitterParser
from core.storer.doc_manager.es_kb_base_manager import ElasticsearchKbBaseManager
from core.storer.doc_manager.es_kb_file_manager import ElasticsearchKbFileManager
from core.storer.doc_manager.knowledge_index import FieldInfo, KBSchema, ParserConfig, check_kb_schema
from core.storer.doc_manager.schema_utils import convert_dataframe_types_by_schema
from core.storer.vector_manager.vearch_manager import VearchManager
from utils.files_process import calculate_file_md5, extract_file_type, is_supported_file
from utils.hash_util import str_to_md5
from utils.type_trans import get_py_type

router = APIRouter(prefix="/kb_base/{kb_id}", tags=["Knowledge Base File Management"])

# Initialize ElasticsearchKbFileManager instance
kb_file_client = ElasticsearchKbFileManager(settings.es_client)
kb_base_client = ElasticsearchKbBaseManager(settings.es_client)


def create_parser_from_config(parser_config: ParserConfig = None) -> NodeParser | ExtensibleSplitterParser:
    """
    Create parser instance based on configuration.
    
    Args:
        parser_config: Parser configuration
        
    Returns:
        Parser instance
    """
    if parser_config is None:
        # Default parser configuration
        logger.info("Using default parser configuration (sentence parser)")
        return ParserFactory.create_parser("sentence", chunk_size=300, chunk_overlap=20)
    
    # Create parser based on configuration
    parser_type = parser_config.parser_type
    
    if parser_type == "extensible":
        return ParserFactory.create_parser(
            parser_type,
            splitter_type=parser_config.splitter_type,
            chunk_size=parser_config.chunk_size,
            chunk_overlap=parser_config.chunk_overlap,
            separator=parser_config.separator
        )
    elif parser_type in ["token", "sentence"]:
        return ParserFactory.create_parser(
            parser_type,
            chunk_size=parser_config.chunk_size,
            chunk_overlap=parser_config.chunk_overlap,
            separator=parser_config.separator
        )
    elif parser_type in ["markdown", "html", "json"]:
        return ParserFactory.create_parser(
            parser_type,
            include_metadata=parser_config.include_metadata,
            include_prev_next_rel=parser_config.include_prev_next_rel
        )
    else:
        logger.warning(f"Unknown parser type: {parser_type}, using default sentence parser")
        return ParserFactory.create_parser("sentence", chunk_size=300, chunk_overlap=20)


@router.get(
    "/kb_file",
    response_model=APIResponse[PaginatedResponse[KnowledgeFileItem]],
    summary="Get all files in the knowledge base",
    description="Returns a list of all files in the specified knowledge base",
    response_description="File list containing file ID, MD5 value, type, path, and other information",
)
def get_kb_files(
        kb_id: str = Path(..., description="Knowledge base ID"),
        pagination: PaginationParams = Depends(get_pagination_params)
) -> APIResponse[PaginatedResponse[KnowledgeFileItem]]:
    """
    Get all files in the knowledge base

    Returns information about all files in the specified knowledge base, including:
    - ori_file_id: Unique identifier of the file in the knowledge base
    - kb_id: Unique identifier of the knowledge base
    - document_md5: MD5 hash value of the file content
    - ori_file_type: File type/extension
    - file_path: Absolute path of the file
    - file_store_mode: File storage mode
    - file_extra_info: Additional information of the file
    - language: Language identifier of the file
    """
    result = kb_file_client.get_kb_files(
        kb_id,
        pagination.page,
        pagination.size
    )

    file_items = [
        KnowledgeFileItem(**file) 
        for file in result.get("items", [])
    ]

    paginated_response = PaginatedResponse[KnowledgeFileItem](
        items=file_items,
        total=result["total"],
        page=result["page"],
        size=result["size"],
        pages=result["pages"],
    )

    # Convert dictionary list to Pydantic model list
    return APIResponse(
        code=200,
        msg="Successfully retrieved file list",
        data=paginated_response
    )


@router.post(
    "/upload_file",
    response_model=APIResponse[FileUploadInfo],
    summary="Upload a single file to a preset directory",
    description="Upload a single file and save it to the preset directory, return the MD5 value of the file",
)
async def upload_kb_file(
        kb_id: str = Path(..., description="Knowledge base ID"),
        file: UploadFile = File(..., description="File to upload")
) -> APIResponse[FileUploadInfo]:
    """
    Upload a single file and return the file's MD5 information
    """
    # File name validation, cannot be empty
    if not file.filename:
        raise HTTPException(status_code=400, detail="File name cannot be empty")

    # File type validation, check if the type is supported
    if not is_supported_file(file.filename):
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {extract_file_type(file.filename)}"
        )

    # Query knowledge base related information
    kb_search_result = kb_base_client.kb_info_search_id(kb_id=kb_id)
    if not kb_search_result:
        raise HTTPException(
            status_code=400,
            detail=f"Knowledge base ID does not exist: {kb_id}"
        )

    temp_dir = tempfile.gettempdir()
    unique_id = str(uuid.uuid4())
    temp_file_path = os.path.join(temp_dir, f"{kb_id}_{unique_id}_{file.filename}")

    # Save to temporary directory
    with open(temp_file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Return temporary file path
    file_path = temp_file_path

    # Calculate document MD5
    try:
        file_md5 = calculate_file_md5(file_path)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"MD5 calculation failed during file upload: {exc}")

    file_size = os.path.getsize(file_path)

    response_data = FileUploadInfo(
        file_id=str_to_md5(file.filename),
        file_name=file.filename,
        file_type=extract_file_type(file.filename),
        file_size=file_size,
        file_path=file_path,
        md5=file_md5,
        upload_time=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    )

    return APIResponse(
        code=200,
        msg="File uploaded successfully",
        data=response_data
    )


def is_temp_file(file_path: str, kb_id: str) -> bool:
    """Check if file is a temporary file created by the system.

    Temporary file format: {kb_id}_{36-char-UUID}_{original_filename}
    Example: kb_12345_550e8400-e29b-41d4-a716-446655440000_test.pdf

    Args:
        file_path: File path to check
        kb_id: Knowledge base ID to match

    Returns:
        True if file matches temporary file format, False otherwise
    """
    filename = os.path.basename(file_path)
    
    # Check if filename starts with {kb_id}_
    prefix = f"{kb_id}_"
    if not filename.startswith(prefix):
        return False
    
    # Extract the part after prefix: {uuid}_{original_filename}
    remaining = filename[len(prefix):]
    
    # Split by first underscore to get UUID part
    # Format: {36-char-UUID}_{original_filename}
    parts = remaining.split('_', 1)
    if len(parts) < 2:
        return False
    
    uuid_part = parts[0]
    
    # Check if UUID part is 36 characters (UUID v4 format: xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx)
    if len(uuid_part) != 36:
        return False
    
    return True


# This file information is the uploaded file information, not the file information already stored in the database
# For file information after storage, use the endpoint /kb_file/{file_id}
@router.get(
    "/upload_file/{file_id}",
    response_model=APIResponse[list[FieldInfo]],
)
def get_uploaded_file_info(
        file_path: str,
        file_type: str
):
    # Read file based on file_type and file_path, and return corresponding data
    if file_type == "csv":
        df = pd.read_csv(file_path)
    elif file_type == "xls" or file_type == "xlsx":
        df = pd.read_excel(file_path)

    else:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {file_type}"
        )
    fields_info = []
    # Get field names and types from file content read into df
    for col_name, dtype in df.dtypes.items():
        # Get Python field type based on df column field type
        front_type = get_py_type(dtype)
        field_info = FieldInfo(
            field_name=col_name,
            field_type=front_type,
            field_desc=""
        )
        fields_info.append(field_info)

    return APIResponse(
        code=200,
        msg="File schema extraction successful",
        data=fields_info
    )


# This endpoint is used to split and store files after upload, writing to ES index and Vearch space
@router.post(
    "/ingest_file",
    response_model=APIResponse[str]
)
def ingest_kb_file(
        kb_id: str = Path(..., description="Knowledge base ID"),
        file_upload_info: FileUploadInfo = Body(..., embed=True),  # Uploaded file information
):
    # Need to get schema based on kb_id
    kb_schema_dict = kb_base_client.get_kb_schema_by_id(kb_id)
    if not file_upload_info.file_id:
        file_upload_info.file_id = str_to_md5(file_upload_info.file_name)
    if not kb_schema_dict:
        raise HTTPException(
            status_code=400,
            detail=f"kb_id: [{kb_id}] The corresponding knowledge base does not have a kb schema"
        )
    # Convert Dict to KBSchema object
    try:
        kb_schema = KBSchema(**kb_schema_dict)
        if not check_kb_schema(kb_schema):
            raise HTTPException(
                status_code=400,
                detail="Current knowledge base kb schema validation failed, please check kb schema!"
            )
    except Exception as e:
        logger.error(f"Failed to parse kb_schema: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Knowledge base schema parsing/validation failed: {str(e)}"
        )

    search_result = kb_base_client.kb_info_search_id(kb_id=kb_id)
    if not search_result:
        raise HTTPException(
            status_code=400,
            detail="Failed to query knowledge base information based on knowledge base id"
        )
    kb_type = search_result[0].get("kb_type", "structured")

    # Query knowledge base name based on knowledge base id
    kb_name = kb_base_client.get_kb_name_by_id(kb_id=kb_id)
    if not kb_name:
        raise HTTPException(
            status_code=400,
            detail="Failed to query knowledge base name based on knowledge base id"
        )

    file_path = file_upload_info.file_path
    try:
        if kb_type == "structured":  # Structured data processing method
            # 1. Read file into memory
            file_type = file_upload_info.file_type

            if file_type == "csv":
                df = pd.read_csv(file_path)
            elif file_type in ["xls", "xlsx"]:
                df = pd.read_excel(file_path)
            else:
                raise HTTPException(
                    status_code=400,
                    detail=f"Unsupported file type: {file_type}"
                )
        elif kb_type == "unstructured":  # Unstructured data processing method
            data = []
            reader = SimpleDirectoryReader(input_files=[file_upload_info.file_path])
            documents = reader.load_data()

            # Create parser based on kb_schema configuration
            if not kb_schema.parser_config:
                logger.warning("Schema has no parser_config defined, use default parser configuration(sentence parser)")
                parser_config = ParserConfig(parser_type='sentence', chunk_size=500, chunk_overlap=50)
            else:
                parser_config = kb_schema.parser_config
            parser = create_parser_from_config(parser_config)
            logger.info(f"Using parser: {type(parser).__name__} with config: {parser_config}")

            nodes = parser.get_nodes_from_documents(documents)
            logger.info(f"node length: {len(nodes)} ")
            for node in nodes:
                data.append({
                    "chunk_to_return": node.text,
                    "chunk_to_emb": node.text,  # This can be replaced with text more suitable for vector matching
                })
            df = pd.DataFrame(data)

    except Exception as e:
        logger.error(f"File reading and processing failed: {file_path}, Error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"File reading and processing failed: {file_path}, Error: {str(e)}"
        )
    finally:
        # Delete temporary file after processing is complete
        # Only delete if it's a temporary file created by system (format: {kb_id}_{36-char-UUID}.{ext})
        if os.path.exists(file_path) and is_temp_file(file_path, kb_id):
            try:
                os.remove(file_path)
                logger.info(f"Temporary file deleted: {file_path}")
            except Exception as e:
                logger.warning(f"Failed to delete temporary file: {file_path}, Error: {e}")
        elif os.path.exists(file_path):
            logger.info(f"File preserved (not a temporary file): {file_path}")

    # 2. Convert DataFrame column types according to schema
    # Keep NaN values first, convert types, then handle NaN based on field type
    df = convert_dataframe_types_by_schema(df, kb_schema)
    
    # 3. Add three fixed columns to each row of data here, which need to be correspondingly added when inferring ES and Vearch schemas
    df['kb_id'] = kb_id
    df['ori_file_id'] = file_upload_info.file_id
    df['chunk_id'] = [str(uuid.uuid4().hex[:16]) for _ in range(len(df))]

    # 2. Write the df data in memory to ES and Vearch
    # 2.1 Write to ES index based on data in df and inferred schema
    es_add_result = kb_file_client.kb_add_df(kb_name=kb_name, df=df)
    if not es_add_result:
        logger.error("add file data into es failed")
        raise HTTPException(
            status_code=500,
            detail="add file data into es failed"
        )
    logger.info(f"add file data into es success")

    # 2.2 First perform embedding processing on some fields based on the inferred schema, save to df, and then write to Vearch space
    if kb_schema.match_rules:
        from core.storer.doc_manager.knowledge_index import VearchVectorMatchPolicy

        # Collect all vector matching policies - iterate through match_rules, then iterate through match_policies in each rule
        vec_matches = [
            policy
            for match_rule in kb_schema.match_rules
            for policy in match_rule.match_policies
            if isinstance(policy, VearchVectorMatchPolicy)
        ]

        if vec_matches:
            vearch_manager = VearchManager(vearch_client=settings.vearch_client)

            # Create embedding model outside the loop to avoid repeated initialization
            embed_model = settings.embedding_model

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
                    embeddings = embed_model.get_text_embedding_batch(texts, show_progress=True)

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
                    # Log failure but don't immediately throw exception, let caller decide whether to rollback
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

    # 3. Write file information to kb_file
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    kb_file_item = {
        "kb_id": kb_id,
        "ori_file_id": file_upload_info.file_id,
        "ori_file_type": file_upload_info.file_type,
        "file_name": file_upload_info.file_name,
        "document_md5": file_upload_info.md5,
        "file_store_mode": "",
        "file_extra_info": {},
        "language": "zh",
        'create_time': current_time,
        'update_time': current_time,
    }
    if not kb_file_client.kb_add_file(kb_file_item):
        logger.error("add kb file info into es failed")
        raise HTTPException(
            status_code=500,
            detail="add kb file info into es failed"
        )

    return APIResponse(
        code=200,
        msg="success",
        data="Data inserted successfully"
    )


@router.post(
    "/ingest_data",
    response_model=APIResponse[str],
)
def ingest_kb_data(
        kb_id: str = Path(..., description="Knowledge base ID"),
        data: dict | list[dict] = Body(..., description="Upload data"),
):
    # Need to get schema based on kb_id
    kb_schema_dict = kb_base_client.get_kb_schema_by_id(kb_id)
    if not kb_schema_dict:
        raise HTTPException(
            status_code=400,
            detail=f"kb_id: [{kb_id}] The corresponding knowledge base does not have a kb schema"
        )
    # Convert Dict to KBSchema object
    try:
        kb_schema = KBSchema(**kb_schema_dict)
        if not check_kb_schema(kb_schema):
            raise HTTPException(
                status_code=400,
                detail="Current knowledge base kb schema validation failed, please check kb schema!"
            )
    except Exception as e:
        logger.error(f"Failed to parse kb_schema: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Knowledge base schema parsing/validation failed: {str(e)}"
        )

    search_result = kb_base_client.kb_info_search_id(kb_id=kb_id)
    if not search_result:
        raise HTTPException(
            status_code=400,
            detail="Failed to query knowledge base information based on knowledge base id"
        )

    # Query knowledge base name based on knowledge base id
    kb_name = kb_base_client.get_kb_name_by_id(kb_id=kb_id)
    if not kb_name:
        raise HTTPException(
            status_code=400,
            detail="Failed to query knowledge base name based on knowledge base id"
        )

    try:
        df = pd.DataFrame(data)
    except ValueError:
        # If it's a scalar dictionary, wrap it in a list
        df = pd.DataFrame([data])
    
    # Convert DataFrame column types according to schema
    # Keep NaN values first, convert types, then handle NaN based on field type
    df = convert_dataframe_types_by_schema(df, kb_schema)
    
    # Add three fixed columns to each row of data here, which need to be correspondingly added when inferring ES and Vearch schemas
    df['kb_id'] = kb_id
    mock_file_id = f"mock_file_{kb_id}"
    df['ori_file_id'] = mock_file_id
    df['chunk_id'] = [str(uuid.uuid4().hex[:16]) for _ in range(len(df))]

    # 2. Write the df data in memory to ES and Vearch
    # 2.1 Write to ES index based on data in df and inferred schema
    es_add_result = kb_file_client.kb_add_df(kb_name=kb_name, df=df)
    if not es_add_result:
        logger.error("add file data into es failed")
        raise HTTPException(
            status_code=500,
            detail="add file data into es failed"
        )
    logger.info(f"add file data into es success")

    # 2.2 First perform embedding processing on some fields based on the inferred schema, save to df, and then write to Vearch space
    if kb_schema.match_rules:
        from core.storer.doc_manager.knowledge_index import VearchVectorMatchPolicy

        # Collect all vector matching policies - iterate through match_rules, then iterate through match_policies in each rule
        vec_matches = [
            policy
            for match_rule in kb_schema.match_rules
            for policy in match_rule.match_policies
            if isinstance(policy, VearchVectorMatchPolicy)
        ]

        if vec_matches:
            vearch_manager = VearchManager(vearch_client=settings.vearch_client)

            # Create embedding model outside the loop to avoid repeated initialization
            embed_model = settings.embedding_model

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
                    embeddings = embed_model.get_text_embedding_batch(texts, show_progress=True)

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
                    # Log failure but don't immediately throw exception, let caller decide whether to rollback
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

    result = kb_file_client.get_kb_file_info(kb_id=kb_id, ori_file_id=mock_file_id)
    ori_file_info = result["_source"] if result else None
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if not ori_file_info:
        # 3. If the current knowledge base does not have this file yet, write the file information to kb_file
        kb_type = search_result[0].get("kb_type", "structured")
        mock_file_type = "md" if kb_type == "unstructured" else "csv"
        kb_file_item = {
            "kb_id": kb_id,
            "ori_file_id": mock_file_id,
            "ori_file_type": mock_file_type,
            "file_name": f"qa_data.{mock_file_type}",  # Add file_name field
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

    return APIResponse(
        code=200,
        msg="success",
        data="Data inserted successfully"
    )
