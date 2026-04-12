"""
File recursive processing utility

Provides recursive traversal of files and directories, MD5 calculation, file type detection, etc.
Supports the document ingestion process of the RAG system.
"""

import hashlib
import json
import os
from pathlib import Path
from typing import List, Dict, Optional, Set

from utils.hash_util import str_to_md5

# Supported file types set
SUPPORTED_FILE_TYPES = {
    "txt",
    "md",
    "markdown",
    "rst",
    "csv",
    "xlsx",
    "xls",
    "pdf",
    "docx",
    "doc",
}


def calculate_file_md5(file_path: str) -> str:
    """
    Calculate MD5 hash value of a file

    Args:
        file_path: Absolute path of the file

    Returns:
        str: MD5 hash value of the file (32-character hexadecimal string)

    Raises:
        FileNotFoundError: File does not exist
        PermissionError: No read permission
    """
    md5_hash = hashlib.md5()

    try:
        with open(file_path, "rb") as f:
            # Read file in chunks to avoid excessive memory usage for large files
            for chunk in iter(lambda: f.read(8192), b""):
                md5_hash.update(chunk)
        return md5_hash.hexdigest()
    except FileNotFoundError:
        raise FileNotFoundError(f"File does not exist: {file_path}")
    except PermissionError:
        raise PermissionError(f"No read permission: {file_path}")


def extract_file_type(file_path: str) -> str:
    """
    Extract file type (extension)

    Args:
        file_path: File path

    Returns:
        str: File extension (lowercase, without dot)
    """
    # Use pathlib to get suffix, remove dot and convert to lowercase
    suffix = Path(file_path).suffix.lstrip(".").lower()
    return suffix


def is_supported_file(
    file_path: str, supported_types: Optional[Set[str]] = None
) -> bool:
    """
    Check if file type is supported

    Args:
        file_path: File path
        supported_types: Set of supported file types, defaults to SUPPORTED_FILE_TYPES

    Returns:
        bool: Whether file type is supported
    """
    if supported_types is None:
        supported_types = SUPPORTED_FILE_TYPES

    file_type = extract_file_type(file_path)
    return file_type in supported_types


def get_file_info(kb_id: str, kb_rel_path: str, file_path: str) -> Dict[str, str]:
    """
    Process a single file and generate file information dictionary

    Args:
        kb_id: Knowledge base ID
        kb_rel_path: File path relative to knowledge base
        file_path: Absolute path of the file

    Returns:
        Dict: Dictionary containing file information, including:
            - ori_file_id: MD5 of file's relative path in knowledge base
            - file_name: File name
            - file_path: Absolute file path
            - document_md5: Document MD5 value for deduplication
            - ori_file_type: File type

    Raises:
        Exception: Raised when file processing fails
    """
    try:
        # Ensure path is absolute
        abs_path = os.path.abspath(file_path)

        # Calculate MD5 for deduplication
        md5_value = calculate_file_md5(abs_path)

        # Extract file information
        file_name = os.path.basename(abs_path)
        file_type = extract_file_type(abs_path)

        # Generate file ID based on relative path in knowledge base
        file_id = str_to_md5(kb_rel_path)

        return {
            "ori_file_id": file_id,
            "kb_id": kb_id,
            "file_name": file_name,
            "file_kb_path": kb_rel_path,
            "file_path": abs_path,
            "document_md5": md5_value,
            "ori_file_type": file_type if file_type else "",
            "file_store_mode": "unstructured",
            "file_extra_info": {},
            "language": "zh"
        }
    except Exception as e:
        raise Exception(f"Failed to process file {file_path}: {str(e)}")

def get_relative_path(directory_path: str, file_path: str) -> str:
    """
    Get relative path of a file
    """
    directory_path = os.path.normpath(directory_path)
    file_path = os.path.normpath(file_path)

    dir_name = os.path.basename(directory_path)

    rel_path = os.path.relpath(file_path, directory_path)

    return os.path.join("/", dir_name, rel_path)


def traverse_directory(
    directory_path: str,
    supported_types: Optional[Set[str]] = None,
    skip_hidden: bool = True,
) -> List[Dict[str, str]]:
    """
    Recursively traverse directory and collect all supported file information

    Args:
        directory_path: Directory path
        supported_types: Set of supported file types, defaults to SUPPORTED_FILE_TYPES
        skip_hidden: Whether to skip hidden files and directories, defaults to True

    Returns:
        List[Dict]: List of file information, each element is a file information dictionary

    Raises:
        NotADirectoryError: Path is not a directory
        PermissionError: No access permission
    """
    if not os.path.isdir(directory_path):
        raise NotADirectoryError(f"Path is not a directory: {directory_path}")

    if supported_types is None:
        supported_types = SUPPORTED_FILE_TYPES

    file_list = []
    errors = []

    kb_id = str_to_md5(Path(directory_path).name)

    # Use os.walk to recursively traverse directory
    for root, dirs, files in os.walk(directory_path):
        # Skip hidden directories
        if skip_hidden:
            # Modify dirs list to skip hidden directories (starting with .)
            dirs[:] = [d for d in dirs if not d.startswith(".")]

        for file_name in files:
            # Skip hidden files
            if skip_hidden and file_name.startswith("."):
                continue

            file_path = os.path.join(root, file_name)

            # Check if file type is supported
            if not is_supported_file(file_path, supported_types):
                continue

            try:
                # Use the path from knowledge base directory to current file as relative path
                # For example, if the input knowledge base path is "/root/.../kb_data/user_kb_1"
                # and the current file path is "/root/.../kb_data/user_kb_1/folder_1/demo.md"
                # then the relative path for this file is "/user_kb_1/folder_1/demo.md"
                kb_rel_path = get_relative_path(directory_path, file_path)
                file_info = get_file_info(kb_id, kb_rel_path, file_path)
                file_list.append(file_info)
            except Exception as e:
                # Log error but continue processing other files
                errors.append(f"Failed to process file {file_path}: {str(e)}")

    # Print warning if there are errors
    if errors:
        print(f"Warning: Encountered {len(errors)} errors during processing:")
        for error in errors[:10]:  # Only print first 10 errors
            print(f"  - {error}")
        if len(errors) > 10:
            print(f"  ... and {len(errors) - 10} other errors")

    return file_list


def save_to_json_file(file_list: List[Dict[str, str]], output_path: str) -> None:
    """
    Save file list as JSON file

    Args:
        file_list: List of file information
        output_path: Output JSON file path
    """
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(file_list, f, ensure_ascii=False, indent=2)
    print(f"Results saved to: {output_path}")
