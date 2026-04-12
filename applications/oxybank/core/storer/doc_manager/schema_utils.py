"""
Utility functions for schema operations.
"""
import pandas as pd
from fastapi import HTTPException
from loguru import logger

from core.storer.doc_manager.knowledge_index import KBSchema


def convert_dataframe_types_by_schema(df: pd.DataFrame, kb_schema: KBSchema) -> pd.DataFrame:
    """
    Convert DataFrame column types according to knowledge base schema.

    Process flow:
    1. Keep NaN values as-is (don't convert to empty string yet)
    2. Convert column types based on schema field_type
    3. For string columns: convert NaN to empty string ""
    4. For integer/float columns: raise error if NaN exists

    Args:
        df: DataFrame to convert
        kb_schema: Knowledge base schema containing field definitions

    Returns:
        Converted DataFrame

    Raises:
        HTTPException: If integer/float columns contain NaN values
    """
    if not kb_schema.fields:
        logger.warning("Schema has no fields defined, skipping type conversion")
        return df

    # Process each field defined in schema
    for field_info in kb_schema.fields:
        field_name = field_info.field_name
        field_type = field_info.field_type

        # Skip if field doesn't exist in DataFrame
        if field_name not in df.columns:
            logger.warning(f"Field '{field_name}' defined in schema but not found in DataFrame, skipping")
            continue

        try:
            if field_type == "string":
                # String type: convert to string, NaN -> ""
                df[field_name] = df[field_name].fillna('').astype(str)
                logger.debug(f"Converted field '{field_name}' to string type")

            elif field_type == "integer":
                # Integer type: convert and check for NaN after conversion
                # First check for existing NaN
                if df[field_name].isnull().any():
                    nan_count = df[field_name].isnull().sum()
                    raise HTTPException(
                        status_code=400,
                        detail=(
                            f"Field '{field_name}' is defined as integer type in schema, "
                            f"but contains {nan_count} empty/null value(s). "
                            f"Integer fields cannot have empty values. "
                            f"Please ensure all rows have valid integer values for field '{field_name}'."
                        )
                    )

                # Convert to integer (use Int64 for nullable integer support)
                # errors='coerce' will convert invalid values to NaN
                converted = pd.to_numeric(df[field_name], errors='coerce')

                # Check if conversion produced any NaN (invalid values)
                if converted.isnull().any():
                    invalid_count = converted.isnull().sum()
                    invalid_values = df[field_name][converted.isnull()].unique().tolist()
                    raise HTTPException(
                        status_code=400,
                        detail=(
                            f"Field '{field_name}' is defined as integer type in schema, "
                            f"but contains {invalid_count} invalid value(s) that cannot be converted to integer: {invalid_values}. "
                            f"Please ensure all values in field '{field_name}' are valid integers."
                        )
                    )

                df[field_name] = converted.astype('Int64')
                logger.debug(f"Converted field '{field_name}' to integer type")

            elif field_type == "float":
                # Float type: convert and check for NaN after conversion
                # First check for existing NaN
                if df[field_name].isnull().any():
                    nan_count = df[field_name].isnull().sum()
                    raise HTTPException(
                        status_code=400,
                        detail=(
                            f"Field '{field_name}' is defined as float type in schema, "
                            f"but contains {nan_count} empty/null value(s). "
                            f"Float fields cannot have empty values. "
                            f"Please ensure all rows have valid float values for field '{field_name}'."
                        )
                    )

                # Convert to float
                # errors='coerce' will convert invalid values to NaN
                converted = pd.to_numeric(df[field_name], errors='coerce')

                # Check if conversion produced any NaN (invalid values)
                if converted.isnull().any():
                    invalid_count = converted.isnull().sum()
                    invalid_values = df[field_name][converted.isnull()].unique().tolist()
                    raise HTTPException(
                        status_code=400,
                        detail=(
                            f"Field '{field_name}' is defined as float type in schema, "
                            f"but contains {invalid_count} invalid value(s) that cannot be converted to float: {invalid_values}. "
                            f"Please ensure all values in field '{field_name}' are valid float numbers."
                        )
                    )

                df[field_name] = converted.astype(float)
                logger.debug(f"Converted field '{field_name}' to float type")

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Failed to convert field '{field_name}' to type '{field_type}': {e}")
            raise HTTPException(
                status_code=400,
                detail=f"Failed to convert field '{field_name}' to type '{field_type}': {str(e)}"
            )

    return df
