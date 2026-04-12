# Convert DataFrame field types to frontend display field types
def get_py_type(dtype: str) -> str:
    if dtype == "object":
        return "string"
    elif dtype == "int32" or dtype == "int64":
        return "integer"
    elif dtype == "float32" or dtype == "float64":
        return "float"
    else:
        # TODO: Need to add support for other types, such as datetime and boolean types
        return "string"

