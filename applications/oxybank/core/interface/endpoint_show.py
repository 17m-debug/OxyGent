import types
from typing import cast, get_origin, Union, get_args

from fastapi.routing import APIRoute
from pydantic import BaseModel, Field
from pydantic.fields import FieldInfo
from starlette.routing import BaseRoute


class QueryAPIInfo(BaseModel):
    name: str = Field(..., description="Interface name")
    path: str = Field(..., description="Interface path")
    params: dict = Field(..., description="Interface body parameters")


def get_route_parameters(route: BaseRoute) -> dict:
    field_info = {}
    api_route = cast(APIRoute, route)
    request_type = api_route.body_field.type_

    # field_type_info type is pydantic.fields.FieldInfo, containing field type information
    for field_name, field_type_info in request_type.model_fields.items():
        field_type = get_field_type(field_type_info)
        field_type_str = field_type.__name__ if field_type is not None and isinstance(field_type, type) else "None"

        field_info.update({
            field_name: field_type_str
        })

    return field_info


def get_query_api_info(routes: list[BaseRoute], base_url: str = None) -> list[dict]:
    result = []
    for route in routes:
        route = cast(APIRoute, route)
        path = route.path
        if base_url:
            full_path = f"{base_url.rstrip('/')}{path}"
        else:
            full_path = path

        query_api_info = QueryAPIInfo(
            name=route.name,
            path=full_path,
            params=get_route_parameters(route)
        )
        result.append(query_api_info)

    return result


def get_field_type(field_type_info: FieldInfo):
    field_annotation = field_type_info.annotation
    origin = get_origin(field_annotation)

    field_type = None

    if origin is not Union:
        if field_annotation is None or field_annotation is type(None) or field_annotation is types.NoneType:
            field_type = None
        else:
            field_type = field_annotation
    else:
        # Get all types in union type
        args = get_args(field_annotation)
        non_none_types = [
            arg for arg in args
            if arg is not None
               and arg is not type(None)
               and arg is not types.NoneType
        ]

        if non_none_types:
            # Default to the first non-None type
            field_type = non_none_types[0]
    return field_type
