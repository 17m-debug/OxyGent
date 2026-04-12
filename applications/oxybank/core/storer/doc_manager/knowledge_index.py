from typing import Literal, Union, Annotated

from loguru import logger
from pydantic import BaseModel, Discriminator, Field as PydanticField
from vearch.schema.field import Field
from vearch.schema.index import ScalarIndex, IvfPQIndex
from vearch.schema.space import SpaceSchema
from vearch.utils import DataType, MetricType

from core.config import settings
from core.storer.doc_manager.es_index_manager import ElasticsearchIndexManager


# Field information has a *:1 relationship with a knowledge base
class FieldInfo(BaseModel):
    """Structured knowledge base field information
    """
    field_name: str
    field_type: Literal["string", "integer", "float"]
    field_desc: str


# Match policy base class and subclass design
class MatchPolicy(BaseModel):
    """Match policy base class

    Use mode field as discriminator to support automatic deserialization to specific policy types
    """
    mode: str
    input_fields: list[str]


class PreciseMatchPolicy(MatchPolicy):
    """Precise match policy

    Match corresponding precise match rules based on input field names to generate final query statements
    Confirm fields to query and return based on output field names
    """
    mode: Literal["precise"] = "precise"


class ESTextMatchPolicy(MatchPolicy):
    """ES full-text search policy

    Configuration strategy for using Elasticsearch for full-text search
    """
    mode: Literal["es_text"] = "es_text"


class VearchVectorMatchPolicy(MatchPolicy):
    """Vearch vector match policy

    Use Vearch vector database for vector similarity matching
    Input parameter type is list[str], but currently limited to one input field
    """
    mode: Literal["vearch_vector"] = "vearch_vector"
    embedding_model: str


# Define union type, use discriminator to implement automatic deserialization
MatchPolicyUnion = Annotated[
    Union[PreciseMatchPolicy, ESTextMatchPolicy, VearchVectorMatchPolicy],
    Discriminator("mode")
]

# Match rules, each match rule contains at least one match policy
class MatchRule(BaseModel):
    match_policies: list[MatchPolicyUnion]
    output_fields: list[str]


class ParserConfig(BaseModel):
    """Parser configuration model for configurable text splitting"""
    parser_type: str = PydanticField(default="sentence", description="Parser type (token, sentence, markdown, html, json, extensible)", example="sentence")
    chunk_size: int = PydanticField(default=500, description="Chunk size for splitting", example=1024)
    chunk_overlap: int = PydanticField(default=50, description="Chunk overlap for splitting", example=200)
    separator: str = PydanticField(default=" ", description="Separator for token splitting", example=" ")
    splitter_type: str = PydanticField(default="sentence", description="Splitter type for extensible parser", example="sentence")
    include_metadata: bool = PydanticField(default=True, description="Include metadata in parsed nodes", example=True)
    include_prev_next_rel: bool = PydanticField(default=True, description="Include previous/next relationships", example=True)



class KBSchema(BaseModel):
    fields: list[FieldInfo]
    match_rules: list[MatchRule] | None = None
    parser_config: ParserConfig | None = None




def check_kb_schema(schema: KBSchema):
    """Check if kb schema meets requirements
    1. Fields used in match_rules should all appear in fields
    2. Each match_rule must have exactly one advanced match policy (es/vearch)
    3. Advanced match policy input field can only be one field
    4. Each match_rule can have 0 or more precise match policies as filters for the advanced match policy

    Args:
        schema: KBSchema object to check

    Raises:
        ValueError: Raised when schema does not meet requirements, error message will detail the problem
    """
    if schema.match_rules is None:
        return False # If no match_rules, check fails

    # Build field name set for quick lookup
    field_names = {field.field_name for field in schema.fields}

    # Iterate through each match_rule for checking
    for rule_idx, match_rule in enumerate(schema.match_rules):
        advanced_policies = []
        precise_policies = []

        # Separate advanced match policies and precise match policies
        for policy in match_rule.match_policies:
            if policy.mode in ["es_text", "vearch_vector"]:
                advanced_policies.append(policy)
            elif policy.mode == "precise":
                precise_policies.append(policy)
            else:
                raise ValueError(
                    f"MatchRule {rule_idx} contains unknown match policy type: {policy.mode}"
                )

        # Rule 2: Each match_rule must have exactly one advanced match policy
        if len(advanced_policies) == 0:
            raise ValueError(
                f"MatchRule {rule_idx} missing advanced match policy (es_text or vearch_vector), "
                f"each match rule must have one advanced match policy as primary query"
            )
        elif len(advanced_policies) > 1:
            raise ValueError(
                f"MatchRule {rule_idx} contains {len(advanced_policies)} advanced match policies, "
                f"each match rule can only have one advanced match policy (es_text or vearch_vector)"
            )

        # Rule 3: Advanced match policy input field can only be one field
        advanced_policy = advanced_policies[0]
        if len(advanced_policy.input_fields) != 1:
            raise ValueError(
                f"MatchRule {rule_idx} advanced match policy ({advanced_policy.mode}) has "
                f"{len(advanced_policy.input_fields)} input fields, must have exactly one field"
            )

        # Rule 1: Check if fields used by advanced match policy exist in fields
        advanced_field_name = advanced_policy.input_fields[0]
        if advanced_field_name not in field_names:
            raise ValueError(
                f"MatchRule {rule_idx} advanced match policy uses non-existent field: {advanced_field_name}, "
                f"field not in schema.fields"
            )

        # Rule 1: Check if fields used by precise match policies exist in fields
        for precise_policy in precise_policies:
            for field_name in precise_policy.input_fields:
                if field_name not in field_names:
                    raise ValueError(
                        f"MatchRule {rule_idx} precise match policy uses non-existent field: {field_name}, "
                        f"field not in schema.fields"
                    )

        # Rule 1: Check if output_fields exist in fields
        for field_name in match_rule.output_fields:
            if field_name not in field_names:
                raise ValueError(
                    f"MatchRule {rule_idx} output_fields contains non-existent field: {field_name}, "
                    f"field not in schema.fields"
                )
    return True


# Create corresponding ES index mapping schema based on structured knowledge base schema. Note to add kb_id, ori_file_id, chunk_id by default
def infer_mapping_from_schema(schema: KBSchema):
    """Infer ES index for storing structured data from structured knowledge base schema
    """
    fields = schema.fields
    index_mapping = {}

    # Collect fields requiring full-text search
    text_match_fields = set()
    if schema.match_rules:
        # Iterate through each match rule
        for match_rule in schema.match_rules:
            # Iterate through each match policy in the rule
            for policy in match_rule.match_policies:
                if isinstance(policy, ESTextMatchPolicy) or policy.mode == "es_text":
                    text_match_fields.update(policy.input_fields)

    # Fast fail when no ES-related rules exist, return None
    if not text_match_fields:
        return None

    for field in fields:
        field_mapping = {}
        if field.field_type == "string":
            # Check if field is in full-text search fields
            if field.field_name in text_match_fields:
                # For fields requiring full-text search, use text type with smartcn Chinese word breaker
                field_mapping["type"] = "text"
                field_mapping["analyzer"] = "smartcn"
                # Keep keyword field for precise matching
                field_mapping["fields"] = {
                    "keyword": {
                        "type": "keyword"
                    }
                }
            else:
                # Ordinary strings use keyword type
                field_mapping["type"] = "keyword"
        elif field.field_type == "integer":
            field_mapping["type"] = "long"
        elif field.field_type == "float":
            field_mapping["type"] = "double"
        index_mapping[field.field_name] = field_mapping

    # Add kb_id, ori_file_id, chunk_id storage fields, uniformly use keyword type
    index_mapping["kb_id"] = {"type": "keyword"}
    index_mapping["ori_file_id"] = {"type": "keyword"}
    index_mapping["chunk_id"] = {"type": "keyword"}

    return {"properties": index_mapping}


def infer_vearch_space_schema(schema: KBSchema, kb_name: str):
    """Infer and create vearch space schema from knowledge base schema, may be empty
    """
    space_fields = []
    space_dimension = settings.vearch_config.vector_dimension
    vector_index = IvfPQIndex(
        index_name="gamma",
        metric_type=MetricType.Inner_product,
        ncentroids=2048,
        nsubvector=64
    )

    # Extract vearch search rules from retrieval rules, divided into vector fields and scalar fields
    if not schema.match_rules:
        return None

    vector_fields = set()
    scalar_fields = set()
    for match_rule in schema.match_rules:  # Iterate through retrieval rules
        vearch_rule_flag = False  # Flag indicating if retrieval rule is vearch
        tmp_fields = set()
        for policy in match_rule.match_policies:  # Iterate through retrieval policies
            # Note: input_fields length is 1 by default, so only take first field of policy input
            if isinstance(policy, VearchVectorMatchPolicy) or policy.mode == "vearch_vector":
                vector_fields.add(policy.input_fields[0])
                vearch_rule_flag = True
            if isinstance(policy, PreciseMatchPolicy) or policy.mode == "precise_match":
                tmp_fields.add(policy.input_fields[0])
        if vearch_rule_flag:
            scalar_fields.update(tmp_fields)

    if not vector_fields:  # Fast fail if no vearch retrieval policy exists
        return None

    # 1. Process all scalar fields, divided into indexed and non-indexed
    for field in schema.fields:
        data_type = DataType.STRING
        if field.field_type == "integer":
            data_type = DataType.INTEGER
        elif field.field_type == "float":
            data_type = DataType.FLOAT

        if field.field_name in scalar_fields:
            space_fields.append(
                Field(
                    name=field.field_name,
                    data_type=data_type,
                    desc=field.field_desc,
                    index=ScalarIndex(f"{field.field_name}_idx")
                )
            )
        else:
            space_fields.append(
                Field(
                    name=field.field_name,
                    data_type=data_type,
                    desc=field.field_desc,
                )
            )

    # 2. Add three scalar index fields by default, these are framework reserved fields, users cannot use these three fields in documents. All three fields are string type
    for fixed_field_name in ["kb_id", "ori_file_id", "chunk_id"]:
        space_fields.append(
            Field(
                name=fixed_field_name,
                data_type=DataType.STRING,
                desc=fixed_field_name,
                index=ScalarIndex(f"{fixed_field_name}_idx")
            )
        )

    # 3. Process vector fields
    for field in vector_fields:
        space_fields.append(
            Field(
                name=f"{field}_vector",  # Vector field naming rule: field_name + "_vector"
                data_type=DataType.VECTOR,
                index=vector_index,
                dimension=space_dimension,
            )
        )

    return SpaceSchema(
        name=kb_name,
        fields=space_fields,
        partition_num=1,
        replica_num=3
    )



kb_info_index = {
    "properties": {
        "kb_id": {
            "type": "keyword"
        },
        "kb_name": {
            "type": "keyword"
        },
        "kb_type": {
            "type": "keyword"
        },
        "kb_description": {
            "type": "text"
        },
        "kb_status": {
            "type": "keyword"
        },
        "create_time": {
            "type": "date",
            "format": "yyyy-MM-dd HH:mm:ss"
        },
        "update_time": {
            "type": "date",
            "format": "yyyy-MM-dd HH:mm:ss"
        },
        "kb_create_user": {
            "type": "keyword"
        },
        "kb_update_user": {
            "type": "keyword"
        },
        "kb_store_type": {
            "type": "keyword"
        },
        "kb_extra_info": {
            "type": "object",
            "dynamic": False
        },
        "kb_schema": {
            "type": "object",
            "dynamic": False
        },
        "auto_bind_query": {
            "type": "boolean"
        }
    }
}

kb_file_index = {
    "properties": {
        "ori_file_id": {
            "type": "keyword",
            "doc_values": True
        },
        "kb_id": {
            "type": "keyword",
            "doc_values": True
        },
        "document_md5": {
            "type": "keyword",
            "doc_values": True,
        },
        "create_time": {
            "type": "date",
            "format": "yyyy-MM-dd HH:mm:ss"
        },
        "update_time": {
            "type": "date",
            "format": "yyyy-MM-dd HH:mm:ss"
        },
        "ori_file_type": {
            "type": "text"
        },
        "file_name": {
            "type": "keyword"
        },
        "file_store_mode": {
            "type": "text"
        },
        "file_extra_info": {
            "type": "object"
        },
        "language": {
            "type": "keyword",
            "doc_values": True,
        }
    }
}

index_configs = {
    "knowledge_base_info": kb_info_index,
    "knowledge_file_info": kb_file_index
}


es_index_manager = ElasticsearchIndexManager(settings.es_client)
# Check and create ES indexes. Includes knowledge base table and knowledge metadata table
for k, v in index_configs.items():
    es_index_manager.ensure_index(k, v)
logger.info("Checked existence of knowledge base table and knowledge metadata table indexes")
