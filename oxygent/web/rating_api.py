from fastapi import APIRouter, HTTPException, Depends
from typing import Optional, Dict, Any
import time

from oxygent.config import Config
from oxygent.databases.db_es.base_es import BaseEs
from oxygent.db_factory import DbFactory

rating_router = APIRouter()

async def get_es_client() -> BaseEs:
    """获取 Elasticsearch 客户端"""
    return DbFactory.get_es_client()

@rating_router.post("/rate")
async def rate_message(
    trace_id: str,
    score: float,
    comment: Optional[str] = None,
    user_id: Optional[str] = None,
    accuracy: Optional[float] = None,
    completeness: Optional[float] = None,
    helpfulness: Optional[float] = None,
    es_client: BaseEs = Depends(get_es_client)
):
    """评分接口"""
    # 校验 trace_id 是否存在于消息索引
    app_name = Config.get_app_name()
    message_index = f"{app_name}_message"
    
    # 检查 trace_id 是否存在
    search_body = {
        "query": {
            "term": {
                "trace_id": trace_id
            }
        },
        "size": 1
    }
    
    result = await es_client.search(message_index, search_body)
    if not result.get("hits", {}).get("hits", []):
        raise HTTPException(status_code=404, detail="Trace ID not found")
    
    # 写入评分记录
    rating_index = f"{app_name}_rating"
    rating_body = {
        "trace_id": trace_id,
        "score": score,
        "comment": comment,
        "user_id": user_id,
        "accuracy": accuracy,
        "completeness": completeness,
        "helpfulness": helpfulness,
        "app_name": app_name,
        "rated_at": int(time.time()),
        "source": "human"
    }
    
    await es_client.index(rating_index, trace_id, rating_body)
    
    return {
        "status": "ok",
        "trace_id": trace_id,
        "score": score
    }

@rating_router.get("/rate/{trace_id}")
async def get_rating(
    trace_id: str,
    es_client: BaseEs = Depends(get_es_client)
):
    """获取指定 trace_id 的评分"""
    app_name = Config.get_app_name()
    rating_index = f"{app_name}_rating"
    
    search_body = {
        "query": {
            "term": {
                "trace_id": trace_id
            }
        },
        "sort": [
            {"rated_at": {"order": "desc"}}
        ],
        "size": 1
    }
    
    result = await es_client.search(rating_index, search_body)
    hits = result.get("hits", {}).get("hits", [])
    
    if not hits:
        return {
            "trace_id": trace_id,
            "score": None,
            "comment": None,
            "rated_at": None,
            "has_rating": False
        }
    
    rating_doc = hits[0]["_source"]
    return {
        "trace_id": trace_id,
        "score": rating_doc.get("score"),
        "comment": rating_doc.get("comment"),
        "rated_at": rating_doc.get("rated_at"),
        "has_rating": True
    }

@rating_router.get("/ratings/stats")
async def get_rating_stats(
    es_client: BaseEs = Depends(get_es_client)
):
    """获取评分统计信息"""
    app_name = Config.get_app_name()
    rating_index = f"{app_name}_rating"
    
    search_body = {
        "size": 0,
        "aggs": {
            "total_ratings": {
                "value_count": {
                    "field": "score"
                }
            },
            "average_score": {
                "avg": {
                    "field": "score"
                }
            },
            "distribution": {
                "terms": {
                    "field": "score",
                    "size": 5
                }
            }
        }
    }
    
    result = await es_client.search(rating_index, search_body)
    aggregations = result.get("aggregations", {})
    
    total_ratings = aggregations.get("total_ratings", {}).get("value", 0)
    average_score = aggregations.get("average_score", {}).get("value", 0)
    
    # 构建评分分布
    distribution = {}
    for i in range(1, 6):
        distribution[str(i)] = 0
    
    buckets = aggregations.get("distribution", {}).get("buckets", [])
    for bucket in buckets:
        score = str(int(bucket["key"]))
        count = bucket["doc_count"]
        if score in distribution:
            distribution[score] = count
    
    return {
        "total_ratings": total_ratings,
        "average_score": round(average_score, 2),
        "distribution": distribution
    }
