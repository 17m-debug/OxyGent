#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
训练数据提取脚本

通过命令行参数提供配置，从 Elasticsearch 中提取训练数据
并转换为 ChatML 格式
"""

import argparse
import asyncio
import json
import os
from typing import Dict, List, Optional, Tuple

from oxygent.config import Config
from oxygent.databases.db_es.base_es import BaseEs
from oxygent.db_factory import DBFactory

async def get_es_client(es_url: str) -> BaseEs:
    """获取 Elasticsearch 客户端"""
    if es_url:
        # 解析 ES URL 并创建客户端
        Config.set_es_config({
            "hosts": [es_url],
            "user": "",
            "password": ""
        })
    db_factory = DBFactory()
    return db_factory.get_instance(
        DBFactory.get_es_class(),
        **Config.get_es_config()
    )

async def get_human_ratings(es_client: BaseEs, app_name: str) -> Dict[str, float]:
    """获取人工评分记录"""
    index_name = f"{app_name}_rating"
    search_body = {
        "query": {
            "term": {
                "source": "human"
            }
        },
        "size": 10000
    }
    result = await es_client.search(index_name, search_body)
    ratings = {}
    for hit in result.get("hits", {}).get("hits", []):
        source = hit.get("_source", {})
        trace_id = source.get("trace_id")
        score = source.get("score")
        if trace_id and score:
            ratings[trace_id] = float(score)
    return ratings

async def get_answer_traces(es_client: BaseEs, app_name: str) -> List[str]:
    """获取所有含最终回答的 trace"""
    index_name = f"{app_name}_message"
    search_body = {
        "query": {
            "term": {
                "message_type": "answer"
            }
        },
        "size": 10000,
        "aggs": {
            "unique_traces": {
                "terms": {
                    "field": "trace_id",
                    "size": 10000
                }
            }
        }
    }
    result = await es_client.search(index_name, search_body)
    buckets = result.get("aggregations", {}).get("unique_traces", {}).get("buckets", [])
    return [bucket.get("key") for bucket in buckets]

async def get_messages_by_trace(es_client: BaseEs, app_name: str, trace_id: str) -> List[Dict]:
    """获取指定 trace 的所有消息"""
    index_name = f"{app_name}_message"
    search_body = {
        "query": {
            "term": {
                "trace_id": trace_id
            }
        },
        "size": 1000,
        "sort": [
            {"message_timestamp": {"order": "asc"}}
        ]
    }
    result = await es_client.search(index_name, search_body)
    messages = []
    for hit in result.get("hits", {}).get("hits", []):
        source = hit.get("_source", {})
        message = json.loads(source.get("message", "{}"))
        message["message_type"] = source.get("message_type")
        messages.append(message)
    return messages

def calculate_rule_score(messages: List[Dict]) -> float:
    """计算规则打分"""
    score = 5.0
    
    # 检查是否有最终回答
    has_answer = any(msg.get("type") == "answer" for msg in messages)
    if not has_answer:
        return 0.5
    
    # 检查回答长度
    answer_msg = next((msg for msg in messages if msg.get("type") == "answer"), None)
    if answer_msg:
        answer_content = answer_msg.get("content", "")
        if len(answer_content) < 15:
            score -= 2.0
        
        # 检查失败关键词
        failure_keywords = ["无法", "不知道", "i don't know", "error"]
        for keyword in failure_keywords:
            if keyword.lower() in answer_content.lower():
                score -= 1.5
                break
    
    # 检查工具执行失败率
    tool_calls = [msg for msg in messages if msg.get("type") == "tool_call"]
    observations = [msg for msg in messages if msg.get("type") == "observation"]
    
    error_count = 0
    for obs in observations:
        output = obs.get("content", {}).get("output", "")
        if "error" in str(output).lower() or "traceback" in str(output).lower():
            error_count += 1
    
    if tool_calls:
        error_rate = error_count / len(tool_calls)
        score -= min(error_rate * 10, 2.0)
    
    # 检查工具调用次数
    if len(tool_calls) > 15:
        score -= 1.5
    elif len(tool_calls) > 10:
        score -= 0.5
    
    # 分数钳制在 [0.0, 5.0]
    return max(0.0, min(5.0, score))

async def calculate_llm_score(judge_api: str, judge_key: str, judge_model: str, user_query: str, assistant_answer: str) -> float:
    """使用 LLM 裁判打分"""
    try:
        import openai
        
        openai.api_key = judge_key
        if judge_api:
            openai.api_base = judge_api
        
        response = openai.ChatCompletion.create(
            model=judge_model,
            messages=[
                {
                    "role": "system",
                    "content": "你是一个公正的裁判，负责对智能体的回答进行评分。请根据以下五个维度对回答进行评估：\n1. 准确性：回答是否正确无误\n2. 完整性：回答是否全面覆盖问题\n3. 有用性：回答对用户是否有帮助\n4. 清晰度：回答是否清晰易懂\n5. 专业性：回答是否专业规范\n请综合考虑这五个维度，给出一个 1-5 之间的整数评分，其中 1 表示非常差，5 表示非常好。只输出数字，不要输出其他内容。"
                },
                {
                    "role": "user",
                    "content": f"问题：{user_query}\n回答：{assistant_answer}"
                }
            ],
            temperature=0,
            max_tokens=1
        )
        
        score = float(response.choices[0].message.content.strip())
        return max(1.0, min(5.0, score))
    except Exception as e:
        print(f"LLM 裁判打分失败: {e}")
        return 3.0

async def calculate_final_score(
    human_ratings: Dict[str, float],
    messages: List[Dict],
    trace_id: str,
    judge_api: Optional[str] = None,
    judge_key: Optional[str] = None,
    judge_model: Optional[str] = None
) -> float:
    """计算最终得分"""
    # 优先使用人工评分
    if trace_id in human_ratings:
        return human_ratings[trace_id]
    
    # 计算规则分
    rule_score = calculate_rule_score(messages)
    
    # 如果规则分 >= 1.5 且配置了 LLM 裁判，则使用 LLM 裁判打分
    if rule_score >= 1.5 and judge_api and judge_key and judge_model:
        # 提取用户问题和智能体回答
        user_query = ""
        assistant_answer = ""
        
        for msg in messages:
            if msg.get("type") == "tool_call" and msg.get("content", {}).get("caller") == "user":
                user_query = msg.get("content", {}).get("arguments", {}).get("query", "")
            elif msg.get("type") == "answer":
                assistant_answer = msg.get("content", "")
        
        if user_query and assistant_answer:
            llm_score = await calculate_llm_score(judge_api, judge_key, judge_model, user_query, assistant_answer)
            return rule_score * 0.4 + llm_score * 0.6
    
    return rule_score

def convert_to_chatml(messages: List[Dict]) -> Optional[Dict]:
    """转换为 ChatML 训练格式"""
    chatml_messages = []
    
    # 提取系统提示（如果存在）
    system_msg = next((msg for msg in messages if msg.get("type") == "system"), None)
    if system_msg:
        chatml_messages.append({
            "role": "system",
            "content": system_msg.get("content", "")
        })
    
    # 提取用户输入
    user_msg = next((msg for msg in messages if msg.get("type") == "tool_call" and msg.get("content", {}).get("caller") == "user"), None)
    if not user_msg:
        return None
    
    chatml_messages.append({
        "role": "user",
        "content": user_msg.get("content", {}).get("arguments", {}).get("query", "")
    })
    
    # 提取工具调用和观察
    tool_calls = []
    for msg in messages:
        if msg.get("type") == "tool_call" and msg.get("content", {}).get("caller") != "user":
            tool_call = msg.get("content", {})
            tool_calls.append({
                "id": tool_call.get("node_id", ""),
                "type": "function",
                "function": {
                    "name": tool_call.get("callee", ""),
                    "arguments": json.dumps(tool_call.get("arguments", {}))
                }
            })
        elif msg.get("type") == "observation":
            if tool_calls:
                chatml_messages.append({
                    "role": "assistant",
                    "content": None,
                    "tool_calls": tool_calls
                })
                chatml_messages.append({
                    "role": "tool",
                    "tool_call_id": tool_calls[-1].get("id", ""),
                    "content": str(msg.get("content", {}).get("output", ""))
                })
                tool_calls = []
    
    # 提取最终回答
    answer_msg = next((msg for msg in messages if msg.get("type") == "answer"), None)
    if not answer_msg:
        return None
    
    chatml_messages.append({
        "role": "assistant",
        "content": answer_msg.get("content", "")
    })
    
    return {"messages": chatml_messages}

def split_dataset(data: List[Dict], train_ratio: float = 0.8, val_ratio: float = 0.1) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """数据集划分"""
    # 按分数降序排列
    data.sort(key=lambda x: x.get("score", 0), reverse=True)
    
    # 随机打乱（保证高分样本优先进入训练集）
    import random
    random.shuffle(data)
    
    total = len(data)
    train_end = int(total * train_ratio)
    val_end = int(total * (train_ratio + val_ratio))
    
    return data[:train_end], data[train_end:val_end], data[val_end:]

def write_jsonl(data: List[Dict], file_path: str):
    """写入 JSONL 文件"""
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

async def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="训练数据提取脚本")
    parser.add_argument("--app", required=True, help="应用名称")
    parser.add_argument("--es", default="http://localhost:9200", help="Elasticsearch 地址")
    parser.add_argument("--out", required=True, help="输出目录")
    parser.add_argument("--min-score", type=float, default=3.5, help="最低分数阈值")
    parser.add_argument("--limit", type=int, default=5000, help="最大样本数")
    parser.add_argument("--judge-api", help="裁判模型 API 地址")
    parser.add_argument("--judge-key", help="裁判模型 API 密钥")
    parser.add_argument("--judge-model", default="gpt-4o-mini", help="裁判模型名称")
    
    args = parser.parse_args()
    
    # 设置应用名称
    Config.set_app_name(args.app)
    
    # 创建输出目录
    os.makedirs(args.out, exist_ok=True)
    
    # 获取 ES 客户端
    es_client = await get_es_client(args.es)
    
    # 获取人工评分
    print("获取人工评分记录...")
    human_ratings = await get_human_ratings(es_client, args.app)
    print(f"获取到 {len(human_ratings)} 条人工评分")
    
    # 获取含最终回答的 trace
    print("获取含最终回答的 trace...")
    trace_ids = await get_answer_traces(es_client, args.app)
    print(f"获取到 {len(trace_ids)} 个含最终回答的 trace")
    
    # 处理每个 trace
    print("处理 trace 数据...")
    samples = []
    from_human = 0
    from_auto = 0
    
    for i, trace_id in enumerate(trace_ids[:args.limit]):
        if i % 100 == 0:
            print(f"处理进度: {i}/{min(len(trace_ids), args.limit)}")
        
        # 获取该 trace 的所有消息
        messages = await get_messages_by_trace(es_client, args.app, trace_id)
        
        # 计算最终得分
        score = await calculate_final_score(
            human_ratings,
            messages,
            trace_id,
            args.judge_api,
            args.judge_key,
            args.judge_model
        )
        
        # 检查分数阈值
        if score < args.min_score:
            continue
        
        # 转换为 ChatML 格式
        chatml = convert_to_chatml(messages)
        if not chatml:
            continue
        
        # 添加分数信息
        chatml["score"] = score
        chatml["trace_id"] = trace_id
        
        samples.append(chatml)
        
        # 统计来源
        if trace_id in human_ratings:
            from_human += 1
        else:
            from_auto += 1
    
    # 数据集划分
    print("数据集划分...")
    train_data, val_data, test_data = split_dataset(samples)
    
    # 写出文件
    print("写出文件...")
    write_jsonl(train_data, os.path.join(args.out, "train.jsonl"))
    write_jsonl(val_data, os.path.join(args.out, "val.jsonl"))
    write_jsonl(test_data, os.path.join(args.out, "test.jsonl"))
    
    # 写出统计信息
    stats = {
        "total": len(samples),
        "splits": {
            "train": len(train_data),
            "val": len(val_data),
            "test": len(test_data)
        },
        "score_mean": sum(s.get("score", 0) for s in samples) / len(samples) if samples else 0,
        "score_min": min(s.get("score", 0) for s in samples) if samples else 0,
        "score_max": max(s.get("score", 0) for s in samples) if samples else 0,
        "from_human": from_human,
        "from_auto": from_auto,
        "generated_at": json.dumps({"$date": "2026-03-20T00:00:00Z"})
    }
    
    with open(os.path.join(args.out, "stats.json"), 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    
    print("\n数据提取完成！")
    print(f"总样本数: {len(samples)}")
    print(f"训练集: {len(train_data)}")
    print(f"验证集: {len(val_data)}")
    print(f"测试集: {len(test_data)}")
    print(f"人工评分样本: {from_human}")
    print(f"自动评分样本: {from_auto}")
    print(f"平均分: {stats['score_mean']:.2f}")
    print(f"最低分: {stats['score_min']:.2f}")
    print(f"最高分: {stats['score_max']:.2f}")
    print(f"输出目录: {args.out}")

if __name__ == "__main__":
    asyncio.run(main())
