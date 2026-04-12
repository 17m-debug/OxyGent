#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
京东企业文化 RAG 集成示例

演示如何将知识库集成到 OxyGent 系统中
默认使用 OpenAI Embedding API
"""

import asyncio
import json
import os
from typing import List, Dict, Any, Optional

import numpy as np

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from oxygent import MAS, ChatAgent, RAGAgent, OpenAILLM
from oxygent.config import Config


class JDKnowledgeRetriever:
    """京东企业文化知识检索器（基于大模型 Embedding API）"""

    def __init__(
        self,
        vector_store_path: str = "./vector_store",
        api_url: str = "https://api.openai.com/v1/embeddings",
        api_key: Optional[str] = None,
        model_name: str = "text-embedding-3-small",
    ):
        self.vector_store_path = vector_store_path
        self.api_url = api_url
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.model_name = model_name
        self.index: List[Dict[str, Any]] = []
        self.vectors: Optional[np.ndarray] = None
        self._local_model = None
        self._load_store()

    def _load_store(self):
        index_path = os.path.join(self.vector_store_path, "index.json")
        vectors_path = os.path.join(self.vector_store_path, "vectors.npy")

        if os.path.exists(index_path):
            with open(index_path, "r", encoding="utf-8") as f:
                self.index = json.load(f)
            print(f"加载了 {len(self.index)} 条知识记录")

        if os.path.exists(vectors_path):
            self.vectors = np.load(vectors_path)
            print(f"加载了向量矩阵: {self.vectors.shape}")

        meta_path = os.path.join(self.vector_store_path, "meta.json")
        if os.path.exists(meta_path):
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            print(f"嵌入模型: {meta.get('embedding_model', 'unknown')}")

    async def get_embedding(self, text: str) -> np.ndarray:
        if self._local_model is not None:
            loop = asyncio.get_event_loop()
            emb = await loop.run_in_executor(None, self._local_model.encode, [text])
            return np.array(emb[0])

        return await self._call_api([text])

    async def _call_api(self, texts: List[str]) -> np.ndarray:
        import aiohttp

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {"input": texts, "model": self.model_name}

        async with aiohttp.ClientSession() as session:
            async with session.post(
                self.api_url, headers=headers, json=payload, timeout=60
            ) as resp:
                if resp.status != 200:
                    body = await resp.text()
                    raise RuntimeError(f"Embedding API 请求失败: {body}")
                result = await resp.json()
                sorted_data = sorted(result["data"], key=lambda x: x["index"])
                embeddings = [item["embedding"] for item in sorted_data]
                if len(embeddings) == 1:
                    return np.array(embeddings[0])
                return np.array(embeddings)

    async def retrieve(self, query: str, top_k: int = 3) -> str:
        if not self.index or self.vectors is None:
            return "知识库暂无数据，请先运行向量化脚本导入数据。"

        query_embedding = await self.get_embedding(query)

        q_norm = np.linalg.norm(query_embedding)
        v_norms = np.linalg.norm(self.vectors, axis=1)
        safe_norms = np.where(v_norms == 0, 1, v_norms)
        similarities = np.dot(self.vectors, query_embedding) / (safe_norms * q_norm)

        top_indices = np.argsort(similarities)[::-1][:top_k]

        results = []
        for idx in top_indices:
            record = self.index[idx]
            score = float(similarities[idx])
            results.append(f"【{record.get('title', '未知标题')}】(相关度: {score:.2f})\n{record.get('content', '')}")

        return "\n\n---\n\n".join(results)


async def retrieve_jd_knowledge(oxy_request) -> str:
    query = oxy_request.arguments.get("query", "")
    vector_store_path = oxy_request.shared_data.get("vector_store_path", "./vector_store")
    api_url = oxy_request.shared_data.get("embedding_api_url", "https://api.openai.com/v1/embeddings")
    api_key = oxy_request.shared_data.get("embedding_api_key", os.environ.get("OPENAI_API_KEY", ""))
    model_name = oxy_request.shared_data.get("embedding_model_name", "text-embedding-3-small")

    retriever = JDKnowledgeRetriever(
        vector_store_path=vector_store_path,
        api_url=api_url,
        api_key=api_key,
        model_name=model_name,
    )
    return await retriever.retrieve(query, top_k=3)


async def create_rag_agent_example():
    print("=" * 60)
    print("OxyGent RAG Agent 示例")
    print("=" * 60)

    llm = OpenAILLM(
        name="default_llm",
        model_name="gpt-4o-mini"
    )

    rag_agent = RAGAgent(
        name="jd_culture_assistant",
        llm_model="default_llm",
        prompt="""你是一个京东企业文化专家助手。请根据以下知识库信息回答用户问题。

知识库信息：
${knowledge}

请基于知识库信息给出准确、专业的回答。如果知识库中没有相关信息，请诚实告知。""",
        func_retrieve_knowledge=retrieve_jd_knowledge
    )

    async with MAS(name="jd_rag_demo", oxy_space=[llm, rag_agent]) as mas:
        test_queries = [
            "京东的企业文化是什么？",
            "京东的核心价值观有哪些？",
            "京东的发展历程是怎样的？"
        ]

        for query in test_queries:
            print(f"\n用户: {query}")
            print("-" * 40)

            response = await mas.chat_with_agent(
                payload={"query": query},
                send_msg_key=""
            )

            print(f"助手: {response.output}")
            print()


async def create_simple_rag_demo():
    print("=" * 60)
    print("简单 RAG 检索演示")
    print("=" * 60)

    knowledge_retriever = JDKnowledgeRetriever("./vector_store")

    test_queries = [
        "京东的企业文化是什么？",
        "京东的核心价值观有哪些？",
        "京东的发展历程"
    ]

    for query in test_queries:
        print(f"\n查询: {query}")
        print("-" * 40)

        knowledge = await knowledge_retriever.retrieve(query, top_k=2)
        print(f"检索结果:\n{knowledge[:500]}...")
        print()


async def main():
    import argparse

    parser = argparse.ArgumentParser(description="京东企业文化 RAG 集成示例")
    parser.add_argument("--mode", choices=["simple", "full"], default="simple",
                       help="运行模式: simple=简单检索演示, full=完整MAS演示")
    parser.add_argument("--vector-store", default="./vector_store",
                       help="向量存储目录")

    args = parser.parse_args()

    if args.mode == "simple":
        await create_simple_rag_demo()
    else:
        await create_rag_agent_example()


if __name__ == "__main__":
    asyncio.run(main())
