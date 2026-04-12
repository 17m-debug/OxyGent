#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
知识库向量化存储脚本

将爬取的文档进行向量化处理并存储到向量数据库
默认使用 OpenAI Embedding API，可选本地模型
"""

import argparse
import asyncio
import json
import os
from typing import List, Dict, Any, Optional

import numpy as np


class LLMEmbedding:
    """基于大模型 API 的文本嵌入"""

    def __init__(
        self,
        api_url: str = "https://api.openai.com/v1/embeddings",
        api_key: Optional[str] = None,
        model_name: str = "text-embedding-3-small",
        local_model_name: Optional[str] = None,
        batch_size: int = 32,
    ):
        self.api_url = api_url
        self.api_key = api_key
        self.model_name = model_name
        self.batch_size = batch_size
        self._local_model = None
        self._use_api = True

        if local_model_name:
            try:
                from sentence_transformers import SentenceTransformer
                self._local_model = SentenceTransformer(local_model_name)
                self._use_api = False
                print(f"使用本地嵌入模型: {local_model_name}")
            except ImportError:
                print("sentence-transformers 未安装，回退到 API 模式")

        if self._use_api:
            if not self.api_key:
                raise ValueError(
                    "使用 API 模式需要提供 --api-key 参数。"
                    "\n用法: python vectorize_knowledge.py --api-key sk-xxx ..."
                    "\n或设置环境变量 OPENAI_API_KEY"
                )
            print(f"使用 API 嵌入模型: {model_name}")

    @property
    def dimension(self) -> int:
        dim_map = {
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
            "text-embedding-ada-002": 1536,
        }
        if self._use_api:
            return dim_map.get(self.model_name, 1536)
        if self._local_model is not None:
            return self._local_model.get_sentence_embedding_dimension()
        return 384

    async def get_embeddings(self, texts: List[str]) -> np.ndarray:
        if self._local_model is not None:
            loop = asyncio.get_event_loop()
            embeddings = await loop.run_in_executor(
                None, self._local_model.encode, texts
            )
            return np.array(embeddings)

        return await self._call_api(texts)

    async def _call_api(self, texts: List[str]) -> np.ndarray:
        import aiohttp

        all_embeddings = []

        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }
            payload = {"input": batch, "model": self.model_name}

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.api_url, headers=headers, json=payload, timeout=60
                ) as resp:
                    if resp.status != 200:
                        body = await resp.text()
                        raise RuntimeError(
                            f"Embedding API 请求失败 (HTTP {resp.status}): {body}"
                        )
                    result = await resp.json()
                    sorted_data = sorted(result["data"], key=lambda x: x["index"])
                    batch_emb = [item["embedding"] for item in sorted_data]
                    all_embeddings.extend(batch_emb)

            if i + self.batch_size < len(texts):
                print(f"  API 批次 {i // self.batch_size + 1} 完成，已处理 {i + self.batch_size}/{len(texts)}")

        return np.array(all_embeddings)


class KnowledgeVectorStore:
    """知识库向量存储"""

    def __init__(self, store_path: str = "./vector_store"):
        self.store_path = store_path
        self.index_file = os.path.join(store_path, "index.json")
        self.vectors_file = os.path.join(store_path, "vectors.npy")
        self.meta_file = os.path.join(store_path, "meta.json")

        os.makedirs(store_path, exist_ok=True)

        self.index: List[Dict[str, Any]] = []
        self.vectors: Optional[np.ndarray] = None

        self._load_index()

    def _load_index(self):
        if os.path.exists(self.index_file):
            with open(self.index_file, "r", encoding="utf-8") as f:
                self.index = json.load(f)
            print(f"加载现有索引: {len(self.index)} 条记录")

        if os.path.exists(self.vectors_file):
            self.vectors = np.load(self.vectors_file)
            print(f"加载现有向量: shape {self.vectors.shape}")

    def _save_index(self, embedding_model_name: str = "", embedding_dim: int = 0):
        with open(self.index_file, "w", encoding="utf-8") as f:
            json.dump(self.index, f, ensure_ascii=False, indent=2)

        if self.vectors is not None:
            np.save(self.vectors_file, self.vectors)

        meta = {
            "embedding_model": embedding_model_name,
            "embedding_dim": embedding_dim or (
                self.vectors.shape[1] if self.vectors is not None else 0
            ),
            "total_docs": len(self.index),
        }
        with open(self.meta_file, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        print(f"索引已保存: {len(self.index)} 条记录, 向量维度: {meta['embedding_dim']}")

    async def add_documents(
        self,
        documents: List[Dict[str, Any]],
        embeddings: np.ndarray,
        embedding_model_name: str = "",
    ):
        start_idx = len(self.index)

        for i, doc in enumerate(documents):
            record = {
                "id": doc.get("id", f"doc_{start_idx + i}"),
                "title": doc.get("title", ""),
                "content": doc.get("content", ""),
                "url": doc.get("url", ""),
                "type": doc.get("type", ""),
                "source": doc.get("source", ""),
                "metadata": doc.get("metadata", {}),
                "vector_idx": start_idx + i,
            }
            self.index.append(record)

        if self.vectors is None:
            self.vectors = embeddings
        else:
            self.vectors = np.vstack([self.vectors, embeddings])

        self._save_index(
            embedding_model_name=embedding_model_name,
            embedding_dim=embeddings.shape[1],
        )

    async def search(
        self, query_embedding: np.ndarray, top_k: int = 5
    ) -> List[Dict[str, Any]]:
        if self.vectors is None or len(self.index) == 0:
            return []

        query_embedding = query_embedding.flatten()

        q_norm = np.linalg.norm(query_embedding)
        v_norms = np.linalg.norm(self.vectors, axis=1)
        safe_norms = np.where(v_norms == 0, 1, v_norms)
        similarities = np.dot(self.vectors, query_embedding) / (safe_norms * q_norm)

        top_indices = np.argsort(similarities)[::-1][:top_k]

        results = []
        for idx in top_indices:
            result = self.index[idx].copy()
            result["score"] = float(similarities[idx])
            results.append(result)

        return results


async def process_knowledge_chunks(
    chunks_file: str,
    output_dir: str,
    api_url: str = "https://api.openai.com/v1/embeddings",
    api_key: Optional[str] = None,
    model_name: str = "text-embedding-3-small",
    local_model_name: Optional[str] = None,
    batch_size: int = 32,
):
    print(f"加载知识块文件: {chunks_file}")
    with open(chunks_file, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    print(f"加载了 {len(chunks)} 个知识块")

    embedding = LLMEmbedding(
        api_url=api_url,
        api_key=api_key,
        model_name=model_name,
        local_model_name=local_model_name,
        batch_size=batch_size,
    )

    vector_store = KnowledgeVectorStore(store_path=output_dir)

    print("开始向量化处理...")
    all_embeddings = []

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        texts = [chunk["content"] for chunk in batch]

        print(f"处理批次 {i // batch_size + 1}/{(len(chunks) + batch_size - 1) // batch_size}")

        emb = await embedding.get_embeddings(texts)
        all_embeddings.append(emb)

    all_embeddings = np.vstack(all_embeddings)
    print(f"向量化完成，shape: {all_embeddings.shape}")

    effective_model = local_model_name if local_model_name else model_name
    await vector_store.add_documents(chunks, all_embeddings, embedding_model_name=effective_model)

    print("\n=== 处理摘要 ===")
    print(f"知识块数: {len(chunks)}")
    print(f"向量维度: {all_embeddings.shape[1]}")
    print(f"嵌入模型: {effective_model}")
    print(f"存储路径: {output_dir}")

    return vector_store, embedding


async def test_search(vector_store: KnowledgeVectorStore, embedding: LLMEmbedding):
    test_queries = [
        "京东的企业文化是什么？",
        "京东的核心价值观有哪些？",
        "京东的发展历程是怎样的？",
    ]

    print("\n=== 测试搜索 ===")
    for query in test_queries:
        print(f"\n查询: {query}")
        query_embedding = await embedding.get_embeddings([query])
        results = await vector_store.search(query_embedding[0], top_k=3)

        for i, result in enumerate(results):
            print(f"  [{i+1}] 分数: {result['score']:.4f}")
            print(f"      标题: {result['title']}")
            print(f"      内容: {result['content'][:100]}...")


async def main():
    parser = argparse.ArgumentParser(description="知识库向量化存储")
    parser.add_argument("--chunks-file", required=True, help="知识块JSON文件路径")
    parser.add_argument("--output-dir", default="./vector_store", help="向量存储目录")
    parser.add_argument(
        "--api-url",
        default="https://api.openai.com/v1/embeddings",
        help="Embedding API 地址",
    )
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY"), help="API 密钥")
    parser.add_argument(
        "--model-name",
        default="text-embedding-3-small",
        help="API 嵌入模型名称 (text-embedding-3-small / text-embedding-3-large / text-embedding-ada-002)",
    )
    parser.add_argument("--local-model", default=None, help="本地模型名称（优先于API）")
    parser.add_argument("--batch-size", type=int, default=32, help="批处理大小")
    parser.add_argument("--test", action="store_true", help="运行搜索测试")

    args = parser.parse_args()

    vector_store, embedding = await process_knowledge_chunks(
        chunks_file=args.chunks_file,
        output_dir=args.output_dir,
        api_url=args.api_url,
        api_key=args.api_key,
        model_name=args.model_name,
        local_model_name=args.local_model,
        batch_size=args.batch_size,
    )

    if args.test:
        await test_search(vector_store, embedding)


if __name__ == "__main__":
    asyncio.run(main())
