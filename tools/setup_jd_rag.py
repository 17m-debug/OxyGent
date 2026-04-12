#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
京东企业文化 RAG 一键部署脚本

整合爬取、向量化、部署的完整流程
"""

import asyncio
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


async def run_step(step_name: str, step_func, *args, **kwargs):
    print(f"\n{'='*60}")
    print(f"步骤: {step_name}")
    print('='*60)
    try:
        result = await step_func(*args, **kwargs)
        print(f"✓ {step_name} 完成")
        return result
    except Exception as e:
        print(f"✗ {step_name} 失败: {e}")
        raise


async def step_crawl(output_dir: str):
    from crawl_jd_culture import JDCultureCrawler, JDCultureKnowledgeBuilder

    crawler = JDCultureCrawler(output_dir=output_dir)
    documents = await crawler.crawl_all()

    if documents:
        crawler.save_documents()
        builder = JDCultureKnowledgeBuilder(documents)
        chunks = builder.build_knowledge_chunks()
        builder.save_knowledge_chunks(output_dir)
        return len(chunks)
    return 0


async def step_vectorize(chunks_file: str, output_dir: str, api_key: str, api_url: str, model_name: str):
    from vectorize_knowledge import process_knowledge_chunks

    vector_store, _ = await process_knowledge_chunks(
        chunks_file=chunks_file,
        output_dir=output_dir,
        api_url=api_url,
        api_key=api_key,
        model_name=model_name,
    )
    return len(vector_store.index)


async def step_test_rag(vector_store_path: str, api_key: str, api_url: str, model_name: str):
    from rag_demo import JDKnowledgeRetriever

    retriever = JDKnowledgeRetriever(
        vector_store_path=vector_store_path,
        api_url=api_url,
        api_key=api_key,
        model_name=model_name,
    )

    test_queries = [
        "京东的企业文化是什么？",
        "京东的核心价值观有哪些？",
    ]

    print("\n测试检索结果:")
    for query in test_queries:
        print(f"\n查询: {query}")
        result = await retriever.retrieve(query, top_k=2)
        print(f"结果: {result[:200]}...")

    return True


async def main():
    parser = argparse.ArgumentParser(description="京东企业文化 RAG 一键部署")
    parser.add_argument("--output-dir", default="./knowledge_data", help="输出目录")
    parser.add_argument("--vector-dir", default="./vector_store", help="向量存储目录")
    parser.add_argument("--api-url", default="https://api.openai.com/v1/embeddings", help="Embedding API 地址")
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", ""), help="API 密钥")
    parser.add_argument("--model-name", default="text-embedding-3-small", help="嵌入模型名称")
    parser.add_argument("--skip-crawl", action="store_true", help="跳过爬取步骤")
    parser.add_argument("--skip-vectorize", action="store_true", help="跳过向量化步骤")
    parser.add_argument("--test-only", action="store_true", help="仅运行测试")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.vector_dir, exist_ok=True)

    chunks_file = os.path.join(args.output_dir, "knowledge_chunks.json")

    if args.test_only:
        await run_step(
            "测试RAG检索", step_test_rag,
            args.vector_dir, args.api_key, args.api_url, args.model_name,
        )
        return

    if not args.skip_crawl:
        await run_step("爬取京东企业文化文档", step_crawl, args.output_dir)

    if not args.skip_vectorize:
        if os.path.exists(chunks_file):
            await run_step(
                "向量化知识库", step_vectorize,
                chunks_file, args.vector_dir, args.api_key, args.api_url, args.model_name,
            )
        else:
            print(f"错误: 知识块文件不存在: {chunks_file}")
            print("请先运行爬取步骤或使用 --skip-crawl 跳过")
            return

    await run_step(
        "测试RAG检索", step_test_rag,
        args.vector_dir, args.api_key, args.api_url, args.model_name,
    )

    print("\n" + "="*60)
    print("RAG 部署完成！")
    print("="*60)
    print(f"\n知识库文件: {chunks_file}")
    print(f"向量存储: {args.vector_dir}")
    print("\n使用方法:")
    print("  python tools/rag_demo.py --mode simple")
    print("  python tools/rag_demo.py --mode full")


if __name__ == "__main__":
    asyncio.run(main())
