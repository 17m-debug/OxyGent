#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
京东企业文化文档爬虫脚本

从京东官网爬取企业文化相关文档，并存储到知识库中
"""

import asyncio
import json
import os
import re
from typing import List, Dict, Any
from datetime import datetime

import aiohttp
from bs4 import BeautifulSoup


class JDCultureCrawler:
    """京东企业文化文档爬虫"""
    
    def __init__(self, output_dir: str = "./knowledge_data"):
        self.output_dir = output_dir
        self.base_url = "https://www.jd.com"
        self.culture_urls = [
            "https://www.jd.com/about",
            "https://corporate.jd.com/aboutUs",
            "https://corporate.jd.com/culture",
            "https://corporate.jd.com/ourHistory",
            "https://corporate.jd.com/responsibility",
        ]
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
        }
        self.documents = []
        
        os.makedirs(output_dir, exist_ok=True)
    
    async def fetch_page(self, session: aiohttp.ClientSession, url: str) -> str:
        """获取页面内容"""
        try:
            async with session.get(url, headers=self.headers, timeout=30) as response:
                if response.status == 200:
                    return await response.text(encoding='utf-8')
                else:
                    print(f"获取页面失败: {url}, 状态码: {response.status}")
                    return ""
        except Exception as e:
            print(f"请求异常: {url}, 错误: {e}")
            return ""
    
    def clean_text(self, text: str) -> str:
        """清理文本内容"""
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'<[^>]+>', '', text)
        text = text.strip()
        return text
    
    def parse_page(self, html: str, url: str) -> List[Dict[str, Any]]:
        """解析页面内容，提取企业文化相关信息"""
        documents = []
        soup = BeautifulSoup(html, 'html.parser')
        
        title = soup.find('title')
        title_text = self.clean_text(title.get_text()) if title else "京东企业文化"
        
        content_selectors = [
            {'selector': 'div.content', 'type': 'content'},
            {'selector': 'div.article-content', 'type': 'article'},
            {'selector': 'div.main-content', 'type': 'main'},
            {'selector': 'article', 'type': 'article'},
            {'selector': 'div.culture-content', 'type': 'culture'},
            {'selector': 'div.about-content', 'type': 'about'},
        ]
        
        for selector_info in content_selectors:
            elements = soup.select(selector_info['selector'])
            for element in elements:
                text = self.clean_text(element.get_text())
                if len(text) > 100:
                    documents.append({
                        'title': title_text,
                        'content': text,
                        'url': url,
                        'type': selector_info['type'],
                        'crawled_at': datetime.now().isoformat(),
                        'source': 'jd_official'
                    })
        
        if not documents:
            paragraphs = soup.find_all(['p', 'div'], class_=re.compile(r'(content|text|desc|intro)', re.I))
            for p in paragraphs:
                text = self.clean_text(p.get_text())
                if len(text) > 50:
                    documents.append({
                        'title': title_text,
                        'content': text,
                        'url': url,
                        'type': 'paragraph',
                        'crawled_at': datetime.now().isoformat(),
                        'source': 'jd_official'
                    })
        
        return documents
    
    async def crawl_all(self):
        """爬取所有预设的文化页面"""
        print("开始爬取京东企业文化文档...")
        
        connector = aiohttp.TCPConnector(limit=5)
        async with aiohttp.ClientSession(connector=connector) as session:
            tasks = [self.fetch_page(session, url) for url in self.culture_urls]
            pages = await asyncio.gather(*tasks)
            
            for url, html in zip(self.culture_urls, pages):
                if html:
                    print(f"解析页面: {url}")
                    docs = self.parse_page(html, url)
                    self.documents.extend(docs)
                    print(f"  提取到 {len(docs)} 个文档片段")
        
        print(f"\n爬取完成，共获取 {len(self.documents)} 个文档片段")
        return self.documents
    
    def save_documents(self, filename: str = "jd_culture_documents.json"):
        """保存文档到JSON文件"""
        output_path = os.path.join(self.output_dir, filename)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.documents, f, ensure_ascii=False, indent=2)
        print(f"文档已保存到: {output_path}")
        return output_path


class JDCultureKnowledgeBuilder:
    """京东企业文化知识库构建器"""
    
    def __init__(self, documents: List[Dict[str, Any]]):
        self.documents = documents
        self.knowledge_chunks = []
    
    def split_content(self, content: str, max_length: int = 500, overlap: int = 50) -> List[str]:
        """将长文本分割成多个块"""
        if len(content) <= max_length:
            return [content]
        
        chunks = []
        start = 0
        while start < len(content):
            end = start + max_length
            chunk = content[start:end]
            
            if end < len(content):
                last_period = chunk.rfind('。')
                last_question = chunk.rfind('？')
                last_exclaim = chunk.rfind('！')
                split_point = max(last_period, last_question, last_exclaim)
                
                if split_point > start + max_length // 2:
                    chunk = content[start:split_point + 1]
                    end = split_point + 1
            
            chunks.append(chunk.strip())
            start = end - overlap if end < len(content) else end
        
        return chunks
    
    def build_knowledge_chunks(self) -> List[Dict[str, Any]]:
        """构建知识库块"""
        print("开始构建知识库块...")
        
        for doc in self.documents:
            chunks = self.split_content(doc['content'])
            
            for i, chunk in enumerate(chunks):
                knowledge_chunk = {
                    'id': f"{doc['source']}_{len(self.knowledge_chunks)}_{i}",
                    'title': doc['title'],
                    'content': chunk,
                    'url': doc['url'],
                    'type': doc['type'],
                    'source': doc['source'],
                    'crawled_at': doc['crawled_at'],
                    'metadata': {
                        'chunk_index': i,
                        'total_chunks': len(chunks),
                        'content_length': len(chunk)
                    }
                }
                self.knowledge_chunks.append(knowledge_chunk)
        
        print(f"构建完成，共 {len(self.knowledge_chunks)} 个知识块")
        return self.knowledge_chunks
    
    def save_knowledge_chunks(self, output_dir: str, filename: str = "knowledge_chunks.json"):
        """保存知识块到文件"""
        output_path = os.path.join(output_dir, filename)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.knowledge_chunks, f, ensure_ascii=False, indent=2)
        print(f"知识块已保存到: {output_path}")
        return output_path


async def main():
    """主函数"""
    output_dir = "./knowledge_data"
    
    crawler = JDCultureCrawler(output_dir=output_dir)
    documents = await crawler.crawl_all()
    
    if documents:
        crawler.save_documents()
        
        builder = JDCultureKnowledgeBuilder(documents)
        chunks = builder.build_knowledge_chunks()
        builder.save_knowledge_chunks(output_dir)
        
        print("\n=== 爬取摘要 ===")
        print(f"原始文档数: {len(documents)}")
        print(f"知识块数: {len(chunks)}")
        
        print("\n=== 示例知识块 ===")
        for i, chunk in enumerate(chunks[:3]):
            print(f"\n[{i+1}] {chunk['title']}")
            print(f"内容: {chunk['content'][:200]}...")
    else:
        print("未获取到任何文档")


if __name__ == "__main__":
    asyncio.run(main())
