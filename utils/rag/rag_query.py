"""
RAG查询接口模块
提供高级的RAG查询功能，包括上下文增强、查询重写等
"""

import asyncio
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import re

from .rag_knowledge_base import RAGKnowledgeBase, RAGDocument
from ..logger import get_logger

logger = get_logger(__name__)


@dataclass
class QueryResult:
    """查询结果"""
    query: str
    documents: List[Tuple[RAGDocument, float]]
    context: str
    answer: Optional[str] = None
    sources: List[Dict[str, Any]] = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.sources is None:
            self.sources = []
        if self.metadata is None:
            self.metadata = {}


@dataclass
class QueryContext:
    """查询上下文"""
    relevant_docs: List[Tuple[RAGDocument, float]]
    expanded_query: Optional[str] = None
    query_intent: Optional[str] = None
    similar_queries: List[str] = None

    def __post_init__(self):
        if self.similar_queries is None:
            self.similar_queries = []


class RAGQueryEngine:
    """RAG查询引擎"""

    def __init__(self,
                 knowledge_base: RAGKnowledgeBase,
                 max_context_length: int = 4000,
                 similarity_threshold: float = 0.6,
                 max_documents: int = 5):
        """
        初始化查询引擎

        Args:
            knowledge_base: 知识库实例
            max_context_length: 最大上下文长度
            similarity_threshold: 相似度阈值
            max_documents: 最大文档数量
        """
        self.knowledge_base = knowledge_base
        self.max_context_length = max_context_length
        self.similarity_threshold = similarity_threshold
        self.max_documents = max_documents

        logger.info("RAG查询引擎初始化完成")

    async def query(self,
                   query_text: str,
                   notebook_id: Optional[str] = None,
                   include_sources: bool = True,
                   **kwargs) -> QueryResult:
        """
        执行查询

        Args:
            query_text: 查询文本
            notebook_id: 笔记本ID过滤
            include_sources: 是否包含来源信息
            **kwargs: 其他查询参数

        Returns:
            QueryResult: 查询结果
        """
        logger.info(f"执行查询: {query_text}")

        # 查询预处理
        processed_query = self._preprocess_query(query_text)

        # 执行相似性搜索
        documents = await self.knowledge_base.search_similar_documents(
            query=processed_query,
            notebook_id=notebook_id,
            n_results=self.max_documents,
            score_threshold=self.similarity_threshold
        )

        # 构建上下文
        context = self._build_context(documents)

        # 构建查询结果
        result = QueryResult(
            query=query_text,
            documents=documents,
            context=context
        )

        # 添加来源信息
        if include_sources:
            result.sources = self._extract_sources(documents)

        # 添加元数据
        result.metadata = {
            "notebook_id": notebook_id,
            "document_count": len(documents),
            "context_length": len(context),
            "processed_query": processed_query
        }

        logger.info(f"查询完成，找到 {len(documents)} 个相关文档")
        return result

    async def contextual_query(self,
                              query_text: str,
                              notebook_id: Optional[str] = None,
                              context_note_ids: Optional[List[str]] = None,
                              **kwargs) -> QueryResult:
        """
        上下文增强查询

        Args:
            query_text: 查询文本
            notebook_id: 笔记本ID
            context_note_ids: 上下文笔记ID列表
            **kwargs: 其他参数

        Returns:
            QueryResult: 查询结果
        """
        logger.info(f"执行上下文增强查询: {query_text}")

        # 首先执行普通查询
        result = await self.query(query_text, notebook_id, **kwargs)

        # 添加指定笔记的上下文
        if context_note_ids:
            context_docs = []
            for note_id in context_note_ids:
                docs = await self.knowledge_base.get_note_context(note_id)
                context_docs.extend([(doc, 1.0) for doc in docs])

            # 合并上下文文档
            all_docs = result.documents + context_docs
            # 去重并重新排序
            unique_docs = self._deduplicate_documents(all_docs)
            result.documents = unique_docs[:self.max_documents]
            result.context = self._build_context(result.documents)

        return result

    async def multi_query(self,
                         queries: List[str],
                         notebook_id: Optional[str] = None,
                         combine_strategy: str = "union",
                         **kwargs) -> QueryResult:
        """
        多查询执行

        Args:
            queries: 查询列表
            notebook_id: 笔记本ID
            combine_strategy: 合并策略 ("union", "intersection", "weighted")
            **kwargs: 其他参数

        Returns:
            QueryResult: 合并的查询结果
        """
        logger.info(f"执行多查询: {queries}")

        all_documents = []
        query_results = []

        # 执行每个查询
        for query in queries:
            result = await self.query(query, notebook_id, **kwargs)
            query_results.append(result)
            all_documents.extend(result.documents)

        # 根据策略合并结果
        if combine_strategy == "union":
            # 并集：去重后按相似度排序
            unique_docs = self._deduplicate_documents(all_documents)
            unique_docs.sort(key=lambda x: x[1], reverse=True)
            documents = unique_docs[:self.max_documents]

        elif combine_strategy == "intersection":
            # 交集：只保留所有查询都返回的文档
            doc_sets = []
            for result in query_results:
                doc_ids = {doc[0].note_id for doc in result.documents}
                doc_sets.append(doc_ids)

            if doc_sets:
                common_ids = set.intersection(*doc_sets)
                documents = [doc for doc in all_documents if doc[0].note_id in common_ids]
                documents = self._deduplicate_documents(documents)
                documents.sort(key=lambda x: x[1], reverse=True)
            else:
                documents = []

        elif combine_strategy == "weighted":
            # 加权：根据出现频率调整权重
            doc_weights = {}
            for doc, similarity in all_documents:
                doc_id = doc.note_id
                if doc_id not in doc_weights:
                    doc_weights[doc_id] = {"doc": doc, "count": 0, "total_sim": 0}
                doc_weights[doc_id]["count"] += 1
                doc_weights[doc_id]["total_sim"] += similarity

            # 计算加权相似度
            weighted_docs = []
            for data in doc_weights.values():
                weighted_sim = data["total_sim"] / data["count"]
                weighted_docs.append((data["doc"], weighted_sim))

            weighted_docs.sort(key=lambda x: x[1], reverse=True)
            documents = weighted_docs[:self.max_documents]

        else:
            documents = all_documents[:self.max_documents]

        # 构建合并结果
        context = self._build_context(documents)
        result = QueryResult(
            query=" | ".join(queries),
            documents=documents,
            context=context,
            sources=self._extract_sources(documents),
            metadata={
                "notebook_id": notebook_id,
                "document_count": len(documents),
                "context_length": len(context),
                "combine_strategy": combine_strategy,
                "original_queries": queries
            }
        )

        logger.info(f"多查询完成，合并策略: {combine_strategy}，文档数量: {len(documents)}")
        return result

    def _preprocess_query(self, query: str) -> str:
        """
        查询预处理

        Args:
            query: 原始查询

        Returns:
            str: 处理后的查询
        """
        # 去除多余空格
        query = re.sub(r'\s+', ' ', query.strip())

        # 转换为小写（保留中文）
        query = query.lower()

        return query

    def _build_context(self, documents: List[Tuple[RAGDocument, float]]) -> str:
        """
        构建上下文文本

        Args:
            documents: 文档列表

        Returns:
            str: 上下文文本
        """
        if not documents:
            return ""

        context_parts = []
        current_length = 0

        for doc, similarity in documents:
            # 格式化文档内容
            doc_text = f"\n## {doc.note_title}\n\n{doc.content}\n"
            doc_text += f"来源: {doc.note_path} (相似度: {similarity:.3f})\n"

            # 检查长度限制
            if current_length + len(doc_text) > self.max_context_length:
                # 截断最后一个文档
                remaining_length = self.max_context_length - current_length - 50
                if remaining_length > 100:
                    truncated_content = doc.content[:remaining_length] + "..."
                    doc_text = f"\n## {doc.note_title}\n\n{truncated_content}\n"
                    doc_text += f"来源: {doc.note_path} (相似度: {similarity:.3f}) [截断]\n"
                    context_parts.append(doc_text)
                break

            context_parts.append(doc_text)
            current_length += len(doc_text)

        return "\n".join(context_parts)

    def _extract_sources(self, documents: List[Tuple[RAGDocument, float]]) -> List[Dict[str, Any]]:
        """
        提取来源信息

        Args:
            documents: 文档列表

        Returns:
            List[Dict[str, Any]]: 来源信息列表
        """
        sources = []
        seen_note_ids = set()

        for doc, similarity in documents:
            if doc.note_id not in seen_note_ids:
                sources.append({
                    "note_id": doc.note_id,
                    "title": doc.note_title,
                    "path": doc.note_path,
                    "similarity": similarity,
                    "notebook_id": doc.notebook_id
                })
                seen_note_ids.add(doc.note_id)

        return sources

    def _deduplicate_documents(self, documents: List[Tuple[RAGDocument, float]]) -> List[Tuple[RAGDocument, float]]:
        """
        文档去重

        Args:
            documents: 文档列表

        Returns:
            List[Tuple[RAGDocument, float]]: 去重后的文档列表
        """
        seen_note_ids = set()
        unique_docs = []

        for doc, similarity in documents:
            if doc.note_id not in seen_note_ids:
                unique_docs.append((doc, similarity))
                seen_note_ids.add(doc.note_id)

        return unique_docs

    async def get_similar_queries(self, query: str, limit: int = 5) -> List[str]:
        """
        获取相似查询建议

        Args:
            query: 原始查询
            limit: 返回数量限制

        Returns:
            List[str]: 相似查询列表
        """
        # 这里可以实现查询扩展逻辑，比如：
        # 1. 基于同义词的扩展
        # 2. 基于历史查询的推荐
        # 3. 基于内容语义的扩展

        # 简单实现：基于关键词的变体
        similar_queries = []

        # 提取关键词
        keywords = re.findall(r'\w+', query)
        if len(keywords) > 1:
            # 生成交叉组合
            for i in range(len(keywords)):
                for j in range(i + 1, len(keywords)):
                    variant = f"{keywords[i]} {keywords[j]}"
                    if variant != query and len(similar_queries) < limit:
                        similar_queries.append(variant)

        return similar_queries


def create_query_engine(knowledge_base: RAGKnowledgeBase,
                       max_context_length: int = 4000,
                       similarity_threshold: float = 0.6,
                       max_documents: int = 5) -> RAGQueryEngine:
    """
    创建查询引擎的便捷函数

    Args:
        knowledge_base: 知识库实例
        max_context_length: 最大上下文长度
        similarity_threshold: 相似度阈值
        max_documents: 最大文档数量

    Returns:
        RAGQueryEngine: 查询引擎实例
    """
    return RAGQueryEngine(
        knowledge_base=knowledge_base,
        max_context_length=max_context_length,
        similarity_threshold=similarity_threshold,
        max_documents=max_documents
    )


async def main():
    """测试代码"""
    # 创建知识库和查询引擎
    from .rag_knowledge_base import create_rag_knowledge_base
    rag_kb = create_rag_knowledge_base()
    query_engine = create_query_engine(rag_kb)

    # 获取笔记本列表
    notebooks = rag_kb.content_extractor.workspace.list_notebooks()
    if not notebooks:
        print("没有找到笔记本")
        return

    test_notebook_id = notebooks[0][0]

    # 测试单查询
    print("=== 测试单查询 ===")
    result = await query_engine.query("测试", test_notebook_id)
    print(f"查询: {result.query}")
    print(f"找到 {len(result.documents)} 个相关文档")
    print(f"上下文长度: {len(result.context)}")
    print("来源:")
    for source in result.sources:
        print(f"  - {source['title']} (相似度: {source['similarity']:.3f})")

    # 测试多查询
    print("\n=== 测试多查询 ===")
    queries = ["测试", "文档", "内容"]
    multi_result = await query_engine.multi_query(queries, test_notebook_id, combine_strategy="union")
    print(f"多查询: {multi_result.metadata['original_queries']}")
    print(f"合并策略: {multi_result.metadata['combine_strategy']}")
    print(f"结果文档数: {len(multi_result.documents)}")


if __name__ == "__main__":
    asyncio.run(main())