"""
RAG知识库模块
用于构建和管理基于思源笔记的RAG知识库
"""

import os
import asyncio
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from pathlib import Path

# 使用共享日志器
from ..logger import get_logger
from ..siyuan.siyuan_content import SiYuanContentExtractor

# 向量数据库
import chromadb
from chromadb.config import Settings as ChromaSettings
from chromadb.utils import embedding_functions

logger = get_logger(__name__)


@dataclass
class RAGDocument:
    """RAG文档"""
    id: str
    content: str
    metadata: Dict[str, Any]
    notebook_id: str
    note_id: str
    note_title: str
    note_path: str

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "id": self.id,
            "content": self.content,
            "metadata": self.metadata,
            "notebook_id": self.notebook_id,
            "note_id": self.note_id,
            "note_title": self.note_title,
            "note_path": self.note_path
        }


class RAGKnowledgeBase:
    """RAG知识库管理器"""

    def __init__(self,
                 persist_directory: Optional[str] = None,
                 embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                 collection_name: str = "siyuan_notes"):
        """
        初始化RAG知识库

        Args:
            persist_directory: 向量数据库持久化目录
            embedding_model: 嵌入模型名称
            collection_name: 集合名称
        """
        self.persist_directory = persist_directory or os.path.join(
            os.getcwd(), "data", "rag_db"
        )
        self.embedding_model = embedding_model
        self.collection_name = collection_name

        # 确保持久化目录存在
        Path(self.persist_directory).mkdir(parents=True, exist_ok=True)

        # 初始化向量数据库
        self._init_chroma_db()

        # 初始化内容提取器
        self.content_extractor = SiYuanContentExtractor()

        logger.info(f"初始化RAG知识库，持久化目录: {self.persist_directory}")

    def _init_chroma_db(self):
        """初始化ChromaDB"""
        try:
            # 创建嵌入函数
            self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=self.embedding_model
            )

            # 初始化ChromaDB客户端
            self.client = chromadb.PersistentClient(
                path=self.persist_directory,
                settings=ChromaSettings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )

            # 获取或创建集合
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                embedding_function=self.embedding_function,
                metadata={"hnsw:space": "cosine"}
            )

            logger.info(f"ChromaDB初始化成功，集合: {self.collection_name}")

        except Exception as e:
            logger.error(f"ChromaDB初始化失败: {e}")
            raise

    async def build_knowledge_base(self,
                                  notebook_id: str,
                                  include_children: bool = True,
                                  chunk_size: int = 1000,
                                  chunk_overlap: int = 200,
                                  batch_size: int = 10) -> int:
        """
        构建知识库

        Args:
            notebook_id: 笔记本ID
            include_children: 是否包含子笔记
            chunk_size: 文档分块大小
            chunk_overlap: 分块重叠大小
            batch_size: 批处理大小

        Returns:
            int: 成功处理的文档数量
        """
        logger.info(f"开始为笔记本 {notebook_id} 构建知识库")

        # 清空现有数据（如果存在）
        await self.clear_notebook_data(notebook_id)

        # 获取所有笔记内容
        note_contents = await self.content_extractor.get_all_note_contents(
            notebook_id, include_children
        )

        if not note_contents:
            logger.warning(f"笔记本 {notebook_id} 中没有找到笔记")
            return 0

        # 处理文档
        documents = []
        for note_content in note_contents:
            # 分块处理
            chunks = self._chunk_text(note_content.content, chunk_size, chunk_overlap)
            for i, chunk in enumerate(chunks):
                doc_id = f"{note_content.id}_chunk_{i}"
                # 构建元数据，确保所有值都不是None
                metadata = {
                    "notebook_id": notebook_id,
                    "note_id": note_content.id,
                    "note_title": note_content.title or "",
                    "note_path": note_content.path or "",
                    "chunk_index": i,
                    "total_chunks": len(chunks)
                }

                # 只有当parent_id不为None时才添加
                if note_content.parent_id is not None:
                    metadata["parent_id"] = note_content.parent_id

                document = RAGDocument(
                    id=doc_id,
                    content=chunk,
                    metadata=metadata,
                    notebook_id=notebook_id,
                    note_id=note_content.id,
                    note_title=note_content.title,
                    note_path=note_content.path
                )
                documents.append(document)

        # 批量插入向量数据库
        total_inserted = 0
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            inserted = await self._insert_documents_batch(batch)
            total_inserted += inserted

            logger.info(f"已处理 {i + len(batch)}/{len(documents)} 个文档块")

        logger.info(f"知识库构建完成，共处理 {total_inserted} 个文档块")
        return total_inserted

    async def _insert_documents_batch(self, documents: List[RAGDocument]) -> int:
        """
        批量插入文档

        Args:
            documents: 文档列表

        Returns:
            int: 成功插入的数量
        """
        try:
            ids = [doc.id for doc in documents]
            contents = [doc.content for doc in documents]
            metadatas = [doc.metadata for doc in documents]

            self.collection.add(
                ids=ids,
                documents=contents,
                metadatas=metadatas
            )

            return len(documents)

        except Exception as e:
            logger.error(f"批量插入文档失败: {e}")
            return 0

    async def clear_notebook_data(self, notebook_id: str):
        """清空指定笔记本的数据"""
        try:
            # 获取该笔记本的所有文档
            results = self.collection.get(
                where={"notebook_id": notebook_id}
            )

            if results["ids"]:
                self.collection.delete(ids=results["ids"])
                logger.info(f"已清空笔记本 {notebook_id} 的现有数据")

        except Exception as e:
            logger.error(f"清空笔记本数据失败: {e}")

    def _chunk_text(self, text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
        """
        文本分块

        Args:
            text: 原始文本
            chunk_size: 块大小
            chunk_overlap: 重叠大小

        Returns:
            List[str]: 文本块列表
        """
        if not text or len(text) <= chunk_size:
            return [text] if text else []

        chunks = []
        start = 0

        while start < len(text):
            end = start + chunk_size

            # 如果不是最后一块，尝试在句号、换行符等位置分割
            if end < len(text):
                # 寻找最近的分割点
                split_chars = ['\n\n', '\n', '。', '！', '？', '.', '!', '?']
                best_split = end

                for char in split_chars:
                    split_pos = text.rfind(char, start, end)
                    if split_pos > start:
                        best_split = split_pos + 1
                        break

                end = best_split

            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)

            start = max(start + 1, end - chunk_overlap)

        return chunks

    async def search_similar_documents(self,
                                     query: str,
                                     notebook_id: Optional[str] = None,
                                     n_results: int = 5,
                                     score_threshold: float = 0.6) -> List[Tuple[RAGDocument, float]]:
        """
        搜索相似文档

        Args:
            query: 查询文本
            notebook_id: 笔记本ID过滤
            n_results: 返回结果数量
            score_threshold: 相似度阈值

        Returns:
            List[Tuple[RAGDocument, float]]: 相似文档和分数列表
        """
        try:
            # 构建过滤条件
            where_filter = None
            if notebook_id:
                where_filter = {"notebook_id": notebook_id}

            # 执行查询
            results = self.collection.query(
                query_texts=[query],
                n_results=n_results * 2,  # 获取更多结果用于过滤
                where=where_filter,
                include=["documents", "metadatas", "distances"]
            )

            # 处理结果
            documents = []
            if results["ids"] and results["ids"][0]:
                for i, doc_id in enumerate(results["ids"][0]):
                    distance = results["distances"][0][i]
                    similarity = 1 - distance  # 转换为相似度

                    if similarity >= score_threshold:
                        metadata = results["metadatas"][0][i]
                        content = results["documents"][0][i]

                        document = RAGDocument(
                            id=doc_id,
                            content=content,
                            metadata=metadata,
                            notebook_id=metadata["notebook_id"],
                            note_id=metadata["note_id"],
                            note_title=metadata["note_title"],
                            note_path=metadata["note_path"]
                        )

                        documents.append((document, similarity))

            # 按相似度排序并限制结果数量
            documents.sort(key=lambda x: x[1], reverse=True)
            return documents[:n_results]

        except Exception as e:
            logger.error(f"搜索相似文档失败: {e}")
            return []

    async def get_note_context(self,
                              note_id: str,
                              max_chunks: int = 10) -> List[RAGDocument]:
        """
        获取指定笔记的上下文

        Args:
            note_id: 笔记ID
            max_chunks: 最大块数量

        Returns:
            List[RAGDocument]: 笔记的文档块列表
        """
        try:
            results = self.collection.get(
                where={"note_id": note_id},
                n_results=max_chunks
            )

            documents = []
            if results["ids"]:
                for i, doc_id in enumerate(results["ids"]):
                    metadata = results["metadatas"][i]
                    content = results["documents"][i]

                    document = RAGDocument(
                        id=doc_id,
                        content=content,
                        metadata=metadata,
                        notebook_id=metadata["notebook_id"],
                        note_id=metadata["note_id"],
                        note_title=metadata["note_title"],
                        note_path=metadata["note_path"]
                    )
                    documents.append(document)

            # 按chunk_index排序
            documents.sort(key=lambda x: x.metadata.get("chunk_index", 0))
            return documents

        except Exception as e:
            logger.error(f"获取笔记上下文失败: {e}")
            return []

    def get_collection_stats(self) -> Dict[str, Any]:
        """获取集合统计信息"""
        try:
            count = self.collection.count()
            return {
                "collection_name": self.collection_name,
                "document_count": count,
                "persist_directory": self.persist_directory,
                "embedding_model": self.embedding_model
            }
        except Exception as e:
            logger.error(f"获取集合统计信息失败: {e}")
            return {}

    async def rebuild_knowledge_base(self, notebook_id: str, **kwargs) -> int:
        """重建知识库"""
        logger.info(f"开始重建笔记本 {notebook_id} 的知识库")
        await self.clear_notebook_data(notebook_id)
        return await self.build_knowledge_base(notebook_id, **kwargs)


def create_rag_knowledge_base(persist_directory: Optional[str] = None) -> RAGKnowledgeBase:
    """
    创建RAG知识库的便捷函数

    Args:
        persist_directory: 持久化目录

    Returns:
        RAGKnowledgeBase: 知识库实例
    """
    return RAGKnowledgeBase(persist_directory)


async def main():
    """测试代码"""
    # 创建知识库
    rag_kb = create_rag_knowledge_base()

    # 获取笔记本列表
    notebooks = rag_kb.content_extractor.workspace.list_notebooks()
    if not notebooks:
        print("没有找到笔记本")
        return

    # 使用第一个笔记本进行测试
    test_notebook_id = notebooks[0][0]
    print(f"测试笔记本: {test_notebook_id}")

    # 构建知识库
    doc_count = await rag_kb.build_knowledge_base(test_notebook_id)
    print(f"构建完成，共处理 {doc_count} 个文档块")

    # 显示统计信息
    stats = rag_kb.get_collection_stats()
    print(f"知识库统计: {stats}")

    # 测试搜索
    test_query = "测试"
    results = await rag_kb.search_similar_documents(test_query, test_notebook_id)
    print(f"\n搜索 '{test_query}' 找到 {len(results)} 个结果:")
    for i, (doc, similarity) in enumerate(results):
        print(f"  {i+1}. {doc.note_title} (相似度: {similarity:.3f})")
        print(f"     内容预览: {doc.content[:100]}...")


if __name__ == "__main__":
    asyncio.run(main())