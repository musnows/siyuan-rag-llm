"""
RAG工具模块
为ReAct Agent提供可调用的RAG查询工具
"""

import json
import asyncio
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
from pydantic import BaseModel, Field

from ..logger import get_logger
from ..rag.rag_knowledge_base import RAGKnowledgeBase, RAGDocument
from ..rag.rag_query import RAGQueryEngine, QueryResult

logger = get_logger(__name__)


@dataclass
class ToolParameter:
    """工具参数定义"""
    name: str
    type: str
    description: str
    required: bool = True
    default: Optional[Any] = None


@dataclass
class ToolDefinition:
    """工具定义"""
    name: str
    description: str
    parameters: List[ToolParameter]
    function: Callable


class RAGSearchTool(BaseModel):
    """RAG搜索工具"""

    query: str = Field(..., description="搜索查询词，用于在知识库中查找相关文档")
    notebook_id: Optional[str] = Field(None, description="可选的笔记本ID，用于限定搜索范围")
    max_results: int = Field(5, description="最大返回结果数量，默认为5")
    similarity_threshold: float = Field(0.6, description="相似度阈值，默认为0.6")

    class Config:
        extra = "forbid"


class RAGContextTool(BaseModel):
    """RAG上下文工具 - 获取特定笔记的完整上下文"""

    note_id: str = Field(..., description="笔记ID，用于获取该笔记的完整上下文内容")
    max_chunks: int = Field(10, description="最大返回块数量，默认为10")

    class Config:
        extra = "forbid"


class RAGMultiQueryTool(BaseModel):
    """RAG多查询工具 - 执行多个相关查询并合并结果"""

    queries: List[str] = Field(..., description="查询列表，支持多个相关查询")
    notebook_id: Optional[str] = Field(None, description="可选的笔记本ID")
    combine_strategy: str = Field("union", description="合并策略：union(并集)、intersection(交集)、weighted(加权)")
    max_results: int = Field(5, description="每个查询的最大结果数量")

    class Config:
        extra = "forbid"


class RAGToolKit:
    """RAG工具包 - 为Agent提供RAG相关的工具"""

    def __init__(self, knowledge_base: RAGKnowledgeBase):
        """
        初始化RAG工具包

        Args:
            knowledge_base: RAG知识库实例
        """
        self.knowledge_base = knowledge_base
        self.query_engine = RAGQueryEngine(knowledge_base)
        self.tools = self._init_tools()
        logger.info("RAG工具包初始化完成")

    def _init_tools(self) -> Dict[str, ToolDefinition]:
        """初始化所有工具"""
        tools = {}

        # RAG搜索工具
        tools["rag_search"] = ToolDefinition(
            name="rag_search",
            description="在RAG知识库中搜索相关文档。根据查询词查找相似的笔记内容，支持相似度过滤和结果数量限制。",
            parameters=[
                ToolParameter(
                    name="query",
                    type="string",
                    description="搜索查询词，用于在知识库中查找相关文档"
                ),
                ToolParameter(
                    name="notebook_id",
                    type="string",
                    description="可选的笔记本ID，用于限定搜索范围",
                    required=False
                ),
                ToolParameter(
                    name="max_results",
                    type="integer",
                    description="最大返回结果数量，默认为5",
                    required=False,
                    default=5
                ),
                ToolParameter(
                    name="similarity_threshold",
                    type="number",
                    description="相似度阈值，默认为0.6，范围0-1",
                    required=False,
                    default=0.6
                )
            ],
            function=self._rag_search
        )

        # RAG上下文工具
        tools["rag_get_context"] = ToolDefinition(
            name="rag_get_context",
            description="获取特定笔记的完整上下文内容。通过笔记ID获取该笔记的所有文档块，按顺序排列。",
            parameters=[
                ToolParameter(
                    name="note_id",
                    type="string",
                    description="笔记ID，用于获取该笔记的完整上下文内容"
                ),
                ToolParameter(
                    name="max_chunks",
                    type="integer",
                    description="最大返回块数量，默认为10",
                    required=False,
                    default=10
                )
            ],
            function=self._rag_get_context
        )

        # RAG多查询工具
        tools["rag_multi_query"] = ToolDefinition(
            name="rag_multi_query",
            description="执行多个相关查询并合并结果。适合复杂问题的多角度搜索，支持不同的合并策略。",
            parameters=[
                ToolParameter(
                    name="queries",
                    type="array",
                    description="查询列表，支持多个相关查询"
                ),
                ToolParameter(
                    name="notebook_id",
                    type="string",
                    description="可选的笔记本ID",
                    required=False
                ),
                ToolParameter(
                    name="combine_strategy",
                    type="string",
                    description="合并策略：union(并集)、intersection(交集)、weighted(加权)",
                    required=False,
                    default="union"
                ),
                ToolParameter(
                    name="max_results",
                    type="integer",
                    description="每个查询的最大结果数量",
                    required=False,
                    default=5
                )
            ],
            function=self._rag_multi_query
        )

        # 获取统计信息工具
        tools["rag_get_stats"] = ToolDefinition(
            name="rag_get_stats",
            description="获取RAG知识库的统计信息，包括文档数量、笔记本分布等。",
            parameters=[],
            function=self._rag_get_stats
        )

        return tools

    async def _rag_search(self, **kwargs) -> Dict[str, Any]:
        """执行RAG搜索"""
        try:
            # 验证参数
            tool_input = RAGSearchTool(**kwargs)

            logger.info(f"执行RAG搜索: {tool_input.query}")

            # 执行搜索
            documents = await self.knowledge_base.search_similar_documents(
                query=tool_input.query,
                notebook_id=tool_input.notebook_id,
                n_results=tool_input.max_results,
                score_threshold=tool_input.similarity_threshold
            )

            # 格式化结果
            results = []
            for doc, similarity in documents:
                results.append({
                    "note_id": doc.note_id,
                    "title": doc.note_title,
                    "path": doc.note_path,
                    "content": doc.content,
                    "similarity": round(similarity, 3),
                    "notebook_id": doc.notebook_id
                })

            return {
                "success": True,
                "query": tool_input.query,
                "results_count": len(results),
                "results": results
            }

        except Exception as e:
            logger.error(f"RAG搜索失败: {e}")
            return {
                "success": False,
                "error": str(e),
                "query": kwargs.get("query", ""),
                "results": []
            }

    async def _rag_get_context(self, **kwargs) -> Dict[str, Any]:
        """获取笔记上下文"""
        try:
            # 验证参数
            tool_input = RAGContextTool(**kwargs)

            logger.info(f"获取笔记上下文: {tool_input.note_id}")

            # 获取上下文
            documents = await self.knowledge_base.get_note_context(
                note_id=tool_input.note_id,
                max_chunks=tool_input.max_chunks
            )

            # 格式化结果
            results = []
            for doc in documents:
                results.append({
                    "note_id": doc.note_id,
                    "title": doc.note_title,
                    "path": doc.note_path,
                    "content": doc.content,
                    "chunk_index": doc.metadata.get("chunk_index", 0),
                    "total_chunks": doc.metadata.get("total_chunks", 1),
                    "notebook_id": doc.notebook_id
                })

            return {
                "success": True,
                "note_id": tool_input.note_id,
                "chunks_count": len(results),
                "results": results
            }

        except Exception as e:
            logger.error(f"获取笔记上下文失败: {e}")
            return {
                "success": False,
                "error": str(e),
                "note_id": kwargs.get("note_id", ""),
                "results": []
            }

    async def _rag_multi_query(self, **kwargs) -> Dict[str, Any]:
        """执行多查询"""
        try:
            # 验证参数
            tool_input = RAGMultiQueryTool(**kwargs)

            logger.info(f"执行多查询: {tool_input.queries}")

            # 执行多查询
            result = await self.query_engine.multi_query(
                queries=tool_input.queries,
                notebook_id=tool_input.notebook_id,
                combine_strategy=tool_input.combine_strategy,
                **{"n_results": tool_input.max_results}
            )

            # 格式化结果
            results = []
            for doc, similarity in result.documents:
                results.append({
                    "note_id": doc.note_id,
                    "title": doc.note_title,
                    "path": doc.note_path,
                    "content": doc.content,
                    "similarity": round(similarity, 3),
                    "notebook_id": doc.notebook_id
                })

            return {
                "success": True,
                "queries": tool_input.queries,
                "combine_strategy": tool_input.combine_strategy,
                "results_count": len(results),
                "results": results
            }

        except Exception as e:
            logger.error(f"多查询失败: {e}")
            return {
                "success": False,
                "error": str(e),
                "queries": kwargs.get("queries", []),
                "results": []
            }

    async def _rag_get_stats(self, **kwargs) -> Dict[str, Any]:
        """获取统计信息"""
        try:
            logger.info("获取RAG统计信息")

            # 获取基本统计
            basic_stats = self.knowledge_base.get_collection_stats()

            # 获取笔记本统计
            notebook_stats = await self.knowledge_base.get_all_notebooks_stats()

            return {
                "success": True,
                "basic_stats": basic_stats,
                "notebook_stats": notebook_stats,
                "total_notebooks": len(notebook_stats)
            }

        except Exception as e:
            logger.error(f"获取统计信息失败: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    def get_tools_schema(self) -> List[Dict[str, Any]]:
        """获取工具的OpenAI函数调用格式"""
        tools_schema = []

        for tool_name, tool_def in self.tools.items():
            # 构建参数schema
            properties = {}
            required = []

            for param in tool_def.parameters:
                param_def = {
                    "type": param.type,
                    "description": param.description
                }

                if param.default is not None:
                    param_def["default"] = param.default

                properties[param.name] = param_def

                if param.required:
                    required.append(param.name)

            # 构建工具schema
            tool_schema = {
                "type": "function",
                "function": {
                    "name": tool_name,
                    "description": tool_def.description,
                    "parameters": {
                        "type": "object",
                        "properties": properties
                    }
                }
            }

            if required:
                tool_schema["function"]["parameters"]["required"] = required

            tools_schema.append(tool_schema)

        return tools_schema

    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """调用指定的工具"""
        if tool_name not in self.tools:
            return {
                "success": False,
                "error": f"未知工具: {tool_name}"
            }

        tool_def = self.tools[tool_name]

        try:
            # 异步调用工具函数
            if asyncio.iscoroutinefunction(tool_def.function):
                result = await tool_def.function(**arguments)
            else:
                result = tool_def.function(**arguments)

            return result

        except Exception as e:
            logger.error(f"工具调用失败 {tool_name}: {e}")
            return {
                "success": False,
                "error": str(e),
                "tool": tool_name,
                "arguments": arguments
            }

    def list_tools(self) -> List[str]:
        """列出所有可用工具"""
        return list(self.tools.keys())

    def get_tool_description(self, tool_name: str) -> Optional[str]:
        """获取工具描述"""
        if tool_name in self.tools:
            return self.tools[tool_name].description
        return None


def create_rag_toolkit(knowledge_base: RAGKnowledgeBase) -> RAGToolKit:
    """
    创建RAG工具包的便捷函数

    Args:
        knowledge_base: RAG知识库实例

    Returns:
        RAGToolKit: 工具包实例
    """
    return RAGToolKit(knowledge_base)