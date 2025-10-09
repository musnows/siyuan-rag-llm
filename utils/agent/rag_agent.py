"""
RAG Agent模块
集成OpenAI API，基于RAG知识库进行智能问答
"""

import os
import asyncio
from typing import Dict, List, Optional, Any, AsyncGenerator
from dataclasses import dataclass, field
import json
from datetime import datetime

# 设置tokenizers并行化以避免警告
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import openai
from openai import AsyncOpenAI

from ..logger import get_logger
from ..rag.rag_knowledge_base import RAGKnowledgeBase
from ..rag.rag_query import RAGQueryEngine, QueryResult, create_query_engine

logger = get_logger(__name__)


@dataclass
class AgentMessage:
    """Agent消息"""
    role: str  # "user", "assistant", "system"
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata
        }


@dataclass
class AgentResponse:
    """Agent响应"""
    answer: str
    sources: List[Dict[str, Any]]
    query_result: QueryResult
    reasoning: Optional[str] = None
    confidence: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "answer": self.answer,
            "sources": self.sources,
            "reasoning": self.reasoning,
            "confidence": self.confidence,
            "metadata": self.metadata,
            "query_result": {
                "query": self.query_result.query,
                "document_count": len(self.query_result.documents),
                "context_length": len(self.query_result.context)
            }
        }


class RAGAgent:
    """RAG智能问答Agent"""

    def __init__(self,
                 knowledge_base: RAGKnowledgeBase,
                 model: str = "gpt-3.5-turbo",
                 max_tokens: int = 2000,
                 temperature: float = 0.1,
                 system_prompt: Optional[str] = None,
                 use_streaming: bool = False):
        """
        初始化RAG Agent

        Args:
            knowledge_base: RAG知识库
            model: OpenAI模型名称
            max_tokens: 最大生成token数
            temperature: 温度参数
            system_prompt: 系统提示词
            use_streaming: 是否使用流式输出
        """
        self.knowledge_base = knowledge_base
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.use_streaming = use_streaming

        # 初始化OpenAI客户端
        self.client = AsyncOpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_BASE_URL")
        )

        # 初始化查询引擎
        self.query_engine = create_query_engine(knowledge_base)

        # 设置系统提示词
        self.system_prompt = system_prompt or self._default_system_prompt()

        # 对话历史
        self.conversation_history: List[AgentMessage] = []

        logger.info(f"RAG Agent初始化完成，模型: {model}")

    def _default_system_prompt(self) -> str:
        """默认系统提示词"""
        return """你是一个基于思源笔记知识库的智能助手。你的任务是：

1. 基于提供的上下文信息回答用户问题
2. 如果上下文中没有相关信息，诚实地说明
3. 回答要准确、简洁、有帮助
4. 引用具体的笔记来源
5. 保持专业和友好的语气

回答格式要求：
- 直接回答用户问题
- 在适当的地方引用来源笔记
- 如果信息不足，明确说明

注意事项：
- 只基于提供的上下文信息回答，不要编造内容
- 如果多个笔记有相关信息，综合多个来源给出完整答案
- 保持回答的客观性和准确性"""

    async def query(self,
                   question: str,
                   notebook_id: Optional[str] = None,
                   context_strategy: str = "simple",
                   **kwargs) -> AgentResponse:
        """
        查询回答

        Args:
            question: 用户问题
            notebook_id: 笔记本ID过滤
            context_strategy: 上下文策略 ("simple", "contextual", "multi_query")
            **kwargs: 其他查询参数

        Returns:
            AgentResponse: Agent响应
        """
        logger.info(f"收到用户查询: {question}")

        # 添加用户消息到历史
        user_message = AgentMessage(
            role="user",
            content=question,
            metadata={"notebook_id": notebook_id}
        )
        self.conversation_history.append(user_message)

        try:
            # 获取相关上下文
            query_result = await self._get_context(question, notebook_id, context_strategy, **kwargs)

            # 生成回答
            response = await self._generate_answer(question, query_result)

            # 添加助手消息到历史
            assistant_message = AgentMessage(
                role="assistant",
                content=response.answer,
                metadata={
                    "sources": response.sources,
                    "confidence": response.confidence,
                    "context_used": len(query_result.documents) > 0
                }
            )
            self.conversation_history.append(assistant_message)

            logger.info(f"生成回答完成，引用 {len(response.sources)} 个来源")
            return response

        except Exception as e:
            logger.error(f"查询处理失败: {e}")
            error_response = AgentResponse(
                answer=f"抱歉，处理您的查询时出现错误：{str(e)}",
                sources=[],
                query_result=QueryResult(query=question, documents=[], context=""),
                metadata={"error": str(e)}
            )
            return error_response

    async def stream_query(self,
                          question: str,
                          notebook_id: Optional[str] = None,
                          context_strategy: str = "simple",
                          **kwargs) -> AsyncGenerator[str, None]:
        """
        流式查询回答

        Args:
            question: 用户问题
            notebook_id: 笔记本ID过滤
            context_strategy: 上下文策略
            **kwargs: 其他查询参数

        Yields:
            str: 流式回答片段
        """
        logger.info(f"收到流式查询: {question}")

        # 获取上下文
        query_result = await self._get_context(question, notebook_id, context_strategy, **kwargs)

        # 构建提示词
        messages = self._build_messages(question, query_result)

        try:
            # 流式生成
            stream = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                stream=True
            )

            full_answer = ""
            async for chunk in stream:
                if chunk.choices[0].delta.content is not None:
                    content = chunk.choices[0].delta.content
                    full_answer += content
                    yield content

            # 添加到对话历史
            user_message = AgentMessage(role="user", content=question)
            assistant_message = AgentMessage(
                role="assistant",
                content=full_answer,
                metadata={"sources": self._extract_sources(query_result)}
            )
            self.conversation_history.extend([user_message, assistant_message])

        except Exception as e:
            logger.error(f"流式查询失败: {e}")
            yield f"抱歉，处理您的查询时出现错误：{str(e)}"

    async def _get_context(self,
                          question: str,
                          notebook_id: Optional[str],
                          strategy: str,
                          **kwargs) -> QueryResult:
        """获取上下文"""
        if strategy == "simple":
            return await self.query_engine.query(question, notebook_id, **kwargs)
        elif strategy == "contextual":
            # 实现上下文增强查询
            context_note_ids = kwargs.get("context_note_ids", [])
            return await self.query_engine.contextual_query(
                question, notebook_id, context_note_ids, **kwargs
            )
        elif strategy == "multi_query":
            # 实现多查询策略
            queries = [question] + kwargs.get("expanded_queries", [])
            return await self.query_engine.multi_query(
                queries, notebook_id, **kwargs
            )
        else:
            return await self.query_engine.query(question, notebook_id, **kwargs)

    async def _generate_answer(self, question: str, query_result: QueryResult) -> AgentResponse:
        """生成回答"""
        messages = self._build_messages(question, query_result)

        try:
            # 调用OpenAI API
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=self.max_tokens,
                temperature=self.temperature
            )

            answer = response.choices[0].message.content or ""
            sources = self._extract_sources(query_result)

            # 计算置信度（基于相似度）
            confidence = self._calculate_confidence(query_result.documents)

            return AgentResponse(
                answer=answer,
                sources=sources,
                query_result=query_result,
                confidence=confidence,
                metadata={
                    "model": self.model,
                    "tokens_used": response.usage.total_tokens if response.usage else None
                }
            )

        except Exception as e:
            logger.error(f"生成回答失败: {e}")
            raise

    def _build_messages(self, question: str, query_result: QueryResult) -> List[Dict[str, str]]:
        """构建消息列表"""
        messages = [
            {"role": "system", "content": self.system_prompt}
        ]

        # 添加上下文
        if query_result.context:
            context_message = f"""基于以下思源笔记内容回答问题：

相关笔记内容：
{query_result.context}

请基于以上内容回答用户的问题。"""
            messages.append({"role": "system", "content": context_message})

        # 添加用户问题
        messages.append({"role": "user", "content": question})

        return messages

    def _extract_sources(self, query_result: QueryResult) -> List[Dict[str, Any]]:
        """提取来源信息"""
        sources = []
        for source in query_result.sources:
            sources.append({
                "note_id": source["note_id"],
                "title": source["title"],
                "path": source["path"],
                "similarity": source["similarity"],
                "notebook_id": source["notebook_id"]
            })
        return sources

    def _calculate_confidence(self, documents: List[tuple]) -> Optional[float]:
        """计算回答置信度"""
        if not documents:
            return 0.0

        # 基于最高相似度和文档数量计算置信度
        max_similarity = max(doc[1] for doc in documents)
        doc_count = len(documents)

        # 简单的置信度计算公式
        confidence = (max_similarity * 0.7 + min(doc_count / 5.0, 1.0) * 0.3)
        return round(confidence, 3)

    def clear_history(self):
        """清空对话历史"""
        self.conversation_history.clear()
        logger.info("对话历史已清空")

    def get_conversation_summary(self) -> Dict[str, Any]:
        """获取对话摘要"""
        return {
            "total_messages": len(self.conversation_history),
            "user_messages": len([m for m in self.conversation_history if m.role == "user"]),
            "assistant_messages": len([m for m in self.conversation_history if m.role == "assistant"]),
            "last_message_time": self.conversation_history[-1].timestamp.isoformat() if self.conversation_history else None
        }

    async def update_knowledge_base(self, notebook_id: str, **kwargs) -> int:
        """更新知识库"""
        logger.info(f"开始更新笔记本 {notebook_id} 的知识库")
        return await self.knowledge_base.rebuild_knowledge_base(notebook_id, **kwargs)

    def set_system_prompt(self, prompt: str):
        """设置系统提示词"""
        self.system_prompt = prompt
        logger.info("系统提示词已更新")


def create_rag_agent(knowledge_base: RAGKnowledgeBase,
                    model: str = "gpt-3.5-turbo",
                    **kwargs) -> RAGAgent:
    """
    创建RAG Agent的便捷函数

    Args:
        knowledge_base: 知识库实例
        model: OpenAI模型名称
        **kwargs: 其他参数

    Returns:
        RAGAgent: Agent实例
    """
    return RAGAgent(knowledge_base, model, **kwargs)


async def main():
    """测试代码"""
    # 创建知识库和Agent
    from ..rag.rag_knowledge_base import create_rag_knowledge_base

    rag_kb = create_rag_knowledge_base()
    agent = create_rag_agent(rag_kb)

    # 获取笔记本列表
    notebooks = rag_kb.content_extractor.workspace.list_notebooks()
    if not notebooks:
        print("没有找到笔记本")
        return

    test_notebook_id = notebooks[0][0]
    print(f"测试笔记本: {test_notebook_id}")

    # 测试查询
    test_questions = [
        "这个笔记本的主要内容是什么？",
        "有没有关于测试的文档？",
        "请总结一下重要的概念"
    ]

    for question in test_questions:
        print(f"\n问题: {question}")
        print("-" * 50)

        response = await agent.query(question, test_notebook_id)
        print(f"回答: {response.answer}")
        print(f"置信度: {response.confidence}")
        print(f"来源数量: {len(response.sources)}")

        # 显示来源
        if response.sources:
            print("来源:")
            for source in response.sources:
                print(f"  - {source['title']} (相似度: {source['similarity']:.3f})")

    # 显示对话摘要
    summary = agent.get_conversation_summary()
    print(f"\n对话摘要: {summary}")


if __name__ == "__main__":
    asyncio.run(main())