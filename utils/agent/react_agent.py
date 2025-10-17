"""
ReAct Agent模块
实现基于Reasoning and Acting模式的智能Agent，支持多轮RAG查询
"""

import os
import asyncio
import json
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import re

import openai
from openai import AsyncOpenAI

from ..logger import get_logger
from ..rag.rag_knowledge_base import RAGKnowledgeBase
from .rag_tools import RAGToolKit, create_rag_toolkit

logger = get_logger(__name__)


@dataclass
class ReActStep:
    """ReAct推理步骤"""
    step_type: str  # "thought", "action", "observation"
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    tool_call: Optional[Dict[str, Any]] = None
    tool_result: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "step_type": self.step_type,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "tool_call": self.tool_call,
            "tool_result": self.tool_result
        }


@dataclass
class ReActResponse:
    """ReAct Agent响应"""
    answer: str
    reasoning: List[ReActStep]
    tool_calls_made: int
    final_confidence: Optional[float] = None
    sources_used: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "answer": self.answer,
            "reasoning": [step.to_dict() for step in self.reasoning],
            "tool_calls_made": self.tool_calls_made,
            "final_confidence": self.final_confidence,
            "sources_used": self.sources_used,
            "metadata": self.metadata
        }


class ReActAgent:
    """ReAct智能Agent - 支持多轮推理和工具调用"""

    def __init__(self,
                 knowledge_base: RAGKnowledgeBase,
                 model: str = None,
                 max_steps: int = 10,
                 max_tool_calls: int = 5,
                 temperature: float = 0.1,
                 system_prompt: Optional[str] = None):
        """
        初始化ReAct Agent

        Args:
            knowledge_base: RAG知识库
            model: OpenAI模型名称
            max_steps: 最大推理步数
            max_tool_calls: 最大工具调用次数
            temperature: 温度参数
            system_prompt: 系统提示词
        """
        self.knowledge_base = knowledge_base
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
        self.max_steps = max_steps
        self.max_tool_calls = max_tool_calls
        self.temperature = temperature

        # 初始化OpenAI客户端
        self.client = AsyncOpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_BASE_URL")
        )

        # 初始化RAG工具包
        self.toolkit = create_rag_toolkit(knowledge_base)

        # 设置系统提示词
        self.system_prompt = system_prompt or self._default_system_prompt()

        logger.info(f"ReAct Agent初始化完成，模型: {model}，最大工具调用: {max_tool_calls}")

    def _default_system_prompt(self) -> str:
        """默认系统提示词"""
        return """你是一个基于ReAct（Reasoning and Acting）模式的智能助手，专门用于处理思源笔记知识库的查询。

你的工作流程是：
1. **Thought（思考）**: 分析用户问题，决定下一步行动
2. **Action（行动）**: 调用相关工具获取信息
3. **Observation（观察）**: 分析工具返回的结果
4. **重复**: 直到有足够信息回答用户问题

可用工具：
- rag_search: 在知识库中搜索相关文档
- rag_get_context: 获取特定笔记的完整上下文
- rag_multi_query: 执行多个相关查询并合并结果
- rag_get_stats: 获取知识库统计信息

回答要求：
- 基于工具调用结果回答问题
- 最多调用5次工具（根据问题复杂度决定）
- 如果找不到相关信息，诚实说明
- 引用具体的笔记来源
- 保持逻辑清晰，推理过程透明

回答格式：
1. 先给出最终答案，必须以“最终答案”开头
2. 然后简要说明推理过程
3. 列出主要信息来源

注意事项：
- 每次工具调用后要仔细分析结果
- 根据结果调整下一步策略
- 避免重复相同的查询
- 优先选择最相关的工具"""

    async def query(self, question: str) -> ReActResponse:
        """
        执行ReAct查询

        Args:
            question: 用户问题

        Returns:
            ReActResponse: Agent响应
        """
        logger.info(f"开始ReAct查询: {question}")

        reasoning_steps = []
        tool_calls_count = 0
        conversation = []

        # 添加初始系统消息
        conversation.append({
            "role": "system",
            "content": self.system_prompt
        })

        # 添加用户问题
        conversation.append({
            "role": "user",
            "content": f"用户问题: {question}\n\n请使用ReAct模式逐步分析并回答这个问题。"
        })

        final_answer = ""
        for step in range(self.max_steps):
            try:
                # 生成下一步行动
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=conversation,
                    temperature=self.temperature,
                    tools=self.toolkit.get_tools_schema(),
                    tool_choice="auto"
                )

                message = response.choices[0].message

                # 处理工具调用
                if message.tool_calls:
                    for tool_call in message.tool_calls:
                        tool_calls_count += 1

                        # 检查工具调用限制
                        if tool_calls_count > self.max_tool_calls:
                            logger.warning(f"达到最大工具调用限制 ({self.max_tool_calls})")
                            thought_step = ReActStep(
                                step_type="thought",
                                content=f"已达到最大工具调用次数 ({self.max_tool_calls})，现在基于已有信息给出答案。"
                            )
                            reasoning_steps.append(thought_step)
                            conversation.append({
                                "role": "assistant",
                                "content": thought_step.content
                            })
                            break

                        # 记录行动步骤
                        action_step = ReActStep(
                            step_type="action",
                            content=f"调用工具: {tool_call.function.name}",
                            tool_call={
                                "name": tool_call.function.name,
                                "arguments": json.loads(tool_call.function.arguments)
                            }
                        )
                        reasoning_steps.append(action_step)

                        # 执行工具调用
                        tool_result = await self.toolkit.call_tool(
                            tool_call.function.name,
                            json.loads(tool_call.function.arguments)
                        )

                        # 记录观察步骤
                        observation_content = self._format_tool_result(tool_result)
                        observation_step = ReActStep(
                            step_type="observation",
                            content=observation_content,
                            tool_result=tool_result
                        )
                        reasoning_steps.append(observation_step)

                        # 添加到对话
                        conversation.append({
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [{
                                "id": tool_call.id,
                                "type": tool_call.type,
                                "function": {
                                    "name": tool_call.function.name,
                                    "arguments": tool_call.function.arguments
                                }
                            }]
                        })

                        conversation.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": json.dumps(tool_result, ensure_ascii=False)
                        })

                else:
                    # 没有工具调用，表示是推理步骤
                    if message.content:
                        thought_step = ReActStep(
                            step_type="thought",
                            content=message.content
                        )
                        reasoning_steps.append(thought_step)
                        conversation.append({
                            "role": "assistant",
                            "content": message.content
                        })

                # 检查是否给出最终答案
                if message.content and self._is_final_answer(message.content):
                    logger.info("检测到最终答案，结束推理")
                    final_answer = message.content
                    break

            except Exception as e:
                logger.error(f"ReAct步骤 {step} 执行失败: {e}")
                error_step = ReActStep(
                    step_type="thought",
                    content=f"执行过程中出现错误: {str(e)}，将基于已有信息给出答案。"
                )
                reasoning_steps.append(error_step)
                final_answer = f"抱歉，在推理过程中出现错误: {str(e)}。基于已有信息给出答案。"
                break

        # 如果没有检测到最终答案，则生成一个
        if not final_answer:
            final_answer = await self._generate_final_answer(conversation)

        # 提取使用的来源
        sources_used = self._extract_sources(reasoning_steps)

        # 计算置信度
        confidence = self._calculate_confidence(reasoning_steps, sources_used)

        response = ReActResponse(
            answer=final_answer,
            reasoning=reasoning_steps,
            tool_calls_made=tool_calls_count,
            final_confidence=confidence,
            sources_used=sources_used,
            metadata={
                "model": self.model,
                "max_steps": self.max_steps,
                "max_tool_calls": self.max_tool_calls,
                "question": question
            }
        )

        logger.info(f"ReAct查询完成，工具调用次数: {tool_calls_count}，置信度: {confidence}")
        return response

    def _format_tool_result(self, result: Dict[str, Any]) -> str:
        """格式化工具结果"""
        if not result.get("success", False):
            return f"工具调用失败: {result.get('error', '未知错误')}"


        if "results" in result:
            # 搜索结果
            results = result["results"]
            if results:
                summary = f"找到 {len(results)} 个相关结果:\n"
                for i, item in enumerate(results[:3]):  # 只显示前3个
                    summary += f"{i+1}. {item.get('title', '无标题')} (相似度: {item.get('similarity', 0):.3f})\n"
                    summary += f"   内容预览: {item.get('content', '')[:100]}...\n"

                if len(results) > 3:
                    summary += f"... 还有 {len(results) - 3} 个结果\n"

                return summary
            else:
                return "未找到相关结果"

        elif "basic_stats" in result:
            # 统计信息
            basic_stats = result.get("basic_stats", {})
            notebook_stats = result.get("notebook_stats", {})
            return f"知识库统计: 总文档数 {basic_stats.get('document_count', 0)}, 分布在 {len(notebook_stats)} 个笔记本中"

        else:
            return "工具调用成功"

    def _is_final_answer(self, content: str) -> bool:
        """检查是否是最终答案"""
        final_indicators = [
            "最终答案",
            "综合以上信息",
            "基于以上分析",
            "答案是",
            "结论是",
            "总结",
            "回答如下",
            "我的回答",
            "根据查询结果"
        ]

        content_lower = content.lower()
        for indicator in final_indicators:
            if indicator in content_lower:
                return True
        return False

    async def _generate_final_answer(self, conversation: List[Dict]) -> str:
        """生成最终答案"""
        try:
            # 添加最终答案生成提示
            final_prompt = """
基于以上的推理过程和工具调用结果，请给出一个清晰、准确的最终答案。

要求：
1. 直接回答用户的原始问题
2. 基于工具调用结果，不要编造信息
3. 如果信息不足，明确说明
4. 引用相关的笔记来源
5. 保持回答简洁明了

请直接给出最终答案："""

            conversation.append({
                "role": "user",
                "content": final_prompt
            })

            response = await self.client.chat.completions.create(
                model=self.model,
                messages=conversation,
                temperature=self.temperature,
                max_tokens=2000
            )

            final_answer = response.choices[0].message.content or "抱歉，无法生成答案。"
            return final_answer.strip()

        except Exception as e:
            logger.error(f"生成最终答案失败: {e}")
            return "抱歉，在生成最终答案时出现错误。"

    def _extract_sources(self, reasoning_steps: List[ReActStep]) -> List[Dict[str, Any]]:
        """从推理步骤中提取信息来源"""
        sources = []
        seen_sources = set()

        for step in reasoning_steps:
            if step.tool_result and step.tool_result.get("success"):
                results = step.tool_result.get("results", [])
                for result in results:
                    source_key = (result.get("note_id"), result.get("title"))
                    if source_key not in seen_sources:
                        sources.append({
                            "note_id": result.get("note_id"),
                            "title": result.get("title"),
                            "path": result.get("path"),
                            "similarity": result.get("similarity"),
                            "notebook_id": result.get("notebook_id")
                        })
                        seen_sources.add(source_key)

        return sources

    def _calculate_confidence(self, reasoning_steps: List[ReActStep], sources: List[Dict]) -> Optional[float]:
        """计算回答置信度"""
        if not sources:
            return 0.1  # 没有来源，置信度很低

        # 基于成功工具调用次数和来源质量计算置信度
        successful_calls = sum(1 for step in reasoning_steps
                             if step.step_type == "observation"
                             and step.tool_result and step.tool_result.get("success"))

        base_confidence = min(successful_calls / 3.0, 1.0) * 0.6  # 基础置信度
        source_confidence = min(len(sources) / 3.0, 1.0) * 0.4   # 来源置信度

        total_confidence = base_confidence + source_confidence
        return round(min(total_confidence, 1.0), 3)

    def get_reasoning_summary(self, response: ReActResponse) -> str:
        """获取推理过程摘要"""
        summary = f"推理过程 (共 {len(response.reasoning)} 步, 工具调用 {response.tool_calls_made} 次):\n"

        for i, step in enumerate(response.reasoning, 1):
            step_type_name = {
                "thought": "思考",
                "action": "行动",
                "observation": "观察"
            }.get(step.step_type, step.step_type)

            summary += f"{i}. {step_type_name}: {step.content[:100]}...\n"

        return summary

    async def update_knowledge_base(self, notebook_id: str, **kwargs) -> int:
        """更新知识库"""
        logger.info(f"开始更新笔记本 {notebook_id} 的知识库")
        return await self.knowledge_base.rebuild_knowledge_base(notebook_id, **kwargs)

    def set_system_prompt(self, prompt: str):
        """设置系统提示词"""
        self.system_prompt = prompt
        logger.info("系统提示词已更新")


def create_react_agent(knowledge_base: RAGKnowledgeBase,
                      model: str = None,
                      **kwargs) -> ReActAgent:
    """
    创建ReAct Agent的便捷函数

    Args:
        knowledge_base: RAG知识库实例
        model: OpenAI模型名称
        **kwargs: 其他参数

    Returns:
        ReActAgent: Agent实例
    """
    return ReActAgent(knowledge_base, model, **kwargs)


async def main():
    """测试代码"""
    # 创建知识库和Agent
    from ..rag.rag_knowledge_base import create_rag_knowledge_base

    rag_kb = create_rag_knowledge_base()
    agent = create_react_agent(rag_kb)

    # 获取笔记本列表
    notebooks = rag_kb.content_extractor.workspace.list_notebooks()
    if not notebooks:
        print("没有找到笔记本")
        return

    test_notebook_id = notebooks[0][0]
    print(f"测试笔记本: {test_notebook_id}")

    # 测试ReAct查询
    test_questions = [
        "这个笔记本的主要内容是什么？",
        "有没有关于API的文档？",
        "请总结一下重要的概念和定义"
    ]

    for question in test_questions:
        print(f"\n问题: {question}")
        print("=" * 60)

        response = await agent.query(question)

        print(f"答案: {response.answer}")
        print(f"工具调用次数: {response.tool_calls_made}")
        print(f"置信度: {response.final_confidence}")
        print(f"使用来源数: {len(response.sources_used)}")

        print("\n推理过程摘要:")
        print(agent.get_reasoning_summary(response))

        if response.sources_used:
            print("\n主要来源:")
            for source in response.sources_used[:3]:
                print(f"  - {source['title']} (相似度: {source.get('similarity', 0):.3f})")


if __name__ == "__main__":
    asyncio.run(main())