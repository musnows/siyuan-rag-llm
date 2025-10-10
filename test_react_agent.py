#!/usr/bin/env python3
"""
ReAct Agent测试脚本
测试基于ReAct模式的智能Agent功能
"""

import asyncio
import os
import sys
import json
from typing import List, Dict, Any

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.rag.rag_knowledge_base import create_rag_knowledge_base
from utils.agent.react_agent import create_react_agent
from utils.agent.rag_tools import create_rag_toolkit
from utils.logger import get_logger

logger = get_logger(__name__)


async def test_rag_tools():
    """测试RAG工具功能"""
    print("=== 测试RAG工具功能 ===")

    # 创建知识库
    rag_kb = create_rag_knowledge_base()

    # 获取笔记本列表
    notebooks = rag_kb.content_extractor.workspace.list_notebooks()
    if not notebooks:
        print("没有找到笔记本，跳过RAG工具测试")
        return False

    test_notebook_id = notebooks[0][0]
    print(f"使用笔记本: {test_notebook_id}")

    # 创建工具包
    toolkit = create_rag_toolkit(rag_kb)

    # 测试搜索工具
    print("\n1. 测试rag_search工具:")
    search_result = await toolkit.call_tool("rag_search", {
        "query": "测试",
        "notebook_id": test_notebook_id,
        "max_results": 3
    })
    print(f"搜索结果: {json.dumps(search_result, ensure_ascii=False, indent=2)}")

    # 测试统计工具
    print("\n2. 测试rag_get_stats工具:")
    stats_result = await toolkit.call_tool("rag_get_stats", {})
    print(f"统计结果: {json.dumps(stats_result, ensure_ascii=False, indent=2)}")

    # 测试多查询工具
    print("\n3. 测试rag_multi_query工具:")
    multi_result = await toolkit.call_tool("rag_multi_query", {
        "queries": ["测试", "文档"],
        "notebook_id": test_notebook_id,
        "combine_strategy": "union"
    })
    print(f"多查询结果: {json.dumps(multi_result, ensure_ascii=False, indent=2)}")

    return True


async def test_react_agent_simple():
    """测试ReAct Agent简单查询"""
    print("\n=== 测试ReAct Agent简单查询 ===")

    # 创建知识库和Agent
    rag_kb = create_rag_knowledge_base()
    agent = create_react_agent(rag_kb, max_tool_calls=3)

    # 获取笔记本列表
    notebooks = rag_kb.content_extractor.workspace.list_notebooks()
    if not notebooks:
        print("没有找到笔记本，跳过Agent测试")
        return False

    test_notebook_id = notebooks[0][0]
    print(f"使用笔记本: {test_notebook_id}")

    # 简单测试问题
    test_questions = [
        "这个笔记本的主要内容是什么？",
        "查找关于测试的文档",
        "有没有API相关的说明？"
    ]

    for i, question in enumerate(test_questions, 1):
        print(f"\n--- 问题 {i}: {question} ---")

        try:
            response = await agent.query(question)

            print(f"答案: {response.answer}")
            print(f"工具调用次数: {response.tool_calls_made}")
            print(f"置信度: {response.final_confidence}")
            print(f"使用来源数: {len(response.sources_used)}")

            # 显示推理过程
            print("\n推理过程:")
            for j, step in enumerate(response.reasoning, 1):
                step_type_name = {
                    "thought": "思考",
                    "action": "行动",
                    "observation": "观察"
                }.get(step.step_type, step.step_type)

                print(f"  {j}. {step_type_name}: {step.content[:100]}...")

            # 显示来源
            if response.sources_used:
                print("\n主要来源:")
                for source in response.sources_used[:3]:
                    print(f"  - {source['title']} (相似度: {source.get('similarity', 0):.3f})")

        except Exception as e:
            print(f"查询失败: {e}")
            logger.error(f"ReAct Agent查询失败: {e}")

    return True


async def test_react_agent_complex():
    """测试ReAct Agent复杂查询"""
    print("\n=== 测试ReAct Agent复杂查询 ===")

    # 创建知识库和Agent
    rag_kb = create_rag_knowledge_base()
    agent = create_react_agent(rag_kb, max_tool_calls=5, max_steps=15)

    # 获取笔记本列表
    notebooks = rag_kb.content_extractor.workspace.list_notebooks()
    if not notebooks:
        print("没有找到笔记本，跳过复杂查询测试")
        return False

    test_notebook_id = notebooks[0][0]
    print(f"使用笔记本: {test_notebook_id}")

    # 复杂测试问题
    complex_questions = [
        "请总结这个笔记本中的所有重要概念和定义",
        "查找关于数据处理流程的相关文档，并说明各个步骤的作用",
        "这个项目中使用了哪些技术栈？请分别说明它们的作用"
    ]

    for i, question in enumerate(complex_questions, 1):
        print(f"\n--- 复杂问题 {i}: {question} ---")

        try:
            response = await agent.query(question)

            print(f"答案: {response.answer}")
            print(f"工具调用次数: {response.tool_calls_made}")
            print(f"推理步数: {len(response.reasoning)}")
            print(f"置信度: {response.final_confidence}")

            # 分析推理模式
            thought_count = sum(1 for step in response.reasoning if step.step_type == "thought")
            action_count = sum(1 for step in response.reasoning if step.step_type == "action")
            observation_count = sum(1 for step in response.reasoning if step.step_type == "observation")

            print(f"推理分析: 思考 {thought_count} 次, 行动 {action_count} 次, 观察 {observation_count} 次")

            # 显示详细推理过程
            print("\n详细推理过程:")
            for j, step in enumerate(response.reasoning, 1):
                step_type_name = {
                    "thought": "思考",
                    "action": "行动",
                    "observation": "观察"
                }.get(step.step_type, step.step_type)

                print(f"\n{j}. [{step_type_name}] {step.content}")

                if step.tool_call:
                    print(f"   工具调用: {step.tool_call['name']}")
                    print(f"   参数: {step.tool_call['arguments']}")

                if step.tool_result:
                    success = step.tool_result.get('success', False)
                    results_count = len(step.tool_result.get('results', []))
                    print(f"   结果: {'成功' if success else '失败'}, 结果数量: {results_count}")

        except Exception as e:
            print(f"复杂查询失败: {e}")
            logger.error(f"ReAct Agent复杂查询失败: {e}")

    return True


async def test_react_agent_comparison():
    """对比测试ReAct Agent和传统Agent"""
    print("\n=== 对比测试ReAct Agent和传统Agent ===")

    # 创建知识库
    rag_kb = create_rag_knowledge_base()

    # 获取笔记本列表
    notebooks = rag_kb.content_extractor.workspace.list_notebooks()
    if not notebooks:
        print("没有找到笔记本，跳过对比测试")
        return False

    test_notebook_id = notebooks[0][0]

    # 创建ReAct Agent和传统Agent
    react_agent = create_react_agent(rag_kb, max_tool_calls=3)

    # 导入传统Agent
    try:
        from utils.agent.rag_agent import create_rag_agent
        traditional_agent = create_rag_agent(rag_kb)
        use_traditional = True
    except Exception as e:
        print(f"无法导入传统Agent: {e}")
        use_traditional = False

    # 测试问题
    test_question = "请总结这个笔记本的主要内容，包括重要的概念和流程"

    print(f"测试问题: {test_question}")

    # 测试ReAct Agent
    print("\n--- ReAct Agent 回答 ---")
    try:
        react_response = await react_agent.query(test_question)
        print(f"ReAct答案: {react_response.answer[:200]}...")
        print(f"工具调用次数: {react_response.tool_calls_made}")
        print(f"置信度: {react_response.final_confidence}")
        print(f"推理步数: {len(react_response.reasoning)}")
    except Exception as e:
        print(f"ReAct Agent失败: {e}")

    # 测试传统Agent
    if use_traditional:
        print("\n--- 传统Agent 回答 ---")
        try:
            traditional_response = await traditional_agent.query(test_question, test_notebook_id)
            print(f"传统答案: {traditional_response.answer[:200]}...")
            print(f"置信度: {traditional_response.confidence}")
            print(f"来源数量: {len(traditional_response.sources)}")
        except Exception as e:
            print(f"传统Agent失败: {e}")

    return True


async def interactive_test():
    """交互式测试"""
    print("\n=== 交互式测试 ===")
    print("输入问题来测试ReAct Agent，输入 'quit' 退出")

    # 创建Agent
    rag_kb = create_rag_knowledge_base()
    agent = create_react_agent(rag_kb, max_tool_calls=5)

    while True:
        try:
            question = input("\n请输入问题: ").strip()

            if question.lower() in ['quit', 'exit', '退出']:
                break

            if not question:
                continue

            print(f"\n正在处理: {question}")
            print("-" * 50)

            response = await agent.query(question)

            print(f"\n答案:\n{response.answer}")
            print(f"\n统计信息:")
            print(f"  - 工具调用次数: {response.tool_calls_made}")
            print(f"  - 推理步数: {len(response.reasoning)}")
            print(f"  - 置信度: {response.final_confidence}")
            print(f"  - 使用来源数: {len(response.sources_used)}")

            if response.sources_used:
                print(f"\n主要来源:")
                for source in response.sources_used[:5]:
                    print(f"  - {source['title']} (相似度: {source.get('similarity', 0):.3f})")

        except KeyboardInterrupt:
            print("\n\n用户中断，退出测试")
            break
        except Exception as e:
            print(f"处理失败: {e}")
            logger.error(f"交互式测试失败: {e}")


async def main():
    """主测试函数"""
    print("开始ReAct Agent功能测试")
    print("=" * 60)

    # 检查环境变量
    if not os.getenv("OPENAI_API_KEY"):
        print("错误: 未设置OPENAI_API_KEY环境变量")
        return

    # 检查思源笔记连接
    try:
        rag_kb = create_rag_knowledge_base()
        notebooks = rag_kb.content_extractor.workspace.list_notebooks()
        if not notebooks:
            print("警告: 未找到思源笔记笔记本，某些测试可能失败")
        else:
            print(f"找到 {len(notebooks)} 个笔记本")
    except Exception as e:
        print(f"思源笔记连接失败: {e}")
        print("某些测试可能会失败")

    # 运行测试
    tests = [
        ("RAG工具测试", test_rag_tools),
        ("ReAct Agent简单查询测试", test_react_agent_simple),
        ("ReAct Agent复杂查询测试", test_react_agent_complex),
        ("ReAct Agent对比测试", test_react_agent_comparison),
        ("交互式测试", interactive_test)
    ]

    results = {}

    for test_name, test_func in tests:
        print(f"\n{'='*60}")
        print(f"运行: {test_name}")
        print('='*60)

        try:
            result = await test_func()
            results[test_name] = "成功" if result else "跳过"
            print(f"\n{test_name}: {'成功' if result else '跳过'}")
        except Exception as e:
            results[test_name] = f"失败: {e}"
            print(f"\n{test_name} 失败: {e}")
            logger.error(f"{test_name} 失败: {e}")

        # 除了交互式测试，其他测试后都暂停一下
        if test_name != "交互式测试":
            input("\n按回车继续下一个测试...")

    # 显示测试总结
    print(f"\n{'='*60}")
    print("测试总结")
    print('='*60)

    for test_name, result in results.items():
        print(f"{test_name}: {result}")

    print("\n测试完成！")


if __name__ == "__main__":
    asyncio.run(main())