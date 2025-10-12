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
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.rag.rag_knowledge_base import create_rag_knowledge_base, create_rag_knowledge_base_with_openai
from utils.agent.react_agent import create_react_agent
from utils.agent.rag_tools import create_rag_toolkit
from utils.logger import get_logger

logger = get_logger(__name__)


class SiYuanConnectionError(Exception):
    """思源笔记连接错误"""
    pass


async def check_siyuan_connection():
    """
    检查思源笔记连接是否正常

    Raises:
        SiYuanConnectionError: 当无法连接到思源笔记时抛出
    """
    try:
        # 创建知识库实例来检查连接
        rag_kb = create_knowledge_base("connection_test")

        # 尝试获取笔记本列表来验证连接
        notebooks = rag_kb.content_extractor.workspace.list_notebooks()
        logger.info(f"✅ 思源笔记连接成功，找到 {len(notebooks)} 个笔记本")

    except ConnectionError as e:
        error_msg = f"无法连接到思源笔记: {e}"
        logger.error(error_msg)
        raise SiYuanConnectionError(error_msg)
    except Exception as e:
        # 检查是否是网络连接相关的错误
        error_str = str(e).lower()
        if any(keyword in error_str for keyword in [
            "connection failed", "connect call failed", "connection refused",
            "timeout", "network", "host", "port", "ssl"
        ]):
            error_msg = f"思源笔记网络连接失败: {e}"
            logger.error(error_msg)
            raise SiYuanConnectionError(error_msg)
        else:
            # 其他类型的错误，重新抛出
            raise


def create_knowledge_base(persist_directory_suffix: str = "", force_rebuild: bool = False):
    """
    根据环境变量创建知识库（不自动构建）

    Args:
        persist_directory_suffix: 持久化目录后缀
        force_rebuild: 是否强制重建知识库（用于异步调用）

    Returns:
        RAGKnowledgeBase: 知识库实例
    """
    # 获取embedding模型配置
    embedding_model = os.getenv("EMBEDDING_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    api_key = os.getenv("OPENAI_API_KEY")
    api_base = os.getenv("OPENAI_BASE_URL")

    # 判断是否使用OpenAI embedding模型
    is_openai_embedding = embedding_model.startswith("text-embedding-") or embedding_model in [
        "text-embedding-ada-002",
        "text-embedding-3-small",
        "text-embedding-3-large"
    ]

    if is_openai_embedding:
        # 使用OpenAI嵌入模型
        if not api_key:
            raise ValueError("使用OpenAI嵌入模型需要设置OPENAI_API_KEY环境变量")

        persist_dir = f"./data/rag_db_openai_{persist_directory_suffix}" if persist_directory_suffix else "./data/rag_db_openai"

        logger.info(f"创建OpenAI嵌入知识库，embedding模型: {embedding_model}")
        return create_rag_knowledge_base_with_openai(
            persist_directory=persist_dir,
            embedding_model=embedding_model,
            api_key=api_key,
            api_base=api_base,
            collection_name=f"siyuan_notes_openai_{persist_directory_suffix}" if persist_directory_suffix else "siyuan_notes_openai"
        )
    else:
        # 使用本地HuggingFace嵌入模型
        persist_dir = f"./data/rag_db_local_{persist_directory_suffix}" if persist_directory_suffix else "./data/rag_db_local"

        logger.info(f"创建本地嵌入知识库，embedding模型: {embedding_model}")
        return create_rag_knowledge_base(
            persist_directory=persist_dir,
            embedding_model=embedding_model,
            use_openai_embedding=False,
            collection_name=f"siyuan_notes_local_{persist_directory_suffix}" if persist_directory_suffix else "siyuan_notes_local"
        )


async def create_and_build_knowledge_base(persist_directory_suffix: str = "", force_rebuild: bool = False):
    """
    创建并构建知识库（异步版本）

    Args:
        persist_directory_suffix: 持久化目录后缀
        force_rebuild: 是否强制重建知识库

    Returns:
        RAGKnowledgeBase: 知识库实例
    """
    # 创建知识库实例
    rag_kb = create_knowledge_base(persist_directory_suffix, force_rebuild)

    # 构建知识库
    await build_knowledge_base_if_needed(rag_kb, force_rebuild)

    return rag_kb


async def select_notebook_and_build(rag_kb) -> bool:
    """
    让用户选择笔记本并构建知识库

    Args:
        rag_kb: 知识库实例

    Returns:
        bool: 是否成功构建
    """
    try:
        # 首先检查思源笔记连接
        print("🔍 检查思源笔记连接...")
        await check_siyuan_connection()

        # 获取笔记本列表
        notebooks = rag_kb.content_extractor.workspace.list_notebooks()
        if not notebooks:
            print("❌ 没有找到思源笔记笔记本")
            return False

        print(f"\n📚 找到 {len(notebooks)} 个笔记本:")
        for i, (nb_id, nb_name) in enumerate(notebooks, 1):
            # 检查该笔记本是否已有数据
            existing_count = await rag_kb.get_notebook_document_count(nb_id)
            status = "✅ 已构建" if existing_count > 0 else "❌ 未构建"
            print(f"  {i}. {nb_name} (ID: {nb_id}) - {status} ({existing_count} 个文档块)")

        print("\n请选择要构建知识库的笔记本:")
        try:
            choice = input("输入笔记本编号 (多个用逗号分隔，如: 1,2,3): ").strip()
            if not choice:
                print("❌ 未选择笔记本")
                return False

            # 解析用户选择
            selected_indices = [int(x.strip()) - 1 for x in choice.split(",")]
            selected_notebooks = []

            for idx in selected_indices:
                if 0 <= idx < len(notebooks):
                    selected_notebooks.append(notebooks[idx])
                else:
                    print(f"⚠️ 编号 {idx + 1} 无效，跳过")

            if not selected_notebooks:
                print("❌ 没有选择有效的笔记本")
                return False

            # 询问分块参数
            print("\n📋 构建参数设置 (直接回车使用默认值):")
            chunk_size_input = input("文档分块大小 (默认1000): ").strip()
            chunk_overlap_input = input("分块重叠大小 (默认200): ").strip()
            batch_size_input = input("批处理大小 (默认10): ").strip()

            chunk_size = int(chunk_size_input) if chunk_size_input.isdigit() else 1000
            chunk_overlap = int(chunk_overlap_input) if chunk_overlap_input.isdigit() else 200
            batch_size = int(batch_size_input) if batch_size_input.isdigit() else 10

            print(f"\n🔧 将使用参数: 分块大小={chunk_size}, 重叠={chunk_overlap}, 批处理={batch_size}")

            # 构建选定的笔记本
            total_docs = 0
            for nb_id, nb_name in selected_notebooks:
                print(f"\n📖 开始构建笔记本: {nb_name} (ID: {nb_id})")

                try:
                    doc_count = await rag_kb.build_knowledge_base(
                        notebook_id=nb_id,
                        include_children=True,
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap,
                        batch_size=batch_size,
                        force_rebuild=True  # 用户主动选择，强制重建
                    )

                    total_docs += doc_count
                    print(f"✅ 笔记本 '{nb_name}' 构建完成，共 {doc_count} 个文档块")

                except Exception as e:
                    print(f"❌ 构建笔记本 '{nb_name}' 失败: {e}")
                    logger.error(f"构建笔记本失败: {e}")

            print(f"\n🎉 知识库构建完成！总计 {total_docs} 个文档块")
            return total_docs > 0

        except ValueError:
            print("❌ 输入格式错误，请输入数字编号")
            return False
        except KeyboardInterrupt:
            print("\n❌ 用户取消操作")
            return False

    except Exception as e:
        print(f"❌ 选择笔记本失败: {e}")
        logger.error(f"选择笔记本失败: {e}")
        return False


async def check_existing_data_and_prompt(rag_kb) -> bool:
    """
    检查现有数据并提示用户是否重建

    Args:
        rag_kb: 知识库实例

    Returns:
        bool: 是否需要重新构建
    """
    try:
        # 首先检查思源笔记连接
        print("🔍 检查思源笔记连接...")
        await check_siyuan_connection()

        # 获取所有笔记本
        notebooks = rag_kb.content_extractor.workspace.list_notebooks()
        if not notebooks:
            return True  # 没有笔记本，需要构建

        # 检查是否有现有数据
        total_existing = 0
        notebooks_with_data = []

        for nb_id, nb_name in notebooks:
            count = await rag_kb.get_notebook_document_count(nb_id)
            if count > 0:
                total_existing += count
                notebooks_with_data.append((nb_id, nb_name, count))

        if total_existing == 0:
            print("🆕 未发现现有知识库数据，需要创建新的知识库")
            return True

        # 显示现有数据状态
        print(f"\n📊 发现现有知识库数据:")
        print(f"总文档块数: {total_existing}")
        print(f"已构建的笔记本:")
        for nb_id, nb_name, count in notebooks_with_data:
            print(f"  - {nb_name} (ID: {nb_id}): {count} 个文档块")

        # 询问用户是否重建
        print("\n是否要重新构建知识库？")
        print("1. 重新构建 (删除现有数据，重新创建)")
        print("2. 使用现有数据 (直接进入ReAct Agent模式)")

        while True:
            choice = input("请选择 (1/2): ").strip()
            if choice == "1":
                print("🔄 选择重新构建知识库")
                return True
            elif choice == "2":
                print("✅ 使用现有知识库数据")
                return False
            else:
                print("❌ 无效选择，请输入 1 或 2")

    except Exception as e:
        print(f"❌ 检查现有数据失败: {e}")
        logger.error(f"检查现有数据失败: {e}")
        return True  # 出错时默认重建


async def build_knowledge_base_if_needed(rag_kb, force_rebuild: bool = False):
    """
    如果需要，构建知识库

    Args:
        rag_kb: 知识库实例
        force_rebuild: 是否强制重建
    """
    try:
        # 获取笔记本列表
        notebooks = rag_kb.content_extractor.workspace.list_notebooks()
        if not notebooks:
            logger.warning("没有找到笔记本，跳过知识库构建")
            return

        logger.info(f"找到 {len(notebooks)} 个笔记本，开始检查知识库状态...")

        total_docs = 0
        for nb_id, nb_name in notebooks:
            # 检查该笔记本是否已有数据
            existing_count = await rag_kb.get_notebook_document_count(nb_id)

            if existing_count > 0 and not force_rebuild:
                logger.info(f"笔记本 '{nb_name}' 已有 {existing_count} 个文档块，跳过构建")
                total_docs += existing_count
                continue

            if existing_count > 0 and force_rebuild:
                logger.info(f"强制重建笔记本 '{nb_name}' 的数据 ({existing_count} 个文档块)")

            # 构建该笔记本的知识库
            logger.info(f"开始构建笔记本 '{nb_name}' 的知识库...")
            doc_count = await rag_kb.build_knowledge_base(
                notebook_id=nb_id,
                include_children=True,
                chunk_size=1000,
                chunk_overlap=200,
                batch_size=10,
                force_rebuild=force_rebuild
            )

            total_docs += doc_count
            logger.info(f"笔记本 '{nb_name}' 构建完成，共 {doc_count} 个文档块")

        # 显示统计信息
        stats = rag_kb.get_collection_stats()
        logger.info(f"知识库构建完成！总计 {total_docs} 个文档块")
        logger.info(f"集合名称: {stats.get('collection_name', 'N/A')}")
        logger.info(f"Embedding模型: {stats.get('embedding_model', 'N/A')}")

    except Exception as e:
        logger.error(f"构建知识库失败: {e}")
        raise


async def test_rag_tools():
    """测试RAG工具功能"""
    print("\n=== 🔧 测试RAG工具功能 ===")

    try:
        # 首先检查思源笔记连接
        print("🔍 检查思源笔记连接...")
        await check_siyuan_connection()
    except SiYuanConnectionError as e:
        print(f"❌ {e}")
        print("跳过RAG工具测试")
        return False

    # 创建知识库实例
    print("🔧 正在创建知识库实例...")
    rag_kb = create_knowledge_base("tools_test")

    # 检查现有数据
    stats = rag_kb.get_collection_stats()
    doc_count = stats.get('document_count', 0)

    if doc_count == 0:
        print("❌ 未发现知识库数据，需要先构建知识库")
        success = await select_notebook_and_build(rag_kb)
        if not success:
            print("❌ 知识库构建失败，跳过RAG工具测试")
            return False
    else:
        print(f"✅ 发现现有知识库数据: {doc_count} 个文档块")

    # 获取统计信息
    print(f"\n📊 知识库信息:")
    print(f"  - 文档总数: {stats.get('document_count', 0)}")
    print(f"  - Embedding模型: {stats.get('embedding_model', 'N/A')}")

    # 获取笔记本列表
    notebooks = rag_kb.content_extractor.workspace.list_notebooks()
    if not notebooks:
        print("❌ 没有找到笔记本，跳过RAG工具测试")
        return False

    # 选择有数据的笔记本
    test_notebook_id = None
    for nb_id, nb_name in notebooks:
        count = await rag_kb.get_notebook_document_count(nb_id)
        if count > 0:
            test_notebook_id = nb_id
            print(f"✅ 选择笔记本: {nb_name} (ID: {nb_id}) - {count} 个文档块")
            break

    if not test_notebook_id:
        print("❌ 没有找到包含数据的笔记本")
        return False

    # 创建工具包
    toolkit = create_rag_toolkit(rag_kb)

    # 测试搜索工具
    print("\n1️⃣ 测试rag_search工具:")
    search_result = await toolkit.call_tool("rag_search", {
        "query": "测试",
        "notebook_id": test_notebook_id,
        "max_results": 3
    })
    print(f"搜索结果: {json.dumps(search_result, ensure_ascii=False, indent=2)}")

    # 测试统计工具
    print("\n2️⃣ 测试rag_get_stats工具:")
    stats_result = await toolkit.call_tool("rag_get_stats", {})
    print(f"统计结果: {json.dumps(stats_result, ensure_ascii=False, indent=2)}")

    # 测试多查询工具
    print("\n3️⃣ 测试rag_multi_query工具:")
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

    # 创建并构建知识库
    print("正在创建并构建知识库...")
    rag_kb = await create_and_build_knowledge_base("simple_test")

    # 获取统计信息
    stats = rag_kb.get_collection_stats()
    print(f"知识库信息:")
    print(f"  - 文档总数: {stats.get('document_count', 0)}")
    print(f"  - Embedding模型: {stats.get('embedding_model', 'N/A')}")

    # 创建Agent
    agent = create_react_agent(rag_kb, max_tool_calls=3)
    print("✅ ReAct Agent创建成功")

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

    # 创建并构建知识库
    print("正在创建并构建知识库...")
    rag_kb = await create_and_build_knowledge_base("complex_test")

    # 获取统计信息
    stats = rag_kb.get_collection_stats()
    print(f"知识库信息:")
    print(f"  - 文档总数: {stats.get('document_count', 0)}")
    print(f"  - Embedding模型: {stats.get('embedding_model', 'N/A')}")

    # 创建Agent
    agent = create_react_agent(rag_kb, max_tool_calls=5, max_steps=15)
    print("✅ ReAct Agent创建成功")

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

    # 创建并构建知识库
    print("正在创建并构建知识库...")
    rag_kb = await create_and_build_knowledge_base("comparison_test")

    # 获取统计信息
    stats = rag_kb.get_collection_stats()
    print(f"知识库信息:")
    print(f"  - 文档总数: {stats.get('document_count', 0)}")
    print(f"  - Embedding模型: {stats.get('embedding_model', 'N/A')}")

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

    # 创建知识库实例
    print("🔧 正在创建知识库实例...")
    rag_kb = create_knowledge_base("interactive_test")

    # 检查现有数据并询问用户
    need_rebuild = await check_existing_data_and_prompt(rag_kb)

    if need_rebuild:
        # 需要重新构建，让用户选择笔记本
        success = await select_notebook_and_build(rag_kb)
        if not success:
            print("❌ 知识库构建失败，退出测试")
            return
    else:
        print("✅ 使用现有知识库数据")

    # 获取统计信息
    stats = rag_kb.get_collection_stats()
    print(f"\n📊 知识库信息:")
    print(f"  - 文档总数: {stats.get('document_count', 0)}")
    print(f"  - Embedding模型: {stats.get('embedding_model', 'N/A')}")

    # 创建Agent
    agent = create_react_agent(rag_kb, max_tool_calls=5)
    print("✅ ReAct Agent创建成功，可以开始提问了！")

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


async def test_embedding_comparison():
    """对比测试不同embedding模型"""
    print("\n=== 对比测试不同Embedding模型 ===")

    # 获取当前配置
    current_embedding_model = os.getenv("EMBEDDING_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    current_openai_model = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")

    print(f"当前配置:")
    print(f"  - Embedding模型: {current_embedding_model}")
    print(f"  - OpenAI模型: {current_openai_model}")
    print(f"  - API Base: {os.getenv('OPENAI_BASE_URL', 'https://api.openai.com/v1')}")

    # 测试查询
    test_query = "思源笔记的主要功能和特点"

    # 创建当前配置的知识库
    try:
        print(f"\n正在创建并构建知识库...")
        rag_kb = await create_and_build_knowledge_base("embedding_test")

        # 获取统计信息
        stats = rag_kb.get_collection_stats()
        print(f"知识库信息:")
        print(f"  - 文档总数: {stats.get('document_count', 0)}")
        print(f"  - Embedding模型: {stats.get('embedding_model', 'N/A')}")

        agent = create_react_agent(rag_kb, max_tool_calls=3)

        print(f"\n使用当前配置测试:")
        print(f"查询: {test_query}")

        response = await agent.query(test_query)

        print(f"✅ 查询成功!")
        print(f"答案: {response.answer[:200]}...")
        print(f"工具调用次数: {response.tool_calls_made}")
        print(f"置信度: {response.final_confidence}")
        print(f"使用来源数: {len(response.sources_used)}")

    except Exception as e:
        print(f"❌ 测试失败: {e}")
        logger.error(f"Embedding对比测试失败: {e}")

    return True


async def main():
    """主测试函数"""
    print("🚀 思源笔记RAG知识库 + ReAct Agent 测试程序")
    print("=" * 60)

    # 首先检查思源笔记连接
    print("🔍 正在检查思源笔记连接...")
    try:
        await check_siyuan_connection()
        print("✅ 思源笔记连接正常")
    except SiYuanConnectionError as e:
        print(f"❌ {e}")
        print("\n🔧 请确保:")
        print("1. 思源笔记正在运行")
        print("2. 端口 6806 可以访问")
        print("3. 环境变量 SIYUAN_API_TOKEN 已正确设置")
        print("\n程序退出")
        return
    except Exception as e:
        print(f"❌ 检查思源笔记连接时发生未知错误: {e}")
        print("\n程序退出")
        return

    print("\n" + "="*60)

    # 显示当前配置
    embedding_model = os.getenv("EMBEDDING_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    openai_model = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
    openai_api_key = os.getenv("OPENAI_API_KEY")

    print(f"📋 当前配置:")
    print(f"  - Embedding模型: {embedding_model}")
    print(f"  - OpenAI模型: {openai_model}")
    print(f"  - API Base: {os.getenv('OPENAI_BASE_URL', 'https://api.openai.com/v1')}")
    print(f"  - API Key: {'已设置' if openai_api_key else '未设置'}")

    # 检查必要的API Key
    if not openai_api_key:
        print("⚠️ 警告: 未设置OPENAI_API_KEY环境变量")
        print("   如需使用OpenAI功能，请设置该环境变量")

    # 判断embedding类型
    is_openai_embedding = embedding_model.startswith("text-embedding-") or embedding_model in [
        "text-embedding-ada-002",
        "text-embedding-3-small",
        "text-embedding-3-large"
    ]

    if is_openai_embedding and not openai_api_key:
        print("❌ 错误: 使用OpenAI embedding模型需要设置OPENAI_API_KEY")
        return

    # 显示菜单
    print(f"\n📋 请选择测试模式:")
    print("1. 交互式测试 (推荐)")
    print("2. 运行所有测试")
    print("3. 仅测试RAG工具")
    print("4. 仅测试ReAct Agent (简单)")
    print("5. 仅测试ReAct Agent (复杂)")
    print("6. Embedding模型对比测试")
    print("0. 退出")

    while True:
        try:
            choice = input("\n请选择 (0-6): ").strip()

            if choice == "0":
                print("👋 退出程序")
                break
            elif choice == "1":
                print("\n🎯 启动交互式测试...")
                await interactive_test()
                break
            elif choice == "2":
                print("\n🧪 运行所有测试...")
                await run_all_tests()
                break
            elif choice == "3":
                print("\n🔧 测试RAG工具...")
                await test_rag_tools()
                break
            elif choice == "4":
                print("\n🤖 测试ReAct Agent (简单)...")
                await test_react_agent_simple()
                break
            elif choice == "5":
                print("\n🤖 测试ReAct Agent (复杂)...")
                await test_react_agent_complex()
                break
            elif choice == "6":
                print("\n📊 Embedding模型对比测试...")
                await test_embedding_comparison()
                break
            else:
                print("❌ 无效选择，请输入 0-6")

        except KeyboardInterrupt:
            print("\n\n👋 用户中断，退出程序")
            break
        except Exception as e:
            print(f"❌ 执行失败: {e}")
            logger.error(f"执行测试失败: {e}")


async def run_all_tests():
    """运行所有测试"""
    # 检查思源笔记连接和知识库状态
    try:
        print("\n🔍 正在检查思源笔记连接和知识库状态...")
        await check_siyuan_connection()

        rag_kb = create_knowledge_base("main_test")
        notebooks = rag_kb.content_extractor.workspace.list_notebooks()
        stats = rag_kb.get_collection_stats()

        if not notebooks:
            print("⚠️ 警告: 未找到思源笔记笔记本，某些测试可能失败")
        else:
            print(f"✅ 找到 {len(notebooks)} 个笔记本")
            print(f"📊 知识库状态: {stats.get('document_count', 0)} 个文档块")
            print(f"🔤 Embedding模型: {stats.get('embedding_model', 'N/A')}")
    except SiYuanConnectionError as e:
        print(f"❌ {e}")
        print("❌ 无法连接思源笔记，跳过所有需要思源笔记的测试")
        return
    except Exception as e:
        print(f"❌ 检查思源笔记时发生未知错误: {e}")
        print("❌ 跳过所有测试")
        return

    # 运行测试
    tests = [
        ("Embedding模型对比测试", test_embedding_comparison),
        ("RAG工具测试", test_rag_tools),
        ("ReAct Agent简单查询测试", test_react_agent_simple),
        ("ReAct Agent复杂查询测试", test_react_agent_complex),
        ("ReAct Agent对比测试", test_react_agent_comparison)
    ]

    results = {}

    for test_name, test_func in tests:
        print(f"\n{'='*60}")
        print(f"🧪 运行: {test_name}")
        print('='*60)

        try:
            result = await test_func()
            results[test_name] = "✅ 成功" if result else "⏭️ 跳过"
            print(f"\n{test_name}: {'成功' if result else '跳过'}")
        except Exception as e:
            results[test_name] = f"❌ 失败: {e}"
            print(f"\n{test_name} 失败: {e}")
            logger.error(f"{test_name} 失败: {e}")

        input("\n⏸️ 按回车继续下一个测试...")

    # 显示测试总结
    print(f"\n{'='*60}")
    print("📊 测试总结")
    print('='*60)

    for test_name, result in results.items():
        print(f"{test_name}: {result}")

    print("\n🎉 所有测试完成！")


if __name__ == "__main__":
    asyncio.run(main())