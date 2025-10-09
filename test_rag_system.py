"""
RAG系统测试脚本
测试RAG知识库构建、查询和Agent问答功能
"""

import asyncio
import os
import sys
from pathlib import Path

# 设置tokenizers并行化以避免警告
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# 添加utils目录到Python路径
sys.path.append(str(Path(__file__).parent / "utils"))

from utils.siyuan.siyuan_content import create_content_extractor
from utils.rag.rag_knowledge_base import create_rag_knowledge_base
from utils.rag.rag_query import create_query_engine
from utils.agent.rag_agent import create_rag_agent
from utils.logger import get_logger

logger = get_logger(__name__)


async def test_siyuan_connection():
    """测试思源笔记连接"""
    print("=" * 60)
    print("测试1: 思源笔记连接")
    print("=" * 60)

    try:
        # 创建内容提取器
        extractor = create_content_extractor()

        # 获取笔记本列表
        notebooks = extractor.workspace.list_notebooks()
        if not notebooks:
            print("❌ 没有找到思源笔记本")
            return None

        print(f"✅ 成功连接思源笔记，找到 {len(notebooks)} 个笔记本:")
        for i, (notebook_id, notebook_name) in enumerate(notebooks):
            print(f"  {i+1}. {notebook_name} (ID: {notebook_id})")

        print(f"📚 开始为笔记本 {notebooks[-1][0]} 构建知识库...")
        return notebooks[-1][0]  # 返回第一个笔记本ID

    except Exception as e:
        print(f"❌ 思源笔记连接失败: {e}")
        return None


async def test_knowledge_base_build(notebook_id: str):
    """测试知识库构建"""
    print("\n" + "=" * 60)
    print("测试2: RAG知识库构建")
    print("=" * 60)

    try:
        # 创建知识库
        rag_kb = create_rag_knowledge_base()

        print(f"📚 开始为笔记本 {notebook_id} 构建知识库...")

        # 构建知识库
        doc_count = await rag_kb.build_knowledge_base(
            notebook_id=notebook_id,
            chunk_size=800,
            chunk_overlap=200,
            batch_size=5
        )

        if doc_count > 0:
            print(f"✅ 知识库构建成功，处理了 {doc_count} 个文档块")

            # 显示统计信息
            stats = rag_kb.get_collection_stats()
            print(f"📊 知识库统计:")
            for key, value in stats.items():
                print(f"  {key}: {value}")

            return rag_kb
        else:
            print("❌ 知识库构建失败，没有处理任何文档")
            return None

    except Exception as e:
        print(f"❌ 知识库构建失败: {e}")
        return None


async def test_rag_query(rag_kb, notebook_id: str):
    """测试RAG查询"""
    print("\n" + "=" * 60)
    print("测试3: RAG查询功能")
    print("=" * 60)

    try:
        # 创建查询引擎
        query_engine = create_query_engine(rag_kb)

        # 测试查询
        test_queries = [
            "什么是主要概念",
            "有哪些重要内容",
            "测试相关的内容",
            "总结要点"
        ]

        for i, query in enumerate(test_queries, 1):
            print(f"\n🔍 查询 {i}: {query}")
            print("-" * 40)

            result = await query_engine.query(
                query_text=query,
                notebook_id=notebook_id,
                n_results=3
            )

            print(f"找到 {len(result.documents)} 个相关文档")
            print(f"上下文长度: {len(result.context)} 字符")

            if result.documents:
                print("最相关的文档:")
                for j, (doc, similarity) in enumerate(result.documents[:3], 1):
                    print(f"  {j}. {doc.note_title}")
                    print(f"     相似度: {similarity:.3f}")
                    print(f"     预览: {doc.content[:100]}...")

                if result.sources:
                    print("来源笔记:")
                    for source in result.sources:
                        print(f"  - {source['title']} (路径: {source['path']})")
            else:
                print("没有找到相关文档")

        return query_engine

    except Exception as e:
        print(f"❌ RAG查询测试失败: {e}")
        return None


async def test_agent_query(rag_kb, notebook_id: str):
    """测试Agent问答"""
    print("\n" + "=" * 60)
    print("测试4: Agent智能问答")
    print("=" * 60)

    try:
        # 检查环境变量
        if not os.getenv("OPENAI_API_KEY"):
            print("⚠️  未设置 OPENAI_API_KEY 环境变量，跳过Agent测试")
            print("请设置环境变量后重新测试")
            return

        # 创建Agent
        agent = create_rag_agent(
            knowledge_base=rag_kb,
            model="gpt-3.5-turbo",
            max_tokens=1000,
            temperature=0.1
        )

        print("🤖 Agent初始化成功")

        # 测试问答
        test_questions = [
            "请简单介绍一下这个笔记本的主要内容",
            "有哪些重要的概念或术语？",
            "可以总结一下关键要点吗？",
            "基于这些内容，有什么建议或见解？"
        ]

        for i, question in enumerate(test_questions, 1):
            print(f"\n❓ 问题 {i}: {question}")
            print("-" * 50)

            try:
                response = await agent.query(
                    question=question,
                    notebook_id=notebook_id,
                    context_strategy="simple"
                )

                print(f"🤖 回答:")
                print(response.answer)

                print(f"\n📊 回答信息:")
                print(f"  置信度: {response.confidence}")
                print(f"  来源数量: {len(response.sources)}")

                if response.sources:
                    print("  引用来源:")
                    for source in response.sources:
                        print(f"    - {source['title']} (相似度: {source['similarity']:.3f})")

            except Exception as e:
                print(f"❌ 问题 {i} 回答失败: {e}")

        # 显示对话摘要
        summary = agent.get_conversation_summary()
        print(f"\n📝 对话摘要:")
        for key, value in summary.items():
            print(f"  {key}: {value}")

    except Exception as e:
        print(f"❌ Agent测试失败: {e}")


async def test_multi_query_strategy(query_engine, notebook_id: str):
    """测试多查询策略"""
    print("\n" + "=" * 60)
    print("测试5: 多查询策略")
    print("=" * 60)

    try:
        # 测试多查询
        queries = ["概念", "内容", "总结"]
        print(f"🔍 多查询测试: {queries}")

        # 测试不同的合并策略
        strategies = ["union", "weighted"]

        for strategy in strategies:
            print(f"\n📋 策略: {strategy}")
            print("-" * 30)

            result = await query_engine.multi_query(
                queries=queries,
                notebook_id=notebook_id,
                combine_strategy=strategy,
                n_results=3
            )

            print(f"合并后文档数: {len(result.documents)}")
            print(f"上下文长度: {len(result.context)}")

            if result.documents:
                print("top 3 文档:")
                for i, (doc, similarity) in enumerate(result.documents[:3], 1):
                    print(f"  {i}. {doc.note_title} (相似度: {similarity:.3f})")

    except Exception as e:
        print(f"❌ 多查询策略测试失败: {e}")


async def main():
    """主测试函数"""
    print("🚀 开始RAG系统完整测试")
    print("测试环境检查...")

    # 检查环境变量
    env_vars = ["OPENAI_API_KEY", "SIYUAN_API_HOST", "SIYUAN_API_TOKEN"]
    missing_vars = []

    for var in env_vars:
        if not os.getenv(var):
            missing_vars.append(var)

    if missing_vars:
        print(f"⚠️  缺少环境变量: {', '.join(missing_vars)}")
        if "OPENAI_API_KEY" in missing_vars:
            print("   Agent测试将被跳过")

    try:
        # 测试1: 思源笔记连接
        notebook_id = await test_siyuan_connection()
        if not notebook_id:
            print("❌ 思源笔记连接失败，测试终止")
            return

        # 测试2: 知识库构建
        rag_kb = await test_knowledge_base_build(notebook_id)
        if not rag_kb:
            print("❌ 知识库构建失败，测试终止")
            return

        # 测试3: RAG查询
        query_engine = await test_rag_query(rag_kb, notebook_id)

        # 测试4: Agent问答（如果有API密钥）
        await test_agent_query(rag_kb, notebook_id)

        # 测试5: 多查询策略
        if query_engine:
            await test_multi_query_strategy(query_engine, notebook_id)

        print("\n" + "=" * 60)
        print("🎉 所有测试完成!")
        print("=" * 60)

    except Exception as e:
        print(f"❌ 测试过程中发生错误: {e}")
        logger.error(f"测试失败: {e}", exc_info=True)


if __name__ == "__main__":
    asyncio.run(main())