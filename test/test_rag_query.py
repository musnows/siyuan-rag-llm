"""
RAG查询测试脚本
测试已有RAG知识库的查询和Agent问答功能，不创建新数据库
"""

import asyncio
import os
import sys
from pathlib import Path

# 设置tokenizers并行化以避免警告
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# 添加utils目录到Python路径
sys.path.append(str(Path(__file__).parent.parent / "utils"))

from utils.siyuan.siyuan_content import create_content_extractor
from utils.rag.rag_knowledge_base import create_rag_knowledge_base
from utils.rag.rag_query import create_query_engine
from utils.agent.rag_agent import create_rag_agent
from utils.logger import get_logger

logger = get_logger(__name__)


async def list_existing_collections():
    """列出所有已有的RAG集合"""
    print("=" * 60)
    print("步骤1: 列出已有RAG集合")
    print("=" * 60)

    try:
        # 创建知识库实例
        rag_kb = create_rag_knowledge_base()

        # 使用ChromaDB客户端获取所有集合
        collections = rag_kb.client.list_collections()

        if not collections:
            print("❌ 没有找到任何RAG集合")
            return []

        print(f"✅ 找到 {len(collections)} 个RAG集合:")
        for i, collection in enumerate(collections, 1):
            print(f"  {i}. {collection.name} (ID: {collection.id})")

            # 获取集合统计信息
            try:
                collection_obj = rag_kb.client.get_collection(collection.name)
                doc_count = collection_obj.count()
                print(f"     文档数量: {doc_count}")
            except Exception as e:
                print(f"     无法获取统计信息: {e}")

        # 转换为字典格式
        collection_dicts = []
        for collection in collections:
            try:
                collection_obj = rag_kb.client.get_collection(collection.name)
                doc_count = collection_obj.count()
                collection_dicts.append({
                    'id': collection.id,
                    'name': collection.name,
                    'document_count': doc_count
                })
            except Exception as e:
                collection_dicts.append({
                    'id': collection.id,
                    'name': collection.name,
                    'document_count': 0
                })

        return collection_dicts

    except Exception as e:
        print(f"❌ 获取RAG集合失败: {e}")
        return []


async def test_collection_loading(collection_id: str, collection_name: str):
    """测试集合加载"""
    print(f"\n" + "=" * 60)
    print(f"步骤2: 验证RAG集合 - {collection_name}")
    print("=" * 60)

    try:
        # 创建知识库实例，指定集合名称
        rag_kb = create_rag_knowledge_base()

        # 尝试获取指定的集合
        try:
            collection = rag_kb.client.get_collection(collection_name)
            doc_count = collection.count()

            if doc_count > 0:
                print(f"✅ 成功访问集合: {collection_name}")
                print(f"📊 集合统计:")
                print(f"  集合名称: {collection_name}")
                print(f"  集合ID: {collection_id}")
                print(f"  文档数量: {doc_count}")
                print(f"  持久化目录: {rag_kb.persist_directory}")
                print(f"  嵌入模型: {rag_kb.embedding_model}")

                # 重新初始化知识库以使用指定集合
                rag_kb.collection = collection
                rag_kb.collection_name = collection_name

                return rag_kb
            else:
                print(f"⚠️  集合 {collection_name} 存在但没有文档")
                return None

        except Exception as e:
            print(f"❌ 无法访问集合 {collection_name}: {e}")
            return None

    except Exception as e:
        print(f"❌ 集合验证失败: {e}")
        return None


async def test_rag_query_existing(rag_kb, collection_id: str, collection_name: str):
    """测试RAG查询已有集合"""
    print(f"\n" + "=" * 60)
    print(f"步骤3: 查询RAG集合 - {collection_name}")
    print("=" * 60)

    try:
        # 创建查询引擎
        query_engine = create_query_engine(rag_kb)

        # 测试查询
        test_queries = [
            "这个集合的主要内容是什么？",
            "有哪些重要的概念？",
            "关键要点总结",
            "核心内容概述"
        ]

        for i, query in enumerate(test_queries, 1):
            print(f"\n🔍 查询 {i}: {query}")
            print("-" * 40)

            try:
                # 获取集合中实际存在的notebook_id
                # 先查询所有文档，获取第一个文档的notebook_id
                sample_results = rag_kb.collection.get(limit=1)
                if sample_results["metadatas"]:
                    actual_notebook_id = sample_results["metadatas"][0].get("notebook_id")
                    if actual_notebook_id:
                        result = await query_engine.query(
                            query_text=query,
                            notebook_id=actual_notebook_id,  # 使用实际的notebook_id
                            n_results=3
                        )
                    else:
                        # 如果没有notebook_id，不使用过滤条件
                        result = await query_engine.query(
                            query_text=query,
                            notebook_id=None,  # 不使用过滤
                            n_results=3
                        )
                else:
                    # 集合为空
                    result = await query_engine.query(
                        query_text=query,
                        notebook_id=None,
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

            except Exception as e:
                print(f"❌ 查询 {i} 失败: {e}")

        return query_engine

    except Exception as e:
        print(f"❌ RAG查询测试失败: {e}")
        return None


async def test_agent_query_existing(rag_kb, collection_id: str, collection_name: str):
    """测试Agent问答已有集合"""
    print(f"\n" + "=" * 60)
    print(f"步骤4: Agent智能问答 - {collection_name}")
    print("=" * 60)

    try:
        # 检查环境变量
        if not os.getenv("OPENAI_API_KEY"):
            print("⚠️  未设置 OPENAI_API_KEY 环境变量，跳过Agent测试")
            print("请设置环境变量后重新测试")
            return

        # 创建查询引擎 - 使用更低的相似度阈值以处理复杂查询
        from utils.rag.rag_query import create_query_engine
        query_engine_for_agent = create_query_engine(
            knowledge_base=rag_kb,
            similarity_threshold=0.3  # 降低相似度阈值，因为Agent查询通常更复杂
        )

        # 创建Agent，使用自定义的查询引擎
        from utils.agent.rag_agent import RAGAgent
        agent = RAGAgent(
            knowledge_base=rag_kb,
            model="gpt-3.5-turbo",
            max_tokens=1000,
            temperature=0.1
        )
        # 替换默认的查询引擎
        agent.query_engine = query_engine_for_agent

        print("🤖 Agent初始化成功")

        # 测试问答
        test_questions = [
            f"请简单介绍一下{collection_name}这个知识库的主要内容",
            "这个知识库中有哪些重要的概念或术语？",
            "可以总结一下这个知识库的关键要点吗？",
            "基于这个知识库的内容，有什么建议或见解？"
        ]

        for i, question in enumerate(test_questions, 1):
            print(f"\n❓ 问题 {i}: {question}")
            print("-" * 50)

            try:
                # 获取集合中实际存在的notebook_id
                sample_results = rag_kb.collection.get(limit=1)
                if sample_results["metadatas"]:
                    actual_notebook_id = sample_results["metadatas"][0].get("notebook_id")
                    response = await agent.query(
                        question=question,
                        notebook_id=actual_notebook_id,  # 使用实际的notebook_id
                        context_strategy="simple"
                    )
                else:
                    response = await agent.query(
                        question=question,
                        notebook_id=None,  # 不使用过滤
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


async def test_multi_query_existing(query_engine, rag_kb, collection_id: str, collection_name: str):
    """测试多查询策略"""
    print(f"\n" + "=" * 60)
    print(f"步骤5: 多查询策略 - {collection_name}")
    print("=" * 60)

    try:
        # 测试多查询
        queries = ["主要内容", "重要概念", "关键要点"]
        print(f"🔍 多查询测试: {queries}")

        # 测试不同的合并策略
        strategies = ["union", "weighted"]

        for strategy in strategies:
            print(f"\n📋 策略: {strategy}")
            print("-" * 30)

            try:
                # 获取集合中实际存在的notebook_id
                sample_results = rag_kb.collection.get(limit=1)
                if sample_results["metadatas"]:
                    actual_notebook_id = sample_results["metadatas"][0].get("notebook_id")
                    result = await query_engine.multi_query(
                        queries=queries,
                        notebook_id=actual_notebook_id,  # 使用实际的notebook_id
                        combine_strategy=strategy,
                        n_results=3
                    )
                else:
                    result = await query_engine.multi_query(
                        queries=queries,
                        notebook_id=None,  # 不使用过滤
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
                print(f"❌ 策略 {strategy} 测试失败: {e}")

    except Exception as e:
        print(f"❌ 多查询策略测试失败: {e}")


async def interactive_query_mode(rag_kb, collection_id: str, collection_name: str):
    """交互式查询模式"""
    print(f"\n" + "=" * 60)
    print(f"交互式查询模式 - {collection_name}")
    print("输入 'quit' 或 'exit' 退出")
    print("=" * 60)

    try:
        # 创建查询引擎
        query_engine = create_query_engine(rag_kb)

        while True:
            print("\n" + "-" * 40)
            query = input("🔍 请输入查询内容: ").strip()

            if query.lower() in ['quit', 'exit', '退出']:
                print("👋 退出交互模式")
                break

            if not query:
                continue

            try:
                # 获取集合中实际存在的notebook_id
                sample_results = rag_kb.collection.get(limit=1)
                if sample_results["metadatas"]:
                    actual_notebook_id = sample_results["metadatas"][0].get("notebook_id")
                    result = await query_engine.query(
                        query_text=query,
                        notebook_id=actual_notebook_id,
                        n_results=3
                    )
                else:
                    result = await query_engine.query(
                        query_text=query,
                        notebook_id=None,
                        n_results=3
                    )

                print(f"\n找到 {len(result.documents)} 个相关文档")

                if result.documents:
                    print("\n最相关的文档:")
                    for i, (doc, similarity) in enumerate(result.documents, 1):
                        print(f"\n{i}. {doc.note_title} (相似度: {similarity:.3f})")
                        print(f"内容: {doc.content[:200]}...")

                    if result.sources:
                        print(f"\n来源笔记:")
                        for source in result.sources:
                            print(f"  - {source['title']} (路径: {source['path']})")
                else:
                    print("没有找到相关文档")

            except Exception as e:
                print(f"❌ 查询失败: {e}")

    except KeyboardInterrupt:
        print("\n\n👋 用户中断，退出交互模式")
    except Exception as e:
        print(f"❌ 交互模式失败: {e}")


async def main():
    """主测试函数"""
    print("🚀 开始RAG数据库查询测试")
    print("注意：此测试仅查询已有数据库，不会创建新数据库")
    print("测试环境检查...")

    # 检查环境变量
    env_vars = ["OPENAI_API_KEY"]
    missing_vars = []

    for var in env_vars:
        if not os.getenv(var):
            missing_vars.append(var)

    if missing_vars:
        print(f"⚠️  缺少环境变量: {', '.join(missing_vars)}")
        if "OPENAI_API_KEY" in missing_vars:
            print("   Agent测试将被跳过")

    try:
        # 步骤1: 列出所有已有集合
        collections = await list_existing_collections()
        if not collections:
            print("❌ 没有找到任何RAG集合，请先运行 test_rag_system.py 创建数据库")
            return

        # 选择要测试的集合
        print(f"\n请选择要测试的集合 (1-{len(collections)}) 或输入 'all' 测试所有集合:")

        # 检查是否在非交互式环境中运行
        try:
            choice = input("选择: ").strip()
        except EOFError:
            # 非交互式环境，自动选择第一个集合
            print("检测到非交互式环境，自动选择第一个集合进行测试")
            choice = "1"

        selected_collections = []
        if choice.lower() == 'all':
            selected_collections = collections
        else:
            try:
                index = int(choice) - 1
                if 0 <= index < len(collections):
                    selected_collections = [collections[index]]
                else:
                    print("❌ 无效选择")
                    return
            except ValueError:
                print("❌ 无效输入")
                return

        # 测试每个选中的集合
        for collection in selected_collections:
            collection_id = collection['id']
            collection_name = collection['name']

            # 步骤2: 加载集合
            rag_kb = await test_collection_loading(collection_id, collection_name)
            if not rag_kb:
                print(f"❌ 集合 {collection_name} 加载失败，跳过")
                continue

            # 步骤3: 测试RAG查询
            query_engine = await test_rag_query_existing(rag_kb, collection_id, collection_name)

            # 步骤4: 测试Agent问答（如果有API密钥）
            await test_agent_query_existing(rag_kb, collection_id, collection_name)

            # 步骤5: 测试多查询策略
            if query_engine:
                await test_multi_query_existing(query_engine, rag_kb, collection_id, collection_name)

            # 询问是否进入交互模式
            if len(selected_collections) == 1:  # 只在单个集合测试时提供交互模式
                try:
                    interactive = input(f"\n是否进入 {collection_name} 的交互查询模式？ (y/n): ").strip().lower()
                    if interactive in ['y', 'yes', '是']:
                        await interactive_query_mode(rag_kb, collection_id, collection_name)
                except EOFError:
                    # 非交互式环境，跳过交互模式
                    print("检测到非交互式环境，跳过交互查询模式")

        print("\n" + "=" * 60)
        print("🎉 RAG数据库查询测试完成!")
        print("=" * 60)

    except Exception as e:
        print(f"❌ 测试过程中发生错误: {e}")
        logger.error(f"测试失败: {e}", exc_info=True)


if __name__ == "__main__":
    asyncio.run(main())