#!/usr/bin/env python3
"""
思源笔记RAG知识库交互式CLI
基于ReAct Agent的交互式问答系统
"""

import asyncio
import os
import sys
import argparse

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.rag.rag_knowledge_base import create_rag_knowledge_base, create_rag_knowledge_base_with_openai
from utils.agent.react_agent import create_react_agent
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
        rag_kb = create_knowledge_base()

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


def create_knowledge_base(persist_directory_suffix: str = ""):
    """
    根据环境变量创建知识库（不自动构建）

    Args:
        persist_directory_suffix: 持久化目录后缀（已弃用，为保持兼容性保留）

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

    # 使用固定的持久化目录
    persist_dir = "./data/rag_db"

    if is_openai_embedding:
        # 使用OpenAI嵌入模型
        if not api_key:
            raise ValueError("使用OpenAI嵌入模型需要设置OPENAI_API_KEY环境变量")

        logger.info(f"创建OpenAI嵌入知识库，embedding模型: {embedding_model}")
        return create_rag_knowledge_base_with_openai(
            persist_directory=persist_dir,
            embedding_model=embedding_model,
            api_key=api_key,
            api_base=api_base,
            collection_name="siyuan_notes"
        )
    else:
        # 使用本地HuggingFace嵌入模型
        logger.info(f"创建本地嵌入知识库，embedding模型: {embedding_model}")
        return create_rag_knowledge_base(
            persist_directory=persist_dir,
            embedding_model=embedding_model,
            use_openai_embedding=False,
            collection_name="siyuan_notes"
        )


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

            # 获取配置参数
            chunk_size = int(os.getenv("RAG_CHUNK_SIZE", "1000"))
            chunk_overlap = int(os.getenv("RAG_CHUNK_OVERLAP", "200"))
            batch_size = int(os.getenv("RAG_BATCH_SIZE", "10"))

            # 构建选定的笔记本
            total_docs = 0
            success_count = 0
            failed_count = 0

            for nb_id, nb_name in selected_notebooks:
                print(f"\n📖 开始构建笔记本: {nb_name} (ID: {nb_id})")

                try:
                    # 首先打开笔记本
                    print(f"🔓 正在打开笔记本: {nb_name}")
                    async with rag_kb.content_extractor.api_client:
                        await rag_kb.content_extractor.api_client.open_notebook(nb_id)
                    print(f"✅ 笔记本 {nb_name} 已打开")

                    doc_count = await rag_kb.build_knowledge_base(
                        notebook_id=nb_id,
                        include_children=True,
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap,
                        batch_size=batch_size,
                        force_rebuild=True  # 用户主动选择，强制重建
                    )

                    total_docs += doc_count
                    success_count += 1
                    print(f"✅ 笔记本 '{nb_name}' 构建完成，共 {doc_count} 个文档块")

                except Exception as e:
                    failed_count += 1
                    print(f"❌ 构建笔记本 '{nb_name}' 失败: {e}")
                    logger.error(f"构建笔记本失败: {e}")

            # 显示构建结果总结
            if success_count > 0 and failed_count == 0:
                print(f"\n🎉 知识库构建完成！总计 {total_docs} 个文档块")
            elif success_count > 0 and failed_count > 0:
                print(f"\n⚠️ 知识库构建部分完成！成功 {success_count} 个笔记本，失败 {failed_count} 个笔记本，总计 {total_docs} 个文档块")
            else:
                print(f"\n❌ 知识库构建失败！所有 {failed_count} 个笔记本构建失败")

            return success_count > 0

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


async def check_existing_data_and_prompt(rag_kb):
    """
    检查现有数据并提示用户操作选择

    Args:
        rag_kb: 知识库实例

    Returns:
        str: 操作类型 ("rebuild", "incremental", "use_existing")
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
            return "rebuild"

        # 显示现有数据状态
        print(f"\n📊 发现现有知识库数据:")
        print(f"总文档块数: {total_existing}")
        print(f"已构建的笔记本:")
        for nb_id, nb_name, count in notebooks_with_data:
            print(f"  - {nb_name} (ID: {nb_id}): {count} 个文档块")

        # 询问用户是否重建或增量更新
        print("\n请选择知识库操作方式？")
        print("1. 重新构建 (删除现有数据，重新创建)")
        print("2. 增量更新 (只更新有修改的文档)")
        print("3. 使用现有数据 (直接进入ReAct Agent模式)")

        while True:
            choice = input("请选择 (1/2/3): ").strip()
            if choice == "1":
                print("🔄 选择重新构建知识库")
                return "rebuild"
            elif choice == "2":
                print("🔄 选择增量更新知识库")
                return "incremental"
            elif choice == "3":
                print("✅ 使用现有知识库数据")
                return "use_existing"
            else:
                print("❌ 无效选择，请输入 1、2 或 3")

    except Exception as e:
        print(f"❌ 检查现有数据失败: {e}")
        logger.error(f"检查现有数据失败: {e}")
        return "rebuild"  # 出错时默认重建


async def select_notebook_for_incremental_update(rag_kb) -> bool:
    """
    让用户选择笔记本进行增量更新

    Args:
        rag_kb: 知识库实例

    Returns:
        bool: 是否成功进行增量更新
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

        # 检查哪些笔记本有现有数据
        notebooks_with_data = []
        for nb_id, nb_name in notebooks:
            existing_count = await rag_kb.get_notebook_document_count(nb_id)
            if existing_count > 0:
                notebooks_with_data.append((nb_id, nb_name, existing_count))

        if not notebooks_with_data:
            print("❌ 没有找到已构建的笔记本，无法进行增量更新")
            print("   请先使用完整构建模式创建知识库")
            return False

        print(f"\n📚 找到 {len(notebooks_with_data)} 个已构建的笔记本:")
        for i, (nb_id, nb_name, count) in enumerate(notebooks_with_data, 1):
            print(f"  {i}. {nb_name} (ID: {nb_id}) - {count} 个文档块")

        print("\n请选择要进行增量更新的笔记本:")
        try:
            choice = input("输入笔记本编号 (多个用逗号分隔，如: 1,2,3)，直接回车选择所有笔记本: ").strip()
            if not choice:
                print("📋 选择所有笔记本进行增量更新")
                selected_notebooks = notebooks_with_data  # 选择所有笔记本
            else:
                # 解析用户选择
                selected_indices = [int(x.strip()) - 1 for x in choice.split(",")]
                selected_notebooks = []

                for idx in selected_indices:
                    if 0 <= idx < len(notebooks_with_data):
                        selected_notebooks.append(notebooks_with_data[idx])
                    else:
                        print(f"⚠️ 编号 {idx + 1} 无效，跳过")

            if not selected_notebooks:
                print("❌ 没有选择有效的笔记本")
                return False

            # 获取配置参数
            chunk_size = int(os.getenv("RAG_CHUNK_SIZE", "1000"))
            chunk_overlap = int(os.getenv("RAG_CHUNK_OVERLAP", "200"))
            batch_size = int(os.getenv("RAG_BATCH_SIZE", "10"))

            # 执行增量更新
            total_updated = 0
            success_count = 0
            failed_count = 0

            for nb_id, nb_name, existing_count in selected_notebooks:
                print(f"\n📖 开始增量更新笔记本: {nb_name} (ID: {nb_id})")
                print(f"   现有文档块: {existing_count}")

                try:
                    # 首先打开笔记本
                    print(f"🔓 正在打开笔记本: {nb_name}")
                    async with rag_kb.content_extractor.api_client:
                        await rag_kb.content_extractor.api_client.open_notebook(nb_id)
                    print(f"✅ 笔记本 {nb_name} 已打开")

                    # 执行增量更新
                    updated_count = await rag_kb.build_knowledge_base_incremental(
                        notebook_id=nb_id,
                        include_children=True,
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap,
                        batch_size=batch_size
                    )

                    total_updated += updated_count
                    success_count += 1
                    print(f"✅ 笔记本 '{nb_name}' 增量更新完成，更新了 {updated_count} 个文档块")

                except Exception as e:
                    failed_count += 1
                    print(f"❌ 增量更新笔记本 '{nb_name}' 失败: {e}")
                    logger.error(f"增量更新笔记本失败: {e}")

            # 显示更新结果总结
            if success_count > 0 and failed_count == 0:
                print(f"\n🎉 增量更新完成！总计更新 {total_updated} 个文档块")
            elif success_count > 0 and failed_count > 0:
                print(f"\n⚠️ 增量更新部分完成！成功 {success_count} 个笔记本，失败 {failed_count} 个笔记本，总计更新 {total_updated} 个文档块")
            else:
                print(f"\n❌ 增量更新失败！所有 {failed_count} 个笔记本更新失败")

            return success_count > 0

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


async def build_notebook_directly(rag_kb, notebook_id: str) -> bool:
    """
    直接构建指定笔记本

    Args:
        rag_kb: 知识库实例
        notebook_id: 笔记本ID

    Returns:
        bool: 是否成功构建
    """
    try:
        # 获取笔记本列表
        notebooks = rag_kb.content_extractor.workspace.list_notebooks()

        # 查找指定笔记本
        target_notebook = None
        for nb_id, nb_name in notebooks:
            if nb_id == notebook_id:
                target_notebook = (nb_id, nb_name)
                break

        if not target_notebook:
            print(f"❌ 未找到笔记本 ID: {notebook_id}")
            return False

        nb_id, nb_name = target_notebook
        print(f"📖 开始构建笔记本: {nb_name} (ID: {nb_id})")

        # 首先打开笔记本
        print(f"🔓 正在打开笔记本: {nb_name}")
        async with rag_kb.content_extractor.api_client:
            await rag_kb.content_extractor.api_client.open_notebook(nb_id)
        print(f"✅ 笔记本 {nb_name} 已打开")

        # 使用默认参数构建
        doc_count = await rag_kb.build_knowledge_base(
            notebook_id=nb_id,
            include_children=True,
            chunk_size=1000,
            chunk_overlap=200,
            batch_size=10,
            force_rebuild=True
        )

        print(f"✅ 笔记本 '{nb_name}' 构建完成，共 {doc_count} 个文档块")
        return doc_count > 0

    except Exception as e:
        print(f"❌ 构建笔记本失败: {e}")
        logger.error(f"构建笔记本失败: {e}")
        return False


async def interactive_cli(notebook_id: str = None, incremental_mode: bool = False):
    """交互式CLI主函数"""
    print("\n=== 思源笔记RAG知识库交互式CLI ===")
    print("输入问题来测试ReAct Agent，输入 'quit' 退出")

    # 创建知识库实例
    print("🔧 正在创建知识库实例...")
    rag_kb = create_knowledge_base()

    # 如果是增量更新模式，直接执行增量更新
    if incremental_mode:
        print("🔄 执行增量更新模式")
        success = await select_notebook_for_incremental_update(rag_kb)
        if not success:
            print("❌ 增量更新失败，退出CLI")
            return
    else:
        # 检查现有数据并询问用户
        action = await check_existing_data_and_prompt(rag_kb)

        if action == "rebuild":
            if notebook_id:
                # 直接构建指定笔记本
                print(f"🔧 直接构建笔记本 ID: {notebook_id}")
                success = await build_notebook_directly(rag_kb, notebook_id)
                if not success:
                    print("❌ 知识库构建失败，退出CLI")
                    return
            else:
                # 需要重新构建，让用户选择笔记本
                success = await select_notebook_and_build(rag_kb)
                if not success:
                    print("❌ 知识库构建失败，退出CLI")
                    return
        elif action == "incremental":
            # 执行增量更新
            success = await select_notebook_for_incremental_update(rag_kb)
            if not success:
                print("❌ 增量更新失败，退出CLI")
                return
        else:  # use_existing
            print("✅ 使用现有知识库数据")

    # 获取统计信息
    stats = rag_kb.get_collection_stats()
    print(f"\n📊 知识库信息:")
    print(f"  - 文档总数: {stats.get('document_count', 0)}")
    print(f"  - Embedding模型: {stats.get('embedding_model', 'N/A')}")

    # 创建Agent
    agent = create_react_agent(rag_kb, max_tool_calls=5)
    print("[SUCCESS] ReAct Agent创建成功，可以开始提问了！")

    while True:
        try:
            print("\n" + "="*80)
            print("[RAG] 请输入您的问题 (输入 'quit' 或 '退出' 结束对话):")
            print("="*80)
            question = input("[?] ").strip()

            if question.lower() in ['quit', 'exit', '退出']:
                print("\n[SYS] 感谢使用思源笔记RAG问答系统，再见！")
                break

            if not question:
                print("[WARN] 请输入有效的问题")
                continue

            print("\n[PROCESSING] 正在处理您的问题...")
            print(f"[QUESTION] {question}")
            print("-" * 80)

            response = await agent.query(question)

            print("\n" + "="*80)
            print("[ANSWER] 答案")
            print("="*80)
            print(f"{response.answer}")

            print("\n[STATS] 处理统计:")
            print(f"  [TOOLS] 工具调用次数: {response.tool_calls_made}")
            print(f"  [REASON] 推理步数: {len(response.reasoning)}")
            print(f"  [CONFIDENCE] 置信度: {response.final_confidence:.2%}")
            print(f"  [SOURCES] 使用来源数: {len(response.sources_used)}")

            if response.sources_used:
                print("\n[REFERENCES] 主要参考来源:")
                for i, source in enumerate(response.sources_used[:5], 1):
                    similarity = source.get('similarity', 0)
                    print(f"  {i}. {source['title']}")
                    print(f"     [SIMILARITY] {similarity:.3f}")

            print("\n" + "="*80)

        except KeyboardInterrupt:
            print("\n\n用户中断，退出CLI")
            break
        except Exception as e:
            print(f"处理失败: {e}")
            logger.error(f"交互式CLI失败: {e}")


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="思源笔记RAG知识库交互式CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 交互式模式
  python cli.py

  # 直接构建指定笔记本
  python cli.py --notebook-id 20230602143452-yt2rrgb

  # 增量更新模式
  python cli.py --incremental

  # 显示笔记本列表
  python cli.py --list-notebooks
        """
    )

    parser.add_argument(
        "--notebook-id",
        type=str,
        help="直接构建指定笔记本ID，跳过交互式选择"
    )

    parser.add_argument(
        "--list-notebooks",
        action="store_true",
        help="显示所有可用的笔记本列表"
    )

    parser.add_argument(
        "--incremental",
        action="store_true",
        help="对已有知识库进行增量更新"
    )

    return parser.parse_args()


async def list_notebooks():
    """显示笔记本列表"""
    print("📚 思源笔记笔记本列表")
    print("=" * 60)

    try:
        rag_kb = create_knowledge_base()
        notebooks = rag_kb.content_extractor.workspace.list_notebooks()

        if not notebooks:
            print("❌ 没有找到思源笔记笔记本")
            return

        print(f"找到 {len(notebooks)} 个笔记本:\n")

        for i, (nb_id, nb_name) in enumerate(notebooks, 1):
            print(f"{i}. {nb_name}")
            print(f"   ID: {nb_id}")
            print()

    except Exception as e:
        print(f"❌ 获取笔记本列表失败: {e}")


async def main():
    """主函数"""
    args = parse_args()

    # 处理列表笔记本请求
    if args.list_notebooks:
        await list_notebooks()
        return

    print("🚀 思源笔记RAG知识库交互式CLI")
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

    # 启动交互式CLI
    await interactive_cli(args.notebook_id, args.incremental)


if __name__ == "__main__":
    asyncio.run(main())