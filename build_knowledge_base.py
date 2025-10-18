#!/usr/bin/env python3
"""
构建思源笔记RAG知识库
使用指定的embedding模型构建向量索引
"""

import asyncio
import os
import sys
from typing import Optional

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.rag.rag_knowledge_base import create_rag_knowledge_base, create_rag_knowledge_base_with_openai
from utils.logger import get_logger

logger = get_logger(__name__)


async def build_knowledge_base(notebook_id: Optional[str] = None,
                             force_rebuild: bool = False,
                             incremental: bool = False,
                             embedding_model: Optional[str] = None):
    """
    构建知识库

    Args:
        notebook_id: 笔记本ID，如果为None则处理所有笔记本
        force_rebuild: 是否强制重建
        incremental: 是否使用增量更新模式
        embedding_model: 指定embedding模型，如果不指定则使用环境变量
    """
    # 获取配置
    embedding_model = embedding_model or os.getenv("EMBEDDING_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    api_key = os.getenv("OPENAI_API_KEY")
    api_base = os.getenv("OPENAI_BASE_URL")

    print("=" * 60)
    print("构建思源笔记RAG知识库")
    print("=" * 60)
    print(f"Embedding模型: {embedding_model}")
    print(f"API Base: {api_base or 'https://api.openai.com/v1'}")
    print(f"API Key: {'已设置' if api_key else '未设置'}")
    print(f"强制重建: {force_rebuild}")
    print(f"增量更新: {incremental}")

    # 判断embedding类型
    is_openai_embedding = embedding_model.startswith("text-embedding-") or embedding_model in [
        "text-embedding-ada-002",
        "text-embedding-3-small",
        "text-embedding-3-large"
    ]

    if is_openai_embedding and not api_key:
        print("❌ 错误: 使用OpenAI embedding模型需要设置OPENAI_API_KEY")
        return

    # 创建知识库实例
    try:
        if is_openai_embedding:
            print("🔧 使用OpenAI嵌入模型创建知识库...")
            rag_kb = create_rag_knowledge_base_with_openai(
                persist_directory="./data/rag_db",
                embedding_model=embedding_model,
                api_key=api_key,
                api_base=api_base,
                collection_name="siyuan_notes"
            )
        else:
            print("🔧 使用本地嵌入模型创建知识库...")
            rag_kb = create_rag_knowledge_base(
                persist_directory="./data/rag_db",
                embedding_model=embedding_model,
                use_openai_embedding=False,
                collection_name="siyuan_notes"
            )

        print("✅ 知识库实例创建成功")

    except Exception as e:
        print(f"❌ 知识库创建失败: {e}")
        logger.error(f"知识库创建失败: {e}")
        return

    # 获取笔记本列表
    try:
        notebooks = rag_kb.content_extractor.workspace.list_notebooks()
        if not notebooks:
            print("❌ 没有找到笔记本")
            return

        print(f"📚 找到 {len(notebooks)} 个笔记本:")
        for i, (nb_id, nb_name) in enumerate(notebooks, 1):
            print(f"  {i}. {nb_name} (ID: {nb_id})")

    except Exception as e:
        print(f"❌ 获取笔记本列表失败: {e}")
        logger.error(f"获取笔记本列表失败: {e}")
        return

    # 选择要处理的笔记本
    notebooks_to_process = []
    if notebook_id:
        # 处理指定笔记本
        for nb_id, nb_name in notebooks:
            if nb_id == notebook_id:
                notebooks_to_process.append((nb_id, nb_name))
                break
        else:
            print(f"❌ 未找到指定的笔记本ID: {notebook_id}")
            return
    else:
        # 处理所有笔记本
        notebooks_to_process = notebooks

    # 构建知识库
    total_docs = 0
    for nb_id, nb_name in notebooks_to_process:
        print(f"\n📖 处理笔记本: {nb_name} (ID: {nb_id})")

        try:
            if incremental and not force_rebuild:
                # 使用增量更新
                print(f"🔄 使用增量更新模式处理笔记本 '{nb_name}'")
                doc_count = await rag_kb.build_knowledge_base_incremental(
                    notebook_id=nb_id,
                    include_children=True,
                    chunk_size=1000,
                    chunk_overlap=200,
                    batch_size=10
                )
            else:
                # 使用完整构建
                if force_rebuild:
                    print(f"🔧 强制重建笔记本 '{nb_name}'")
                else:
                    print(f"📝 构建笔记本 '{nb_name}'")
                doc_count = await rag_kb.build_knowledge_base(
                    notebook_id=nb_id,
                    include_children=True,
                    chunk_size=1000,
                    chunk_overlap=200,
                    batch_size=10,
                    force_rebuild=force_rebuild
                )

            total_docs += doc_count
            print(f"✅ 笔记本 '{nb_name}' 处理完成，共 {doc_count} 个文档块")

        except Exception as e:
            print(f"❌ 处理笔记本 '{nb_name}' 失败: {e}")
            logger.error(f"处理笔记本失败: {e}")
            raise  # 重新抛出异常，终止整个流程

    # 显示统计信息
    print(f"\n📊 知识库构建完成!")
    print(f"总文档块数: {total_docs}")

    try:
        stats = rag_kb.get_collection_stats()
        print(f"集合名称: {stats.get('collection_name', 'N/A')}")
        print(f"文档总数: {stats.get('document_count', 'N/A')}")
        print(f"持久化目录: {stats.get('persist_directory', 'N/A')}")
        print(f"Embedding模型: {stats.get('embedding_model', 'N/A')}")

        # 显示各笔记本统计
        notebook_stats = await rag_kb.get_all_notebooks_stats()
        if notebook_stats:
            print(f"\n📚 各笔记本文档数量:")
            for nb_id, count in notebook_stats.items():
                # 找到笔记本名称
                nb_name = next((name for nid, name in notebooks if nid == nb_id), nb_id)
                print(f"  - {nb_name}: {count} 个文档块")

    except Exception as e:
        print(f"⚠️ 获取统计信息失败: {e}")

    print(f"\n🎉 知识库构建完成! 可以开始进行RAG查询了。")


async def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description="构建思源笔记RAG知识库")
    parser.add_argument("--notebook", "-n", type=str, help="指定笔记本ID（不指定则处理所有笔记本）")
    parser.add_argument("--force", "-f", action="store_true", help="强制重建现有知识库")
    parser.add_argument("--incremental", "-i", action="store_true", help="使用增量更新模式（只更新已有RAG数据且有修改的文档）")
    parser.add_argument("--model", "-m", type=str, help="指定embedding模型")

    args = parser.parse_args()

    await build_knowledge_base(
        notebook_id=args.notebook,
        force_rebuild=args.force,
        incremental=args.incremental,
        embedding_model=args.model
    )


if __name__ == "__main__":
    asyncio.run(main())