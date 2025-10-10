#!/usr/bin/env python3
"""
思源笔记RAG+Agent交互式查询系统
提供完整的知识库构建和智能问答功能
"""

import os
import sys
import asyncio
import json
from typing import Optional, Dict, Any
from pathlib import Path

# 设置tokenizers并行化以避免警告
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# 添加utils目录到Python路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))

from utils.logger import get_logger
from utils.siyuan.siyuan_workspace import SiYuanWorkspace
from utils.rag.rag_knowledge_base import create_rag_knowledge_base
from utils.agent.rag_agent import create_rag_agent

logger = get_logger(__name__)


class InteractiveRAGSystem:
    """交互式RAG查询系统"""

    def __init__(self):
        """初始化交互式系统"""
        self.workspace = None
        self.knowledge_base = None
        self.agent = None
        self.current_notebook = None
        self.running = False

    def initialize(self):
        """初始化系统组件"""
        try:
            print("🚀 正在初始化思源笔记RAG系统...")

            # 初始化工作空间
            print("📁 连接思源笔记工作空间...")
            self.workspace = SiYuanWorkspace()
            print(f"✅ 工作空间已连接: {self.workspace.workspace_path}")

            # 初始化知识库
            print("📚 初始化RAG知识库...")
            self.knowledge_base = create_rag_knowledge_base()
            print("✅ RAG知识库已初始化")

            # 初始化Agent
            print("🤖 初始化智能问答Agent...")
            self.agent = create_rag_agent(self.knowledge_base)
            print("✅ Agent已初始化")

            print("\n🎉 系统初始化完成！")
            return True

        except Exception as e:
            print(f"❌ 初始化失败: {e}")
            logger.error(f"系统初始化失败: {e}")
            return False

    def show_banner(self):
        """显示系统横幅"""
        banner = """
╔══════════════════════════════════════════════════════════════╗
║                思源笔记 RAG + Agent 交互式查询系统                ║
║                                                              ║
║  📚 基于思源笔记构建RAG知识库                                  ║
║  🤖 智能Agent问答助手                                         ║
║  🔍 支持语义搜索和上下文查询                                   ║
╚══════════════════════════════════════════════════════════════╝
"""
        print(banner)

    async def select_notebook(self) -> bool:
        """选择笔记本"""
        try:
            print("\n📋 获取可用笔记本列表...")
            notebooks = self.workspace.list_notebooks()

            if not notebooks:
                print("❌ 没有找到符合条件的笔记本")
                print("请确保笔记本ID符合日期格式：YYYYMMDDHHMMSS-xxxxxx")
                return False

            print("\n📖 可用笔记本列表:")
            print("-" * 80)
            for i, (notebook_id, name) in enumerate(notebooks, 1):
                print(f"  {i:2d}. {notebook_id} - {name}")
            print("-" * 80)

            while True:
                try:
                    choice = input(f"\n请选择笔记本 (1-{len(notebooks)}) 或输入 'q' 退出: ").strip()

                    if choice.lower() == 'q':
                        return False

                    choice_num = int(choice)
                    if 1 <= choice_num <= len(notebooks):
                        selected_id, selected_name = notebooks[choice_num - 1]
                        self.current_notebook = selected_id

                        print(f"\n✅ 已选择笔记本: {selected_name} ({selected_id})")
                        return True
                    else:
                        print(f"❌ 请输入 1-{len(notebooks)} 之间的数字")

                except ValueError:
                    print("❌ 请输入有效的数字")

        except Exception as e:
            print(f"❌ 选择笔记本失败: {e}")
            logger.error(f"选择笔记本失败: {e}")
            return False

    async def build_knowledge_base(self, force_rebuild: bool = False) -> bool:
        """构建RAG知识库"""
        if not self.current_notebook:
            print("❌ 请先选择笔记本")
            return False

        try:
            print(f"\n🔨 开始为笔记本构建RAG知识库...")
            print(f"📝 笔记本ID: {self.current_notebook}")

            # 检查该笔记本是否已经存在
            existing_count = await self.knowledge_base.get_notebook_document_count(self.current_notebook)
            if existing_count > 0:
                if force_rebuild:
                    print(f"⚠️  笔记本已存在 {existing_count} 个文档块，将强制重建")
                else:
                    print(f"✅ 笔记本已存在 {existing_count} 个文档块，跳过构建")
                    return True

            print("📊 正在处理笔记内容，请稍候...")

            # 构建知识库
            doc_count = await self.knowledge_base.build_knowledge_base(
                self.current_notebook,
                chunk_size=1000,
                chunk_overlap=200,
                batch_size=10,
                force_rebuild=force_rebuild
            )

            if doc_count > 0:
                print(f"✅ 知识库构建完成！")
                print(f"📄 共处理 {doc_count} 个文档块")

                # 显示所有笔记本统计信息
                await self.show_all_notebooks_stats()

                return True
            else:
                print("⚠️  没有找到可处理的笔记内容")
                return False

        except Exception as e:
            print(f"❌ 构建知识库失败: {e}")
            logger.error(f"构建知识库失败: {e}")
            return False

    async def show_all_notebooks_stats(self):
        """显示所有笔记本的统计信息"""
        try:
            notebook_stats = await self.knowledge_base.get_all_notebooks_stats()

            if notebook_stats:
                print(f"\n📚 知识库中的笔记本:")
                print("-" * 60)
                for notebook_id, doc_count in notebook_stats.items():
                    # 获取笔记本名称
                    notebooks = self.workspace.list_notebooks()
                    notebook_name = next((name for nid, name in notebooks if nid == notebook_id), notebook_id)
                    print(f"  📖 {notebook_name} ({notebook_id[:12]}...): {doc_count} 个文档块")
                print("-" * 60)
            else:
                print("\n📚 知识库中暂无笔记本数据")
        except Exception as e:
            print(f"❌ 获取笔记本统计信息失败: {e}")

    def show_help(self):
        """显示帮助信息"""
        help_text = """
🔍 交互式查询帮助:

命令:
  直接输入问题      - 进行智能问答
  /help 或 /?       - 显示此帮助信息
  /stats            - 显示知识库统计信息
  /notebooks        - 显示所有笔记本统计信息
  /clear            - 清空对话历史
  /notebook         - 重新选择笔记本
  /rebuild          - 重建当前笔记本知识库
  /test             - 运行测试查询
  /quit 或 /exit    - 退出程序

示例问题:
  "这个笔记本的主要内容是什么？"
  "有没有关于Python的文档？"
  "请总结一下重要的概念"
  "查找关于测试的笔记"

📚 支持多笔记本: 知识库可以同时包含多个笔记本的内容
"""
        print(help_text)

    async def show_stats(self):
        """显示统计信息"""
        if not self.knowledge_base:
            print("❌ 知识库未初始化")
            return

        try:
            stats = self.knowledge_base.get_collection_stats()
            conversation_stats = self.agent.get_conversation_summary() if self.agent else {}

            print("\n📊 系统统计信息:")
            print("-" * 50)
            print(f"📚 集合名称: {stats.get('collection_name', 'N/A')}")
            print(f"📄 文档数量: {stats.get('document_count', 0)}")
            print(f"💾 持久化目录: {stats.get('persist_directory', 'N/A')}")
            print(f"🧠 嵌入模型: {stats.get('embedding_model', 'N/A')}")

            if self.current_notebook:
                current_count = await self.knowledge_base.get_notebook_document_count(self.current_notebook)
                print(f"📖 当前笔记本: {self.current_notebook} ({current_count} 个文档块)")

            if conversation_stats:
                print(f"💬 对话消息数: {conversation_stats.get('total_messages', 0)}")
                print(f"👤 用户消息数: {conversation_stats.get('user_messages', 0)}")
                print(f"🤖 助手消息数: {conversation_stats.get('assistant_messages', 0)}")

            print("-" * 50)

            # 显示所有笔记本统计
            await self.show_all_notebooks_stats()

        except Exception as e:
            print(f"❌ 获取统计信息失败: {e}")

    async def run_test_queries(self):
        """运行测试查询"""
        if not self.agent or not self.current_notebook:
            print("❌ 系统未完全初始化")
            return

        test_queries = [
            "这个笔记本的主要内容是什么？",
            "笔记本中最重要的概念是什么？",
            "有哪些技术文档或说明？"
        ]

        print("\n🧪 运行测试查询...")
        print("=" * 60)

        for i, query in enumerate(test_queries, 1):
            print(f"\n🔍 测试查询 {i}: {query}")
            print("-" * 40)

            try:
                response = await self.agent.query(query, self.current_notebook)
                print(f"💬 回答: {response.answer}")

                if response.confidence:
                    print(f"🎯 置信度: {response.confidence:.3f}")

                if response.sources:
                    print(f"📚 来源 ({len(response.sources)} 个):")
                    for source in response.sources[:3]:  # 只显示前3个来源
                        print(f"   • {source['title']} (相似度: {source['similarity']:.3f})")

            except Exception as e:
                print(f"❌ 查询失败: {e}")

        print("\n" + "=" * 60)
        print("✅ 测试查询完成")

    async def handle_query(self, query: str):
        """处理用户查询"""
        if not self.agent or not self.current_notebook:
            print("❌ 系统未完全初始化，请先选择笔记本并构建知识库")
            return

        try:
            print(f"\n🔍 查询中...")

            # 执行查询
            response = await self.agent.query(query, self.current_notebook)

            print(f"\n💬 回答:")
            print(f"{response.answer}")

            # 显示来源信息
            if response.sources:
                print(f"\n📚 参考来源 ({len(response.sources)} 个):")
                for i, source in enumerate(response.sources, 1):
                    print(f"  {i}. {source['title']}")
                    print(f"     路径: {source['path']}")
                    print(f"     相似度: {source['similarity']:.3f}")

            # 显示置信度
            if response.confidence is not None:
                confidence_level = "高" if response.confidence > 0.8 else "中" if response.confidence > 0.6 else "低"
                print(f"\n🎯 置信度: {response.confidence:.3f} ({confidence_level})")

            # 显示元数据
            if response.metadata:
                tokens_used = response.metadata.get("tokens_used")
                if tokens_used:
                    print(f"🔢 Token使用量: {tokens_used}")

        except Exception as e:
            print(f"❌ 查询处理失败: {e}")
            logger.error(f"查询处理失败: {e}")

    async def interactive_loop(self):
        """交互式主循环"""
        self.running = True

        print("\n🎯 进入交互式查询模式")
        print("输入问题进行查询，输入 '/help' 查看帮助，输入 '/quit' 退出")

        while self.running:
            try:
                # 获取用户输入
                user_input = input("\n💭 请输入您的问题或命令: ").strip()

                if not user_input:
                    continue

                # 处理命令
                if user_input.startswith('/'):
                    await self.handle_command(user_input.lower())
                else:
                    # 处理查询
                    await self.handle_query(user_input)

            except KeyboardInterrupt:
                print("\n\n👋 检测到中断信号，正在退出...")
                self.running = False
            except EOFError:
                print("\n\n👋 输入结束，正在退出...")
                self.running = False
            except Exception as e:
                print(f"\n❌ 发生错误: {e}")
                logger.error(f"交互循环错误: {e}")

    async def handle_command(self, command: str):
        """处理系统命令"""
        cmd_parts = command.split()
        cmd = cmd_parts[0]

        if cmd in ['/quit', '/exit']:
            print("\n👋 正在退出系统...")
            self.running = False

        elif cmd in ['/help', '/?']:
            self.show_help()

        elif cmd == '/stats':
            await self.show_stats()

        elif cmd == '/notebooks':
            await self.show_all_notebooks_stats()

        elif cmd == '/clear':
            if self.agent:
                self.agent.clear_history()
                print("✅ 对话历史已清空")
            else:
                print("❌ Agent未初始化")

        elif cmd == '/notebook':
            if await self.select_notebook():
                await self.build_knowledge_base()
            else:
                print("❌ 重新选择笔记本失败")

        elif cmd == '/rebuild':
            print("⚠️  这将重建当前笔记本的知识库数据")
            choice = input("确认重建当前笔记本吗？(y/N): ").strip().lower()
            if choice == 'y':
                if await self.build_knowledge_base(force_rebuild=True):
                    print("✅ 知识库重建完成")
                else:
                    print("❌ 知识库重建失败")
            else:
                print("🚫 已取消重建操作")

        elif cmd == '/test':
            await self.run_test_queries()

        else:
            print(f"❌ 未知命令: {command}")
            print("输入 '/help' 查看可用命令")

    async def run(self):
        """运行主程序"""
        try:
            # 显示横幅
            self.show_banner()

            # 初始化系统
            if not self.initialize():
                return False

            # 选择笔记本
            if not await self.select_notebook():
                print("👋 已退出")
                return False

            # 构建知识库
            if not await self.build_knowledge_base():
                print("❌ 无法构建知识库，程序退出")
                return False

            # 进入交互循环
            await self.interactive_loop()

            print("\n🎉 感谢使用思源笔记RAG系统！")
            return True

        except Exception as e:
            print(f"❌ 程序运行出错: {e}")
            logger.error(f"程序运行出错: {e}")
            return False
        finally:
            # 清理资源
            if self.knowledge_base:
                print("🧹 清理资源...")
                # 这里可以添加资源清理逻辑


async def main():
    """主函数"""
    try:
        # 检查环境变量
        required_env_vars = ['SIYUAN_WORKSPACE_PATH', 'OPENAI_API_KEY']
        missing_vars = []

        for var in required_env_vars:
            if not os.getenv(var):
                missing_vars.append(var)

        if missing_vars:
            print("❌ 缺少必需的环境变量:")
            for var in missing_vars:
                if var == 'SIYUAN_WORKSPACE_PATH':
                    print(f"   - {var}: 思源笔记工作空间路径")
                elif var == 'OPENAI_API_KEY':
                    print(f"   - {var}: OpenAI API密钥")
            print("\n请设置环境变量后重试")
            return False

        # 创建并运行交互式系统
        system = InteractiveRAGSystem()
        success = await system.run()
        return success

    except KeyboardInterrupt:
        print("\n\n👋 程序被用户中断")
        return False
    except Exception as e:
        print(f"❌ 程序启动失败: {e}")
        logger.error(f"程序启动失败: {e}")
        return False


if __name__ == "__main__":
    # 运行主程序
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
