"""
思源笔记 API 使用示例
"""

import asyncio
from datetime import datetime

from utils.siyuan_api import SiYuanAPIClient, SiYuanError


async def basic_usage_example():
    """基本使用示例"""
    # 使用新的 from_env 方法自动从环境变量加载配置
    async with SiYuanAPIClient.from_env() as client:
        try:
            # 1. 获取系统信息
            print("=== 系统信息 ===")
            version = await client.get_version()
            current_time = await client.get_current_time()
            print(f"思源笔记版本: {version}")
            print(f"当前时间: {datetime.fromtimestamp(current_time / 1000)}")

            # 2. 列出笔记本
            print("\n=== 笔记本列表 ===")
            notebooks = await client.ls_notebooks()
            for notebook in notebooks:
                print(f"- {notebook['name']} (ID: {notebook['id']})")

            if not notebooks:
                print("没有找到笔记本，请先在思源笔记中创建一个笔记本")
                return

            # 3. 获取倒数第一个笔记本
            first_notebook = notebooks[-1]
            notebook_id = first_notebook['id']
            print(f"\n使用笔记本: {first_notebook['name']}")

            # 4. 创建文档
            print("\n=== 创建文档 ===")
            doc_path = f"/api_test/{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            markdown_content = """# 测试文档

这是一个通过 API 创建的测试文档。

## 功能列表

1. 支持异步操作
2. 完整的错误处理
3. 类型提示支持
4. 日志记录

## 代码示例

```python
async with SiYuanAPIClient() as client:
    notebooks = await client.ls_notebooks()
    print(f"找到 {len(notebooks)} 个笔记本")
```

> 思源笔记是一个功能强大的知识管理工具。
"""

            doc_id = await client.create_doc_with_md(
                notebook_id=notebook_id,
                path=doc_path,
                markdown=markdown_content
            )
            print(f"创建文档成功，ID: {doc_id}")

            # 5. 查询文档内容
            print("\n=== 查询文档 ===")
            hPath = await client.get_hpath_by_id(doc_id)
            print(f"文档人类可读路径: {hPath}")

            # 6. SQL 查询
            print("\n=== SQL 查询 ===")
            # 查询最近创建的文档
            sql = f"SELECT * FROM blocks WHERE id = '{doc_id}'"
            results = await client.query_sql(sql)
            if results:
                print(f"查询到 {len(results)} 条记录")
                for result in results:
                    print(f"- ID: {result.get('id', 'N/A')}")
                    print(f"  内容: {result.get('content', 'N/A')[:50]}...")

            # 7. 设置块属性
            print("\n=== 设置块属性 ===")
            await client.set_block_attrs(doc_id, {
                "custom-api-created": "true",
                "custom-create-time": datetime.now().isoformat()
            })
            print("设置块属性成功")

            # 8. 获取块属性
            attrs = await client.get_block_attrs(doc_id)
            print(f"块属性: {attrs}")

            # 9. 插入子块
            print("\n=== 插入子块 ===")
            child_block = await client.append_block(
                parent_id=doc_id,
                data="**这是通过 API 插入的子块**",
                data_type="markdown"
            )
            print(f"插入子块成功，ID: {child_block['id']}")

            # 10. 推送消息
            print("\n=== 推送通知 ===")
            msg_id = await client.push_msg("API 测试完成！", timeout=3000)
            print(f"推送消息成功，ID: {msg_id}")

            print("\n✅ 所有测试操作完成！")

        except SiYuanError as e:
            print(f"❌ 思源笔记 API 错误: {e}")
        except Exception as e:
            print(f"❌ 未知错误: {e}")


async def advanced_usage_example():
    """高级使用示例"""
    # 使用新的 from_env 方法自动从环境变量加载配置
    async with SiYuanAPIClient.from_env() as client:
        try:
            # 1. 批量操作
            print("=== 批量操作示例 ===")

            # 获取所有笔记本
            notebooks = await client.ls_notebooks()
            if not notebooks:
                print("没有找到笔记本")
                return

            notebook_id = notebooks[-1]['id']

            # 创建多个文档
            doc_ids = []
            for i in range(3):
                path = f"/batch_test/doc_{i+1}"
                content = f"# 文档 {i+1}\n\n这是第 {i+1} 个测试文档。"
                doc_id = await client.create_doc_with_md(notebook_id, path, content)
                doc_ids.append(doc_id)
                print(f"创建文档 {i+1}: {doc_id}")

            # 2. 文件操作
            print("\n=== 文件操作示例 ===")

            # 写入文件
            test_content = "这是一个测试文件内容。"
            await client.put_file("/temp/test.txt", test_content)
            print("写入文件成功")

            # 读取文件
            file_content = await client.get_file("/temp/test.txt")
            print(f"读取文件内容: {file_content}")

            # 3. 搜索和过滤
            print("\n=== 搜索示例 ===")

            # 搜索包含特定内容的块
            search_sql = "SELECT * FROM blocks WHERE content LIKE '%测试%' LIMIT 5"
            search_results = await client.query_sql(search_sql)
            print(f"找到 {len(search_results)} 个包含'测试'的块")

            # 4. 导出功能
            print("\n=== 导出示例 ===")

            if doc_ids:
                # 导出第一个文档为 Markdown
                export_result = await client.export_md_content(doc_ids[0])
                print("导出: ",export_result)
                print(f"导出文档路径: {export_result['hPath']}")
                print(f"导出内容长度: {len(export_result['content'])} 字符")

            print("\n✅ 高级示例完成！")

        except SiYuanError as e:
            print(f"❌ 思源笔记 API 错误: {e}")
        except Exception as e:
            print(f"❌ 未知错误: {e}")


async def error_handling_example():
    """错误处理示例"""
    # 使用新的 from_env 方法自动从环境变量加载配置
    async with SiYuanAPIClient.from_env() as client:
        # 1. 尝试操作不存在的文档
        print("=== 错误处理示例 ===")
        try:
            await client.get_hpath_by_id("non-existent-id")
        except SiYuanError as e:
            print(f"捕获到预期的错误: {e}")

        # 2. 尝试使用无效的 SQL
        try:
            await client.query_sql("INVALID SQL SYNTAX")
        except SiYuanError as e:
            print(f"捕获到 SQL 错误: {e}")

        # 3. 尝试访问不存在的文件
        try:
            await client.get_file("/non/existent/file.txt")
        except SiYuanError as e:
            print(f"捕获到文件访问错误: {e}")


async def main():
    """主函数"""
    print("🚀 开始思源笔记 API 测试...")

    print("\n" + "="*50)
    await basic_usage_example()

    print("\n" + "="*50)
    await advanced_usage_example()

    print("\n" + "="*50)
    await error_handling_example()

    print("\n🎉 所有测试完成！")


if __name__ == "__main__":
    # 运行示例
    asyncio.run(main())