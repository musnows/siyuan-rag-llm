# 思源笔记 API 工具包

一个基于 `aiohttp` 的思源笔记 API 客户端工具包，提供了完整、类型安全、异步的思源笔记 API 接口封装。

## 特性

- ✅ **完整的 API 覆盖** - 实现了思源笔记官方 API 文档中的所有接口
- ✅ **异步支持** - 基于 `aiohttp` 的异步客户端，支持高并发操作
- ✅ **类型安全** - 使用 `pydantic` 进行数据验证和类型提示
- ✅ **错误处理** - 完善的异常处理机制
- ✅ **上下文管理** - 支持异步上下文管理器，自动管理会话
- ✅ **日志记录** - 内置日志记录功能
- ✅ **易于使用** - 简洁的 API 设计，易于集成

## 安装

### 环境要求

- Python 3.13+
- uv 包管理器

### 安装步骤

1. 克隆项目：
```bash
git clone <repository-url>
cd siyuan-rag-llm
```

2. 安装依赖：
```bash
uv sync
```

3. 激活虚拟环境：
```bash
source .venv/bin/activate  # Linux/Mac
# 或
.venv\Scripts\activate     # Windows
```

## 快速开始

### 基本使用

```python
import asyncio
from siyuan_api import SiYuanAPIClient

async def main():
    # 创建客户端
    async with SiYuanAPIClient(
        host="127.0.0.1",
        port=6806,
        token="your-api-token"  # 在思源笔记设置-关于中查看
    ) as client:
        # 获取版本信息
        version = await client.get_version()
        print(f"思源笔记版本: {version}")

        # 列出笔记本
        notebooks = await client.ls_notebooks()
        print(f"找到 {len(notebooks)} 个笔记本")

if __name__ == "__main__":
    asyncio.run(main())
```

### 环境变量配置

```bash
export SIYUAN_HOST="127.0.0.1"
export SIYUAN_PORT="6806"
export SIYUAN_TOKEN="your-api-token"
```

### 创建文档

```python
async def create_document():
    async with SiYuanAPIClient() as client:
        # 获取第一个笔记本
        notebooks = await client.ls_notebooks()
        if not notebooks:
            print("没有找到笔记本")
            return

        notebook_id = notebooks[0]['id']

        # 创建文档
        doc_id = await client.create_doc_with_md(
            notebook_id=notebook_id,
            path="/test/my-doc",
            markdown="# 我的文档\n\n这是通过 API 创建的文档。"
        )
        print(f"文档创建成功，ID: {doc_id}")
```

## API 功能模块

### 1. 笔记本管理

```python
# 列出笔记本
notebooks = await client.ls_notebooks()

# 创建笔记本
notebook = await client.create_notebook("新笔记本")

# 打开/关闭笔记本
await client.open_notebook(notebook_id)
await client.close_notebook(notebook_id)

# 重命名笔记本
await client.rename_notebook(notebook_id, "新名称")

# 删除笔记本
await client.remove_notebook(notebook_id)
```

### 2. 文档管理

```python
# 创建文档
doc_id = await client.create_doc_with_md(
    notebook_id=notebook_id,
    path="/path/to/doc",
    markdown="# 标题\n\n内容"
)

# 重命名文档
await client.rename_doc(notebook_id, "/path/to/doc", "新标题")

# 移动文档
await client.move_docs(["/path/to/doc"], to_notebook_id, "/new/path")

# 删除文档
await client.remove_doc(notebook_id, "/path/to/doc")
```

### 3. 块操作

```python
# 插入块
block = await client.insert_block(
    data="这是一个新块",
    data_type="markdown",
    parent_id=doc_id
)

# 更新块
await client.update_block(block_id, "更新后的内容")

# 删除块
await client.delete_block(block_id)

# 移动块
await client.move_block(block_id, previous_id=prev_id, parent_id=parent_id)

# 获取子块
children = await client.get_child_blocks(parent_id)
```

### 4. 属性管理

```python
# 设置块属性
await client.set_block_attrs(block_id, {
    "custom-tag": "重要",
    "custom-priority": "high"
})

# 获取块属性
attrs = await client.get_block_attrs(block_id)
```

### 5. SQL 查询

```python
# 执行 SQL 查询
results = await client.query_sql(
    "SELECT * FROM blocks WHERE content LIKE '%关键词%' LIMIT 10"
)

# 提交事务
await client.flush_transaction()
```

### 6. 文件操作

```python
# 写入文件
await client.put_file("/temp/test.txt", "文件内容")

# 读取文件
content = await client.get_file("/temp/test.txt")

# 列出目录
files = await client.read_dir("/temp/")

# 删除文件
await client.remove_file("/temp/test.txt")
```

### 7. 导出功能

```python
# 导出 Markdown
export_data = await client.export_md_content(doc_id)
print(export_data["content"])

# 导出资源文件
zip_path = await client.export_resources(["/path/to/files"], "export-name")
```

## 运行示例

项目包含了完整的使用示例：

```bash
# 运行基本示例
python examples.py

# 运行测试
python test_siyuan_api.py
```

## 错误处理

```python
from siyuan_api import SiYuanAPIClient, SiYuanError

async def handle_errors():
    async with SiYuanAPIClient() as client:
        try:
            # 尝试操作不存在的资源
            await client.get_hpath_by_id("non-existent-id")
        except SiYuanError as e:
            print(f"API 错误 {e.code}: {e.msg}")
        except Exception as e:
            print(f"其他错误: {e}")
```

## 配置说明

### API Token 获取

1. 打开思源笔记
2. 进入 **设置** → **关于**
3. 复制 **API token**

### 网络配置

默认配置：
- 主机：`127.0.0.1`
- 端口：`6806`
- 协议：`http`

如需修改，可在创建客户端时指定：

```python
client = SiYuanAPIClient(
    host="192.168.1.100",
    port=6806,
    token="your-token",
    timeout=60  # 超时时间（秒）
)
```

## 项目结构

```
siyuan-rag-llm/
├── siyuan_api.py      # 主要的 API 客户端
├── examples.py        # 使用示例
├── test_siyuan_api.py # 单元测试
├── main.py           # 项目入口
├── pyproject.toml    # 项目配置
├── README.md         # 项目文档
└── API.md           # 思源笔记 API 文档
```

## 开发计划

- [ ] 添加更多高级功能示例
- [ ] 支持批量操作优化
- [ ] 添加缓存机制
- [ ] 实现重试机制
- [ ] 添加 WebSocket 支持

## 贡献指南

欢迎提交 Issue 和 Pull Request！

## 许可证

MIT License

## 相关链接

- [思源笔记官网](https://b3log.org/siyuan/)
- [思源笔记 API 文档](https://github.com/siyuan-note/siyuan/blob/master/API_zh_CN.md)
- [aiohttp 文档](https://docs.aiohttp.org/)
- [pydantic 文档](https://pydantic-docs.helpmanual.io/)