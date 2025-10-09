# 思源笔记RAG知识库系统

基于思源笔记的RAG（检索增强生成）知识库系统，支持从思源笔记自动构建向量知识库，并提供智能问答功能。

## 功能特性

- 📚 **自动知识库构建**: 从思源笔记自动提取markdown内容，构建向量索引
- 🔍 **智能检索**: 支持语义相似度搜索和多查询策略
- 🤖 **智能问答**: 集成OpenAI API，基于知识库内容进行智能问答
- 📝 **上下文管理**: 支持上下文增强查询和对话历史
- ⚡ **高性能**: 支持批量处理和流式输出
- 🔧 **可配置**: 灵活的参数配置和扩展性

## 系统架构

```
思源笔记 → 内容提取 → 文档分块 → 向量化 → ChromaDB
                                    ↓
用户查询 → 相似度检索 → 上下文构建 → OpenAI API → 智能回答
```

## 环境要求

- Python 3.13+
- 思源笔记（需要开启API）
- OpenAI API密钥（用于智能问答）

## 安装依赖

```bash
# 使用uv安装依赖
uv sync

# 或使用pip
pip install -e .
```

## 环境配置

创建 `.env` 文件并配置以下环境变量：

```env
# 思源笔记API配置
SIYUAN_API_HOST=http://127.0.0.1:6806
SIYUAN_API_TOKEN=your_api_token_here

# OpenAI API配置（用于智能问答）
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_BASE_URL=https://api.openai.com/v1  # 可选，自定义API地址
```

### 获取思源笔记API Token

1. 打开思源笔记设置 → API → 生成Token
2. 确保 API 服务已启用（默认端口 6806）
3. 复制生成的 Token 到环境变量

## 快速开始

### 1. 基本使用

```python
import asyncio
from utils.siyuan.siyuan_content import create_content_extractor
from utils.rag.rag_knowledge_base import create_rag_knowledge_base
from utils.agent.rag_agent import create_rag_agent

async def main():
    # 1. 连接思源笔记
    extractor = create_content_extractor()
    notebooks = extractor.workspace.list_notebooks()
    notebook_id = notebooks[0][0]  # 使用第一个笔记本

    # 2. 构建知识库
    rag_kb = create_rag_knowledge_base()
    doc_count = await rag_kb.build_knowledge_base(notebook_id)
    print(f"知识库构建完成，处理了 {doc_count} 个文档块")

    # 3. 创建智能Agent
    agent = create_rag_agent(rag_kb)

    # 4. 智能问答
    response = await agent.query(
        question="这个笔记本的主要内容是什么？",
        notebook_id=notebook_id
    )
    print(f"回答: {response.answer}")
    print(f"置信度: {response.confidence}")
    print(f"来源: {response.sources}")

if __name__ == "__main__":
    asyncio.run(main())
```

### 2. 运行测试

```bash
# 运行完整测试
python test_rag_system.py

# 运行使用示例
python example_usage.py
```

## 详细功能

### RAG知识库构建

```python
from utils.rag.rag_knowledge_base import create_rag_knowledge_base

# 创建知识库（自定义配置）
rag_kb = create_rag_knowledge_base(
    persist_directory="./data/my_rag_db",  # 持久化目录
    embedding_model="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

# 构建知识库（高级配置）
doc_count = await rag_kb.build_knowledge_base(
    notebook_id="your_notebook_id",
    include_children=True,      # 包含子笔记
    chunk_size=1000,           # 文档分块大小
    chunk_overlap=200,         # 分块重叠
    batch_size=10              # 批处理大小
)
```

### 智能检索

```python
from utils.rag.rag_query import create_query_engine

# 创建查询引擎
query_engine = create_query_engine(rag_kb)

# 单查询
result = await query_engine.query(
    query_text="搜索关键词",
    notebook_id="notebook_id",
    n_results=5
)

# 多查询策略
multi_result = await query_engine.multi_query(
    queries=["概念", "内容", "总结"],
    notebook_id="notebook_id",
    combine_strategy="union"  # union, intersection, weighted
)

# 上下文增强查询
context_result = await query_engine.contextual_query(
    query_text="问题",
    notebook_id="notebook_id",
    context_note_ids=["note_id_1", "note_id_2"]
)
```

### 智能Agent

```python
from utils.agent.rag_agent import create_rag_agent

# 创建Agent（自定义配置）
agent = create_rag_agent(
    knowledge_base=rag_kb,
    model="gpt-4",              # 使用的模型
    max_tokens=2000,            # 最大生成token数
    temperature=0.1,            # 温度参数
    system_prompt="自定义系统提示词"
)

# 普通问答
response = await agent.query(
    question="您的问题",
    notebook_id="notebook_id",
    context_strategy="simple"   # simple, contextual, multi_query
)

# 流式问答
async for chunk in agent.stream_query(
    question="您的问题",
    notebook_id="notebook_id"
):
    print(chunk, end="")

# 对话历史管理
agent.clear_history()  # 清空历史
summary = agent.get_conversation_summary()  # 获取摘要
```

## 配置参数

### 知识库配置

- `persist_directory`: 向量数据库持久化目录
- `embedding_model`: 嵌入模型（默认支持中英文）
- `chunk_size`: 文档分块大小（默认1000字符）
- `chunk_overlap`: 分块重叠大小（默认200字符）
- `batch_size`: 批处理大小（默认10）

### 查询配置

- `max_context_length`: 最大上下文长度（默认4000字符）
- `similarity_threshold`: 相似度阈值（默认0.6）
- `max_documents`: 最大文档数量（默认5）

### Agent配置

- `model`: OpenAI模型（默认gpt-3.5-turbo）
- `max_tokens`: 最大生成token数（默认2000）
- `temperature`: 温度参数（默认0.1）
- `use_streaming`: 是否使用流式输出（默认False）

## 使用场景

1. **个人知识管理**: 构建个人笔记的知识库，快速检索和问答
2. **企业知识库**: 基于企业文档构建智能问答系统
3. **学习辅助**: 自动总结学习内容，回答相关问题
4. **研究助手**: 快速检索文献内容，生成研究摘要

## 注意事项

1. **思源笔记API**: 确保思源笔记API已启用并正确配置
2. **网络连接**: 首次使用嵌入模型时需要下载模型文件
3. **API费用**: 使用OpenAI API会产生费用，请注意控制使用量
4. **性能优化**: 大量文档建议分批处理，避免内存溢出

## 故障排除

### 常见问题

1. **连接思源笔记失败**
   - 检查思源笔记是否启动
   - 验证API Token是否正确
   - 确认网络连接正常

2. **知识库构建失败**
   - 检查笔记本中是否有内容
   - 确认文件权限
   - 查看详细错误日志

3. **Agent回答质量差**
   - 调整相似度阈值
   - 增加上下文长度
   - 优化系统提示词

4. **性能问题**
   - 减少批处理大小
   - 调整分块参数
   - 使用更快的嵌入模型

### 日志调试

```python
import logging
from utils.logger import get_logger

logger = get_logger(__name__)
logger.setLevel(logging.DEBUG)  # 启用详细日志
```

## 扩展开发

系统采用模块化设计，支持自定义扩展：

1. **自定义嵌入模型**: 继承`RAGKnowledgeBase`类
2. **自定义查询策略**: 扩展`RAGQueryEngine`类
3. **自定义Agent行为**: 继承`RAGAgent`类
4. **添加新的数据源**: 实现`ContentExtractor`接口

## 许可证

本项目采用 MIT 许可证，详见 LICENSE 文件。