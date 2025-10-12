# 思源笔记 RAG + Agent 智能问答系统

基于思源笔记的RAG知识库构建和智能问答系统，通过思源笔记API获取笔记内容，构建向量索引，并通过AI Agent进行智能问答。

## 特性

- 🚀 **完整的RAG系统** - 从思源笔记提取内容，构建向量知识库
- 🤖 **智能Agent问答** - 基于OpenAI GPT的智能对话助手
- 📚 **多笔记本支持** - 同时管理多个笔记本的知识库
- 🔍 **语义搜索** - 基于embedding的相似度搜索
- 💾 **持久化存储** - 使用ChromaDB持久化向量数据
- 🎯 **上下文感知** - 支持上下文增强查询
- 📊 **统计分析** - 完整的知识库统计和监控
- 🛠️ **灵活配置** - 支持本地和OpenAI embedding模型
- 🔄 **增量更新** - 支持知识库的增量更新和重建

## 安装

### 环境要求

- Python 3.13+
- uv 包管理器
- 思源笔记（需要开启API服务）
- OpenAI API Key（用于GPT对话）

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

### 1. 环境配置

创建 `.env` 文件并配置必要的环境变量，参考 [.env.example](./.env.example)

####  必需环境变量

- `SIYUAN_WORKSPACE_PATH`: 思源笔记工作空间路径
- `OPENAI_API_KEY`: OpenAI API密钥

#### 可选环境变量

- `OPENAI_BASE_URL`: OpenAI API基础URL（默认：https://api.openai.com/v1）
- `SIYUAN_HOST`: 思源笔记服务地址（默认：127.0.0.1）
- `SIYUAN_PORT`: 思源笔记服务端口（默认：6806）
- `SIYUAN_TOKEN`: 思源笔记API Token
- `EMBEDDING_MODEL`: Embedding模型名称

#### 配置来源

思源笔记配置：

1. **获取工作空间路径**：
   - 打开思源笔记
   - 进入 **设置** → **关于**
   - 查看工作空间路径

2. **启用API服务**：
   - 进入 **设置** → **API**
   - 开启 "打开 API"
   - 复制 API Token

Embedding模型选择：

**默认模型**（不推荐，效果很差）：
- `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`（默认，支持中文）
- `sentence-transformers/all-MiniLM-L6-v2`

**OpenAI模型**（建议本地LM Studio或ollama部署embedding模型）：
- `text-embedding-qwen3-embedding-0.6b`
- `text-embedding-qwen3-embedding-4b`

### 2. 启动交互式问答系统

```bash
uv run python cli.py
```

系统会自动：
- 连接思源笔记工作空间
- 初始化RAG知识库
- 让您选择要处理的笔记本
- 构建向量索引
- 启动交互式问答界面

### 3. 命令行使用

```bash
# 交互式问答系统（推荐）
uv run python cli.py

# 直接构建指定笔记本
uv run python cli.py --notebook-id YOUR_NOTEBOOK_ID

# 显示笔记本列表
uv run python cli.py --list-notebooks

# 构建知识库（传统方式）
uv run python build_knowledge_base.py

# 指定特定笔记本构建
uv run python build_knowledge_base.py --notebook YOUR_NOTEBOOK_ID

# 强制重建现有知识库
uv run python build_knowledge_base.py --force

# 使用特定embedding模型
uv run python build_knowledge_base.py --model text-embedding-3-small
```

## 系统架构

### 核心组件

1. **思源笔记连接器** (`utils/siyuan/`)
   - `siyuan_api.py`: 完整的思源笔记API客户端封装
   - `siyuan_workspace.py`: 工作空间管理和笔记遍历
   - `siyuan_content.py`: 内容提取和预处理

2. **RAG知识库** (`utils/rag/`)
   - `rag_knowledge_base.py`: 向量知识库管理
   - `rag_query.py`: 查询引擎和搜索算法
   - 支持ChromaDB持久化存储

3. **智能Agent** (`utils/agent/`)
   - `rag_agent.py`: 基于GPT的对话助手
   - `rag_tools.py`: 工具函数和辅助功能
   - `react_agent.py`: ReAct模式的推理代理

4. **嵌入模型** (`utils/embeddings/`)
   - 支持本地sentence-transformers模型
   - 支持OpenAI embedding API
   - 灵活的模型切换机制

### 交互式界面功能

- 📋 **笔记本选择**: 智能识别和选择可用笔记本
- 🔨 **知识库构建**: 自动化文档分块和向量化
- 💬 **智能问答**: 基于RAG的上下文问答
- 📊 **统计监控**: 实时知识库状态监控
- 🔄 **动态更新**: 支持知识库增量更新
- 🧪 **测试查询**: 内置测试功能验证系统状态
- 🚀 **CLI增强功能**:
  - 命令行参数支持 (`--notebook-id`, `--list-notebooks`)
  - 直接构建指定笔记本
  - 笔记本列表显示
  - 增强的错误处理和状态反馈

### 交互式命令

在交互式界面中，您可以使用以下命令：

```
/help 或 /?        # 显示帮助信息
/stats             # 显示知识库统计信息
/notebooks         # 显示所有笔记本统计信息
/clear             # 清空对话历史
/notebook          # 重新选择笔记本
/rebuild           # 重建当前笔记本知识库
/test              # 运行测试查询
/quit 或 /exit     # 退出程序
```

### 项目结构

```
siyuan-rag-llm/
├── cli.py                           # 交互式CLI系统入口（推荐）
├── build_knowledge_base.py          # 知识库构建工具
├── pyproject.toml                   # 项目配置和依赖
├── .env.example                     # 环境变量示例
├── README.md                        # 项目文档
├── SIYUAN_API.md                    # 思源笔记API文档
├── utils/                           # 工具模块
│   ├── siyuan/                      # 思源笔记相关
│   │   ├── siyuan_api.py           # API客户端
│   │   ├── siyuan_workspace.py     # 工作空间管理
│   │   └── siyuan_content.py       # 内容提取
│   ├── rag/                         # RAG知识库
│   │   ├── rag_knowledge_base.py   # 知识库管理
│   │   ├── rag_query.py            # 查询引擎
│   │   └── __init__.py
│   ├── agent/                       # 智能Agent
│   │   ├── rag_agent.py            # RAG对话助手
│   │   ├── rag_tools.py            # 工具函数
│   │   ├── react_agent.py          # ReAct代理
│   │   └── __init__.py
│   ├── embeddings/                  # 嵌入模型
│   │   └── openai_embedding.py     # OpenAI嵌入
│   ├── content_filter.py            # 内容过滤
│   └── logger.py                    # 日志工具
├── test/                            # 测试文件
│   ├── test_react_agent.py         # ReAct Agent测试
│   ├── test_rag_system.py          # RAG系统测试
│   ├── test_rag_query.py           # 查询功能测试
│   ├── siyuan_api_examples.py      # API使用示例
│   └── docs/                        # 测试文档
└── data/                           # 数据目录
    └── rag_db/                     # 向量数据库存储
```


## 常见问题

### Q: 如何处理大型笔记本？
A: 系统会自动进行分块处理，但对于特别大的笔记本，建议：
- 分批构建知识库
- 增加批处理大小
- 使用更高效的embedding模型

### Q: 查询结果不准确怎么办？
A: 可以尝试：
- 调整相似度阈值
- 重建知识库并使用不同的分块策略
- 使用更高质量的embedding模型
- 优化问题的表述方式

### Q: 如何更新知识库？
A: 使用 `/rebuild` 命令或重新运行构建程序，系统会自动检测变化并更新。


## 贡献指南

欢迎提交 Issue 和 Pull Request！

1. Fork 项目
2. 创建特性分支
3. 提交更改
4. 发起 Pull Request

## 许可证

本项目采用 [MIT 许可证](LICENSE)。

## 相关链接

- [思源笔记官网](https://b3log.org/siyuan/)
- [思源笔记 API 文档](https://github.com/siyuan-note/siyuan/blob/master/API_zh_CN.md)
- [ChromaDB 文档](https://docs.trychroma.com/)
- [OpenAI API 文档](https://platform.openai.com/docs)
- [Sentence Transformers](https://www.sbert.net/)