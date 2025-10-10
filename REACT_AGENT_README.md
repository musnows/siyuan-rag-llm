# ReAct Agent - 智能推理助手

## 概述

ReAct Agent是基于Reasoning and Acting模式的智能助手，它能够通过多轮推理和工具调用来回答复杂问题。相比传统的单次查询RAG Agent，ReAct Agent具有更强的推理能力和信息整合能力。

## 核心特性

### 🧠 ReAct推理模式
- **思考(Think)**: 分析问题，制定下一步策略
- **行动(Act)**: 调用合适的工具获取信息
- **观察(Observe)**: 分析工具返回的结果
- **循环**: 直到有足够信息给出答案

### 🔧 丰富的RAG工具
- `rag_search`: 在知识库中搜索相关文档
- `rag_get_context`: 获取特定笔记的完整上下文
- `rag_multi_query`: 执行多个相关查询并合并结果
- `rag_get_stats`: 获取知识库统计信息

### 📊 智能控制
- 最多5次工具调用限制（可配置）
- 最多10步推理限制（可配置）
- 自动置信度评估
- 详细的推理过程记录

## 安装和配置

### 环境要求
- Python 3.8+
- OpenAI API Key
- 思源笔记（可选，用于构建知识库）

### 环境变量
```bash
export OPENAI_API_KEY="your-openai-api-key"
export OPENAI_BASE_URL="https://api.openai.com/v1"  # 可选
```

### 安装依赖
```bash
pip install -r requirements.txt
```

## 快速开始

### 基本使用

```python
import asyncio
from utils.rag.rag_knowledge_base import create_rag_knowledge_base
from utils.agent.react_agent import create_react_agent

async def main():
    # 创建知识库
    rag_kb = create_rag_knowledge_base()

    # 创建ReAct Agent
    agent = create_react_agent(
        knowledge_base=rag_kb,
        model="gpt-3.5-turbo",
        max_tool_calls=5
    )

    # 执行查询
    response = await agent.query("请分析项目中的数据处理流程")

    # 查看结果
    print(f"答案: {response.answer}")
    print(f"工具调用次数: {response.tool_calls_made}")
    print(f"推理步骤: {len(response.reasoning)}")
    print(f"置信度: {response.final_confidence}")

asyncio.run(main())
```

### 高级配置

```python
# 自定义系统提示词
custom_prompt = """
你是一个专业的技术文档分析师。
请仔细分析用户问题，使用合适的工具查找信息，
并给出详细、准确的技术分析。
"""

agent = create_react_agent(
    knowledge_base=rag_kb,
    model="gpt-4",
    max_tool_calls=8,
    max_steps=15,
    temperature=0.1,
    system_prompt=custom_prompt
)
```

## 工具详解

### 1. rag_search - 智能搜索
在知识库中搜索相关文档，支持相似度过滤和结果数量限制。

```python
result = await toolkit.call_tool("rag_search", {
    "query": "数据处理流程",
    "notebook_id": "20231231120000-xxx",  # 可选
    "max_results": 5,
    "similarity_threshold": 0.6
})
```

### 2. rag_get_context - 获取上下文
获取特定笔记的完整上下文内容，按文档块顺序排列。

```python
result = await toolkit.call_tool("rag_get_context", {
    "note_id": "20231231120100-yyy",
    "max_chunks": 10
})
```

### 3. rag_multi_query - 多查询合并
执行多个相关查询并合并结果，支持不同合并策略。

```python
result = await toolkit.call_tool("rag_multi_query", {
    "queries": ["API接口", "数据处理", "错误处理"],
    "combine_strategy": "union",  # union, intersection, weighted
    "max_results": 5
})
```

### 4. rag_get_stats - 统计信息
获取知识库的统计信息。

```python
result = await toolkit.call_tool("rag_get_stats", {})
```

## 推理过程示例

ReAct Agent的推理过程如下：

```
1. [思考] 用户询问数据处理流程，我需要先搜索相关文档
2. [行动] 调用rag_search工具，查询"数据处理流程"
3. [观察] 找到了3个相关文档，其中包含数据预处理、转换和加载的步骤
4. [思考] 需要更详细的信息，让我查找具体的技术实现
5. [行动] 调用rag_multi_query工具，查询["数据预处理", "数据转换", "数据加载"]
6. [观察] 获得了更详细的技术文档，包括算法和代码示例
7. [思考] 现在有足够信息，可以给出完整答案了
8. [最终答案] 基于收集的信息，详细说明数据处理流程...
```

## 配置参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `model` | "gpt-3.5-turbo" | 使用的OpenAI模型 |
| `max_tool_calls` | 5 | 最大工具调用次数 |
| `max_steps` | 10 | 最大推理步数 |
| `temperature` | 0.1 | 生成温度参数 |
| `system_prompt` | 默认提示词 | 系统提示词 |

## 性能对比

| 特性 | 传统RAG Agent | ReAct Agent |
|------|---------------|-------------|
| 查询模式 | 单次查询 | 多轮推理 |
| 工具使用 | 固定流程 | 动态选择 |
| 信息整合 | 有限 | 智能整合 |
| 推理过程 | 不透明 | 完全透明 |
| 复杂问题处理 | 较弱 | 较强 |
| 延迟 | 较低 | 较高 |

## 使用场景

### 适合使用ReAct Agent的场景：
- **复杂问题分析**: 需要从多个角度分析的问题
- **技术文档查询**: 需要整合多个技术文档的信息
- **流程说明**: 需要逐步说明复杂流程
- **概念解释**: 需要从多个来源解释概念
- **对比分析**: 需要对比不同技术或方法

### 适合使用传统Agent的场景：
- **简单事实查询**: 直接查找特定信息
- **快速响应**: 对延迟要求高的场景
- **单一来源**: 信息来源相对固定

## 测试和示例

### 运行测试
```bash
python test_react_agent.py
```

### 运行示例
```bash
python example_react_usage.py
```

### 交互式测试
```bash
python test_react_agent.py
# 选择"交互式测试"选项
```

## 最佳实践

### 1. 提问技巧
- **明确具体**: 避免模糊的问题描述
- **分步骤**: 对复杂问题可以分步提问
- **提供上下文**: 给出相关的背景信息

### 2. 配置优化
- **模型选择**: 复杂问题使用gpt-4，简单问题使用gpt-3.5-turbo
- **工具调用限制**: 根据问题复杂度调整max_tool_calls
- **温度参数**: 事实性查询使用低温度(0.1)，创造性查询可使用中等温度(0.7)

### 3. 结果评估
- **查看置信度**: 关注final_confidence值
- **检查推理过程**: 通过reasoning了解Agent的思考过程
- **验证信息来源**: 查看sources_used确认信息可靠性

## 故障排除

### 常见问题

1. **工具调用失败**
   - 检查知识库是否正确初始化
   - 确认思源笔记连接正常
   - 查看日志了解具体错误

2. **推理循环过长**
   - 调整max_tool_calls和max_steps参数
   - 优化系统提示词
   - 使用更明确的问题表述

3. **答案质量不佳**
   - 尝试使用更强的模型(gpt-4)
   - 调整相似度阈值
   - 检查知识库内容质量

### 调试模式

```python
# 启用详细日志
import logging
logging.basicConfig(level=logging.DEBUG)

# 查看详细推理过程
response = await agent.query(question)
for step in response.reasoning:
    print(f"[{step.step_type}] {step.content}")
    if step.tool_call:
        print(f"  工具: {step.tool_call}")
    if step.tool_result:
        print(f"  结果: {step.tool_result}")
```

## 贡献和反馈

欢迎提交问题和改进建议！

## 许可证

本项目采用MIT许可证。