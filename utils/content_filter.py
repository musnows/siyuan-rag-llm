"""
内容过滤工具
用于过滤空白笔记内容和处理front-matter
"""

import re
import os
from typing import Optional


def remove_front_matter(content: str) -> str:
    """
    移除markdown内容的front-matter

    Args:
        content: 原始markdown内容

    Returns:
        str: 移除front-matter后的内容
    """
    if not content:
        return ""

    # 匹配YAML front-matter格式
    # 以---开头，以---结尾，中间可以包含任何内容（包括换行）
    front_matter_pattern = r'^---\s*\n.*?\n---\s*\n'

    # 使用re.DOTALL标志使.匹配包括换行符在内的所有字符
    cleaned_content = re.sub(front_matter_pattern, '', content, flags=re.DOTALL)

    return cleaned_content.strip()


def is_empty_content(content: str, remove_front_matter_first: bool = True) -> bool:
    """
    检查内容是否为空白

    Args:
        content: 要检查的内容
        remove_front_matter_first: 是否先移除front-matter再检查

    Returns:
        bool: True表示内容为空白，False表示有内容
    """
    if not content:
        return True

    # 如果需要，先移除front-matter
    if remove_front_matter_first:
        content = remove_front_matter(content)

    # 移除多余的空白字符，只保留空格和换行来判断
    cleaned_content = re.sub(r'\s+', ' ', content).strip()

    # 如果只有空格或者完全为空，则认为是空白内容
    return len(cleaned_content) == 0


def should_skip_empty_notes() -> bool:
    """
    检查是否应该跳过空白笔记

    通过环境变量RAG_SKIP_EMPTY控制，支持字符串和整数格式

    Returns:
        bool: True表示跳过空白笔记，False表示保留空白笔记
    """
    skip_empty = os.getenv('RAG_SKIP_EMPTY', '0')

    # 兼容各种输入格式
    if isinstance(skip_empty, str):
        return skip_empty.lower() in ('1', 'true', 'yes', 'on')
    elif isinstance(skip_empty, (int, float)):
        return int(skip_empty) == 1
    else:
        return bool(skip_empty)


def filter_empty_content(content: str, remove_front_matter_first: bool = True) -> Optional[str]:
    """
    过滤空白内容，如果内容为空白则返回None，否则返回清理后的内容

    Args:
        content: 要过滤的内容
        remove_front_matter_first: 是否先移除front-matter

    Returns:
        Optional[str]: 如果内容不为空白则返回清理后的内容，否则返回None
    """
    if not content:
        return None

    # 如果需要，先移除front-matter
    if remove_front_matter_first:
        content = remove_front_matter(content)

    # 检查是否为空白内容
    if is_empty_content(content, remove_front_matter_first=False):
        return None

    return content.strip() if content else None


def clean_content(content: str, remove_front_matter_first: bool = True) -> str:
    """
    清理内容，移除front-matter并标准化空白字符

    Args:
        content: 要清理的内容
        remove_front_matter_first: 是否先移除front-matter

    Returns:
        str: 清理后的内容
    """
    if not content:
        return ""

    # 移除front-matter
    if remove_front_matter_first:
        content = remove_front_matter(content)

    # 标准化空白字符，将多个连续空白字符替换为单个空格
    content = re.sub(r'\s+', ' ', content)

    return content.strip()


# 测试函数
def _test_content_filtering():
    """测试内容过滤功能"""

    # 测试front-matter移除
    test_content_1 = """---
title: 测试笔记
tags: [test, demo]
---

# 这是一个测试笔记

这里有一些内容。

## 子标题

更多内容。"""

    print("=== 测试front-matter移除 ===")
    print("原始内容:")
    print(repr(test_content_1))
    print("\n移除front-matter后:")
    print(repr(remove_front_matter(test_content_1)))
    print("\n清理后的内容:")
    print(clean_content(test_content_1))

    # 测试空白内容检测
    test_content_2 = """---
title: 空白笔记
---

"""

    print("\n=== 测试空白内容检测 ===")
    print("只有front-matter的内容:")
    print(f"是否为空白: {is_empty_content(test_content_2)}")

    test_content_3 = """

    \t\n
    """

    print("只有空白字符的内容:")
    print(f"是否为空白: {is_empty_content(test_content_3)}")

    test_content_4 = """
# 有实际内容

这里有一些文字内容。
"""

    print("有实际内容的内容:")
    print(f"是否为空白: {is_empty_content(test_content_4)}")

    # 测试环境变量
    print("\n=== 测试环境变量 ===")
    print(f"当前是否跳过空白笔记: {should_skip_empty_notes()}")

    # 设置环境变量测试
    os.environ['RAG_SKIP_EMPTY'] = '1'
    print(f"设置RAG_SKIP_EMPTY=1后: {should_skip_empty_notes()}")

    os.environ['RAG_SKIP_EMPTY'] = 'true'
    print(f"设置RAG_SKIP_EMPTY=true后: {should_skip_empty_notes()}")

    # 恢复默认
    os.environ['RAG_SKIP_EMPTY'] = '0'
    print(f"恢复默认后: {should_skip_empty_notes()}")


if __name__ == "__main__":
    _test_content_filtering()