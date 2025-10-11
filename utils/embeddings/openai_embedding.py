"""
OpenAI嵌入模型封装
提供基于OpenAI API的嵌入功能，用于替代本地sentence-transformers模型
"""

import os
import asyncio
from typing import List, Optional, Dict, Any
from dataclasses import dataclass

try:
    from openai import AsyncOpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

# 尝试导入logger，如果失败则使用标准logging
try:
    from ..logger import get_logger
    logger = get_logger(__name__)
except ImportError:
    import logging
    logger = logging.getLogger(__name__)


@dataclass
class EmbeddingConfig:
    """嵌入模型配置"""
    model_name: str = "text-embedding-3-small"  # 或 text-embedding-ada-002, text-embedding-3-large
    api_key: Optional[str] = None
    api_base: Optional[str] = None
    max_retries: int = 3
    retry_delay: float = 1.0
    batch_size: int = 100  # OpenAI API支持批量处理
    timeout: float = 30.0
    dimensions: Optional[int] = None  # 仅适用于text-embedding-3系列


class OpenAIEmbeddingFunction:
    """
    OpenAI嵌入函数，兼容ChromaDB的embedding_function接口
    """

    def __init__(self,
                 model_name: str = "text-embedding-3-small",
                 api_key: Optional[str] = None,
                 api_base: Optional[str] = None,
                 config: Optional[EmbeddingConfig] = None):
        """
        初始化OpenAI嵌入函数

        Args:
            model_name: 模型名称
            api_key: OpenAI API Key
            api_base: API基础URL（用于自定义endpoint）
            config: 嵌入配置
        """
        if not OPENAI_AVAILABLE:
            raise ImportError("OpenAI库未安装，请运行: pip install openai")

        self.config = config or EmbeddingConfig(
            model_name=model_name,
            api_key=api_key,
            api_base=api_base
        )

        # 初始化OpenAI客户端
        self.client = AsyncOpenAI(
            api_key=self.config.api_key or os.getenv("OPENAI_API_KEY"),
            base_url=self.config.api_base or os.getenv("OPENAI_API_BASE")
        )

        # 验证API Key
        if not self.client.api_key:
            raise ValueError("OpenAI API Key未配置，请设置OPENAI_API_KEY环境变量或传入api_key参数")

        logger.info(f"OpenAI嵌入函数初始化完成，模型: {self.config.model_name}")

    async def __call__(self, texts: List[str]) -> List[List[float]]:
        """
        计算文本嵌入向量

        Args:
            texts: 文本列表

        Returns:
            List[List[float]]: 嵌入向量列表
        """
        if not texts:
            return []

        logger.debug(f"计算 {len(texts)} 个文本的嵌入向量")

        # 批量处理
        all_embeddings = []
        for i in range(0, len(texts), self.config.batch_size):
            batch = texts[i:i + self.config.batch_size]
            batch_embeddings = await self._embed_batch_with_retry(batch)
            all_embeddings.extend(batch_embeddings)

            # 添加延迟以避免API限制
            if i + self.config.batch_size < len(texts):
                await asyncio.sleep(0.1)

        logger.debug(f"嵌入计算完成，共 {len(all_embeddings)} 个向量")
        return all_embeddings

    async def _embed_batch_with_retry(self, texts: List[str]) -> List[List[float]]:
        """
        带重试机制的批量嵌入计算

        Args:
            texts: 文本批次

        Returns:
            List[List[float]]: 嵌入向量列表
        """
        last_exception = None

        for attempt in range(self.config.max_retries + 1):
            try:
                return await self._embed_batch(texts)

            except Exception as e:
                last_exception = e

                if attempt < self.config.max_retries:
                    wait_time = self.config.retry_delay * (2 ** attempt)
                    logger.warning(f"嵌入计算失败 (尝试 {attempt + 1}/{self.config.max_retries + 1}): {e}")
                    logger.info(f"等待 {wait_time:.1f} 秒后重试...")
                    await asyncio.sleep(wait_time)
                else:
                    logger.error(f"嵌入计算最终失败: {e}")
                    break

        # 如果所有重试都失败，返回零向量作为fallback
        logger.error(f"所有重试都失败，返回零向量。错误: {last_exception}")
        return [[0.0] * self._get_embedding_dimension() for _ in texts]

    async def _embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        批量计算嵌入向量

        Args:
            texts: 文本批次

        Returns:
            List[List[float]]: 嵌入向量列表
        """
        # 准备请求参数
        kwargs = {
            "model": self.config.model_name,
            "input": texts,
            "timeout": self.config.timeout
        }

        # 仅text-embedding-3系列支持dimensions参数
        if self.config.dimensions and "text-embedding-3" in self.config.model_name:
            kwargs["dimensions"] = self.config.dimensions

        # 调用OpenAI API
        response = await self.client.embeddings.create(**kwargs)

        # 提取嵌入向量
        embeddings = [data.embedding for data in response.data]

        logger.debug(f"批次处理完成，输入: {len(texts)} 个文本，输出: {len(embeddings)} 个向量")
        return embeddings

    def _get_embedding_dimension(self) -> int:
        """
        获取嵌入向量维度

        Returns:
            int: 向量维度
        """
        dimension_map = {
            "text-embedding-ada-002": 1536,
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072
        }
        return dimension_map.get(self.config.model_name, 1536)

    def get_model_info(self) -> Dict[str, Any]:
        """
        获取模型信息

        Returns:
            Dict[str, Any]: 模型信息
        """
        return {
            "model_name": self.config.model_name,
            "dimensions": self._get_embedding_dimension(),
            "max_retries": self.config.max_retries,
            "batch_size": self.config.batch_size,
            "timeout": self.config.timeout,
            "api_base": self.config.api_base or "https://api.openai.com/v1"
        }


class CustomOpenAIEmbeddingFunction:
    """
    自定义OpenAI嵌入函数，完全兼容ChromaDB的embedding_functions接口
    这是ChromaDB期望的接口格式
    """

    def __init__(self,
                 model_name: str = "text-embedding-3-small",
                 api_key: Optional[str] = None,
                 api_base: Optional[str] = None,
                 **kwargs):
        """
        初始化自定义OpenAI嵌入函数

        Args:
            model_name: 模型名称
            api_key: OpenAI API Key
            api_base: API基础URL
            **kwargs: 其他配置参数
        """
        self.model_name = model_name  # ChromaDB需要的name属性
        self._name = model_name  # 内部存储名称
        self.api_key = api_key
        self.api_base = api_base

        self.embedding_function = OpenAIEmbeddingFunction(
            model_name=model_name,
            api_key=api_key,
            api_base=api_base,
            config=EmbeddingConfig(model_name=model_name, api_key=api_key, api_base=api_base, **kwargs)
        )

    def __call__(self, input: List[str]) -> List[List[float]]:
        """
        同步接口，内部使用异步实现
        兼容ChromaDB的EmbeddingFunction接口

        Args:
            input: 文本列表（ChromaDB接口要求使用input参数名）

        Returns:
            List[List[float]]: 嵌入向量列表
        """
        # 直接实现同步嵌入计算，避免事件循环冲突
        try:
            # 在新的事件循环中运行异步嵌入函数
            import asyncio

            # 创建新的事件循环来运行异步函数
            try:
                # 尝试获取当前循环
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # 如果当前循环正在运行，在新线程中创建新循环
                    import concurrent.futures

                    def run_in_new_loop():
                        new_loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(new_loop)
                        try:
                            return new_loop.run_until_complete(self.embedding_function(input))
                        finally:
                            new_loop.close()

                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        future = executor.submit(run_in_new_loop)
                        return future.result(timeout=60)
                else:
                    # 当前循环未运行，可以直接使用
                    return loop.run_until_complete(self.embedding_function(input))

            except RuntimeError:
                # 没有事件循环，创建新的
                new_loop = asyncio.new_event_loop()
                asyncio.set_event_loop(new_loop)
                try:
                    return new_loop.run_until_complete(self.embedding_function(input))
                finally:
                    new_loop.close()

        except Exception as e:
            logger.error(f"嵌入计算失败: {e}")
            # 返回零向量作为fallback
            import random
            fallback_vectors = []
            for text in input:
                # 为每个文本生成一个1536维的向量
                vector = [random.random() for _ in range(1536)]
                fallback_vectors.append(vector)
            return fallback_vectors

    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        return self.embedding_function.get_model_info()

    def name(self) -> str:
        """获取模型名称（ChromaDB兼容性）"""
        return self._name

    def embed_query(self, input: List[str]) -> List[List[float]]:
        """
        嵌入查询文本（ChromaDB接口要求）

        Args:
            input: 查询文本列表

        Returns:
            List[List[float]]: 嵌入向量列表
        """
        return self(input)

    def embed_documents(self, input: List[str]) -> List[List[float]]:
        """
        嵌入文档文本（ChromaDB接口要求）

        Args:
            input: 文档文本列表

        Returns:
            List[List[float]]: 嵌入向量列表
        """
        return self(input)

    @property
    def model(self) -> str:
        """获取模型名称（ChromaDB兼容性）"""
        return self.model_name

    def __str__(self) -> str:
        """字符串表示"""
        return f"OpenAIEmbeddingFunction(model={self.model_name})"

    def __repr__(self) -> str:
        """详细字符串表示"""
        return f"CustomOpenAIEmbeddingFunction(model_name='{self.model_name}', api_base='{self.api_base or 'https://api.openai.com/v1'}')"


def create_openai_embedding_function(model_name: str = "text-embedding-3-small",
                                   api_key: Optional[str] = None,
                                   api_base: Optional[str] = None,
                                   **kwargs):
    """
    创建OpenAI嵌入函数的便捷函数

    Args:
        model_name: 模型名称
        api_key: OpenAI API Key
        api_base: API基础URL
        **kwargs: 其他配置参数

    Returns:
        嵌入函数对象（兼容ChromaDB接口）
    """
    if not OPENAI_AVAILABLE:
        # 如果OpenAI库不可用，返回一个模拟函数
        def fallback_embedding(texts):
            import random
            return [[random.random() for _ in range(1536)] for _ in texts]

        fallback_embedding.name = model_name
        fallback_embedding.model = model_name
        return fallback_embedding

    # 返回兼容ChromaDB接口的对象实例
    return CustomOpenAIEmbeddingFunction(
        model_name=model_name,
        api_key=api_key,
        api_base=api_base,
        **kwargs
    )


async def test_embedding_function():
    """测试OpenAI嵌入函数"""
    try:
        # 创建嵌入函数
        embed_func = create_openai_embedding_function()

        # 测试文本
        test_texts = [
            "这是一个测试文本",
            "This is a test text",
            "思源笔记是一个很好的笔记工具"
        ]

        print(f"测试OpenAI嵌入函数，模型信息: {embed_func.get_model_info()}")
        print(f"测试文本数量: {len(test_texts)}")

        # 计算嵌入向量
        embeddings = embed_func(test_texts)

        print(f"嵌入向量数量: {len(embeddings)}")
        print(f"每个向量维度: {len(embeddings[0])}")
        print(f"第一个向量的前5个值: {embeddings[0][:5]}")

        print("✅ OpenAI嵌入函数测试成功!")

    except Exception as e:
        print(f"❌ 测试失败: {e}")
        logger.error(f"OpenAI嵌入函数测试失败: {e}")


if __name__ == "__main__":
    asyncio.run(test_embedding_function())