"""
共享日志工具
提供统一的日志配置和管理
"""

import logging
import os
import sys
from pathlib import Path
from typing import Optional
from datetime import datetime

# 加载环境变量
from dotenv import load_dotenv
load_dotenv(override=True)


class SiYuanLogger:
    """思源笔记项目专用日志工具"""

    _instance = None
    _configured = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if self._configured:
            return

        # 获取日志配置
        self.log_level = os.getenv("LOG_LEVEL", "INFO").upper()
        self.log_file = os.getenv("LOG_FILE")
        self.log_format = os.getenv(
            "LOG_FORMAT",
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        self.enable_console = os.getenv("LOG_CONSOLE", "true").lower() == "true"

        # 配置根日志器
        self._configure_root_logger()
        self._configured = True

    def _configure_root_logger(self):
        """配置根日志器"""
        root_logger = logging.getLogger()
        root_logger.setLevel(getattr(logging, self.log_level, logging.INFO))

        # 清除现有处理器
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)

        # 创建格式器
        formatter = logging.Formatter(self.log_format)

        # 控制台处理器
        if self.enable_console:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(formatter)
            root_logger.addHandler(console_handler)

        # 文件处理器
        if self.log_file:
            self._ensure_log_directory()
            file_handler = logging.FileHandler(self.log_file, encoding='utf-8')
            file_handler.setFormatter(formatter)
            root_logger.addHandler(file_handler)

    def _ensure_log_directory(self):
        """确保日志目录存在"""
        if self.log_file:
            log_path = Path(self.log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)

    @classmethod
    def get_logger(cls, name: str) -> logging.Logger:
        """
        获取日志器实例

        Args:
            name: 日志器名称，通常使用 __name__

        Returns:
            logging.Logger: 日志器实例
        """
        # 确保已配置
        if not cls._configured:
            cls()

        return logging.getLogger(name)

    @classmethod
    def configure(cls,
                 log_level: str = "INFO",
                 log_file: Optional[str] = None,
                 log_format: Optional[str] = None,
                 enable_console: bool = True):
        """
        手动配置日志器

        Args:
            log_level: 日志级别
            log_file: 日志文件路径
            log_format: 日志格式
            enable_console: 是否启用控制台输出
        """
        # 设置环境变量
        os.environ["LOG_LEVEL"] = log_level
        if log_file:
            os.environ["LOG_FILE"] = log_file
        if log_format:
            os.environ["LOG_FORMAT"] = log_format
        os.environ["LOG_CONSOLE"] = str(enable_console).lower()

        # 重新创建实例
        cls._instance = None
        cls._configured = False
        cls()


def get_logger(name: str) -> logging.Logger:
    """
    获取日志器的便捷函数

    Args:
        name: 日志器名称

    Returns:
        logging.Logger: 日志器实例
    """
    return SiYuanLogger.get_logger(name)


def configure_logger(log_level: str = "INFO",
                    log_file: Optional[str] = None,
                    log_format: Optional[str] = None,
                    enable_console: bool = True):
    """
    配置日志的便捷函数

    Args:
        log_level: 日志级别
        log_file: 日志文件路径
        log_format: 日志格式
        enable_console: 是否启用控制台输出
    """
    SiYuanLogger.configure(log_level, log_file, log_format, enable_console)


# 默认配置 - 使用环境变量或默认值
configure_logger(
    log_level=os.getenv("LOG_LEVEL", "INFO"),
    log_file=os.getenv("LOG_FILE"),
    enable_console=os.getenv("LOG_CONSOLE", "true").lower() == "true"
)


# 创建项目专用的日志器
siyuan_api_logger = get_logger("siyuan_api")
siyuan_workspace_logger = get_logger("siyuan_workspace")
rag_logger = get_logger("rag")
agent_logger = get_logger("agent")


if __name__ == "__main__":
    # 测试日志器
    test_logger = get_logger("test")

    test_logger.debug("这是一条调试信息")
    test_logger.info("这是一条信息")
    test_logger.warning("这是一条警告")
    test_logger.error("这是一条错误")
    test_logger.critical("这是一条严重错误")

    # 测试专用日志器
    siyuan_api_logger.info("API日志器测试")
    siyuan_workspace_logger.info("工作空间日志器测试")