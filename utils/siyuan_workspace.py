"""
思源笔记工作空间工具
用于通过本地文件系统遍历获取笔记结构和ID信息
"""

import json
import os
import re
from pathlib import Path
from typing import List, Optional, Tuple
from dataclasses import dataclass

# 使用共享日志器（已在logger.py中加载环境变量）
from .logger import get_logger

logger = get_logger(__name__)

# 思源笔记笔记本ID的日期格式：YYYYMMDDHHMMSS-xxxxxx
NOTEBOOK_ID_PATTERN = re.compile(r'^\d{14}-\w+$')


@dataclass
class NoteInfo:
    """笔记信息"""
    id: str
    title: str
    path: str
    parent_id: Optional[str] = None
    children: Optional[List['NoteInfo']] = None

    def __post_init__(self):
        if self.children is None:
            self.children = []


class SiYuanWorkspace:
    """思源笔记工作空间操作工具"""

    def __init__(self, workspace_path: Optional[str] = None):
        """
        初始化工作空间工具

        Args:
            workspace_path: 思源笔记工作空间路径，默认从环境变量 SIYUAN_WORKSPACE_PATH 获取
        """
        self.workspace_path = workspace_path or os.getenv("SIYUAN_WORKSPACE_PATH")
        if not self.workspace_path:
            raise ValueError("请提供工作空间路径或设置环境变量 SIYUAN_WORKSPACE_PATH")
        # 替换波浪线用户家目录
        if self.workspace_path.startswith("~"):
            self.workspace_path = os.path.expanduser(self.workspace_path)

        self.data_path = Path(self.workspace_path) / "data"
        if not self.data_path.exists():
            raise ValueError(f"工作空间数据目录不存在: {self.data_path}")

        logger.info(f"初始化思源笔记工作空间: {self.workspace_path}")

    def list_notebooks(self) -> List[Tuple[str, str]]:
        """
        列出所有笔记本（仅包含以日期格式开头的笔记本）

        Returns:
            List[Tuple[str, str]]: 笔记本ID和名称的列表
        """
        notebooks = []
        for item in self.data_path.iterdir():
            if item.is_dir() and NOTEBOOK_ID_PATTERN.match(item.name):
                # 只处理符合日期格式的笔记本ID
                # 读取笔记本配置文件获取名称
                conf_file = item / ".siyuan" / "conf.json"
                if conf_file.exists():
                    try:
                        with open(conf_file, 'r', encoding='utf-8') as f:
                            conf = json.load(f)
                            name = conf.get("name", item.name)
                    except Exception as e:
                        logger.warning(f"读取笔记本配置失败: {conf_file}, 错误: {e}")
                        name = item.name
                else:
                    name = item.name

                notebooks.append((item.name, name))

        logger.info(f"找到 {len(notebooks)} 个符合条件的笔记本")
        return notebooks

    def get_all_notes_from_notebook(self, notebook_id: str) -> List[NoteInfo]:
        """
        从笔记本获取所有笔记信息

        Args:
            notebook_id: 笔记本ID

        Returns:
            List[NoteInfo]: 笔记信息列表，包含层级结构
        """
        notebook_path = self.data_path / notebook_id
        if not notebook_path.exists():
            raise ValueError(f"笔记本不存在: {notebook_id}")

        logger.info(f"开始遍历笔记本: {notebook_id}")

        # 构建笔记树结构
        notes_dict = {}
        root_notes = []

        # 遍历所有文件和目录
        for root, _, files in os.walk(notebook_path):
            root_path = Path(root)

            # 处理每个.ssy文件（笔记文件）
            for file in files:
                if file.endswith('.sy'):
                    note_path = root_path / file
                    relative_path = note_path.relative_to(notebook_path)
                    note_id = self._extract_note_id_from_path(relative_path)

                    # 获取笔记标题
                    title = self._extract_note_title(note_path)

                    # 确定父笔记ID
                    parent_id = self._get_parent_note_id(relative_path)

                    note_info = NoteInfo(
                        id=note_id,
                        title=title,
                        path=str(relative_path),
                        parent_id=parent_id
                    )

                    notes_dict[note_id] = note_info

                    # 如果没有父ID，则是根笔记
                    if not parent_id:
                        root_notes.append(note_info)

        # 构建层级关系
        for note_id, note_info in notes_dict.items():
            if note_info.parent_id and note_info.parent_id in notes_dict:
                parent = notes_dict[note_info.parent_id]
                parent.children.append(note_info)

        logger.info(f"笔记本 {notebook_id} 共找到 {len(notes_dict)} 个笔记")
        return root_notes

    def get_all_note_ids(self, notebook_id: str) -> List[str]:
        """
        获取笔记本中所有笔记ID（扁平列表）

        Args:
            notebook_id: 笔记本ID

        Returns:
            List[str]: 所有笔记ID列表
        """
        root_notes = self.get_all_notes_from_notebook(notebook_id)
        all_ids = []

        def collect_ids(notes: List[NoteInfo]):
            for note in notes:
                all_ids.append(note.id)
                if note.children:
                    collect_ids(note.children)

        collect_ids(root_notes)
        return all_ids

    def get_note_path_by_id(self, notebook_id: str, note_id: str) -> Optional[str]:
        """
        根据笔记ID获取笔记路径

        Args:
            notebook_id: 笔记本ID
            note_id: 笔记ID（文件名，不含扩展名）

        Returns:
            Optional[str]: 笔记相对路径，如果不存在则返回None
        """
        notebook_path = self.data_path / notebook_id
        if not notebook_path.exists():
            return None

        # 在笔记本目录中查找对应的笔记文件
        for root, _, files in os.walk(notebook_path):
            for file in files:
                if file == f"{note_id}.sy":
                    file_path = Path(root) / file
                    relative_path = file_path.relative_to(notebook_path)
                    return str(relative_path)

        return None

    def _extract_note_id_from_path(self, relative_path: Path) -> str:
        """
        从相对路径提取笔记ID

        Args:
            relative_path: 相对于笔记本根目录的路径

        Returns:
            str: 笔记ID（文件名，不含扩展名）
        """
        # 笔记ID就是文件名（去掉.sy扩展名）
        return relative_path.stem

    def _extract_note_title(self, note_path: Path) -> str:
        """
        从笔记文件提取标题

        Args:
            note_path: 笔记文件路径

        Returns:
            str: 笔记标题
        """
        try:
            with open(note_path, 'r', encoding='utf-8') as f:
                content = f.read()

                # 解析JSON格式的内容
                try:
                    data = json.loads(content)
                    # 从Properties中获取title，如果没有则使用ID作为标题
                    title = data.get("Properties", {}).get("title", note_path.stem)
                    return title or "未命名笔记"
                except json.JSONDecodeError:
                    # 如果JSON解析失败，使用文件名作为标题
                    return note_path.stem

        except Exception as e:
            logger.warning(f"读取笔记标题失败: {note_path}, 错误: {e}")
            return note_path.stem or "未命名笔记"

    def _get_parent_note_id(self, relative_path: Path) -> Optional[str]:
        """
        获取父笔记ID

        Args:
            relative_path: 笔记文件的相对路径

        Returns:
            Optional[str]: 父笔记ID，如果是根笔记则返回None
        """
        # 如果笔记直接在笔记本根目录下，则没有父笔记
        if len(relative_path.parts) == 1:
            return None

        # 获取父目录
        parent_dir = relative_path.parent

        # 查找父目录中的index.sy作为父笔记
        index_file = parent_dir / "index.sy"
        if index_file.exists():
            return index_file.stem

        # 如果没有index.sy，则使用父目录的第一个.sy文件
        for file in parent_dir.glob("*.sy"):
            return file.stem

        return None

    def get_note_content(self, notebook_id: str, note_id: str) -> Optional[str]:
        """
        获取笔记内容

        Args:
            notebook_id: 笔记本ID
            note_id: 笔记ID

        Returns:
            Optional[str]: 笔记内容，如果不存在则返回None
        """
        note_path = self.get_note_path_by_id(notebook_id, note_id)
        if not note_path:
            return None

        file_path = self.data_path / notebook_id / note_path
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            logger.error(f"读取笔记内容失败: {file_path}, 错误: {e}")
            return None

    def get_note_info_by_id(self, notebook_id: str, note_id: str) -> Optional[NoteInfo]:
        """
        根据笔记ID获取笔记信息

        Args:
            notebook_id: 笔记本ID
            note_id: 笔记ID（文件名，不含扩展名）

        Returns:
            Optional[NoteInfo]: 笔记信息
        """
        # 获取完整的笔记结构
        root_notes = self.get_all_notes_from_notebook(notebook_id)
        all_notes = self._flatten_note_tree(root_notes)

        # 查找匹配的笔记
        for note in all_notes:
            if note.id == note_id:
                return note

        return None

    def _flatten_note_tree(self, notes: List[NoteInfo]) -> List[NoteInfo]:
        """
        将笔记树展平为列表

        Args:
            notes: 根笔记列表

        Returns:
            List[NoteInfo]: 展平后的笔记列表
        """
        result = []

        def traverse(notes: List[NoteInfo]):
            for note in notes:
                result.append(note)
                if note.children:
                    traverse(note.children)

        traverse(notes)
        return result


def create_workspace_from_env() -> SiYuanWorkspace:
    """
    从环境变量创建工作空间实例

    Returns:
        SiYuanWorkspace: 工作空间实例
    """
    return SiYuanWorkspace()


if __name__ == "__main__":
    # 测试代码
    workspace = SiYuanWorkspace("~/data/notes/siyuan")

    print("笔记本列表:")
    notebooks = workspace.list_notebooks()
    for notebook_id, name in notebooks:
        print(f"  {notebook_id}: {name}")

    if notebooks:
        # 测试第一个笔记本
        test_notebook_id = notebooks[0][0]
        print(f"\n获取笔记本 '{test_notebook_id}' 的所有笔记:")

        note_ids = workspace.get_all_note_ids(test_notebook_id)
        print(f"找到 {len(note_ids)} 个笔记:")
        for i, note_id in enumerate(note_ids[:10]):  # 只显示前10个
            print(f"  {i+1}. {note_id}")

        if len(note_ids) > 10:
            print(f"  ... 还有 {len(note_ids) - 10} 个笔记")