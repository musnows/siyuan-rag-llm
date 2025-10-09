"""
思源笔记内容工具
用于获取笔记本中所有笔记的markdown内容
"""

from typing import Dict, List, Optional, Iterator, AsyncIterator
from dataclasses import dataclass

# 使用共享日志器（已在logger.py中加载环境变量）
from .logger import get_logger
from .siyuan_workspace import SiYuanWorkspace, NoteInfo
from .siyuan_api import SiYuanAPIClient

logger = get_logger(__name__)


@dataclass
class NoteContent:
    """笔记内容"""
    id: str
    title: str
    path: str
    content: str
    parent_id: Optional[str] = None
    children_ids: Optional[List[str]] = None

    def __post_init__(self):
        if self.children_ids is None:
            self.children_ids = []


class SiYuanContentExtractor:
    """思源笔记内容提取器"""

    def __init__(self, workspace_path: Optional[str] = None):
        """
        初始化内容提取器

        Args:
            workspace_path: 思源笔记工作空间路径，默认从环境变量获取
        """
        self.workspace = SiYuanWorkspace(workspace_path)
        self.api_client = SiYuanAPIClient.from_env()
        logger.info(f"初始化思源笔记内容提取器") 

    async def get_all_note_contents(self, notebook_id: str, include_children: bool = True) -> List[NoteContent]:
        """
        获取笔记本中所有笔记的markdown内容

        Args:
            notebook_id: 笔记本ID
            include_children: 是否包含子笔记，如果为False则只获取根笔记

        Returns:
            List[NoteContent]: 所有笔记内容列表
        """
        logger.info(f"开始获取笔记本 {notebook_id} 的所有笔记内容")

        # 获取笔记结构
        if include_children:
            root_notes = self.workspace.get_all_notes_from_notebook(notebook_id)
            all_notes = self._flatten_note_tree(root_notes)
        else:
            # 只获取根笔记ID
            root_notes = self.workspace.get_all_notes_from_notebook(notebook_id)
            all_notes = root_notes

        # 获取每个笔记的内容
        note_contents = []
        async with self.api_client:
            for note in all_notes:
                try:
                    # 使用API获取markdown内容
                    md_data = await self.api_client.export_md_content(note.id)
                    content = md_data.get("content", "")

                    note_content = NoteContent(
                        id=note.id,
                        title=note.title,
                        path=note.path,
                        content=content,
                        parent_id=note.parent_id,
                        children_ids=[child.id for child in note.children] if note.children else []
                    )
                    note_contents.append(note_content)
                except Exception as e:
                    logger.warning(f"无法获取笔记内容: {note.id}, 错误: {e}")

        logger.info(f"成功获取 {len(note_contents)} 个笔记的内容")
        return note_contents

    async def iterate_note_contents(self, notebook_id: str, include_children: bool = True) -> AsyncIterator[NoteContent]:
        """
        迭代获取笔记内容（生成器模式，节省内存）

        Args:
            notebook_id: 笔记本ID
            include_children: 是否包含子笔记

        Yields:
            NoteContent: 笔记内容
        """
        logger.info(f"开始迭代获取笔记本 {notebook_id} 的笔记内容")

        # 获取笔记结构
        if include_children:
            root_notes = self.workspace.get_all_notes_from_notebook(notebook_id)
            all_notes = self._flatten_note_tree(root_notes)
        else:
            root_notes = self.workspace.get_all_notes_from_notebook(notebook_id)
            all_notes = root_notes

        # 逐个获取笔记内容
        async with self.api_client:
            for note in all_notes:
                try:
                    # 使用API获取markdown内容
                    md_data = await self.api_client.export_md_content(note.id)
                    content = md_data.get("content", "")

                    note_content = NoteContent(
                        id=note.id,
                        title=note.title,
                        path=note.path,
                        content=content,
                        parent_id=note.parent_id,
                        children_ids=[child.id for child in note.children] if note.children else []
                    )
                    yield note_content
                except Exception as e:
                    logger.warning(f"无法获取笔记内容: {note.id}, 错误: {e}")

    async def get_note_contents_dict(self, notebook_id: str, include_children: bool = True) -> Dict[str, NoteContent]:
        """
        获取笔记内容字典（以笔记ID为键）

        Args:
            notebook_id: 笔记本ID
            include_children: 是否包含子笔记

        Returns:
            Dict[str, NoteContent]: 笔记内容字典
        """
        note_contents = await self.get_all_note_contents(notebook_id, include_children)
        return {note.id: note for note in note_contents}

    async def get_note_contents_by_path(self, notebook_id: str, include_children: bool = True) -> Dict[str, NoteContent]:
        """
        获取笔记内容字典（以路径为键）

        Args:
            notebook_id: 笔记本ID
            include_children: 是否包含子笔记

        Returns:
            Dict[str, NoteContent]: 笔记内容字典
        """
        note_contents = await self.get_all_note_contents(notebook_id, include_children)
        return {note.path: note for note in note_contents}

    async def get_note_content_by_id(self, notebook_id: str, note_id: str) -> Optional[NoteContent]:
        """
        根据ID获取单个笔记内容

        Args:
            notebook_id: 笔记本ID
            note_id: 笔记ID

        Returns:
            Optional[NoteContent]: 笔记内容，如果不存在则返回None
        """
        logger.info(f"获取笔记内容: {note_id}")

        # 获取笔记路径
        note_path = self.workspace.get_note_path_by_id(notebook_id, note_id)
        if not note_path:
            logger.warning(f"找不到笔记路径: {note_id}")
            return None

        # 提取标题
        note_info = self.workspace.get_note_info_by_id(notebook_id, note_id)
        if not note_info:
            logger.warning(f"无法获取笔记信息: {note_id}")
            return None

        # 使用API获取内容
        try:
            async with self.api_client:
                md_data = await self.api_client.export_md_content(note_id)
                content = md_data.get("content", "")

                return NoteContent(
                    id=note_id,
                    title=note_info.title,
                    path=note_path,
                    content=content,
                    parent_id=note_info.parent_id,
                    children_ids=[child.id for child in note_info.children] if note_info.children else []
                )
        except Exception as e:
            logger.error(f"获取笔记内容失败: {note_id}, 错误: {e}")
            return None

    async def get_markdown_text_dict(self, notebook_id: str, include_children: bool = True) -> Dict[str, str]:
        """
        获取简化的markdown文本字典（只包含标题和内容）

        Args:
            notebook_id: 笔记本ID
            include_children: 是否包含子笔记

        Returns:
            Dict[str, str]: 笔记ID到markdown内容的映射
        """
        note_contents = await self.get_all_note_contents(notebook_id, include_children)
        return {note.id: note.content for note in note_contents}

    async def search_notes_by_content(self, notebook_id: str, keyword: str, include_children: bool = True) -> List[NoteContent]:
        """
        根据关键词搜索笔记内容

        Args:
            notebook_id: 笔记本ID
            keyword: 搜索关键词
            include_children: 是否包含子笔记

        Returns:
            List[NoteContent]: 包含关键词的笔记列表
        """
        logger.info(f"在笔记本 {notebook_id} 中搜索关键词: {keyword}")

        matching_notes = []
        async for note_content in self.iterate_note_contents(notebook_id, include_children):
            if keyword.lower() in note_content.content.lower() or keyword.lower() in note_content.title.lower():
                matching_notes.append(note_content)

        logger.info(f"找到 {len(matching_notes)} 个匹配的笔记")
        return matching_notes

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

    

def create_content_extractor(workspace_path: Optional[str] = None) -> SiYuanContentExtractor:
    """
    创建内容提取器的便捷函数

    Args:
        workspace_path: 工作空间路径

    Returns:
        SiYuanContentExtractor: 内容提取器实例
    """
    return SiYuanContentExtractor(workspace_path)


async def main():
    """测试代码"""
    extractor = create_content_extractor("~/data/notes/siyuan")

    # 获取所有笔记本
    notebooks = extractor.workspace.list_notebooks()
    if notebooks:
        # 测试第一个笔记本
        test_notebook_id = notebooks[-1][0]
        print(f"测试笔记本: {test_notebook_id}")

        # 获取前3个笔记的内容
        contents = await extractor.get_all_note_contents(test_notebook_id)
        print(f"获取到 {len(contents)} 个笔记")

        # 显示前3个笔记的基本信息
        for i, note_content in enumerate(contents[:3]):
            print(f"\n笔记 {i+1}:")
            print(f"  ID: {note_content.id}")
            print(f"  标题: {note_content.title}")
            print(f"  路径: {note_content.path}")
            print(f"  内容长度: {len(note_content.content)} 字符")
            print(f"  内容预览: {note_content.content[:100]}...")

        # 测试搜索功能
        keyword = "测试"
        search_results = await extractor.search_notes_by_content(test_notebook_id, keyword)
        print(f"\n搜索关键词 '{keyword}' 找到 {len(search_results)} 个结果")

        # 测试迭代器
        print(f"\n使用迭代器遍历前3个笔记:")
        count = 0
        async for note_content in extractor.iterate_note_contents(test_notebook_id):
            if count >= 3:
                break
            print(f"  {count+1}. {note_content.title} ({len(note_content.content)} 字符)")
            count += 1


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())