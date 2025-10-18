"""
思源笔记 API 文档遍历工具
基于思源笔记 API 遍历获取笔记结构和ID信息，替代本地文件系统遍历
"""

from typing import List, Optional, Dict, Any
from dataclasses import dataclass

from .siyuan_api import SiYuanAPIClient
from ..logger import get_logger

logger = get_logger(__name__)


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


class SiYuanAPITraversal:
    """基于API的思源笔记文档遍历工具"""

    def __init__(self, client: Optional[SiYuanAPIClient] = None):
        """
        初始化API遍历工具

        Args:
            client: 思源笔记API客户端，如果为None则创建新客户端
        """
        self.client = client

    async def list_notebooks(self) -> List[Dict[str, Any]]:
        """
        列出所有笔记本

        Returns:
            List[Dict[str, Any]]: 笔记本信息列表
        """
        if not self.client:
            raise ValueError("需要提供API客户端")

        return await self.client.ls_notebooks()

    async def get_all_notes_from_notebook(self, notebook_id: str) -> List[NoteInfo]:
        """
        从笔记本获取所有笔记信息（使用API）

        Args:
            notebook_id: 笔记本ID

        Returns:
            List[NoteInfo]: 笔记信息列表，包含层级结构
        """
        if not self.client:
            raise ValueError("需要提供API客户端")

        logger.info(f"开始通过API遍历笔记本: {notebook_id}")

        # 获取文档树结构
        doc_tree = await self.client.list_doc_tree(notebook_id, "/")

        # 转换为NoteInfo结构
        root_notes = []

        for node in doc_tree:
            note_info = await self._convert_tree_node_to_note_info(notebook_id, node)
            if note_info:
                root_notes.append(note_info)

        logger.info(f"笔记本 {notebook_id} 通过API找到文档，获取了 {len(root_notes)} 个根笔记")
        return root_notes

    async def get_all_note_ids(self, notebook_id: str) -> List[str]:
        """
        获取笔记本中所有笔记ID（扁平列表）

        Args:
            notebook_id: 笔记本ID

        Returns:
            List[str]: 所有笔记ID列表
        """
        root_notes = await self.get_all_notes_from_notebook(notebook_id)
        all_ids = []

        def collect_ids(notes: List[NoteInfo]):
            for note in notes:
                all_ids.append(note.id)
                if note.children:
                    collect_ids(note.children)

        logger.info(f"开始进行笔记本 {notebook_id} 的笔记遍历获取...")
        collect_ids(root_notes)
        return all_ids

    async def _convert_tree_node_to_note_info(self, notebook_id: str, node: Dict[str, Any],
                                            parent_id: Optional[str] = None) -> Optional[NoteInfo]:
        """
        将文档树节点转换为NoteInfo对象

        Args:
            notebook_id: 笔记本ID
            node: 文档树节点
            parent_id: 父笔记ID

        Returns:
            Optional[NoteInfo]: 笔记信息对象
        """
        if not node or "id" not in node:
            return None

        note_id = node["id"]

        # 简化处理：只使用文档树提供的信息，不额外调用API
        # 因为listDocTree API返回的可能是目录或文件夹ID，不一定是文档ID
        title = note_id  # 默认使用ID作为标题
        path = f"{notebook_id}/"  # 构造基本的路径信息

        # 如果有父ID，说明是子文档，可以尝试构造相对路径
        if parent_id:
            path += f"{parent_id}/{note_id}"
        else:
            path += note_id

        note_info = NoteInfo(
            id=note_id,
            title=title,
            path=path,
            parent_id=parent_id
        )

        # 处理子节点
        if "children" in node and node["children"]:
            for child_node in node["children"]:
                child_note = await self._convert_tree_node_to_note_info(
                    notebook_id, child_node, note_id
                )
                if child_note:
                    note_info.children.append(child_note)

        return note_info

    async def get_note_info_by_id(self, notebook_id: str, note_id: str) -> Optional[NoteInfo]:
        """
        根据笔记ID获取笔记信息

        Args:
            notebook_id: 笔记本ID
            note_id: 笔记ID

        Returns:
            Optional[NoteInfo]: 笔记信息
        """
        # 获取完整的笔记结构
        root_notes = await self.get_all_notes_from_notebook(notebook_id)
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


async def create_api_traversal_from_env() -> SiYuanAPITraversal:
    """
    从环境变量创建API遍历工具实例

    Returns:
        SiYuanAPITraversal: API遍历工具实例
    """
    client = SiYuanAPIClient.from_env()
    return SiYuanAPITraversal(client)


if __name__ == "__main__":
    # 测试代码
    import asyncio

    async def test_api_traversal():
        """测试API遍历工具"""
        traversal = await create_api_traversal_from_env()

        print("笔记本列表:")
        notebooks = await traversal.list_notebooks()
        for notebook in notebooks:
            print(f"  ID: {notebook['id']}, 名称: {notebook['name']}")

        if notebooks:
            # 测试第一个笔记本
            test_notebook_id = notebooks[0]['id']
            print(f"\n通过API获取笔记本 '{test_notebook_id}' 的所有笔记:")

            note_ids = await traversal.get_all_note_ids(test_notebook_id)
            print(f"找到 {len(note_ids)} 个笔记:")
            for i, note_id in enumerate(note_ids[:10]):  # 只显示前10个
                print(f"  {i+1}. {note_id}")

            if len(note_ids) > 10:
                print(f"  ... 还有 {len(note_ids) - 10} 个笔记")

    asyncio.run(test_api_traversal())