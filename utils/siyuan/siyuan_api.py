"""
思源笔记 API 客户端工具
基于思源笔记官方 API 文档实现：https://github.com/siyuan-note/siyuan/blob/master/API_zh_CN.md
"""

import json
import os
from typing import Any, Dict, List, Optional, Union

import aiohttp
from pydantic import BaseModel, Field

# 使用共享日志器（已在logger.py中加载环境变量）
from ..logger import get_logger

logger = get_logger(__name__)


class SiYuanAPIResponse(BaseModel):
    """思源笔记 API 响应模型"""
    code: int = Field(description="响应码，0 表示成功")
    msg: str = Field(description="响应消息")
    data: Any = Field(description="响应数据")


class SiYuanError(Exception):
    """思源笔记 API 错误"""
    def __init__(self, code: int, msg: str):
        self.code = code
        self.msg = msg
        super().__init__(f"SiYuan API Error {code}: {msg}")


class SiYuanAPIClient:
    """思源笔记 API 客户端"""

    def __init__(self, host: Optional[str] = None, port: Optional[int] = None,
                 token: Optional[str] = None, timeout: int = 30):
        """
        初始化思源笔记 API 客户端

        Args:
            host: 思源笔记服务地址，默认从环境变量 SIYUAN_HOST 获取，否则为 127.0.0.1
            port: 思源笔记服务端口，默认从环境变量 SIYUAN_PORT 获取，否则为 6806
            token: API token，默认从环境变量 SIYUAN_TOKEN 获取，在思源笔记设置-关于中查看
            timeout: 请求超时时间（秒）
        """
        # 从环境变量获取配置，如果参数未提供
        self.host = host or os.getenv("SIYUAN_HOST", "127.0.0.1")
        self.port = port or int(os.getenv("SIYUAN_PORT", "6806"))
        self.token = token or os.getenv("SIYUAN_TOKEN", "")

        self.base_url = f"http://{self.host}:{self.port}"
        self.timeout = aiohttp.ClientTimeout(total=timeout)
        self._session: Optional[aiohttp.ClientSession] = None

    async def __aenter__(self):
        """异步上下文管理器入口"""
        await self._ensure_session()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """异步上下文管理器出口"""
        await self.close()

    async def _ensure_session(self):
        """确保会话存在"""
        if self._session is None or self._session.closed:
            headers = {}
            if self.token:
                headers["Authorization"] = f"Token {self.token}"
            self._session = aiohttp.ClientSession(
                base_url=self.base_url,
                headers=headers,
                timeout=self.timeout
            )

    async def close(self):
        """关闭会话"""
        if self._session and not self._session.closed:
            await self._session.close()

    @classmethod
    def from_env(cls, timeout: int = 30) -> "SiYuanAPIClient":
        """
        从环境变量创建客户端实例

        Args:
            timeout: 请求超时时间（秒）

        Returns:
            SiYuanAPIClient: 客户端实例
        """
        return cls(
            host=os.getenv("SIYUAN_HOST"),
            port=int(os.getenv("SIYUAN_PORT", "6806")) if os.getenv("SIYUAN_PORT") else None,
            token=os.getenv("SIYUAN_TOKEN"),
            timeout=timeout
        )

    async def _request(self, endpoint: str, data: Optional[Dict[str, Any]] = None) -> SiYuanAPIResponse:
        """
        发送 API 请求

        Args:
            endpoint: API 端点
            data: 请求数据

        Returns:
            API 响应对象

        Raises:
            SiYuanError: API 返回错误时抛出
        """
        await self._ensure_session()

        url = f"{self.base_url}{endpoint}"

        try:
            async with self._session.post(url, json=data) as response:
                response_data = await response.json()
                result = SiYuanAPIResponse(**response_data)

                if result.code != 0:
                    raise SiYuanError(result.code, result.msg)

                logger.debug(f"API 请求成功: {endpoint}")
                return result

        except aiohttp.ClientError as e:
            logger.error(f"网络请求失败: {endpoint}, 错误: {e}")
            raise
        except json.JSONDecodeError as e:
            logger.error(f"JSON 解析失败: {endpoint}, 错误: {e}")
            raise
        except Exception as e:
            logger.error(f"API 请求异常: {endpoint}, 错误: {e}")
            raise

    # ========== 笔记本相关 API ==========

    async def ls_notebooks(self) -> List[Dict[str, Any]]:
        """列出笔记本"""
        response = await self._request("/api/notebook/lsNotebooks")
        return response.data.get("notebooks", [])

    async def open_notebook(self, notebook_id: str) -> None:
        """打开笔记本"""
        await self._request("/api/notebook/openNotebook", {"notebook": notebook_id})

    async def close_notebook(self, notebook_id: str) -> None:
        """关闭笔记本"""
        await self._request("/api/notebook/closeNotebook", {"notebook": notebook_id})

    async def rename_notebook(self, notebook_id: str, name: str) -> None:
        """重命名笔记本"""
        await self._request("/api/notebook/renameNotebook", {
            "notebook": notebook_id,
            "name": name
        })

    async def create_notebook(self, name: str) -> Dict[str, Any]:
        """创建笔记本"""
        response = await self._request("/api/notebook/createNotebook", {"name": name})
        return response.data["notebook"]

    async def remove_notebook(self, notebook_id: str) -> None:
        """删除笔记本"""
        await self._request("/api/notebook/removeNotebook", {"notebook": notebook_id})

    async def get_notebook_conf(self, notebook_id: str) -> Dict[str, Any]:
        """获取笔记本配置"""
        response = await self._request("/api/notebook/getNotebookConf", {"notebook": notebook_id})
        return response.data

    async def set_notebook_conf(self, notebook_id: str, conf: Dict[str, Any]) -> Dict[str, Any]:
        """保存笔记本配置"""
        response = await self._request("/api/notebook/setNotebookConf", {
            "notebook": notebook_id,
            "conf": conf
        })
        return response.data

    # ========== 文档相关 API ==========

    async def create_doc_with_md(self, notebook_id: str, path: str, markdown: str) -> str:
        """通过 Markdown 创建文档"""
        response = await self._request("/api/filetree/createDocWithMd", {
            "notebook": notebook_id,
            "path": path,
            "markdown": markdown
        })
        return response.data

    async def rename_doc(self, notebook_id: str, path: str, title: str) -> None:
        """重命名文档"""
        await self._request("/api/filetree/renameDoc", {
            "notebook": notebook_id,
            "path": path,
            "title": title
        })

    async def rename_doc_by_id(self, doc_id: str, title: str) -> None:
        """通过 ID 重命名文档"""
        await self._request("/api/filetree/renameDocByID", {
            "id": doc_id,
            "title": title
        })

    async def remove_doc(self, notebook_id: str, path: str) -> None:
        """删除文档"""
        await self._request("/api/filetree/removeDoc", {
            "notebook": notebook_id,
            "path": path
        })

    async def remove_doc_by_id(self, doc_id: str) -> None:
        """通过 ID 删除文档"""
        await self._request("/api/filetree/removeDocByID", {"id": doc_id})

    async def move_docs(self, from_paths: List[str], to_notebook_id: str, to_path: str) -> None:
        """移动文档"""
        await self._request("/api/filetree/moveDocs", {
            "fromPaths": from_paths,
            "toNotebook": to_notebook_id,
            "toPath": to_path
        })

    async def move_docs_by_id(self, from_ids: List[str], to_id: str) -> None:
        """通过 ID 移动文档"""
        await self._request("/api/filetree/moveDocsByID", {
            "fromIDs": from_ids,
            "toID": to_id
        })

    async def get_hpath_by_path(self, notebook_id: str, path: str) -> str:
        """根据路径获取人类可读路径"""
        response = await self._request("/api/filetree/getHPathByPath", {
            "notebook": notebook_id,
            "path": path
        })
        return response.data

    async def get_hpath_by_id(self, block_id: str) -> str:
        """根据 ID 获取人类可读路径"""
        response = await self._request("/api/filetree/getHPathByID", {"id": block_id})
        return response.data

    async def get_path_by_id(self, block_id: str) -> Dict[str, str]:
        """根据 ID 获取存储路径"""
        response = await self._request("/api/filetree/getPathByID", {"id": block_id})
        return response.data

    async def get_ids_by_hpath(self, path: str, notebook_id: str) -> List[str]:
        """根据人类可读路径获取 IDs"""
        response = await self._request("/api/filetree/getIDsByHPath", {
            "path": path,
            "notebook": notebook_id
        })
        return response.data

    async def list_doc_tree(self, notebook_id: str, path: str = "/") -> List[Dict[str, Any]]:
        """列出文档树"""
        response = await self._request("/api/filetree/listDocTree", {
            "notebook": notebook_id,
            "path": path
        })
        return response.data.get("tree", [])

    # ========== 块相关 API ==========

    async def insert_block(self, data: str, data_type: str = "markdown",
                          next_id: str = "", previous_id: str = "", parent_id: str = "") -> Dict[str, Any]:
        """插入块"""
        response = await self._request("/api/block/insertBlock", {
            "dataType": data_type,
            "data": data,
            "nextID": next_id,
            "previousID": previous_id,
            "parentID": parent_id
        })
        return response.data[0]["doOperations"][0]

    async def prepend_block(self, data: str, data_type: str = "markdown", parent_id: str = "") -> Dict[str, Any]:
        """插入前置子块"""
        response = await self._request("/api/block/prependBlock", {
            "data": data,
            "dataType": data_type,
            "parentID": parent_id
        })
        return response.data[0]["doOperations"][0]

    async def append_block(self, data: str, data_type: str = "markdown", parent_id: str = "") -> Dict[str, Any]:
        """插入后置子块"""
        response = await self._request("/api/block/appendBlock", {
            "data": data,
            "dataType": data_type,
            "parentID": parent_id
        })
        return response.data[0]["doOperations"][0]

    async def update_block(self, block_id: str, data: str, data_type: str = "markdown") -> Dict[str, Any]:
        """更新块"""
        response = await self._request("/api/block/updateBlock", {
            "id": block_id,
            "data": data,
            "dataType": data_type
        })
        return response.data[0]["doOperations"][0]

    async def delete_block(self, block_id: str) -> None:
        """删除块"""
        await self._request("/api/block/deleteBlock", {"id": block_id})

    async def move_block(self, block_id: str, previous_id: str = "", parent_id: str = "") -> None:
        """移动块"""
        await self._request("/api/block/moveBlock", {
            "id": block_id,
            "previousID": previous_id,
            "parentID": parent_id
        })

    async def fold_block(self, block_id: str) -> None:
        """折叠块"""
        await self._request("/api/block/foldBlock", {"id": block_id})

    async def unfold_block(self, block_id: str) -> None:
        """展开块"""
        await self._request("/api/block/unfoldBlock", {"id": block_id})

    async def get_block_kramdown(self, block_id: str) -> Dict[str, Any]:
        """获取块 kramdown 源码"""
        response = await self._request("/api/block/getBlockKramdown", {"id": block_id})
        return response.data

    async def get_child_blocks(self, block_id: str) -> List[Dict[str, Any]]:
        """获取子块"""
        response = await self._request("/api/block/getChildBlocks", {"id": block_id})
        return response.data

    async def transfer_block_ref(self, from_id: str, to_id: str, ref_ids: Optional[List[str]] = None) -> None:
        """转移块引用"""
        data = {"fromID": from_id, "toID": to_id}
        if ref_ids:
            data["refIDs"] = ref_ids
        await self._request("/api/block/transferBlockRef", data)

    # ========== 属性相关 API ==========

    async def set_block_attrs(self, block_id: str, attrs: Dict[str, str]) -> None:
        """设置块属性"""
        await self._request("/api/attr/setBlockAttrs", {
            "id": block_id,
            "attrs": attrs
        })

    async def get_block_attrs(self, block_id: str) -> Dict[str, Any]:
        """获取块属性"""
        response = await self._request("/api/attr/getBlockAttrs", {"id": block_id})
        return response.data

    async def get_block_updated_time(self, block_id: str) -> Optional[int]:
        """
        获取块更新时间

        Args:
            block_id: 块ID

        Returns:
            Optional[int]: 更新时间戳，格式如20230601162812，如果获取失败返回None
        """
        try:
            attrs = await self.get_block_attrs(block_id)
            updated = attrs.get("updated")
            if updated:
                return int(updated)
            return None
        except Exception as e:
            logger.error(f"获取块更新时间失败: {block_id}, 错误: {e}")
            return None

    # ========== SQL 相关 API ==========

    async def query_sql(self, stmt: str) -> List[Dict[str, Any]]:
        """执行 SQL 查询"""
        response = await self._request("/api/query/sql", {"stmt": stmt})
        return response.data

    async def flush_transaction(self) -> None:
        """提交事务"""
        await self._request("/api/sqlite/flushTransaction")

    # ========== 模板相关 API ==========

    async def render_template(self, doc_id: str, template_path: str) -> Dict[str, Any]:
        """渲染模板"""
        response = await self._request("/api/template/render", {
            "id": doc_id,
            "path": template_path
        })
        return response.data

    async def render_sprig(self, template: str) -> str:
        """渲染 Sprig 模板"""
        response = await self._request("/api/template/renderSprig", {"template": template})
        return response.data

    # ========== 文件相关 API ==========

    async def get_file(self, path: str) -> str:
        """获取文件内容"""
        await self._ensure_session()
        url = f"{self.base_url}/api/file/getFile"

        try:
            async with self._session.post(url, json={"path": path}) as response:
                if response.status == 200:
                    return await response.text()
                else:
                    error_data = await response.json()
                    raise SiYuanError(error_data.get("code", response.status),
                                    error_data.get("msg", f"HTTP {response.status}"))
        except aiohttp.ClientError as e:
            logger.error(f"获取文件失败: {path}, 错误: {e}")
            raise

    async def put_file(self, path: str, content: Union[str, bytes], is_dir: bool = False,
                      mod_time: Optional[int] = None) -> None:
        """写入文件"""
        await self._ensure_session()
        url = f"{self.base_url}/api/file/putFile"

        data = aiohttp.FormData()
        data.add_field("path", path)
        if is_dir:
            data.add_field("isDir", "true")
        if mod_time:
            data.add_field("modTime", str(mod_time))
        if isinstance(content, bytes):
            data.add_field("file", content, filename="file")
        else:
            data.add_field("file", content, filename="file", content_type="text/plain")

        async with self._session.post(url, data=data) as response:
            result = await response.json()
            response_obj = SiYuanAPIResponse(**result)
            if response_obj.code != 0:
                raise SiYuanError(response_obj.code, response_obj.msg)

    async def remove_file(self, path: str) -> None:
        """删除文件"""
        await self._request("/api/file/removeFile", {"path": path})

    async def rename_file(self, path: str, new_path: str) -> None:
        """重命名文件"""
        await self._request("/api/file/renameFile", {
            "path": path,
            "newPath": new_path
        })

    async def read_dir(self, path: str) -> List[Dict[str, Any]]:
        """列出目录"""
        response = await self._request("/api/file/readDir", {"path": path})
        return response.data

    # ========== 导出相关 API ==========

    async def export_md_content(self, doc_id: str) -> Dict[str, str]:
        """导出 Markdown 文本"""
        response = await self._request("/api/export/exportMdContent", {"id": doc_id})
        return response.data

    async def export_resources(self, paths: List[str], name: Optional[str] = None) -> str:
        """导出文件与目录"""
        data = {"paths": paths}
        if name:
            data["name"] = name
        response = await self._request("/api/export/exportResources", data)
        return response.data["path"]

    # ========== 转换相关 API ==========

    async def convert_pandoc(self, dir_name: str, args: List[str]) -> str:
        """Pandoc 转换"""
        response = await self._request("/api/convert/pandoc", {
            "dir": dir_name,
            "args": args
        })
        return response.data["path"]

    # ========== 通知相关 API ==========

    async def push_msg(self, msg: str, timeout: int = 7000) -> str:
        """推送消息"""
        response = await self._request("/api/notification/pushMsg", {
            "msg": msg,
            "timeout": timeout
        })
        return response.data["id"]

    async def push_err_msg(self, msg: str, timeout: int = 7000) -> str:
        """推送报错消息"""
        response = await self._request("/api/notification/pushErrMsg", {
            "msg": msg,
            "timeout": timeout
        })
        return response.data["id"]

    # ========== 网络相关 API ==========

    async def forward_proxy(self, url: str, method: str = "GET", timeout: int = 7000,
                          content_type: str = "application/json", headers: Optional[List[Dict[str, str]]] = None,
                          payload: Optional[Union[Dict, str]] = None, payload_encoding: str = "text",
                          response_encoding: str = "text") -> Dict[str, Any]:
        """正向代理"""
        data = {
            "url": url,
            "method": method,
            "timeout": timeout,
            "contentType": content_type,
            "headers": headers or [],
            "payload": payload or {},
            "payloadEncoding": payload_encoding,
            "responseEncoding": response_encoding
        }
        response = await self._request("/api/network/forwardProxy", data)
        return response.data

    # ========== 系统相关 API ==========

    async def get_boot_progress(self) -> Dict[str, Any]:
        """获取启动进度"""
        response = await self._request("/api/system/bootProgress")
        return response.data

    async def get_version(self) -> str:
        """获取系统版本"""
        response = await self._request("/api/system/version")
        return response.data

    async def get_current_time(self) -> int:
        """获取系统当前时间（毫秒）"""
        response = await self._request("/api/system/currentTime")
        return response.data