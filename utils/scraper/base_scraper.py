# -*- coding: utf-8 -*-
"""
utils/scraper/base_scraper.py
==============================
爬虫基类模块，为各平台招聘爬虫提供统一的网络请求能力。

功能特性
--------
- 抽象基类设计：定义 crawl() 接口契约，强制派生类实现
- Session 复用：全局 requests.Session，提升请求效率
- 工业级浏览器指纹伪装：UA + Sec-Ch-Ua + Sec-Ch-Ua-Platform 全匹配轮换
- Brotli 压缩解压：自动处理 Content-Encoding: br（requests 默认不支持）
- 生产级容错：tenacity 驱动指数退避重试，覆盖 5xx/连接超时/ReadTimeout
- 日志追踪：每轮重试输出 WARNING 日志，便于线上问题定位

依赖
----
    pip install requests tenacity brotli

架构约定
--------
所有大厂爬虫须继承 BaseScraper，并实现：

    class ByteDanceScraper(BaseScraper):
        def crawl(self) -> list[dict]:
            # 返回该平台所有原始岗位字典列表
            ...

Author: HireInsight-Agent
"""

from __future__ import annotations

import logging
import random
import time
from abc import ABC, abstractmethod
from typing import Any

import requests
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

# ---------------------------------------------------------------------------
# Brotli 解压支持（可选依赖，缺失时自动降级）
# ---------------------------------------------------------------------------
try:
    import brotli  # type: ignore[import-untyped]

    _BROTLI_AVAILABLE = True
    _brotli_decompress = brotli.decompress
except ImportError:  # pragma: no cover
    _BROTLI_AVAILABLE = False
    _brotli_decompress = None   # type: ignore[assignment]

# ---------------------------------------------------------------------------
# 日志配置
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 常量定义
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# 完整浏览器指纹池（工业级对抗大厂 WAF）
#
# 设计原则：
#   1. 每个指纹包含 UA + Sec-Ch-Ua + Sec-Ch-Ua-Mobile + Sec-Ch-Ua-Platform，
#      四者必须为同一浏览器的同一版本，否则 WAF 的 JS Challenge 检测到
#      UA 声称 Chrome 但 Sec-Ch-Ua 缺或无匹配，直接标记为 Bot 流量。
#   2. 覆盖 Windows / macOS / Linux 三大平台，模拟真实访客地理分布。
#   3. 版本号接近最新稳定版，避免被标记为"过时浏览器"可疑特征。
# ---------------------------------------------------------------------------
_BROWSER_FINGERPRINTS: list[dict[str, str]] = [
    # Chrome 131 on Windows 11
    {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/131.0.0.0 Safari/537.36"
        ),
        "Sec-Ch-Ua": '"Google Chrome";v="131", "Chromium";v="131", "Not A(Brand";v="24"',
        "Sec-Ch-Ua-Mobile": "?0",
        "Sec-Ch-Ua-Platform": '"Windows"',
    },
    # Chrome 130 on macOS
    {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/130.0.0.0 Safari/537.36"
        ),
        "Sec-Ch-Ua": '"Google Chrome";v="130", "Chromium";v="130", "Not A(Brand";v="99"',
        "Sec-Ch-Ua-Mobile": "?0",
        "Sec-Ch-Ua-Platform": '"macOS"',
    },
    # Firefox 133 on Windows
    {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:133.0) "
            "Gecko/20100101 Firefox/133.0"
        ),
        # Firefox 不发送 Sec-Ch-Ua，该项设为空以区别于 Chrome 系
        "Sec-Ch-Ua": "",
        "Sec-Ch-Ua-Mobile": "",
        "Sec-Ch-Ua-Platform": "",
    },
    # Edge 131 on Windows (Chromium 内核，Sec-Ch-Ua 与 Chrome 同源)
    {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/131.0.0.0 Safari/537.36 Edg/131.0.0.0"
        ),
        "Sec-Ch-Ua": '"Microsoft Edge";v="131", "Chromium";v="131", "Not A(Brand";v="24"',
        "Sec-Ch-Ua-Mobile": "?0",
        "Sec-Ch-Ua-Platform": '"Windows"',
    },
    # Chrome 129 on Linux (模拟开发者/服务器场景)
    {
        "User-Agent": (
            "Mozilla/5.0 (X11; Linux x86_64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/129.0.0.0 Safari/537.36"
        ),
        "Sec-Ch-Ua": '"Google Chrome";v="129", "Chromium";v="129", "Not A(Brand";v="99"',
        "Sec-Ch-Ua-Mobile": "?0",
        "Sec-Ch-Ua-Platform": '"Linux"',
    },
]

# 默认请求头骨架（可被子类覆盖）
# Accept-Encoding 根据 brotli 可用性动态决定是否声明 br
_DEFAULT_HEADERS: dict[str, str] = {
    "Accept": "application/json, text/plain, */*",
    "Accept-Encoding": "gzip, deflate" + (", br" if _BROTLI_AVAILABLE else ""),
    "Accept-Language": "zh-CN,zh;q=0.9,en-US;q=0.8,en;q=0.7",
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    "DNT": "1",
    "Sec-Fetch-Dest": "empty",
    "Sec-Fetch-Mode": "cors",
    "Sec-Fetch-Site": "same-origin",
    "Upgrade-Insecure-Requests": "1",
}

# 网络超时配置（秒）
_CONNECT_TIMEOUT: int = 10   # 建立连接超时
_READ_TIMEOUT: int = 30     # 读取响应超时

# 重试配置
_MAX_RETRIES: int = 3       # 最大重试次数
_WAIT_MULTIPLIER: int = 2   # 退避基数（秒）
_WAIT_MAX: int = 8          # 最大等待时间（秒）


# ---------------------------------------------------------------------------
# 自定义异常
# ---------------------------------------------------------------------------

class ScraperError(Exception):
    """爬虫基础异常，所有爬虫相关错误的基类。"""
    pass


class NetworkError(ScraperError):
    """网络层异常（连接超时 / 读取超时 / 5xx 响应）。"""
    pass


class ParseError(ScraperError):
    """数据解析异常（JSON 解码失败 / 字段缺失）。"""
    pass


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------

def _build_retry_decorator() -> Any:
    """
    构建 tenacity retry 装饰器。

    重试触发条件
    ------------
    1. requests.exceptions.ConnectTimeout      —— 连接建立超时
    2. requests.exceptions.ReadTimeout        —— 服务端未在 ReadTimeout 内响应
    3. requests.exceptions.ConnectionError     —— 连接被拒绝/网络不可达
    4. requests.exceptions.HTTPError (5xx)    —— 服务器内部错误

    退避策略：指数退避 (exponential back-off)，首次 2s，最高 8s
    """
    return retry(
        retry=retry_if_exception_type(
            (
                requests.exceptions.ConnectTimeout,
                requests.exceptions.ReadTimeout,
                requests.exceptions.ConnectionError,
                requests.exceptions.HTTPError,   # 仅 5xx 自动重试
            )
        ),
        stop=stop_after_attempt(_MAX_RETRIES),
        wait=wait_exponential(
            multiplier=_WAIT_MULTIPLIER,
            min=_WAIT_MULTIPLIER,
            max=_WAIT_MAX,
        ),
        reraise=True,           # 重试耗尽后仍抛出原异常
        before_sleep=lambda retry_state: logger.warning(
            "[%s] 请求失败，第 %d/%d 次重试中... "
            "等待 %.1f 秒后重试。（错误：%s）",
            time.strftime("%Y-%m-%d %H:%M:%S"),
            retry_state.attempt_number,
            _MAX_RETRIES,
            retry_state.next_action.sleep if retry_state.next_action else 0,
            retry_state.outcome.exception() if retry_state.outcome else "unknown",
        ),
        after=logger.info(
            "[%s] 请求成功，恢复执行。",
            time.strftime("%Y-%m-%d %H:%M:%S"),
        ),
    )


# ---------------------------------------------------------------------------
# 抽象基类
# ---------------------------------------------------------------------------

class BaseScraper(ABC):
    """
    招聘爬虫抽象基类。

    设计原则
    --------
    1. **接口契约**：派生类必须实现 crawl() 方法，返回 list[dict]
    2. **Session 复用**：子类共享同一个 Session 实例（连接池复用）
    3. **自防御网络层**：所有 HTTP 请求自动携带 UA + 重试逻辑
    4. **零爬取逻辑**：基类本身不实现任何业务爬取逻辑

    使用示例
    --------
    ```python
    class MyScraper(BaseScraper):
        def crawl(self) -> list[dict]:
            return self._get(url="https://example.com/api/jobs")

    jobs = MyScraper().crawl()
    ```

    Attributes
    ----------
    source : str
        平台标识符，由子类在 __init__ 中设置（例：`"bytedance"`）
    """

    def __init__(self, source: str = "unknown") -> None:
        """
        初始化爬虫实例。

        Parameters
        ----------
        source : str
            数据来源标识，用于日志和错误追踪
        """
        self.source = source
        self._session: requests.Session | None = None
        self._logger = logging.getLogger(f"{__name__}.{source}")

        # 子类初始化的 headers（子类可覆盖）
        self._base_headers: dict[str, str] = _DEFAULT_HEADERS.copy()

        # 工业级指纹：从指纹池随机选取一条完整浏览器指纹
        # （UA + Sec-Ch-Ua + Sec-Ch-Ua-Platform + Sec-Ch-Ua-Mobile 四者同源）
        self._rotate_fingerprint(init=True)

    # ------------------------------------------------------------------
    # Session 管理
    # ------------------------------------------------------------------

    @property
    def session(self) -> requests.Session:
        """
        延迟初始化的 requests.Session（线程不安全，单线程使用）。

        同一实例多次调用返回同一个 Session 对象，确保连接池复用。
        """
        if self._session is None or not isinstance(self._session, requests.Session):
            self._session = requests.Session()
            # 将 base_headers 合并到 Session 默认 headers
            self._session.headers.update(self._base_headers)
            self._logger.debug(
                "[%s] Session 已初始化，UA = %s",
                self.source,
                self._session.headers.get("User-Agent", "unknown"),
            )
        return self._session

    def close(self) -> None:
        """显式关闭 Session，释放连接池资源。推荐在爬虫使用完毕后调用。"""
        if self._session is not None:
            self._session.close()
            self._session = None
            self._logger.debug("[%s] Session 已关闭。", self.source)

    # ------------------------------------------------------------------
    # UA 管理
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # 浏览器指纹管理（工业级：UA + Sec-Ch-Ua 全匹配轮换）
    # ------------------------------------------------------------------

    @staticmethod
    def _random_fingerprint() -> dict[str, str]:
        """从指纹池中随机选取一条完整浏览器指纹（UA + Sec-Ch-Ua 系列）。"""
        return random.choice(_BROWSER_FINGERPRINTS).copy()

    def _rotate_fingerprint(self, *, init: bool = False) -> dict[str, str]:
        """
        随机更换当前 Session 的完整浏览器指纹。

        与旧版 rotate_ua() 的关键区别：不仅换 UA，同时刷新 Sec-Ch-Ua /
        Sec-Ch-Ua-Platform / Sec-Ch-Ua-Mobile，确保四者同源匹配，
        避免 WAF 检测到 UA 与 Sec-Ch-Ua 不一致而触发 Bot 标记。

        Parameters
        ----------
        init : bool
            是否为首次初始化（True 时不打 debug 日志，减少噪音）

        Returns
        -------
        dict[str, str]
            新选中的指纹字典
        """
        fingerprint = self._random_fingerprint()
        self._base_headers.update(fingerprint)
        if self._session is not None:
            self._session.headers.update(fingerprint)
        if not init:
            self._logger.debug(
                "[%s] 浏览器指纹已轮换: UA=Chrome/%s", self.source,
                fingerprint.get("Sec-Ch-Ua", "")[:30]
            )
        return fingerprint

    # 向后兼容：保留旧 API（内部委托至新方法）
    @staticmethod
    def _random_ua() -> str:
        """[已废弃] 从指纹池中随机选取 User-Agent。保留以兼容旧调用。"""
        return _BROWSER_FINGERPRINTS[0]["User-Agent"]

    def rotate_ua(self) -> str:
        """
        [已废弃] 随机更换完整浏览器指纹（已升级为 rotate_fingerprint）。

        为保持 API 兼容性，内部委托至 _rotate_fingerprint。
        """
        self._rotate_fingerprint()
        return self._base_headers.get("User-Agent", "")

    # ------------------------------------------------------------------
    # 通用网络请求（已包裹 tenacity 重试）
    # ------------------------------------------------------------------

    @_build_retry_decorator()
    def _request(
        self,
        method: str,
        url: str,
        *,
        headers: dict[str, str] | None = None,
        params: dict[str, Any] | None = None,
        json: dict[str, Any] | None = None,
        data: dict[str, Any] | None = None,
        timeout: tuple[int, int] | int | None = None,
        allow_redirects: bool = True,
    ) -> requests.Response:
        """
        通用 HTTP 请求方法（已内置 tenacity 重试）。

        Parameters
        ----------
        method : str
            HTTP 方法：`GET`、`POST`、`HEAD` 等
        url : str
            目标 URL
        headers : dict, optional
            覆盖请求头（与 base_headers 合并）
        params : dict, optional
            URL 查询参数（自动编码）
        json : dict, optional
            JSON 请求体（自动设置 Content-Type）
        data : dict, optional
            表单编码请求体
        timeout : tuple | int, optional
            (connect_timeout, read_timeout)，若传入 int 则同时作为两者
        allow_redirects : bool
            是否跟随 3xx 重定向，默认 True

        Returns
        -------
        requests.Response
            响应对象（未做 raise_for_status，需调用方自行判断）

        Raises（经 tenacity 重试后仍失败的异常）
        ----------------------------------------
        requests.exceptions.ConnectTimeout
        requests.exceptions.ReadTimeout
        requests.exceptions.ConnectionError
        requests.exceptions.HTTPError（仅 5xx）
        """
        # 合并 headers
        merged_headers = self._base_headers.copy()
        if headers:
            merged_headers.update(headers)

        # 解析 timeout
        if timeout is None:
            resolved_timeout = (_CONNECT_TIMEOUT, _READ_TIMEOUT)
        elif isinstance(timeout, int):
            resolved_timeout = (timeout, timeout)
        else:
            resolved_timeout = timeout  # type: ignore[assignment]

        self._logger.debug(
            "[%s] >>> %s %s | params=%s",
            self.source,
            method.upper(),
            url,
            params,
        )

        response = self.session.request(
            method=method.upper(),
            url=url,
            headers=merged_headers,
            params=params,
            json=json,
            data=data,
            timeout=resolved_timeout,
            allow_redirects=allow_redirects,
        )

        # ------------------------------------------------------------------
        # Brotli 解压：requests/urllib3 默认仅处理 gzip 和 deflate，
        # 对 Content-Encoding: br 的响应不作解压处理，导致 response.content
        # 为原始 Brotli 二进制流。此处拦截并原地解压。
        #
        # 防御策略：
        #   1. 仅当响应体非空时尝试解压
        #   2. 解压失败时回退为 gzip 解压（部分 CDN 声称 br 但实际发 gzip）
        #   3. 全部失败时清空 _content 缓存，让调用方从 raw socket 重读
        # ------------------------------------------------------------------
        content_encoding = response.headers.get("Content-Encoding", "").lower()
        if "br" in content_encoding and _BROTLI_AVAILABLE and _brotli_decompress is not None:
            raw_body: bytes = response.content
            if raw_body and len(raw_body) > 0:
                try:
                    decompressed: bytes = _brotli_decompress(raw_body)
                    response._content = decompressed  # type: ignore[attr-defined]
                    self._logger.debug(
                        "[%s] Brotli 解压完成: %.1f KB → %.1f KB",
                        self.source,
                        len(raw_body) / 1024,
                        len(decompressed) / 1024,
                    )
                except Exception:
                    # brotli 失败 → 尝试 gzip 回退（某些 CDN/代理误标 Content-Encoding）
                    import gzip as _gz
                    try:
                        decompressed = _gz.decompress(raw_body)
                        response._content = decompressed  # type: ignore[attr-defined]
                        self._logger.debug(
                            "[%s] Brotli 失败，gzip 回退成功: %.1f KB → %.1f KB",
                            self.source,
                            len(raw_body) / 1024,
                            len(decompressed) / 1024,
                        )
                    except Exception:
                        # 全部失败：清空 _content 缓存，让调用方收到原始字节流
                        response._content = raw_body  # type: ignore[attr-defined]
                        self._logger.debug(
                            "[%s] 解压失败（br+gzip），保留 %d 字节原始数据",
                            self.source, len(raw_body),
                        )
        elif "br" in content_encoding and not _BROTLI_AVAILABLE:
            self._logger.warning(
                "[%s] 响应为 Brotli 压缩，但 brotli 库未安装（"
                "请执行 pip install brotli），响应内容可能为乱码。",
                self.source,
            )

        # 仅 5xx 进入 tenacity 重试，4xx 直接返回（由调用方处理）
        if response.status_code >= 500:
            self._logger.warning(
                "[%s] 服务端返回 %d，触发重试机制。",
                self.source,
                response.status_code,
            )
            response.raise_for_status()   # 抛出 HTTPError，触发 tenacity

        self._logger.debug(
            "[%s] <<< %d %s | size=%d bytes",
            self.source,
            response.status_code,
            response.url,
            len(response.content),
        )

        return response

    def get(
        self,
        url: str,
        **kwargs: Any,
    ) -> requests.Response:
        """
        发送 GET 请求（等同于 self._request("GET", url, **kwargs)）。
        """
        return self._request("GET", url, **kwargs)

    def post(
        self,
        url: str,
        **kwargs: Any,
    ) -> requests.Response:
        """
        发送 POST 请求（等同于 self._request("POST", url, **kwargs)）。
        """
        return self._request("POST", url, **kwargs)

    # ------------------------------------------------------------------
    # JSON 解析辅助
    # ------------------------------------------------------------------

    def _parse_json(self, response: requests.Response) -> dict[str, Any]:
        """
        将响应内容解析为 JSON 字典。

        Parameters
        ----------
        response : requests.Response

        Returns
        -------
        dict
            解析后的字典

        Raises
        ------
        ParseError
            JSON 解码失败时抛出，并附带响应文本前 200 字符供调试
        """
        try:
            return response.json()
        except Exception as e:
            preview = response.text[:200] if response.text else "(empty)"
            raise ParseError(
                f"[{self.source}] JSON 解析失败: {e} | "
                f"响应预览: {preview}"
            ) from e

    # ------------------------------------------------------------------
    # 抽象方法（子类必须实现）
    # ------------------------------------------------------------------

    @abstractmethod
    def crawl(self) -> list[dict]:
        """
        执行完整的数据爬取流程。

        Returns
        -------
        list[dict]
            **原始**岗位数据字典列表。每个字典的字段结构由各平台原始 API 决定，
            后续由 data_transformer.py 统一转换为标准格式。

        Raises
        ------
        ScraperError
            爬取过程中发生的所有业务错误
        """
        ...  # pragma: no cover

    # ------------------------------------------------------------------
    # 上下文管理（支持 with 语句）
    # ------------------------------------------------------------------

    def __enter__(self) -> "BaseScraper":
        """支持 `with MyScraper() as scraper:` 自动管理 Session。"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """退出时自动关闭 Session。"""
        self.close()

    # ------------------------------------------------------------------
    # 通用工具方法（供子类复用）
    # ------------------------------------------------------------------

    def _paginate(
        self,
        fetch_one_page: Any,
        page_start: int = 1,
        page_end: int | None = None,
        max_pages: int = 100,
        stop_if_empty: bool = True,
    ) -> list[dict]:
        """
        通用的分页爬取模板方法。

        Parameters
        ----------
        fetch_one_page : Callable[[int], list[dict]]
            单页抓取函数签名：接收页码 int，返回当页数据 list
        page_start : int
            起始页码（从 1 开始）
        page_end : int, optional
            终止页码，若为 None 则依赖 max_pages 或 stop_if_empty
        max_pages : int
            最大页数上限（防止无限循环）
        stop_if_empty : bool
            若某页返回空列表，是否立即停止，默认 True

        Returns
        -------
        list[dict]
            所有页数据的合并列表
        """
        all_data: list[dict] = []
        page = page_start

        while page_end is None or page <= page_end:
            if len(all_data) >= max_pages * 100:   # 保守保护
                self._logger.warning(
                    "[%s] 已达到最大页数上限 %d，停止分页。",
                    self.source,
                    max_pages,
                )
                break

            self._logger.info(
                "[%s] 正在抓取第 %d 页 ...",
                self.source,
                page,
            )
            page_data = fetch_one_page(page)

            if not page_data:
                if stop_if_empty:
                    self._logger.info(
                        "[%s] 第 %d 页为空，停止分页。",
                        self.source,
                        page,
                    )
                    break
            else:
                all_data.extend(page_data)
                self._logger.info(
                    "[%s] 第 %d 页抓取成功，本批累计 %d 条记录。",
                    self.source,
                    page,
                    len(all_data),
                )

            page += 1

        self._logger.info(
            "[%s] 分页爬取完成，共 %d 页，合计 %d 条记录。",
            self.source,
            page - 1,
            len(all_data),
        )
        return all_data
