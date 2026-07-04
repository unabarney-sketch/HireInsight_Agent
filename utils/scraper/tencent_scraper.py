# -*- coding: utf-8 -*-
"""
utils/scraper/tencent_scraper.py
=================================
腾讯招聘官网爬虫。

数据契约
--------
    返回值：list[dict]
    每个 dict = 列表页基础字段 + 详情页扩展字段（字典深度合并）
    完整字段覆盖 Salary / Requirement / DegreeName 等列表页缺失的脏数据核心字段

API 端点
--------
    列表接口 : GET  https://careers.tencent.com/tencentcareer/api/post/Query
              ?keyword=<>&pageIndex=<>&pageSize=10&language=zh-cn&area=cn
    详情接口 : GET  https://careers.tencent.com/tencentcareer/api/post/ByPostId
              ?postId=<>&language=zh-cn

双请求流说明
-----------
    腾讯列表接口仅返回 RecruitPostName / Responsibility 等基础字段，
    Salary / Requirement / DegreeName 等核心脏数据字段藏于详情接口。
    本爬虫采用"列表页拉取 postId → 批量详情页串行请求 → 字典合并"的二级拉取模式。

高频防抖策略
------------
    详情请求后强制随机睡眠 0.3~0.8s，防止 IP 被腾讯风控标记。

依赖
----
    - base_scraper.BaseScraper
    - requests（由 base_scraper 引入）
    - time / random（标准库）

Author: HireInsight-Agent
"""

from __future__ import annotations

import logging
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from .base_scraper import (
    BaseScraper,
    ParseError,
    ScraperError,
)

# ---------------------------------------------------------------------------
# 常量定义
# ---------------------------------------------------------------------------

# 腾讯招聘 API 地址
_LIST_API_URL: str = "https://careers.tencent.com/tencentcareer/api/post/Query"
_DETAIL_API_URL: str = "https://careers.tencent.com/tencentcareer/api/post/ByPostId"

# 列表页默认值
_DEFAULT_PAGE_SIZE: int = 10        # 每页条目数
_DEFAULT_MAX_PAGES: int = 50        # 最大页数上限

# 详情请求防抖区间（秒）—— 并发模式下仅作线程内微小抖动，批量间隔由 _BATCH_JITTER 控制
_DETAIL_SLEEP_MIN: float = 0.03
_DETAIL_SLEEP_MAX: float = 0.10

# 并发线程池配置
_CONCURRENT_WORKERS: int = 8        # 线程池最大工作线程数
_BATCH_JITTER: float = 0.3          # 每批次并发完成后额外等待（秒），避免瞬时高频


# ---------------------------------------------------------------------------
# 异常定义
# ---------------------------------------------------------------------------

class TencentScraperError(ScraperError):
    """腾讯爬虫专用异常。"""
    pass


class DetailFetchError(TencentScraperError):
    """详情页抓取失败的专用异常（不影响主流程）。"""
    pass


# ---------------------------------------------------------------------------
# 爬虫实现
# ---------------------------------------------------------------------------

class TencentScraper(BaseScraper):
    """
    腾讯招聘官网爬虫。

    执行流程
    --------
        crawl()
          └─► _paginate(_fetch_list_page)       # 基类通用分页模板
                └─► _fetch_list_page(page)
                      ├─► GET /api/post/Query         # 列表页
                      │     └─► 提取 postId 列表
                      └─► [_fetch_detail(post_id)]    # 每条详情页
                            ├─► GET /api/post/ByPostId
                            ├─► random.sleep(0.3~0.8s) # 防抖
                            └─► 异常捕获 + warning 日志
                └─► Dict Merge：列表基础字段 ⋃ 详情扩展字段

    Attributes
    ----------
    keyword : str | None
        搜索关键词（可选），传入后仅返回匹配岗位
    page_size : int
        每页条目数，默认 10
    max_pages : int
        最大翻页页数上限
    language : str
        接口语言，默认 zh-cn

    Example
    -------
    ```python
    with TencentScraper() as scraper:
        raw_jobs = scraper.crawl()
        print(f"共抓取 {len(raw_jobs)} 条（含详情）原始岗位数据")
    ```
    """

    def __init__(
        self,
        keyword: str | None = None,
        page_size: int = _DEFAULT_PAGE_SIZE,
        max_pages: int = _DEFAULT_MAX_PAGES,
        language: str = "zh-cn",
    ) -> None:
        """
        初始化腾讯招聘爬虫。

        Parameters
        ----------
        keyword : str, optional
            岗位搜索关键词，不传则抓取全量
        page_size : int
            每页条目数，建议 10
        max_pages : int
            最大翻页页数上限
        language : str
            接口语言，默认 zh-cn（国内）
        """
        super().__init__(source="tencent")
        self.keyword = keyword
        self.page_size = page_size
        self.max_pages = max_pages
        self.language = language

        # 覆盖基类默认 Accept（腾讯接口 + 浏览器同源 API 伪装）
        self._base_headers.update({
            "Accept": "application/json, text/plain, */*",
            "Accept-Language": f"{language},zh;q=0.9,en;q=0.8",
            "Origin": "https://careers.tencent.com",
            "Referer": "https://careers.tencent.com/search.html",
            # 模拟同源 AJAX 调用
            "Sec-Fetch-Dest": "empty",
            "Sec-Fetch-Mode": "cors",
            "Sec-Fetch-Site": "same-origin",
        })

        # 运行时统计
        self._total_fetched: int = 0      # 列表页抓取总数
        self._total_with_details: int = 0  # 成功获取详情的数量
        self._detail_failures: int = 0    # 详情页失败数量

    # ------------------------------------------------------------------
    # 列表页抓取（分页回调）
    # ------------------------------------------------------------------

    def _fetch_list_page(self, page: int) -> list[dict]:
        """
        抓取指定页码的岗位列表页，并返回完整合并后的岗位数据。

        内部流程
        --------
            GET /api/post/Query
              └─► 提取 Posts[].postId
                    └─► 遍历 postId
                          ├─► _fetch_detail(post_id)  → dict（失败则返回 {}）
                          └─► 合并：基础字段 ⋃ 详情字段
                              （详情字段优先级更高）

        Parameters
        ----------
        page : int
            页码（从 1 开始）

        Returns
        -------
        list[dict]
            当前页所有岗位的完整数据（含列表字段 + 详情字段）
        """
        params: dict[str, Any] = {
            "keyword": self.keyword or "",
            "pageIndex": page,
            "pageSize": self.page_size,
            "language": self.language,
            "area": "cn",
        }

        self._logger.debug(
            "[tencent] 第 %d 页 | params=%s", page, params
        )

        # 跨页间隔随机延迟（0.3~0.6s）
        if page > 1:
            time.sleep(random.uniform(0.3, 0.6))
            # 每 5 页轮换一次浏览器指纹，降低长会话指纹锁定风险
            if page % 5 == 0:
                self._rotate_fingerprint()

        response = self.get(
            url=_LIST_API_URL,
            params=params,
        )

        raw = self._parse_json(response)

        # 防御性解析
        data = raw.get("Data")
        if data is None:
            self._logger.warning(
                "[tencent] 第 %d 页 Data 字段为空，响应 code=%s",
                page, raw.get("ErrCode"),
            )
            return []

        posts: list[dict] = data.get("Posts", [])
        count: int = data.get("Count", 0)

        if page == 1:
            self._total_fetched = count
            self._logger.info(
                "[tencent] 接口返回岗位总数: %d", self._total_fetched
            )

        if not posts:
            self._logger.debug(
                "[tencent] 第 %d 页 Posts 列表为空", page
            )
            return []

        # 收集有效 postId 和对应 post（过滤无 id 条目）
        valid_pairs: list[tuple[str, dict]] = []
        for post in posts:
            post_id: str = str(
                post.get("PostId") or post.get("postId") or ""
            ).strip()
            if not post_id:
                self._logger.warning(
                    "[tencent] 第 %d 页跳过无效 postId 条目", page
                )
                continue
            valid_pairs.append((post_id, post))

        if not valid_pairs:
            return []

        # ------------------------------------------------------------------
        # 并发详情拉取（ThreadPoolExecutor）
        #
        # 原串行模式：10 条 × (0.5s 请求 + 0.5s 睡眠) ≈ 10 秒/页
        # 并发模式：10 条 / 8 线程 ≈ 1-2 秒/页（5-10x 提速）
        #
        # 线程安全说明：
        #   - self._base_headers 只读访问，线程安全
        #   - 不使用 self.session（requests.Session 非线程安全），
        #     改为每线程独立 requests.get()
        #   - _DETAIL_API_URL / self.language 只读，线程安全
        # ------------------------------------------------------------------
        page_success = 0
        page_fail = 0
        results: dict[str, dict[str, Any]] = {}  # postId → merged dict

        # 独立于 self.session 的线程安全请求函数
        def _fetch_single(pid: str, base_post: dict) -> dict[str, Any]:
            import requests as _requests
            params = {"postId": pid, "language": self.language}
            try:
                resp = _requests.get(
                    _DETAIL_API_URL,
                    params=params,
                    headers=self._base_headers,
                    timeout=(10, 30),
                )
                # 手动 JSON 解析（避免依赖 self._parse_json 的 session）
                raw = resp.json()
                detail = raw.get("Data", {})
                # 线程内微小抖动，避免多线程完全同步击中
                time.sleep(random.uniform(_DETAIL_SLEEP_MIN, _DETAIL_SLEEP_MAX))
                if detail:
                    return {**base_post, **detail}
                return {**base_post}
            except Exception:
                return {**base_post}   # 失败时仅保留列表字段

        with ThreadPoolExecutor(max_workers=_CONCURRENT_WORKERS) as executor:
            future_map = {
                executor.submit(_fetch_single, pid, post): pid
                for pid, post in valid_pairs
            }
            for future in as_completed(future_map):
                pid = future_map[future]
                try:
                    result = future.result()
                    results[pid] = result
                    # 判断是否成功获取了详情（含详情特有字段）
                    if any(k in result for k in ("Salary", "Requirement", "DegreeName")):
                        page_success += 1
                    else:
                        page_fail += 1
                except Exception as e:
                    page_fail += 1
                    self._detail_failures += 1
                    self._logger.debug(
                        "[tencent] postId=%s 详情拉取失败: %s", pid, e
                    )
                    results[pid] = {**valid_pairs[[p[0] for p in valid_pairs].index(pid)][1]}

        # 保持原始顺序 + 更新全局计数器
        merged_jobs: list[dict] = []
        for pid, post in valid_pairs:
            merged = results.get(pid, {**post})
            merged_jobs.append(merged)

        self._total_with_details += page_success
        self._detail_failures += page_fail

        self._logger.info(
            "[tencent] 第 %d 页完成 | 成功: %d | 详情失败: %d",
            page, page_success, page_fail
        )

        # 批量完成后统一休眠，减缓整体请求频率
        time.sleep(random.uniform(_BATCH_JITTER, _BATCH_JITTER * 2))

        return merged_jobs

    # ------------------------------------------------------------------
    # 详情页抓取（内部方法）
    # ------------------------------------------------------------------

    def _fetch_detail(self, post_id: str) -> dict[str, Any]:
        """
        根据 postId 请求详情接口，返回扩展字段字典。

        Parameters
        ----------
        post_id : str
            岗位唯一标识

        Returns
        -------
        dict[str, Any]
            详情接口返回的扩展字段字典，异常时抛出 DetailFetchError
            （调用方负责捕获）

        Raises
        ------
        DetailFetchError
            详情接口返回非 200 / JSON 解析失败 / 字段异常
        """
        params: dict[str, str] = {
            "postId": post_id,
            "language": self.language,
        }

        try:
            response = self.get(
                url=_DETAIL_API_URL,
                params=params,
            )
            raw = self._parse_json(response)

            detail_data = raw.get("Data", {})
            if not detail_data:
                raise DetailFetchError(
                    f"postId={post_id} 详情 Data 字段为空"
                )

            return detail_data

        except (ParseError, DetailFetchError):
            raise
        except Exception as e:
            raise DetailFetchError(
                f"postId={post_id} 详情请求异常: {e}"
            ) from e

    # ------------------------------------------------------------------
    # 核心接口
    # ------------------------------------------------------------------

    def crawl(self) -> list[dict]:
        """
        执行完整爬取流程：列表分页 → 详情页拉取 → 字典合并 → 去重返回。

        Returns
        -------
        list[dict]
            完整合并后的岗位字典列表，每个 dict 包含：
            - 列表页字段：RecruitPostName / LocationName / Responsibility / ...
            - 详情页字段：Salary / Requirement / DegreeName / ...

        Raises
        ------
        TencentScraperError
            列表页整体抓取失败
        """
        self._logger.info(
            "[tencent] ========== 腾讯招聘爬虫启动 =========="
        )
        self._logger.info(
            "[tencent] keyword=%s | page_size=%d | max_pages=%d",
            self.keyword or "(全量)", self.page_size, self.max_pages
        )

        # 重置计数器
        self._total_fetched = 0
        self._total_with_details = 0
        self._detail_failures = 0

        # 分页拉取（列表页内部已完成详情合并）
        # 长会话期间每 5 页轮换一次浏览器指纹，降低指纹锁定风险
        all_jobs: list[dict] = self._paginate(
            fetch_one_page=self._fetch_list_page,
            page_start=1,
            page_end=None,          # 由 stop_if_empty / max_pages 控制
            max_pages=self.max_pages,
            stop_if_empty=True,
        )

        # 去重（基于 PostId / postId，大小写兼容）
        seen: set[str] = set()
        unique_jobs: list[dict] = []
        for job in all_jobs:
            post_id = str(job.get("PostId") or job.get("postId") or "").strip()
            if post_id and post_id not in seen:
                seen.add(post_id)
                # 注入 source 标识，确保 app.py 策略路由可靠
                job["source"] = self.source
                unique_jobs.append(job)

        dup_count = len(all_jobs) - len(unique_jobs)
        if dup_count > 0:
            self._logger.warning(
                "[tencent] 检测到 %d 条重复记录（已去重）", dup_count
            )

        self._logger.info(
            "[tencent] ========== 爬虫结束 =========="
        )
        self._logger.info(
            "[tencent] 列表总条数: %d | 成功拉取详情: %d | 详情失败: %d | "
            "去重后总计: %d",
            self._total_fetched,
            self._total_with_details,
            self._detail_failures,
            len(unique_jobs),
        )

        return unique_jobs

    # ------------------------------------------------------------------
    # 便捷属性
    # ------------------------------------------------------------------

    @property
    def total_fetched(self) -> int:
        """列表接口返回的岗位总条数。"""
        return self._total_fetched

    @property
    def detail_success_rate(self) -> float:
        """
        详情页请求成功率（百分比）。
        若列表页为空则返回 0.0。
        """
        total_attempted = self._total_with_details + self._detail_failures
        if total_attempted == 0:
            return 0.0
        return round(self._total_with_details / total_attempted * 100, 2)
