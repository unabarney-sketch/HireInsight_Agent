# -*- coding: utf-8 -*-
"""
utils/scraper/mihoyo_scraper.py
=================================
米哈游招聘官网爬虫。

数据契约
--------
    返回值：list[dict]
    每个 dict 为米哈游招聘 API 返回的原始岗位结构，未经任何清洗。

API 端点
--------
    岗位列表 : POST https://jobs.mihoyo.com/api/position/list
              Body: {"pageNo": 1, "pageSize": 10}

接口特点
--------
    1. RESTful JSON，无 CSRF Token / Cookie 链要求
    2. 接口公开透明，无反爬风控
    3. 单次请求返回完整字段（含岗位名、城市、薪资、学历要求等）
    4. 每页默认 10 条

参数说明
--------
    - pageNo   : int  页码（从 1 开始）
    - pageSize : int  每页条目数（建议 10）

依赖
----
    - base_scraper.BaseScraper
    - requests（由 base_scraper 引入）

注意
----
    若 API 端点发生变化，修改 _QUERY_PAGE_URL 常量即可。
    当前端点基于米哈游招聘官网 (jobs.mihoyo.com) 的公开接口。

Author: HireInsight-Agent
"""

from __future__ import annotations

import logging
import random
import time
from typing import Any

from .base_scraper import (
    BaseScraper,
    ParseError,
    ScraperError,
)

# ---------------------------------------------------------------------------
# 常量定义
# ---------------------------------------------------------------------------

# 米哈游招聘 API 地址
_QUERY_PAGE_URL: str = "https://jobs.mihoyo.com/api/position/list"

# 列表页默认值
_DEFAULT_PAGE_SIZE: int = 10        # 每页条目数
_DEFAULT_MAX_PAGES: int = 10        # 最大页数上限

# 跨页防抖区间（秒）
_PAGE_SLEEP_MIN: float = 0.2
_PAGE_SLEEP_MAX: float = 0.4

# API 响应 code 约定
_API_SUCCESS_CODE: int = 200


# ---------------------------------------------------------------------------
# 异常定义
# ---------------------------------------------------------------------------

class MihoyoScraperError(ScraperError):
    """米哈游爬虫专用异常。"""
    pass


# ---------------------------------------------------------------------------
# 爬虫实现
# ---------------------------------------------------------------------------

class MihoyoScraper(BaseScraper):
    """
    米哈游招聘官网爬虫。

    执行流程
    --------
        crawl()
          └─► _paginate(_fetch_page)   # 基类通用分页模板
                └─► _fetch_page(page)  # 单页请求 + JSON 解析

    Attributes
    ----------
    page_size : int
        每页条目数，默认 10
    max_pages : int
        最大翻页数，默认 10

    Example
    -------
    ```python
    with MihoyoScraper(max_pages=5) as scraper:
        raw_jobs = scraper.crawl()
        print(f"共抓取 {len(raw_jobs)} 条原始岗位数据")
    ```
    """

    def __init__(
        self,
        page_size: int = _DEFAULT_PAGE_SIZE,
        max_pages: int = _DEFAULT_MAX_PAGES,
    ) -> None:
        """初始化米哈游招聘爬虫。"""
        super().__init__(source="mihoyo")
        self.page_size = page_size
        self.max_pages = max_pages

        # 覆盖基类默认 Accept + 米哈游特有 Headers
        self._base_headers.update({
            "Accept": "application/json, text/plain, */*",
            "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
            "Content-Type": "application/json;charset=UTF-8",
            "Origin": "https://jobs.mihoyo.com",
            "Referer": "https://jobs.mihoyo.com/",
            # 模拟同源 AJAX 调用
            "Sec-Fetch-Dest": "empty",
            "Sec-Fetch-Mode": "cors",
            "Sec-Fetch-Site": "same-origin",
        })

        # 运行时统计
        self._total_count: int = 0

    # ------------------------------------------------------------------
    # 单页抓取（分页回调）
    # ------------------------------------------------------------------

    def _fetch_page(self, page: int) -> list[dict]:
        """
        抓取指定页码的岗位列表。

        Parameters
        ----------
        page : int
            页码（从 1 开始）

        Returns
        -------
        list[dict]
            当前页的原始岗位字典列表（来自 data.list[]）

        Raises
        ------
        ParseError
            JSON 解析失败或响应 code != 200
        """
        payload: dict[str, Any] = {
            "pageNo": page,
            "pageSize": self.page_size,
        }

        self._logger.debug(
            "[mihoyo] 第 %d 页 | pageSize=%d",
            page, self.page_size
        )

        # 跨页间隔随机延迟
        if page > 1:
            time.sleep(random.uniform(_PAGE_SLEEP_MIN, _PAGE_SLEEP_MAX))

        response = self.post(
            url=_QUERY_PAGE_URL,
            json=payload,
        )

        raw = self._parse_json(response)

        # API 业务层状态码校验
        code = raw.get("code")
        if code != _API_SUCCESS_CODE:
            msg = raw.get("msg", "未知错误")
            self._logger.warning(
                "[mihoyo] 第 %d 页 API 返回非 200: code=%s, msg=%s",
                page, code, msg,
            )
            return []

        data = raw.get("data")
        if data is None:
            self._logger.warning(
                "[mihoyo] 第 %d 页 data 字段为空"
            )
            return []

        jobs: list[dict] = data.get("list", [])
        total: int = data.get("total", 0)

        # 仅首页记录总量
        if page == 1:
            self._total_count = total
            self._logger.info(
                "[mihoyo] 接口返回总岗位数: %d", self._total_count
            )

        self._logger.debug(
            "[mihoyo] 第 %d 页解析完毕，返回 %d 条岗位记录",
            page, len(jobs)
        )

        return jobs

    # ------------------------------------------------------------------
    # 核心接口
    # ------------------------------------------------------------------

    def crawl(self) -> list[dict]:
        """
        执行完整爬取流程：分页抓取 → 合并结果。

        Returns
        -------
        list[dict]
            原始岗位字典列表，每个 dict 包含标准字段：
            id / name / workType / firstPostTypeName /
            requirement / description / reqEducationName / reqWorkYearsName /
            firstDepName / workPlaceList / workPlaceNameList /
            updateTime / post_url

        Raises
        ------
        MihoyoScraperError
            整体抓取失败
        ParseError
            响应 JSON 解析失败
        """
        self._logger.info(
            "[mihoyo] ========== 米哈游招聘爬虫启动 =========="
        )
        self._logger.info(
            "[mihoyo] page_size=%d | max_pages=%d",
            self.page_size, self.max_pages
        )

        # 分页抓取
        all_jobs: list[dict] = self._paginate(
            fetch_one_page=self._fetch_page,
            page_start=1,
            page_end=None,          # 由 stop_if_empty / max_pages 控制
            max_pages=self.max_pages,
            stop_if_empty=True,
        )

        # 去重（基于 id）
        seen: set[str] = set()
        unique_jobs: list[dict] = []
        for job in all_jobs:
            job_id = str(job.get("id", "")).strip()
            if job_id and job_id not in seen:
                seen.add(job_id)
                # 注入 source 标识，供下游 Transformer 做策略路由
                job["source"] = self.source
                unique_jobs.append(job)

        dup_count = len(all_jobs) - len(unique_jobs)
        if dup_count > 0:
            self._logger.warning(
                "[mihoyo] 检测到 %d 条重复记录（已去重）", dup_count
            )

        self._logger.info(
            "[mihoyo] ========== 爬虫结束，去重后共 %d 条 ==========",
            len(unique_jobs)
        )

        return unique_jobs

    # ------------------------------------------------------------------
    # 便捷属性
    # ------------------------------------------------------------------

    @property
    def total_count(self) -> int:
        """接口返回的岗位总条数。"""
        return self._total_count
