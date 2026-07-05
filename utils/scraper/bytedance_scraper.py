# -*- coding: utf-8 -*-
"""
utils/scraper/bytedance_scraper.py
===================================
字节跳动招聘爬虫（Playwright 拦截 API 响应模式）。

API 端点
--------
    GET https://jobs.bytedance.com/api/v1/search/job/posts
    Params: keyword="", limit=10, offset=0

技术方案
--------
    字节跳动招聘页为 SPA 应用，API 需要浏览器 Cookie 认证。
    本爬虫使用 Playwright 启动无头 Chromium，拦截 API 响应获取数据。
    每次 crawl() 调用独立启停浏览器。

参数说明
--------
    - keyword : str  搜索关键词（可选，空字符串=全量）
    - limit   : int  每页条数（建议 10）
    - offset  : int  偏移量（分页用）

依赖
----
    - playwright

Author: HireInsight-Agent
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

_API_URL = "https://jobs.bytedance.com/experienced/position"
_API_PATTERN = "api/v1/search/job/posts"
_PAGE_SIZE = 10
_DEFAULT_MAX_PAGES = 10


class ByteDanceScraper:
    """字节跳动招聘爬虫（Playwright 拦截模式）。"""

    def __init__(
        self,
        page_size: int = _PAGE_SIZE,
        max_pages: int = _DEFAULT_MAX_PAGES,
        keyword: str | None = None,
    ) -> None:
        self.page_size = page_size
        self.max_pages = max_pages
        self.keyword = keyword
        self._logger = logging.getLogger(f"{__name__}.bytedance")

    def crawl(self) -> list[dict]:
        return asyncio.run(self._crawl_async())

    async def _crawl_async(self) -> list[dict]:
        from playwright.async_api import async_playwright

        kw_info = f" | keyword={self.keyword}" if self.keyword else " (全量)"
        print(f"[ByteDance] ========== 字节跳动招聘爬虫启动 ==========")
        print(f"[ByteDance] page_size={self.page_size} | max_pages={self.max_pages}{kw_info}")

        all_jobs: list[dict] = []
        seen_ids: set[str] = set()

        async def handle_resp(response):
            if _API_PATTERN in response.url and response.status == 200:
                try:
                    ct = response.headers.get("content-type", "")
                    if "json" in ct:
                        data = await response.json()
                        dl = data.get("data", {})
                        jobs = dl.get("job_post_list", [])
                        if not jobs and isinstance(data.get("data"), list):
                            jobs = data["data"]
                        for job in jobs:
                            jid = str(job.get("id", ""))
                            if jid and jid not in seen_ids:
                                seen_ids.add(jid)
                                job["source"] = "bytedance"
                                all_jobs.append(job)
                        print(f"[ByteDance] Captured {len(jobs)} jobs (total collected={len(all_jobs)})")
                except Exception:
                    pass

        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            page = await browser.new_page()
            page.on("response", handle_resp)

            kw = self.keyword or ""
            url = f"https://jobs.bytedance.com/experienced/position?keywords={kw}" if kw else _API_URL
            await page.goto(url, wait_until="domcontentloaded", timeout=30000)
            await page.wait_for_timeout(4000)

            # Scroll to trigger lazy loading
            for pg in range(1, self.max_pages):
                await page.evaluate(f"window.scrollTo(0, {pg * 4000})")
                await page.wait_for_timeout(2500)

            await browser.close()

        print(f"[ByteDance] ========== 爬虫结束，去重后共 {len(all_jobs)} 条 ==========")
        return all_jobs
