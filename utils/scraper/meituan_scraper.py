# -*- coding: utf-8 -*-
"""
utils/scraper/meituan_scraper.py
=================================
美团招聘爬虫（Playwright 拦截 API 响应模式）。

API 端点
--------
    POST https://zhaopin.meituan.com/api/official/job/getJobList
    Body: {pageNo:1, pageSize:10, ...}

技术方案
--------
    美团招聘 SPA 需要浏览器 Cookie 认证。
    使用 Playwright 无头 Chromium 拦截 API 响应获取数据。

依赖
----
    - playwright

Author: HireInsight-Agent
"""

from __future__ import annotations

import asyncio
import logging

_API_URL = "https://zhaopin.meituan.com/web/campus"
_API_PATTERN = "api/official/job/getJobList"
_PAGE_SIZE = 10
_DEFAULT_MAX_PAGES = 10


class MeituanScraper:
    """美团招聘爬虫（Playwright 拦截模式）。"""

    def __init__(
        self,
        page_size: int = _PAGE_SIZE,
        max_pages: int = _DEFAULT_MAX_PAGES,
        keyword: str | None = None,
    ) -> None:
        self.page_size = page_size
        self.max_pages = max_pages
        self.keyword = keyword
        self._logger = logging.getLogger(f"{__name__}.meituan")

    def crawl(self) -> list[dict]:
        try:
            loop = asyncio.get_running_loop()
            # Already in event loop, use run_coroutine_threadsafe or nest
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(asyncio.run, self._crawl_async())
                return future.result()
        except RuntimeError:
            return asyncio.run(self._crawl_async())

    async def _crawl_async(self) -> list[dict]:
        from playwright.async_api import async_playwright

        kw_info = f" | keyword={self.keyword}" if self.keyword else " (全量)"
        print(f"[Meituan] ========== 美团招聘爬虫启动 ==========")
        print(f"[Meituan] page_size={self.page_size} | max_pages={self.max_pages}{kw_info}")

        all_jobs: list[dict] = []
        seen_ids: set[str] = set()

        async def handle_resp(response):
            if _API_PATTERN in response.url and response.status == 200:
                try:
                    ct = response.headers.get("content-type", "")
                    if "json" in ct:
                        data = await response.json()
                        dl = data.get("data", {})
                        jobs = dl.get("list", []) or dl.get("records", []) or (data.get("data") if isinstance(data.get("data"), list) else [])
                        for job in jobs:
                            jid = str(job.get("jobUnionId") or job.get("id", ""))
                            if jid and jid not in seen_ids:
                                seen_ids.add(jid)
                                job["source"] = "meituan"
                                all_jobs.append(job)
                        total = dl.get("total", dl.get("count", 0))
                        print(f"[Meituan] Captured {len(jobs)} jobs, total_pool={total}, collected={len(all_jobs)}")
                except Exception:
                    pass

        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            page = await browser.new_page()
            page.on("response", handle_resp)

            await page.goto(_API_URL, wait_until="domcontentloaded", timeout=30000)
            await page.wait_for_timeout(5000)

            # Try clicking next page
            for _ in range(self.max_pages - 1):
                try:
                    for sel in [".btn-next:not([disabled])", "button:has-text('>')",
                                ".ant-pagination-next:not(.ant-pagination-disabled)"]:
                        btn = page.locator(sel).first
                        if await btn.count() > 0 and await btn.is_visible():
                            disabled = await btn.get_attribute("disabled")
                            if disabled is None or disabled == "false":
                                await btn.click()
                                await page.wait_for_timeout(2000)
                                break
                except Exception:
                    break

            await browser.close()

        print(f"[Meituan] ========== 爬虫结束，去重后共 {len(all_jobs)} 条 ==========")
        return all_jobs
