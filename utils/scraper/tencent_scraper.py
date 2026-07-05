# -*- coding: utf-8 -*-
"""
utils/scraper/tencent_scraper.py
=================================
腾讯招聘官网爬虫（Playwright 拦截 API 响应模式）。

数据契约
--------
    返回值：list[dict]
    每个 dict 为腾讯招聘 API 返回的原始岗位结构，未经任何清洗。

API 端点
--------
    岗位查询 : POST https://careers.tencent.com/tencentcareer/api/post/Query
              Body: {"timestamp": <ms>, "pageIndex": 1, "pageSize": 10}

技术方案
--------
    腾讯招聘 API 需要浏览器 Cookie 认证，单纯 HTTP 请求返回 404。
    本爬虫使用 Playwright 启动无头 Chromium，拦截浏览器发出的 API 响应，
    以此获取结构化 JSON 数据。每次 crawl() 调用独立启停浏览器。

接口特点
--------
    1. 需要浏览器会话 Cookie（Playwright 自动处理）
    2. 返回完整字段（岗位名、城市、BG/部门、产品、经验要求等）
    3. 每页默认 10 条
    4. 返回的 Posts 列表即完整数据

参数说明
--------
    - pageIndex : int  页码（从 1 开始）
    - pageSize  : int  每页条目数（建议 10）
    - timestamp : int  毫秒时间戳（防缓存）

依赖
----
    - playwright（pip install playwright && python -m playwright install chromium）

Author: HireInsight-Agent
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

# ---------------------------------------------------------------------------
# 常量定义
# ---------------------------------------------------------------------------

_QUERY_PAGE_URL: str = (
    "https://careers.tencent.com/tencentcareer/api/post/Query"
)
_CAREER_URL: str = "https://careers.tencent.com/search.html"

_DEFAULT_PAGE_SIZE: int = 10
_DEFAULT_MAX_PAGES: int = 15

_API_RESPONSE_PATTERN: str = "tencentcareer/api/post/Query"


# ---------------------------------------------------------------------------
# 爬虫实现
# ---------------------------------------------------------------------------

class TencentScraper:
    """
    腾讯招聘官网爬虫（Playwright 拦截模式）。

    与 BaseScraper 不同，本爬虫不继承抽象基类，而是直接使用 Playwright
    异步 API 拦截浏览器网络请求。对外暴露同步的 crawl() 方法。

    Example
    -------
    ```python
    scraper = TencentScraper(max_pages=5, page_size=10)
    raw_jobs = scraper.crawl()
    print(f"共抓取 {len(raw_jobs)} 条原始岗位数据")
    ```
    """

    def __init__(
        self,
        page_size: int = _DEFAULT_PAGE_SIZE,
        max_pages: int = _DEFAULT_MAX_PAGES,
    ) -> None:
        self.page_size = page_size
        self.max_pages = max_pages
        self._logger = logging.getLogger(f"{__name__}.tencent")
        self._total_count: int = 0

    # ------------------------------------------------------------------
    # 核心接口
    # ------------------------------------------------------------------

    def crawl(self) -> list[dict]:
        """
        执行完整爬取流程：启动浏览器 → 拦截 API → 分页抓取 → 合并结果。

        Returns
        -------
        list[dict]
            原始岗位字典列表，每条注入 source="tencent"
        """
        return asyncio.run(self._crawl_async())

    async def _crawl_async(self) -> list[dict]:
        from playwright.async_api import async_playwright

        print("[Tencent] ========== 腾讯招聘爬虫启动 (Playwright) ==========")
        print(f"[Tencent] page_size={self.page_size} | max_pages={self.max_pages}")

        all_jobs: list[dict] = []
        seen_ids: set[str] = set()

        async def handle_response(response):
            if _API_RESPONSE_PATTERN in response.url and response.status == 200:
                try:
                    ct = response.headers.get("content-type", "")
                    if "json" in ct:
                        data = await response.json()
                        posts = data.get("Data", {}).get("Posts", [])
                        pg_idx = data.get("Data", {}).get("PageIndex", 0)
                        total = data.get("Data", {}).get("Count", 0)
                        for job in posts:
                            pid = str(job.get("PostId", ""))
                            if pid and pid not in seen_ids:
                                seen_ids.add(pid)
                                job["source"] = "tencent"
                                all_jobs.append(job)
                        print(f"[Tencent Debug] Page {pg_idx}: {len(posts)} posts, total_pool={total}, collected={len(all_jobs)}")
                except Exception:
                    pass

        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            page = await browser.new_page()
            page.on("response", handle_response)

            try:
                # 打开首页
                await page.goto(
                    "https://careers.tencent.com/search.html",
                    wait_until="domcontentloaded",
                    timeout=30000,
                )
                await page.wait_for_timeout(3000)

                # 逐页点击"下一页"按钮触发 SPA 分页
                for attempt in range(self.max_pages - 1):
                    clicked = False
                    for selector in [
                        ".btn-next:not([disabled])",
                        "button.btn-next:not([disabled])",
                        ".ant-pagination-next:not(.ant-pagination-disabled)",
                        "li.ant-pagination-next:not(.ant-pagination-disabled)",
                        ".t-pagination__btn-next:not(.t-is-disabled)",
                        "button:has-text('>')",
                    ]:
                        try:
                            btn = page.locator(selector).first
                            if await btn.count() > 0 and await btn.is_visible():
                                is_disabled = await btn.get_attribute("disabled")
                                if is_disabled is None or is_disabled == "false":
                                    await btn.click()
                                    await page.wait_for_timeout(2000)
                                    clicked = True
                                    break
                        except Exception:
                            continue

                    if not clicked:
                        print(f"[Tencent] Page {attempt+2}: no clickable next button, stopping")
                        break

            except Exception as e:
                print(f"[Tencent] Error: {e}")
            finally:
                await browser.close()

        self._total_count = len(seen_ids)
        print(f"[Tencent] ========== 爬虫结束，去重后共 {len(all_jobs)} 条 ==========")
        return all_jobs

    @property
    def total_count(self) -> int:
        return self._total_count
