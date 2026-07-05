# -*- coding: utf-8 -*-
"""
utils/scraper/didi_scraper.py
==============================
滴滴出行招聘爬虫（直接 HTTP REST API）。

数据契约
--------
    返回值：list[dict]
    每个 dict 为滴滴招聘 API 返回的原始岗位结构（data.items[] 内单个对象）。

API 端点
--------
    GET https://talent.didiglobal.com/recruit-portal-service/api/job/front/list
    Params: page=1, rows=16, recruitType=1

接口特点
--------
    1. 公开 RESTful JSON API，无需认证/Cookie
    2. 单次返回 16 条（rows 参数控制）
    3. 返回字段：jobName, workArea, deptName, jobType, createTime 等
    4. 总岗位池 1000+ 条

参数说明
--------
    - page       : int  页码（从 1 开始）
    - rows       : int  每页条目数（默认 16）
    - recruitType: int  招聘类型（1=社招）

依赖
----
    - requests

Author: HireInsight-Agent
"""

from __future__ import annotations

import logging
import time
from typing import Any

import requests

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------
_API_URL: str = (
    "https://talent.didiglobal.com/recruit-portal-service/api/job/front/list"
)
_DEFAULT_PAGE_SIZE: int = 16
_DEFAULT_MAX_PAGES: int = 10

# ---------------------------------------------------------------------------
# 爬虫实现
# ---------------------------------------------------------------------------

class DidiScraper:
    """滴滴出行招聘爬虫（直接 HTTP GET）。"""

    def __init__(
        self,
        page_size: int = _DEFAULT_PAGE_SIZE,
        max_pages: int = _DEFAULT_MAX_PAGES,
        keyword: str | None = None,
    ) -> None:
        self.page_size = page_size
        self.max_pages = max_pages
        self.keyword = keyword
        self._logger = logging.getLogger(f"{__name__}.didi")

    def crawl(self) -> list[dict]:
        """
        执行分页爬取。

        Returns
        -------
        list[dict]
            原始岗位字典列表，每条注入 source="didi"
        """
        print("[Didi] ========== 滴滴招聘爬虫启动 ==========")
        kw_info = f" | keyword={self.keyword}" if self.keyword else " (全量)"
        print(f"[Didi] page_size={self.page_size} | max_pages={self.max_pages}{kw_info}")

        all_jobs: list[dict] = []
        seen: set[str] = set()

        session = requests.Session()
        session.headers.update({
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
            "Accept": "application/json, text/plain, */*",
            "Accept-Language": "zh-CN,zh;q=0.9",
            "Referer": "https://talent.didiglobal.com/",
        })

        for page in range(1, self.max_pages + 1):
            params: dict[str, Any] = {
                "page": page,
                "rows": self.page_size,
                "recruitType": 1,
            }
            if self.keyword:
                params["keyword"] = self.keyword

            try:
                r = session.get(_API_URL, params=params, timeout=15)
                if r.status_code != 200:
                    print(f"[Didi] Page {page}: HTTP {r.status_code}")
                    break

                data = r.json()
                items = data.get("data", {}).get("items", [])
                total = data.get("data", {}).get("total", 0)

                for job in items:
                    jid = str(job.get("id") or job.get("jdId", ""))
                    if jid and jid not in seen:
                        seen.add(jid)
                        job["source"] = "didi"
                        all_jobs.append(job)

                print(f"[Didi] Page {page}: {len(items)} items, total_pool={total}, collected={len(all_jobs)}")

                if page == 1:
                    print(f"[Didi] 接口返回总岗位数: {total}")

                if len(items) == 0:
                    break

                time.sleep(0.3)

            except Exception as e:
                print(f"[Didi] Page {page} ERROR: {e}")
                break

        print(f"[Didi] ========== 爬虫结束，去重后共 {len(all_jobs)} 条 ==========")
        return all_jobs
