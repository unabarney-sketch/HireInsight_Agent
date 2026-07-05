"""
utils/scraper - 大厂招聘爬虫包

模块列表
--------
- base_scraper      : 抽象基类（Brotli 解压 + 浏览器指纹伪装 + tenacity 重试）
- netease_scraper   : 网易招聘（POST RESTful JSON，无认证要求）
- tencent_scraper   : 腾讯招聘（Playwright 拦截 API 响应模式）
- mihoyo_scraper    : 米哈游招聘（已废弃，保留接口兼容）

使用示例
--------
```python
from utils.scraper.netease_scraper import NetEaseScraper
from utils.scraper.tencent_scraper import TencentScraper

with NetEaseScraper(max_pages=5) as scraper:
    jobs = scraper.crawl()

ts = TencentScraper(max_pages=5)
jobs = ts.crawl()  # 腾讯使用 Playwright，非上下文管理器模式
```
"""
