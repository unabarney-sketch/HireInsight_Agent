"""
utils/scraper - 大厂招聘爬虫包

模块列表
--------
- base_scraper      : 抽象基类（Brotli 解压 + 浏览器指纹伪装 + tenacity 重试）
- netease_scraper   : 网易招聘（单次 RESTful JSON 请求，无需 CSRF/详情二级拉取）
- mihoyo_scraper    : 米哈游招聘（接口公开透明，无反爬风控）

使用示例
--------
```python
from utils.scraper.netease_scraper import NetEaseScraper
from utils.scraper.mihoyo_scraper import MihoyoScraper

with NetEaseScraper(max_pages=5) as scraper:
    jobs = scraper.crawl()

with MihoyoScraper(max_pages=5) as scraper:
    jobs = scraper.crawl()
```
"""
