"""
utils/scraper - 大厂招聘爬虫包

模块列表
--------
- base_scraper      : 抽象基类（Brotli 解压 + 浏览器指纹伪装 + tenacity 重试）
- netease_scraper   : 网易招聘（单次 RESTful JSON 请求，无需 CSRF/详情二级拉取）
- tencent_scraper   : 腾讯招聘（列表→详情二级拉取 + 指纹定期轮换）

使用示例
--------
```python
from utils.scraper.bytedance_scraper import ByteDanceScraper
from utils.scraper.tencent_scraper import TencentScraper

with ByteDanceScraper(max_pages=5) as scraper:
    jobs = scraper.crawl()
```
"""
