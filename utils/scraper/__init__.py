"""
utils/scraper - 大厂招聘爬虫包

模块列表
--------
- base_scraper        : 抽象基类（Brotli 解压 + 浏览器指纹伪装 + tenacity 重试）
- netease_scraper     : 网易招聘（POST RESTful JSON，无认证要求）
- tencent_scraper     : 腾讯招聘（Playwright 拦截 API 响应模式）
- bytedance_scraper   : 字节跳动招聘（Playwright 拦截 API 响应模式）
- didi_scraper        : 滴滴出行招聘（GET RESTful JSON，无认证要求）
- mihoyo_scraper      : 米哈游招聘（已废弃）

使用示例
--------
```python
from utils.scraper.netease_scraper import NetEaseScraper
from utils.scraper.tencent_scraper import TencentScraper
from utils.scraper.bytedance_scraper import ByteDanceScraper
from utils.scraper.didi_scraper import DidiScraper

with NetEaseScraper(max_pages=5) as s: jobs = s.crawl()
td = TencentScraper(max_pages=5); jobs = td.crawl()
bd = ByteDanceScraper(max_pages=5); jobs = bd.crawl()
dd = DidiScraper(max_pages=5); jobs = dd.crawl()
```
"""
