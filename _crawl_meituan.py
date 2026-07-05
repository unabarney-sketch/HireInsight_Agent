# -*- coding: utf-8 -*-
import sys, os, sqlite3
sys.path.insert(0, os.path.dirname(__file__))
DB = "data/hireinsight.db"

from utils.scraper.meituan_scraper import MeituanScraper
from utils.data_transformer import transform_jobs
from utils.data_cleaner import clean_job_data
from utils.data_persistence import init_sqlite_db, save_to_sqlite
import pandas as pd

print("Crawling Meituan...")
mt = MeituanScraper(max_pages=8, page_size=10)
raw = mt.crawl()
print(f"Raw: {len(raw)}")

mt_jobs = [j for j in raw if j.get("source") == "meituan"]
t = transform_jobs(mt_jobs, source="meituan")
df = pd.DataFrame(t)
clean = clean_job_data(df)
init_sqlite_db(DB)
n = save_to_sqlite(clean, DB)
print(f"Inserted: {n}")

conn = sqlite3.connect(DB)
cur = conn.cursor()
cur.execute("SELECT source, COUNT(*) FROM job_positions GROUP BY source ORDER BY COUNT(*) DESC")
for src, cnt in cur.fetchall():
    print(f"  {src}: {cnt}")
conn.close()
