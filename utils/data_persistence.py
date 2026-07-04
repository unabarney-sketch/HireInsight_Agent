# -*- coding: utf-8 -*-
"""
utils/data_persistence.py
==========================
SQLite 持久化层 —— 建表、Upsert 写入、条件查询。

核心功能
--------
    1. init_sqlite_db()        —— 建表 + 联合唯一索引
    2. save_to_sqlite()        —— 原生 UPSERT 批量入库（ACID 事务）
    3. load_from_sqlite()      —— 全量/条件读取
    4. query_jobs_by_filters() —— 按城市/学历/经验精确筛选

数据契约
--------
    表名：job_positions（30 字段，严格对齐 v1.0 数据结构设计文档 §2.1.1）
    UNIQUE 索引：(original_id, source) —— 硬件级去重

UPSERT 语义
------------
    使用 INSERT OR REPLACE 语法：
    - original_id + source 不存在 → INSERT 新记录
    - original_id + source 已存在 → REPLACE 整行（保留旧数据 + 更新新字段）

事务保障
--------
    所有写入操作包裹在 db.commit() / db.rollback() 中，
    确保 ACID 原子性：全部成功提交，任一失败全量回滚。

依赖
----
    sqlite3（内置）, pandas >= 1.5.0

Author: HireInsight-Agent
"""

from __future__ import annotations

import logging
import os
import sqlite3
from datetime import datetime
from typing import Any

import pandas as pd

# ---------------------------------------------------------------------------
# 日志
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------

_DEFAULT_DB_PATH: str = "data/hireinsight.db"
_DEFAULT_TABLE_NAME: str = "job_positions"

# ---------------------------------------------------------------------------
# DDL 模板（严格对齐设计文档 §2.1.1 字段表，共 30 列）
# ---------------------------------------------------------------------------

_CREATE_TABLE_SQL: str = """
CREATE TABLE IF NOT EXISTS job_positions (
    id              TEXT PRIMARY KEY NOT NULL,
    source          TEXT NOT NULL,
    original_id     TEXT NOT NULL,
    title           TEXT NOT NULL,
    title_raw       TEXT,
    company         TEXT NOT NULL,
    department      TEXT,
    category        TEXT NOT NULL,
    sub_category    TEXT,
    city            TEXT NOT NULL,
    city_raw        TEXT,
    district        TEXT,
    salary_raw      TEXT,
    salary_min      REAL,
    salary_max      REAL,
    salary_bonus    TEXT,
    experience_min  INTEGER,
    experience_max  INTEGER,
    experience_raw  TEXT,
    degree          TEXT,
    degree_raw      TEXT,
    work_type       TEXT,
    duty            TEXT,
    requirement     TEXT,
    skills          TEXT,
    post_url        TEXT NOT NULL,
    published_at    TEXT,
    updated_at      TEXT,
    created_at      TEXT NOT NULL,
    crawl_batch     TEXT NOT NULL
)
"""

# 联合唯一索引（original_id + source 硬件级去重）
_CREATE_INDEX_SQL: str = """
CREATE UNIQUE INDEX IF NOT EXISTS idx_original_source
ON job_positions (original_id, source)
"""

# ---------------------------------------------------------------------------
# UPSERT 时使用的 INSERT 列清单
# ---------------------------------------------------------------------------

_INSERT_COLUMNS: list[str] = [
    "id", "source", "original_id",
    "title", "title_raw", "company",
    "department", "category", "sub_category",
    "city", "city_raw", "district",
    "salary_raw", "salary_min", "salary_max", "salary_bonus",
    "experience_min", "experience_max", "experience_raw",
    "degree", "degree_raw",
    "work_type",
    "duty", "requirement",
    "skills",
    "post_url",
    "published_at", "updated_at", "created_at",
    "crawl_batch",
]

_COLUMNS_CSV: str = ", ".join(_INSERT_COLUMNS)
_PLACEHOLDERS: str = ", ".join(["?" for _ in _INSERT_COLUMNS])

# ON CONFLICT 更新子句（除 id / source / original_id 外全部更新）
_UPDATE_SET_CLAUSES: list[str] = []
for _col in _INSERT_COLUMNS:
    if _col in ("id", "source", "original_id"):
        continue
    _UPDATE_SET_CLAUSES.append(f'"{_col}" = excluded."{_col}"')
_UPDATE_SET_SQL = ",\n    ".join(_UPDATE_SET_CLAUSES)

_INSERT_OR_REPLACE_SQL: str = f"""
INSERT OR REPLACE INTO job_positions ({_COLUMNS_CSV})
VALUES ({_PLACEHOLDERS})
"""

_INSERT_ON_CONFLICT_SQL: str = f"""
INSERT INTO job_positions ({_COLUMNS_CSV})
VALUES ({_PLACEHOLDERS})
ON CONFLICT (original_id, source)
DO UPDATE SET
    {_UPDATE_SET_SQL}
"""


# ---------------------------------------------------------------------------
# 1. 建表与索引初始化
# ---------------------------------------------------------------------------

def init_sqlite_db(db_path: str = _DEFAULT_DB_PATH) -> str:
    """
    初始化 SQLite 数据库：自动创建目录、建表、建索引。

    Parameters
    ----------
    db_path : str
        数据库文件路径，默认 "data/hireinsight.db"

    Returns
    -------
    str
        数据库文件的绝对路径（确认创建成功）

    幂等性
    ------
        使用 CREATE TABLE IF NOT EXISTS，重复调用安全。
    """
    # 自动创建 data/ 目录
    db_dir = os.path.dirname(db_path)
    if db_dir and not os.path.exists(db_dir):
        os.makedirs(db_dir, exist_ok=True)
        logger.info("自动创建目录: %s", db_dir)

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    try:
        # 建表
        cursor.execute(_CREATE_TABLE_SQL)
        logger.debug("表 job_positions 已就绪（幂等）")

        # 建联合唯一索引
        cursor.execute(_CREATE_INDEX_SQL)
        logger.debug("索引 idx_original_source 已就绪（幂等）")

        conn.commit()
        logger.info("数据库初始化完成: %s", os.path.abspath(db_path))
    except Exception as e:
        conn.rollback()
        logger.error("数据库初始化失败: %s", e)
        raise
    finally:
        conn.close()

    return os.path.abspath(db_path)


# ---------------------------------------------------------------------------
# 2. 批量 UPSERT 写入（ACID 事务）
# ---------------------------------------------------------------------------

def save_to_sqlite(
    df: pd.DataFrame,
    db_path: str = _DEFAULT_DB_PATH,
    table_name: str = _DEFAULT_TABLE_NAME,
    batch_size: int = 500,
) -> int:
    """
    将清洗后的 DataFrame 批量 Upsert 写入 SQLite。

    写入策略
    --------
        - 使用原生 SQLite3 ON CONFLICT(original_id, source) DO UPDATE SET
        - cursor.executemany() 批量执行，批大小为 batch_size
        - 自动补齐缺失字段（如 created_at）
        - db.commit() / db.rollback() 保证 ACID 事务原子性

    Parameters
    ----------
    df : pd.DataFrame
        清洗后的岗位数据 DataFrame（须至少包含 original_id, source 列）
    db_path : str
        数据库文件路径
    table_name : str
        目标表名，默认 "job_positions"
    batch_size : int
        单次 execute 的批次大小，默认 500

    Returns
    -------
    int
        实际写入成功的记录数
    """
    if df.empty:
        logger.warning("DataFrame 为空，跳过写入。")
        return 0

    # ---- 确保 table 已存在 ----
    if not os.path.exists(db_path):
        init_sqlite_db(db_path)

    df = df.copy()
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # ---- 补齐缺失字段（数据库 NOT NULL 约束兜底） ----
    _fill_defaults(df, now)

    # ---- 准备数据：只保留 INSERT 列清单中的字段 ----
    available_cols = [c for c in _INSERT_COLUMNS if c in df.columns]
    df_subset = df[available_cols].copy()

    # 补充缺失列（以 None 填充）
    for col in _INSERT_COLUMNS:
        if col not in df_subset.columns:
            df_subset[col] = None

    # 转换为原生 Python 类型列表（executemany 专用）
    records: list[tuple] = []
    for _, row in df_subset.iterrows():
        row_values = []
        for col in _INSERT_COLUMNS:
            val = row[col]
            # 将 numpy/pandas 特殊类型转为 Python 原生类型
            if pd.isna(val):
                row_values.append(None)
            elif hasattr(val, "item"):  # numpy scalar
                row_values.append(val.item())
            else:
                row_values.append(val)
        records.append(tuple(row_values))

    # ---- 事务性批量写入 ----
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    total_inserted = 0

    try:
        for i in range(0, len(records), batch_size):
            batch = records[i : i + batch_size]
            cursor.executemany(_INSERT_ON_CONFLICT_SQL, batch)
            total_inserted += len(batch)
            logger.debug(
                "批次写入: %d/%d 条", total_inserted, len(records)
            )

        conn.commit()
        logger.info(
            "save_to_sqlite 完成: %d 条记录已写入 %s",
            total_inserted, db_path,
        )
    except Exception as e:
        conn.rollback()
        logger.error(
            "save_to_sqlite 写入失败（已回滚），批次=%d/%d: %s",
            total_inserted, len(records), e,
        )
        raise
    finally:
        conn.close()

    return total_inserted


def _fill_defaults(df: pd.DataFrame, now: str) -> None:
    """
    补齐 DataFrame 中的必填默认值。

    Parameters
    ----------
    df : pd.DataFrame
        输入 DataFrame（会被原地修改）
    now : str
        当前时间字符串，格式 YYYY-MM-DD HH:MM:SS
    """
    if "created_at" not in df.columns or df["created_at"].isna().all():
        df["created_at"] = now

    if "crawl_batch" not in df.columns or df["crawl_batch"].isna().all():
        df["crawl_batch"] = datetime.now().strftime("%Y%m%d_%H%M%S")

    # salary_min / salary_max 需要确保是 float
    for col in ("salary_min", "salary_max"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # experience_min / experience_max 需要确保是 int
    for col in ("experience_min", "experience_max"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")


# ---------------------------------------------------------------------------
# 3. 全量读取
# ---------------------------------------------------------------------------

def load_from_sqlite(
    db_path: str = _DEFAULT_DB_PATH,
    table_name: str = _DEFAULT_TABLE_NAME,
) -> pd.DataFrame:
    """
    从 SQLite 读取全部岗位数据。

    Parameters
    ----------
    db_path : str
        数据库文件路径
    table_name : str
        表名

    Returns
    -------
    pd.DataFrame
        全部岗位数据（空库时返回空 DataFrame）

    Raises
    ------
    FileNotFoundError
        数据库文件不存在
    """
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"数据库文件不存在: {db_path}")

    conn = sqlite3.connect(db_path)
    try:
        df = pd.read_sql_query(
            f"SELECT * FROM {table_name} ORDER BY created_at DESC",
            conn,
        )
        logger.info("load_from_sqlite: 读取 %d 条记录", len(df))
        return df
    except Exception as e:
        logger.error("读取数据库失败: %s", e)
        raise
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# 4. 条件筛选查询
# ---------------------------------------------------------------------------

# 合法筛选键白名单（防 SQL 注入）
_ALLOWED_FILTERS: dict[str, str] = {
    "city":      "city",
    "degree":    "degree",
    "company":   "company",
    "category":  "category",
    "work_type": "work_type",
    "source":    "source",
    "experience_min": "experience_min",
    "experience_max": "experience_max",
    "salary_min":    "salary_min",
    "salary_max":    "salary_max",
    "keyword":   "title",  # 关键词模糊匹配标题
}


def query_jobs_by_filters(
    db_path: str = _DEFAULT_DB_PATH,
    filters: dict[str, Any] | None = None,
    table_name: str = _DEFAULT_TABLE_NAME,
    limit: int | None = 500,
) -> pd.DataFrame:
    """
    按组合条件筛选岗位数据。

    支持的筛选键
    ------------
        city            —— 精确匹配（如 "北京"）
        degree          —— 精确匹配（如 "本科"）
        company         —— 精确匹配（如 "字节跳动"）
        category        —— 精确匹配（如 "技术"）
        work_type       —— 精确匹配（如 "全职"）
        source          —— 精确匹配（如 "netease"）
        experience_min  —— 经验年限下限（>=）
        experience_max  —— 经验年限上限（<=）
        salary_min      —— 薪资下限（>=）
        salary_max      —— 薪资上限（<=）
        keyword         —— 职位标题模糊搜索（LIKE %keyword%）

    Parameters
    ----------
    db_path : str
        数据库文件路径
    filters : dict, optional
        筛选条件字典，如 {"city": "深圳", "degree": "本科"}
    table_name : str
        表名
    limit : int, optional
        返回条数上限，默认 500，传 None 则不限

    Returns
    -------
    pd.DataFrame
        过滤后的岗位数据

    Raises
    ------
    ValueError
        使用了不支持的筛选键（防注入）
    FileNotFoundError
        数据库文件不存在
    Examples
    --------
    >>> df = query_jobs_by_filters(
    ...     filters={"city": "深圳", "degree": "本科", "keyword": "Python"},
    ...     limit=20
    ... )
    """
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"数据库文件不存在: {db_path}")

    conn = sqlite3.connect(db_path)
    conditions: list[str] = []
    params: list[Any] = []

    if filters:
        for key, value in filters.items():
            if key not in _ALLOWED_FILTERS:
                raise ValueError(
                    f"不支持的筛选键: '{key}'。"
                    f"允许的键: {list(_ALLOWED_FILTERS.keys())}"
                )
            col = _ALLOWED_FILTERS[key]

            if key == "keyword":
                # 模糊搜索
                conditions.append(f'"{col}" LIKE ?')
                params.append(f"%{value}%")
            elif key in ("experience_min", "salary_min"):
                # 下限条件
                conditions.append(f'"{col}" >= ?')
                params.append(value)
            elif key in ("experience_max", "salary_max"):
                # 上限条件
                conditions.append(f'"{col}" <= ?')
                params.append(value)
            else:
                # 精确匹配
                conditions.append(f'"{col}" = ?')
                params.append(value)

    where_clause = ""
    if conditions:
        where_clause = "WHERE " + " AND ".join(conditions)

    limit_clause = ""
    if limit is not None and limit > 0:
        limit_clause = f"LIMIT {int(limit)}"

    query_sql = (
        f"SELECT * FROM {table_name} "
        f"{where_clause} "
        f"ORDER BY created_at DESC "
        f"{limit_clause}"
    )

    logger.debug("查询 SQL: %s | params: %s", query_sql, params)

    try:
        df = pd.read_sql_query(query_sql, conn, params=params)
        logger.info("query_jobs_by_filters: 返回 %d 条", len(df))
        return df
    except Exception as e:
        logger.error("查询失败: %s", e)
        raise
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# 5. 便捷统计查询
# ---------------------------------------------------------------------------

def get_distinct_values(
    db_path: str = _DEFAULT_DB_PATH,
    column: str = "city",
    table_name: str = _DEFAULT_TABLE_NAME,
) -> list[str]:
    """
    获取某列的去重值列表（用于前端下拉框填充）。

    Parameters
    ----------
    db_path : str
        数据库文件路径
    column : str
        列名（city / degree / company / category / work_type / source）
    table_name : str
        表名

    Returns
    -------
    list[str]
        去重值列表（按字母排序）

    Raises
    ------
    ValueError
        column 不在允许列表中
    """
    allowed = {"city", "degree", "company", "category", "work_type", "source"}
    if column not in allowed:
        raise ValueError(
            f"不支持的列: '{column}'。允许: {sorted(allowed)}"
        )

    if not os.path.exists(db_path):
        return []

    conn = sqlite3.connect(db_path)
    try:
        cursor = conn.cursor()
        cursor.execute(
            f'SELECT DISTINCT "{column}" FROM {table_name} '
            f'WHERE "{column}" IS NOT NULL AND "{column}" != "" '
            f'ORDER BY "{column}"'
        )
        return [row[0] for row in cursor.fetchall()]
    except Exception as e:
        logger.error("get_distinct_values 失败: %s", e)
        return []
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# 本地测试
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import tempfile

    # 使用临时库测试
    test_db = os.path.join(tempfile.gettempdir(), "hireinsight_test.db")

    print("=" * 60)
    print("1. 初始化数据库")
    print("=" * 60)
    db_path = init_sqlite_db(test_db)
    print(f"   数据库路径: {db_path}")

    print("\n" + "=" * 60)
    print("2. 模拟写入测试数据")
    print("=" * 60)
    test_df = pd.DataFrame([
        {
            "id": "netease_001", "source": "netease", "original_id": "001",
            "title": "Python后端", "title_raw": "Python后端开发工程师",
            "company": "网易", "department": "研发部",
            "category": "技术", "sub_category": "后端",
            "city": "深圳", "city_raw": "深圳", "district": "南山区",
            "salary_raw": "25k-45k·15薪", "salary_min": 25.0, "salary_max": 45.0,
            "salary_bonus": "15薪",
            "experience_min": 3, "experience_max": 5, "experience_raw": "3-5年",
            "degree": "本科", "degree_raw": "本科及以上",
            "work_type": "全职",
            "duty": "负责网易推荐系统开发", "requirement": "熟悉Python/Golang",
            "skills": "Python, Go, MySQL",
            "post_url": "https://hr.163.com/job-detail.html?id=001",
            "published_at": "2025-06-01", "updated_at": "2025-06-15",
            "created_at": "2025-07-02 21:00:00", "crawl_batch": "20250702_210000",
        },
        {
            "id": "mihoyo_002", "source": "mihoyo", "original_id": "002",
            "title": "Python后台", "title_raw": "Python后台开发",
            "company": "米哈游", "department": "游戏研发部",
            "category": "技术", "sub_category": "",
            "city": "深圳", "city_raw": "深圳市", "district": "",
            "salary_raw": "200-300元/天", "salary_min": 4.4, "salary_max": 6.6,
            "salary_bonus": "",
            "experience_min": 0, "experience_max": 0, "experience_raw": "在校生/应届生",
            "degree": "本科", "degree_raw": "本科",
            "work_type": "实习",
            "duty": "负责米哈游游戏后台开发", "requirement": "熟悉Python",
            "skills": "Python, C++",
            "post_url": "https://jobs.mihoyo.com/job-detail/002",
            "published_at": "", "updated_at": "2025-06-15 10:30:00",
            "created_at": "2025-07-02 21:00:00", "crawl_batch": "20250702_210000",
        },
        # 意图重复记录：测试 Upsert 更新逻辑
        {
            "id": "netease_001_v2", "source": "netease", "original_id": "001",
            "title": "Python后端(更新)", "title_raw": "Python后端开发工程师(更新)",
            "company": "网易", "department": "研发部",
            "category": "技术", "sub_category": "后端",
            "city": "深圳", "city_raw": "深圳", "district": "南山区",
            "salary_raw": "30k-50k·15薪", "salary_min": 30.0, "salary_max": 50.0,
            "salary_bonus": "15薪",
            "experience_min": 3, "experience_max": 5, "experience_raw": "3-5年",
            "degree": "本科", "degree_raw": "本科及以上",
            "work_type": "全职",
            "duty": "负责网易推荐系统开发", "requirement": "熟悉Python/Golang",
            "skills": "Python, Go, MySQL",
            "post_url": "https://hr.163.com/job-detail.html?id=001",
            "published_at": "2025-07-01", "updated_at": "2025-07-02",
            "created_at": "2025-07-03 09:00:00", "crawl_batch": "20250703_090000",
        },
    ])
    n = save_to_sqlite(test_df, test_db)
    print(f"   写入: {n} 条")

    print("\n" + "=" * 60)
    print("3. 全量读取")
    print("=" * 60)
    all_data = load_from_sqlite(test_db)
    print(f"   总条数: {len(all_data)}")
    # bytedance_001 应已被 Upsert 为 salary_min=30.0
    bytedance_row = all_data[all_data["source"] == "bytedance"].iloc[0]
    print(f"   bytedance_001 salary_min = {bytedance_row['salary_min']} "
          f"(预期 30.0, Upsert {'成功' if bytedance_row['salary_min'] == 30.0 else '失败'})")

    print("\n" + "=" * 60)
    print("4. 条件筛选")
    print("=" * 60)
    filtered = query_jobs_by_filters(
        test_db,
        filters={"city": "深圳", "degree": "本科"},
        limit=10,
    )
    print(f"   筛选结果: {len(filtered)} 条")

    keyword_df = query_jobs_by_filters(
        test_db,
        filters={"keyword": "Python"},
    )
    print(f"   关键词 Python: {len(keyword_df)} 条")

    print("\n" + "=" * 60)
    print("5. 去重值查询")
    print("=" * 60)
    cities = get_distinct_values(test_db, "city")
    companies = get_distinct_values(test_db, "company")
    print(f"   城市: {cities}")
    print(f"   公司: {companies}")

    # 清理测试库
    os.remove(test_db)
    print("\n测试完成，临时库已清理。")
