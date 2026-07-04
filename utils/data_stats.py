# -*- coding: utf-8 -*-
"""
utils/data_stats.py
====================
数据统计层 —— 四大看板维度聚合 + AI Agent 文本生成器。

四大看板维度
-----------
    1. 薪资水平   —— 全盘 min/max/median/avg 区间
    2. 地区分布   —— Top10 城市 + 城市等级占比
    3. 学历水平   —— 标准化枚举数量与百分比
    4. 工作经验   —— 经验区间分布

双源输入支持
-----------
    - 直接传入 pd.DataFrame（内存模式）
    - 传入 db_path 从 SQLite 自动读取最新全量数据

AI Agent 输出
-------------
    generate_agent_prompt_summary() → Markdown 摘要文本
    此文本可直接喂给 DeepSeek Market_Agent，作为市场分析报告依据。

编程契约
--------
    - 零显式循环：纯 Pandas 向量化操作
    - 空值容错：所有统计均自动过滤 NaN/None
    - 类型稳定：输出 dict 值均为 Python 原生类型

依赖
----
    pandas >= 1.5.0

Author: HireInsight-Agent
"""

from __future__ import annotations

import logging
import os
from typing import Any, Optional

import pandas as pd

# ---------------------------------------------------------------------------
# 日志
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------

# 默认库路径
_DEFAULT_DB_PATH: str = "data/hireinsight.db"

# 城市等级中文标签
_CITY_TIER_LABELS: dict[int, str] = {
    0: "远程/全国",
    1: "一线城市",
    2: "新一线/二线城市",
    3: "其他城市",
}

# 城市等级映射表（同步自 data_cleaner.py）
_CITY_TIER_MAP: dict[str, int] = {
    "北京": 1, "上海": 1, "深圳": 1, "广州": 1,
    "杭州": 2, "成都": 2, "南京": 2, "武汉": 2,
    "西安": 2, "长沙": 2, "济南": 2, "苏州": 2,
    "重庆": 2, "天津": 2, "厦门": 2, "郑州": 2,
    "合肥": 2, "福州": 2, "东莞": 2, "佛山": 2,
    "远程": 0, "全国": 0,
}

# 经验区间分组（label 与条件表达式）
_EXPERIENCE_BINS: list[dict[str, Any]] = [
    {"label": "应届/实习", "min": -1, "max": 0, "desc": "0 年"},
    {"label": "1 年以下", "min": 0, "max": 1, "desc": "< 1 年"},
    {"label": "1-3 年",   "min": 1, "max": 3, "desc": "1 ~ 3 年"},
    {"label": "3-5 年",   "min": 3, "max": 5, "desc": "3 ~ 5 年"},
    {"label": "5-10 年",  "min": 5, "max": 10, "desc": "5 ~ 10 年"},
    {"label": "10 年以上","min": 10, "max": 99, "desc": ">= 10 年"},
    {"label": "经验不限",  "min": -99, "max": -1, "desc": "经验不限"},
]


# ---------------------------------------------------------------------------
# 辅助：从 DB 加载 DataFrame
# ---------------------------------------------------------------------------

def _load_df_from_db(db_path: str) -> pd.DataFrame:
    """从 SQLite 加载全量岗位数据（延迟导入避免循环依赖）。"""
    from utils.data_persistence import load_from_sqlite

    if not os.path.exists(db_path):
        raise FileNotFoundError(f"数据库文件不存在: {db_path}")
    return load_from_sqlite(db_path)


# ---------------------------------------------------------------------------
# 辅助：city → tier 向量化映射
# ---------------------------------------------------------------------------

def _assign_city_tier(df: pd.DataFrame) -> pd.DataFrame:
    """对 DataFrame 追加 city_tier 列（向量化映射）。"""
    df = df.copy()
    df["city_tier"] = df["city"].map(_CITY_TIER_MAP).fillna(3).astype(int)
    return df


# ---------------------------------------------------------------------------
# 1. 薪资维度统计
# ---------------------------------------------------------------------------

def _calc_salary_stats(df: pd.DataFrame) -> dict[str, Any]:
    """计算薪资维度指标。"""
    # 过滤掉面议（salary_min/salary_max 都为 NaN 的记录）
    valid = df.dropna(subset=["salary_min", "salary_max"])

    if valid.empty:
        return {"status": "无有效薪资数据（可能全部为面议）", "count": 0}

    result: dict[str, Any] = {"count": int(len(valid))}

    result["salary_min_overall"] = round(float(valid["salary_min"].min()), 1)
    result["salary_max_overall"] = round(float(valid["salary_max"].max()), 1)
    result["salary_min_median"]  = round(float(valid["salary_min"].median()), 1)
    result["salary_max_median"]  = round(float(valid["salary_max"].median()), 1)
    result["salary_min_mean"]    = round(float(valid["salary_min"].mean()), 1)
    result["salary_max_mean"]    = round(float(valid["salary_max"].mean()), 1)

    # 平均值区间（最常见的薪资水平）
    result["avg_interval"] = f"{result['salary_min_mean']}k - {result['salary_max_mean']}k"

    # describe 统计摘要（供 LOB / RAG 使用）
    desc = valid[["salary_min", "salary_max"]].describe()
    result["percentiles"] = {
        "p25_min": round(float(desc["salary_min"]["25%"]), 1),
        "p50_min": round(float(desc["salary_min"]["50%"]), 1),
        "p75_min": round(float(desc["salary_min"]["75%"]), 1),
        "p25_max": round(float(desc["salary_max"]["25%"]), 1),
        "p50_max": round(float(desc["salary_max"]["50%"]), 1),
        "p75_max": round(float(desc["salary_max"]["75%"]), 1),
    }

    # 薪资段分布
    bins = [0, 5, 10, 20, 30, 50, 100]
    labels = ["< 5k", "5-10k", "10-20k", "20-30k", "30-50k", "> 50k"]
    valid = valid.copy()
    valid["salary_bucket"] = pd.cut(
        valid["salary_max"], bins=bins, labels=labels, right=False
    )
    bucket_counts = valid["salary_bucket"].value_counts().sort_index()
    result["salary_distribution"] = {
        str(k): int(v) for k, v in bucket_counts.items()
    }

    return result


# ---------------------------------------------------------------------------
# 2. 地区维度统计
# ---------------------------------------------------------------------------

def _calc_city_stats(df: pd.DataFrame) -> dict[str, Any]:
    """计算地区分布维度指标。"""
    city_counts = df["city"].value_counts()
    result: dict[str, Any] = {
        "unique_cities": int(city_counts.count()),
        "top10": {},
    }

    for city, cnt in city_counts.head(10).items():
        result["top10"][str(city)] = int(cnt)

    # 城市等级占比
    df_tier = _assign_city_tier(df)
    tier_counts = df_tier["city_tier"].value_counts().sort_index()
    total = int(len(df_tier))

    result["tier_distribution"] = {}
    for tier_id, cnt in tier_counts.items():
        label = _CITY_TIER_LABELS.get(int(tier_id), f"Tier {tier_id}")
        result["tier_distribution"][label] = {
            "count": int(cnt),
            "percentage": round(int(cnt) / total * 100, 1) if total > 0 else 0.0,
        }

    return result


# ---------------------------------------------------------------------------
# 3. 学历维度统计
# ---------------------------------------------------------------------------

def _calc_degree_stats(df: pd.DataFrame) -> dict[str, Any]:
    """计算学历水平维度指标。"""
    degree_counts = df["degree"].value_counts()
    total = int(len(df))

    result: dict[str, Any] = {
        "unique_degrees": int(degree_counts.count()),
        "distribution": {},
    }

    for degree, cnt in degree_counts.items():
        result["distribution"][str(degree)] = {
            "count": int(cnt),
            "percentage": round(int(cnt) / total * 100, 1) if total > 0 else 0.0,
        }

    return result


# ---------------------------------------------------------------------------
# 4. 工作经验维度统计
# ---------------------------------------------------------------------------

def _calc_experience_stats(df: pd.DataFrame) -> dict[str, Any]:
    """计算工作经验维度指标。"""
    result: dict[str, Any] = {"distribution": {}}
    total = int(len(df))

    for bin_def in _EXPERIENCE_BINS:
        label = bin_def["label"]
        lo = bin_def["min"]
        hi = bin_def["max"]

        if lo == -99:  # 经验不限
            subset = df[
                (df["experience_min"] == 0) & (df["experience_max"] >= 90)
            ]
        elif lo == -1:  # 应届
            subset = df[
                (df["experience_min"] == 0) & (df["experience_max"] <= 0)
            ]
        else:
            subset = df[
                (df["experience_min"] >= lo)
                & (df["experience_max"] <= hi)
                & ~((df["experience_min"] == 0) & (df["experience_max"] >= 90))  # 排除经验不限
            ]

        cnt = len(subset)
        result["distribution"][label] = {
            "count": int(cnt),
            "percentage": round(int(cnt) / total * 100, 1) if total > 0 else 0.0,
        }

    return result


# ---------------------------------------------------------------------------
# 5. 主统计函数
# ---------------------------------------------------------------------------

def calculate_market_metrics(
    source: pd.DataFrame | str | None = None,
    db_path: str = _DEFAULT_DB_PATH,
) -> dict[str, Any]:
    """
    计算四大看板维度的全量市场指标。

    Parameters
    ----------
    source : pd.DataFrame | str | None
        - pd.DataFrame: 直接使用传入的 DataFrame 计算
        - str: 视为 db_path，从 SQLite 读取
        - None: 默认读取 data/hireinsight.db
    db_path : str
        数据库路径（当 source 不是 DataFrame 时使用）

    Returns
    -------
    dict[str, Any]
        四大维度的统计字典，结构：
        {
            "meta":         { "total_count": 156, "source_distribution": {...} },
            "salary":       { "count": 120, "salary_min_median": 25.0, ... },
            "city":         { "unique_cities": 18, "top10": {...}, "tier_distribution": {...} },
            "degree":       { "distribution": {"本科": {"count": 80, "percentage": 51.3}, ...} },
            "experience":   { "distribution": {"1-3年": {"count": 60, "percentage": 38.5}, ...} },
        }
    """
    # ---- 数据源处理 ----
    if isinstance(source, pd.DataFrame):
        df = source.copy()
    elif isinstance(source, str):
        df = _load_df_from_db(source)
    else:
        df = _load_df_from_db(db_path)

    if df.empty:
        logger.warning("数据源为空，返回空统计。")
        return {"meta": {"total_count": 0, "error": "数据源为空"}}

    logger.info("开始计算市场指标，总记录数: %d", len(df))

    # ---- 元信息 ----
    meta: dict[str, Any] = {
        "total_count": int(len(df)),
    }
    if "source" in df.columns:
        src_counts = df["source"].value_counts()
        meta["source_distribution"] = {
            str(k): int(v) for k, v in src_counts.items()
        }

    # ---- 四大维度 ----
    stats: dict[str, Any] = {"meta": meta}

    stats["salary"]     = _calc_salary_stats(df)
    stats["city"]       = _calc_city_stats(df)
    stats["degree"]     = _calc_degree_stats(df)
    stats["experience"] = _calc_experience_stats(df)

    logger.info(
        "市场指标计算完成 | 薪资记录: %s | 城市数: %s | 学历种类: %s",
        stats["salary"].get("count", 0),
        stats["city"].get("unique_cities", 0),
        stats["degree"].get("unique_degrees", 0),
    )

    return stats


# ---------------------------------------------------------------------------
# 6. AI Agent 白话文摘要生成器
# ---------------------------------------------------------------------------

def generate_agent_prompt_summary(
    stats_dict: dict[str, Any],
    target_position: str = "",
) -> str:
    """
    将统计字典组装为 Markdown 格式的市场摘要文本。

    该文本可直接作为 System Prompt 角色描述，喂给 DeepSeek API，
    供 Market_Agent 据此撰写《就业形势分析报告》。

    Parameters
    ----------
    stats_dict : dict
        calculate_market_metrics() 返回的完整统计字典
    target_position : str
        目标岗位名称（可选），用于个性化开头

    Returns
    -------
    str
        Markdown 格式的市场数据摘要
    """
    # 安全获取嵌套值
    def _s(d: dict, *keys: Any, default: Any = "N/A") -> Any:
        for k in keys:
            if isinstance(d, dict):
                d = d.get(k, {})
            else:
                return default
        return d if d != {} else default

    meta = stats_dict.get("meta", {})
    salary = stats_dict.get("salary", {})
    city = stats_dict.get("city", {})
    degree = stats_dict.get("degree", {})
    experience = stats_dict.get("experience", {})

    total_jobs = meta.get("total_count", 0)
    valid_salary_count = salary.get("count", 0)

    # ---- 组装 Markdown ----
    lines: list[str] = []

    # 标题
    if target_position:
        lines.append(f"# 📊 {target_position} 岗位市场数据看板")
    else:
        lines.append("# 📊 全量岗位市场数据看板")
    lines.append(f"> 总岗位数：**{total_jobs}** 条 | 有效薪资数据：**{valid_salary_count}** 条")
    lines.append("")

    # ---- 薪资 ----
    lines.append("## 💰 一、薪资水平")
    lines.append("")
    if valid_salary_count > 0:
        lines.append(f"- **全盘最低月薪**：{_s(salary, 'salary_min_overall')} k")
        lines.append(f"- **全盘最高月薪**：{_s(salary, 'salary_max_overall')} k")
        lines.append(f"- **薪资中位数区间**：{_s(salary, 'salary_min_median')}k - {_s(salary, 'salary_max_median')}k")
        lines.append(f"- **平均薪资区间**：{_s(salary, 'avg_interval')}")

        pct = salary.get("percentiles", {})
        if pct:
            lines.append(f"- **25% 分位**：{_s(pct, 'p25_min')}k - {_s(pct, 'p25_max')}k")
            lines.append(f"- **75% 分位**：{_s(pct, 'p75_min')}k - {_s(pct, 'p75_max')}k")

        dist = salary.get("salary_distribution", {})
        if dist:
            lines.append("")
            lines.append("**薪资段分布**：")
            lines.append("")
            lines.append("| 薪资段 | 岗位数 |")
            lines.append("|--------|--------|")
            for bracket, cnt in dist.items():
                lines.append(f"| {bracket} | {cnt} |")
    else:
        lines.append("> ⚠️ 当前数据源中没有有效薪资数据（可能全为面议岗位）。")
    lines.append("")

    # ---- 城市 ----
    lines.append("## 🌆 二、地区分布")
    lines.append("")
    lines.append(f"- 覆盖城市数：**{_s(city, 'unique_cities')}** 个")
    lines.append("")

    top10 = city.get("top10", {})
    if top10:
        lines.append("**岗位需求 Top 10 城市**：")
        lines.append("")
        lines.append("| 排名 | 城市 | 岗位数 |")
        lines.append("|------|------|--------|")
        for rank, (c, cnt) in enumerate(top10.items(), 1):
            lines.append(f"| {rank} | {c} | {cnt} |")
        lines.append("")

    tier_dist = city.get("tier_distribution", {})
    if tier_dist:
        lines.append("**城市等级占比**：")
        lines.append("")
        lines.append("| 城市等级 | 岗位数 | 占比 |")
        lines.append("|----------|--------|------|")
        for tier_label, info in tier_dist.items():
            lines.append(f"| {tier_label} | {info['count']} | {info['percentage']}% |")
    lines.append("")

    # ---- 学历 ----
    lines.append("## 🎓 三、学历要求")
    lines.append("")
    deg_dist = degree.get("distribution", {})
    if deg_dist:
        lines.append("| 学历 | 岗位数 | 占比 |")
        lines.append("|------|--------|------|")
        # 按学历等级排序
        degree_order = ["博士", "硕士", "本科", "大专", "高中", "中专及以下", "不限"]
        for d in degree_order:
            if d in deg_dist:
                info = deg_dist[d]
                lines.append(f"| {d} | {info['count']} | {info['percentage']}% |")
        for d, info in deg_dist.items():
            if d not in degree_order:
                lines.append(f"| {d} | {info['count']} | {info['percentage']}% |")
    lines.append("")

    # ---- 经验 ----
    lines.append("## ⏳ 四、工作经验要求")
    lines.append("")
    exp_dist = experience.get("distribution", {})
    if exp_dist:
        lines.append("| 经验区间 | 岗位数 | 占比 |")
        lines.append("|----------|--------|------|")
        for label, info in exp_dist.items():
            lines.append(f"| {label} | {info['count']} | {info['percentage']}% |")
    lines.append("")

    # ---- 尾部：供 Agent 使用的指令 ----
    lines.append("---")
    lines.append("")
    lines.append("## 🤖 AI Agent 指令")
    lines.append("")
    lines.append(
        "请基于以上真实市场统计数据，为求职者撰写一份结构清晰的 "
        "**《就业形势分析报告》**。报告需包含以下部分："
    )
    lines.append("1. **市场概览**：总结当前目标岗位的整体供需情况")
    lines.append("2. **薪资竞争力分析**：解读薪资中位数与分位数，评估竞争力")
    lines.append("3. **城市选择建议**：根据 Top 城市需求与城市等级给出建议")
    lines.append("4. **学历与经验门槛**：分析学历和工作经验的门槛现状")
    lines.append("5. **求职策略建议**：基于数据给出 3 条可执行的求职建议")
    lines.append("")
    lines.append("请使用专业且易读的中文，避免过度使用技术术语。")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 7. 便捷函数：一步加载 + 统计
# ---------------------------------------------------------------------------

def load_and_calculate(db_path: str = _DEFAULT_DB_PATH) -> dict[str, Any]:
    """
    从数据库加载数据并一步完成全线统计。
    
    Parameters
    ----------
    db_path : str
        数据库文件路径

    Returns
    -------
    dict
        calculate_market_metrics() 的完整结果
    """
    return calculate_market_metrics(source=db_path)


# ---------------------------------------------------------------------------
# 本地测试
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # 构造模拟测试数据
    import numpy as np

    np.random.seed(42)
    n = 200

    # 构造随机 cities / degrees / experiences
    cities_choices = ["深圳", "北京", "上海", "杭州", "成都", "武汉", "南京", "西安", "广州", "长沙"]
    degree_choices = ["本科", "硕士", "博士", "大专", "高中", "不限"]
    exp_choices = [
        ("应届/实习", 0, 0),
        ("1年以下", 0, 1),
        ("1-3年", 1, 3),
        ("3-5年", 3, 5),
        ("5-10年", 5, 10),
        ("10年以上", 10, 99),
        ("经验不限", 0, 99),
    ]

    test_df = pd.DataFrame({
        "city": np.random.choice(cities_choices, n),
        "degree": np.random.choice(degree_choices, n),
        "source": np.random.choice(["bytedance", "tencent"], n),
        "salary_min": np.random.uniform(5, 40, n),
        "salary_max": np.random.uniform(10, 80, n),
        "experience_min": np.random.choice([e[1] for e in exp_choices], n),
        "experience_max": np.random.choice([e[2] for e in exp_choices], n),
    })

    # 注入 10% 面议岗位（salary_min/max 为 NaN）
    mask = np.random.choice([True, False], n, p=[0.1, 0.9])
    test_df.loc[mask, "salary_min"] = np.nan
    test_df.loc[mask, "salary_max"] = np.nan

    print("=" * 60)
    print("市场指标计算测试")
    print("=" * 60)

    stats = calculate_market_metrics(source=test_df)

    print(f"\n  总岗位数: {stats['meta']['total_count']}")
    print(f"  有效薪资: {stats['salary']['count']}")
    print(f"  薪资中位数: {stats['salary']['salary_min_median']}k - {stats['salary']['salary_max_median']}k")
    print(f"  城市数: {stats['city']['unique_cities']}")
    print(f"  学历种类: {stats['degree']['unique_degrees']}")

    print("\n" + "=" * 60)
    print("AI Agent 摘要生成测试（前 30 行预览）")
    print("=" * 60)

    summary = generate_agent_prompt_summary(stats, target_position="Python 后端开发")
    lines = summary.split("\n")
    # 打印前 30 行，替换 emoji 以防 Windows GBK 控制台报错
    for line in lines[:30]:
        try:
            print(line)
        except UnicodeEncodeError:
            print(line.encode("ascii", errors="replace").decode("ascii"))
    print("...")
    print(f"\n(总 {len(lines)} 行 Markdown)")
