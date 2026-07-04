# -*- coding: utf-8 -*-
"""
utils/data_cleaner.py
======================
数据清洗工具 —— 消费 Transformer 输出的标准中间字典，产出清洗后可入库的 DataFrame。

清洗管线
--------
    标准中间 dict → clean_job_data(df) → 清洗后 DataFrame
        ├─ parse_salary()       # 薪资解析 → salary_min / salary_max / salary_bonus
        ├─ normalize_city()     # 城市规范化 → city
        ├─ normalize_education()# 学历规范化 → degree
        ├─ parse_experience()   # 经验解析 → experience_min / experience_max
        └─ clean_text()         # 文本清理 → duty / requirement

生产级约束
----------
    - Pandas 向量化 .apply() 轴向映射（非逐行循环）
    - 所有正则使用 re.compile() 预编译
    - 清理函数内部零 try-except（由 clean_job_data 外层统一捕获）
    - 关键字段缺失的记录直接过滤剔除

依赖
----
    pandas >= 1.5.0

Author: HireInsight-Agent (工业级重构版)
"""

from __future__ import annotations

import logging
import re
from typing import Any, Tuple

import pandas as pd

# ---------------------------------------------------------------------------
# 日志
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 预编译正则（性能关键路径）
# ---------------------------------------------------------------------------

# 薪资正则
# "25k-45k" / "20-35k" / "25K~45K" / "25K-45K"
_RE_SALARY_K_RANGE = re.compile(
    r"(?P<min>\d+)\s*[kK]?\s*[-~到至]\s*(?P<max>\d+)\s*[kK]",
)
# "25000-45000" 纯数字范围（可能不含 K）
_RE_SALARY_YUAN_RANGE = re.compile(
    r"(?P<min>\d{5,6})\s*[-~到至]\s*(?P<max>\d{5,6})(?!\s*[kK])",
)
# 绩效倍数："14薪" / "16薪" / "*14"
_RE_BONUS = re.compile(
    r"(?:[·\*×]\s*(?P<bonus>\d+)薪|\b(?P<bonus2>\d+)薪\b|[\*×](?P<bonus3>\d+))",
)
# 日薪："200-300元/天" / "200元/天"
_RE_DAILY_SALARY = re.compile(
    r"(?P<min>\d+)\s*[-~到至]?\s*(?P<max>\d+)?\s*元?\s*/\s*[天日]",
)
# 年薪区间："25万-45万/年"
_RE_ANNUAL_RANGE = re.compile(
    r"(?P<min>\d+)\s*万\s*[-~到至]\s*(?P<max>\d+)\s*万\s*/\s*年",
)
# 年薪单值："45万/年" 或 "25-45万/年"
_RE_ANNUAL_SINGLE = re.compile(
    r"(?P<min>\d+)\s*[-~到至]?\s*(?P<max>\d+)?\s*万\s*/\s*年",
)

# 经验正则
_RE_EXP_RANGE = re.compile(r"(?P<min>\d+)\s*[-~到至]\s*(?P<max>\d+)\s*年?")
_RE_EXP_ABOVE = re.compile(r"(?P<min>\d+)\s*年以上")
_RE_EXP_BELOW = re.compile(r"(?P<max>\d+)\s*年以下")
_RE_EXP_SINGLE = re.compile(r"^(?P<num>\d+)\s*年$")

# 城市清理
_RE_CITY_SUFFIX = re.compile(r"[省市县区]$")
_RE_CITY_AREA = re.compile(r"[\(（].+?[\)）]")      # 去除括号内区域名

# 文本清理
_RE_HTML_TAG = re.compile(r"<[^>]+>")
_RE_MULTI_SPACE = re.compile(r"\s{2,}")
_RE_DUTY_PREFIX = re.compile(r"^(岗位职责|工作职责|职位描述|工作内容)[:：]?\s*", re.IGNORECASE)
_RE_REQ_PREFIX = re.compile(r"^(任职要求|职位要求|岗位要求|工作要求)[:：]?\s*", re.IGNORECASE)

# ---------------------------------------------------------------------------
# 城市映射表（带等级分类）
# ---------------------------------------------------------------------------

_CITY_NORMALIZE_TABLE: dict[str, str] = {
    # 直辖市 → 去"市"
    "北京市": "北京", "上海市": "上海",
    "天津市": "天津", "重庆市": "重庆",
    # 省会 → 去"市"
    "广州市": "广州", "深圳市": "深圳",
    "杭州市": "杭州", "成都市": "成都",
    "南京市": "南京", "武汉市": "武汉",
    "西安市": "西安", "长沙市": "长沙",
    "济南市": "济南", "郑州市": "郑州",
    "合肥市": "合肥", "福州市": "福州",
    # 热门非省会
    "苏州市": "苏州", "厦门市": "厦门",
    "东莞市": "东莞", "佛山市": "佛山",
    # 英文拼音
    "Beijing": "北京", "Shanghai": "上海",
    "Shenzhen": "深圳", "Guangzhou": "广州",
    "Hangzhou": "杭州", "Chengdu": "成都",
    # 远程/全国
    "远程": "远程", "全国": "全国",
}

# 城市等级分类（统计看板用）
_CITY_TIER: dict[str, int] = {
    "北京": 1, "上海": 1, "深圳": 1, "广州": 1,            # 一线
    "杭州": 2, "成都": 2, "南京": 2, "武汉": 2,
    "西安": 2, "长沙": 2, "济南": 2, "苏州": 2,
    "重庆": 2, "天津": 2, "厦门": 2, "郑州": 2,
    "合肥": 2, "福州": 2, "东莞": 2, "佛山": 2,           # 新一线/二线
    "远程": 0, "全国": 0,
}

# ---------------------------------------------------------------------------
# 学历映射表
# ---------------------------------------------------------------------------

_DEGREE_NORMALIZE_TABLE: dict[str, str] = {}
for _raw, _std in [
    # 博士
    ("博士", "博士"), ("博士研究生", "博士"), ("博士及以上", "博士"),
    # 硕士
    ("硕士", "硕士"), ("硕士研究生", "硕士"), ("硕士及以上", "硕士"),
    # 本科
    ("本科", "本科"), ("本科及以上", "本科"), ("大学本科", "本科"),
    ("统招本科", "本科"), ("全日制本科", "本科"),
    # 大专
    ("大专", "大专"), ("大专及以上", "大专"), ("专科", "大专"),
    ("专科及以上", "大专"), ("高等专科", "大专"),
    # 高中
    ("高中", "高中"), ("高中及以上", "高中"), ("中专", "中专及以下"),
    ("中技", "中专及以下"), ("初中", "中专及以下"),
    # 不限
    ("学历不限", "不限"), ("无要求", "不限"),
]:
    _DEGREE_NORMALIZE_TABLE[_raw] = _std


# ---------------------------------------------------------------------------
# 薪资通用标记
# ---------------------------------------------------------------------------

_NEGOTIABLE_KEYWORDS: frozenset = frozenset({
    "面议", "薪资面议", "薪资面谈", "待遇面议", "工资面议",
    "薪资无上限", "无上限", "不限",
})


# ---------------------------------------------------------------------------
# 1. 增强版薪资解析
# ---------------------------------------------------------------------------

def parse_salary(
    salary_raw: str,
    daily_to_monthly_multiplier: int = 22,
) -> Tuple[float | None, float | None, str]:
    """
    将异构原始薪资串解析为 min / max / bonus 三元组。

    支持的格式（按优先级）
    -----------------------
     1. "25k-45k"          → (25.0, 45.0, "")
     2. "25k-40k·14薪"     → (25.0, 40.0, "14薪")
     3. "25000-45000"      → (25.0, 45.0, "")
     4. "25-45k"           → (25.0, 45.0, "")
     5. "200-300元/天"     → (4.4, 6.6, "")  (日薪 × 22)
     6. "25万-45万/年"     → (2.1, 3.8, "")  (年薪 ÷ 12)
     7. "面议"             → (None, None, "")

    Parameters
    ----------
    salary_raw : str
        原始薪资字符串（保留脏数据）
    daily_to_monthly_multiplier : int
        日薪折算月薪天数，默认 22

    Returns
    -------
    tuple[float | None, float | None, str]
        (salary_min, salary_max, salary_bonus)
        salary_min/max 单位为千元/月
    """
    # 快速路径：空值 / 面议
    if not salary_raw:
        return (None, None, "")

    s = str(salary_raw).strip()
    if not s or s in _NEGOTIABLE_KEYWORDS:
        return (None, None, "")

    # 提取绩效倍数
    bonus = ""
    bonus_match = _RE_BONUS.search(s)
    if bonus_match:
        for group_name in ("bonus", "bonus2", "bonus3"):
            val = bonus_match.group(group_name)
            if val:
                bonus = f"{val}薪"
                break
        # 从原始串中移除绩效部分，简化后续匹配
        s = _RE_BONUS.sub("", s).strip()

    min_val: float | None = None
    max_val: float | None = None

    # Case 1: "25k-45k" / "20-35k" / "25K~45K"
    match = _RE_SALARY_K_RANGE.search(s)
    if match:
        min_val = float(match.group("min"))
        max_val = float(match.group("max"))
        return (min_val, max_val, bonus)

    # Case 2: "25000-45000"（5-6 位数的纯数字范围）
    match = _RE_SALARY_YUAN_RANGE.search(s)
    if match:
        min_val = float(match.group("min")) / 1000.0
        max_val = float(match.group("max")) / 1000.0
        return (round(min_val, 1), round(max_val, 1), bonus)

    # Case 3: 日薪 "200-300元/天"
    match = _RE_DAILY_SALARY.search(s)
    if match:
        _min = int(match.group("min"))
        _max = match.group("max")
        if _max is None:
            _max = _min
        else:
            _max = int(_max)
        min_val = round(_min * daily_to_monthly_multiplier / 1000.0, 1)
        max_val = round(int(_max) * daily_to_monthly_multiplier / 1000.0, 1)
        return (min_val, max_val, bonus)

    # Case 4: 年薪区间 "25万-45万/年"
    match = _RE_ANNUAL_RANGE.search(s)
    if match:
        _min = int(match.group("min"))
        _max = int(match.group("max"))
        min_val = round(_min * 10 / 12.0, 1)
        max_val = round(_max * 10 / 12.0, 1)
        return (min_val, max_val, bonus)

    # Case 5: 年薪单值 "45万/年" 或 "25-45万/年"（兜底）
    match = _RE_ANNUAL_SINGLE.search(s)
    if match:
        _min = int(match.group("min"))
        _max = match.group("max")
        if _max is None:
            _max = _min
        else:
            _max = int(_max)
        min_val = round(_min * 10 / 12.0, 1)        # 万 → 千元/月
        max_val = round(int(_max) * 10 / 12.0, 1)
        return (min_val, max_val, bonus)

    # 兜底：完全无法解析
    logger.debug("薪资解析兜底: %s", salary_raw[:50])
    return (None, None, bonus)


# ---------------------------------------------------------------------------
# 2. 城市规范化
# ---------------------------------------------------------------------------

def normalize_city(city_raw: str) -> str:
    """
    规范化城市名称，如 "北京市" → "北京"、"Hangzhou" → "杭州"。

    Parameters
    ----------
    city_raw : str
        原始城市字段

    Returns
    -------
    str
        标准化后的城市名，未命中映射表则返回去后缀版本
    """
    if not city_raw:
        return "未知"

    c = str(city_raw).strip()

    # 先查规范化表
    standard = _CITY_NORMALIZE_TABLE.get(c)
    if standard:
        return standard

    # 去除括号及内容："深圳(南山区)" → "深圳"
    c = _RE_CITY_AREA.sub("", c).strip()

    # 再次查表
    standard = _CITY_NORMALIZE_TABLE.get(c)
    if standard:
        return standard

    # 尾部去"市"
    c = _RE_CITY_SUFFIX.sub("", c)
    return c if c else "未知"


def get_city_tier(city: str) -> int:
    """
    返回城市等级（1=一线, 2=新一线/二线, 3=其他, 0=特殊）

    Parameters
    ----------
    city : str
        normalize_city() 标准化后的城市名

    Returns
    -------
    int
        城市等级
    """
    return _CITY_TIER.get(city, 3)


# ---------------------------------------------------------------------------
# 3. 学历规范化
# ---------------------------------------------------------------------------

def normalize_education(degree_raw: str) -> str:
    """
    学历标准化映射。

    Parameters
    ----------
    degree_raw : str
        原始学历字段（如 "本科及以上"）

    Returns
    -------
    str
        标准化枚举值：博士 / 硕士 / 本科 / 大专 / 高中 / 中专及以下 / 不限
    """
    if not degree_raw:
        return "不限"

    d = str(degree_raw).strip()

    # 优先精确匹配
    if d in _DEGREE_NORMALIZE_TABLE:
        return _DEGREE_NORMALIZE_TABLE[d]

    # 模糊匹配：包含关键词即判定
    _dl = d.lower()
    if "博士" in _dl:
        return "博士"
    if "硕士" in _dl:
        return "硕士"
    if "本科" in _dl or "学士" in _dl:
        return "本科"
    if "大专" in _dl or "专科" in _dl:
        return "大专"
    if "高中" in _dl:
        return "高中"
    if "中专" in _dl or "中技" in _dl or "初中" in _dl:
        return "中专及以下"

    return d if d else "不限"


# ---------------------------------------------------------------------------
# 4. 经验解析
# ---------------------------------------------------------------------------

# 经验通用标记
_EXP_UNLIMITED_KEYWORDS: frozenset = frozenset({
    "经验不限", "不限", "无要求", "不限经验", "无经验要求",
})

_EXP_FRESH_KEYWORDS: frozenset = frozenset({
    "在校生/应届生", "应届生", "应届毕业生", "应届", "实习",
    "在校生", "实习生",
})


def parse_experience(experience_raw: str) -> Tuple[int, int]:
    """
    将经验要求字符串解析为 (min, max) 整数元组。

    Parameters
    ----------
    experience_raw : str
        原始经验字段（如 "3-5年", "经验不限", "1年以下"）

    Returns
    -------
    tuple[int, int]
        (experience_min, experience_max)，不限时 max=99
    """
    if not experience_raw:
        return (0, 99)

    s = str(experience_raw).strip()

    # 经验不限
    if s in _EXP_UNLIMITED_KEYWORDS:
        return (0, 99)

    # 应届生
    if s in _EXP_FRESH_KEYWORDS:
        return (0, 0)

    # "3-5年" / "3年-5年"
    match = _RE_EXP_RANGE.search(s)
    if match:
        return (int(match.group("min")), int(match.group("max")))

    # "3年以上"
    match = _RE_EXP_ABOVE.search(s)
    if match:
        return (int(match.group("min")), 99)

    # "1年以下" / "3年以下"
    match = _RE_EXP_BELOW.search(s)
    if match:
        return (0, int(match.group("max")))

    # "5年" 单值（高阈值，判定为 min）
    match = _RE_EXP_SINGLE.search(s)
    if match:
        num = int(match.group("num"))
        return (num, num + 1)

    return (0, 99)


# ---------------------------------------------------------------------------
# 5. 文本清理
# ---------------------------------------------------------------------------

def clean_text(text: str) -> str:
    """
    清洗岗位文本（去除 HTML 标签 / 多余空白 / 特定前缀）。

    Parameters
    ----------
    text : str
        原始文本字段（duty / requirement）

    Returns
    -------
    str
        清洗后的纯文本
    """
    if not text:
        return ""

    t = str(text)
    t = _RE_HTML_TAG.sub("", t)
    t = _RE_DUTY_PREFIX.sub("", t)
    t = _RE_REQ_PREFIX.sub("", t)
    t = _RE_MULTI_SPACE.sub(" ", t)
    return t.strip()


# ---------------------------------------------------------------------------
# 6. 主清洗管道
# ---------------------------------------------------------------------------

# 关键非空字段（缺失则整行剔除）
_CRITICAL_COLUMNS: list[str] = [
    "title_raw", "company", "category", "city_raw", "post_url",
]


def clean_job_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    主清洗管道：接收 Transformer 输出的 DataFrame，完成全量清洗。

    清洗步骤
    --------
        1. 过滤：剔除关键字段（title / company / category / city / post_url）缺失的行
        2. 生成 title：title_raw → title 降级拷贝
        3. 清洗 city_raw → city
        4. 清洗 salary_raw → salary_min / salary_max / salary_bonus
        5. 清洗 degree_raw → degree
        6. 清洗 experience_raw → experience_min / experience_max
        7. 清洗 duty / requirement → 去除 HTML + 多余空格
        8. 生成 id 字段（source_original_id）

    Parameters
    ----------
    df : pd.DataFrame
        Transformer 输出的标准中间 DataFrame

    Returns
    -------
    pd.DataFrame
        全量清洗后的 DataFrame（新增清算字段，剔除脏行）
    """
    if df.empty:
        logger.warning("DataFrame 为空，跳过清洗。")
        return df

    df = df.copy()
    before_count = len(df)

    # ---- Step 1: 剔除关键字段缺失的行 ----
    for col in _CRITICAL_COLUMNS:
        if col in df.columns:
            df = df[df[col].notna() & (df[col].astype(str).str.strip() != "")]
    after_filter = len(df)
    if before_count - after_filter > 0:
        logger.info(
            "关键字段过滤: %d → %d 条（剔除 %d 条）",
            before_count, after_filter, before_count - after_filter,
        )

    # ---- Step 2: 生成 title 字段 ----
    if "title" not in df.columns:
        # 从 title_raw 降级拷贝（清洗层职责：确保 title 字段在流向存储层前已完备）
        df["title"] = df.get("title_raw", pd.Series("", index=df.index)).fillna("").astype(str)
    else:
        df["title"] = df["title"].fillna(df.get("title_raw", pd.Series("", index=df.index))).fillna("").astype(str)

    # ---- Step 3: 城市规范化 ----
    if "city_raw" in df.columns:
        df["city"] = df["city_raw"].astype(str).apply(normalize_city)
    else:
        df["city"] = "未知"

    # ---- Step 4: 薪资解析 ----
    if "salary_raw" in df.columns:
        parsed = df["salary_raw"].astype(str).apply(parse_salary)
        df["salary_min"] = parsed.apply(lambda x: x[0] if x else None)
        df["salary_max"] = parsed.apply(lambda x: x[1] if x else None)
        df["salary_bonus"] = parsed.apply(lambda x: x[2] if x else "")
    else:
        df["salary_min"] = None
        df["salary_max"] = None
        df["salary_bonus"] = ""

    # ---- Step 5: 学历规范化 ----
    if "degree_raw" in df.columns:
        df["degree"] = df["degree_raw"].astype(str).apply(normalize_education)
    else:
        df["degree"] = "不限"

    # ---- Step 6: 经验解析 ----
    if "experience_raw" in df.columns:
        exp_parsed = df["experience_raw"].astype(str).apply(parse_experience)
        df["experience_min"] = exp_parsed.apply(lambda x: x[0])
        df["experience_max"] = exp_parsed.apply(lambda x: x[1])
    else:
        df["experience_min"] = 0
        df["experience_max"] = 99

    # ---- Step 7: 文本清洗 ----
    if "duty" in df.columns:
        df["duty"] = df["duty"].astype(str).apply(clean_text)
    if "requirement" in df.columns:
        df["requirement"] = df["requirement"].astype(str).apply(clean_text)

    # ---- Step 8: 生成 id 字段 ----
    if "source" in df.columns and "original_id" in df.columns:
        df["id"] = df["source"].astype(str) + "_" + df["original_id"].astype(str)

    logger.info(
        "清洗管道完成: 输入 %d → 输出 %d 条记录",
        before_count, len(df),
    )

    return df


# ---------------------------------------------------------------------------
# 7. 数据统计聚合
# ---------------------------------------------------------------------------

def aggregate_job_stats(df: pd.DataFrame) -> dict[str, Any]:
    """
    生成岗位数据统计摘要（供数据大屏 / LLM 市场分析使用）。

    Parameters
    ----------
    df : pd.DataFrame
        清洗后的 DataFrame

    Returns
    -------
    dict
        包含各维度统计的字典
    """
    stats: dict[str, Any] = {"total_count": len(df)}

    if df.empty:
        return stats

    # 薪资统计
    if "salary_min" in df.columns and "salary_max" in df.columns:
        valid = df.dropna(subset=["salary_min", "salary_max"])
        if len(valid) > 0:
            stats["salary"] = {
                "avg_min": round(float(valid["salary_min"].mean()), 1),
                "avg_max": round(float(valid["salary_max"].mean()), 1),
                "median_min": round(float(valid["salary_min"].median()), 1),
                "median_max": round(float(valid["salary_max"].median()), 1),
            }

    # 城市分布
    if "city" in df.columns:
        stats["city_top10"] = df["city"].value_counts().head(10).to_dict()

    # 学历分布
    if "degree" in df.columns:
        stats["degree"] = df["degree"].value_counts().to_dict()

    # 经验分布
    if "experience_raw" in df.columns:
        stats["experience"] = df["experience_raw"].value_counts().to_dict()

    # 公司分布
    if "company" in df.columns:
        stats["company"] = df["company"].value_counts().to_dict()

    # source 分布
    if "source" in df.columns:
        stats["source"] = df["source"].value_counts().to_dict()

    return stats


# ---------------------------------------------------------------------------
# 本地测试
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("薪资解析测试")
    print("=" * 60)
    test_salaries = [
        "25k-45k",
        "25K-45K",
        "25k-40k·14薪",
        "25000-45000",
        "200-300元/天",
        "200元/天",
        "25万-45万/年",
        "面议",
        "",
        "15k~25k",
        "30K-50K*16",
    ]
    for s in test_salaries:
        print(f"  {s!r:25s} → {parse_salary(s)}")

    print("\n" + "=" * 60)
    print("城市规范化测试")
    print("=" * 60)
    test_cities = ["北京市", "深圳市", "Hangzhou", "苏州", "深圳(南山区)", ""]
    for c in test_cities:
        print(f"  {c!r:25s} → {normalize_city(c)!r}")

    print("\n" + "=" * 60)
    print("学历规范化测试")
    print("=" * 60)
    test_degrees = ["本科及以上", "硕士", "博士", "大专及以上", "高中", "学历不限", ""]
    for d in test_degrees:
        print(f"  {d!r:25s} → {normalize_education(d)!r}")

    print("\n" + "=" * 60)
    print("经验解析测试")
    print("=" * 60)
    test_exps = ["3-5年", "1-3年", "经验不限", "在校生/应届生", "5年以上", "1年以下", ""]
    for e in test_exps:
        print(f"  {e!r:25s} → {parse_experience(e)}")

    print("\n" + "=" * 60)
    print("文本清洗测试")
    print("=" * 60)
    dirty = "<p>岗位职责：\n1. 负责开发...\n2. 维护系统...</p>"
    print(f"  原始: {dirty!r}")
    print(f"  清洗: {clean_text(dirty)!r}")

    print("\n" + "=" * 60)
    print("主清洗管道测试")
    print("=" * 60)
    sample_df = pd.DataFrame([
        {
            "source": "bytedance", "original_id": "001",
            "title_raw": "Python后端", "company": "字节跳动",
            "department": "抖音研发部", "category": "技术",
            "sub_category": "后端", "city_raw": "深圳",
            "district": "南山区", "salary_raw": "25k-45k·15薪",
            "experience_raw": "3-5年", "degree_raw": "本科及以上",
            "work_type": "全职",
            "duty": "1. 负责系统开发\n2. 维护系统", "requirement": "1. 本科以上\n2. Python",
            "skills": "Python, Go", "post_url": "https://...",
        },
        {
            "source": "tencent", "original_id": "002",
            "title_raw": "Python后台开发", "company": "腾讯",
            "department": "CSIG", "category": "技术",
            "sub_category": "", "city_raw": "深圳市",
            "district": "", "salary_raw": "200-300元/天",
            "experience_raw": "经验不限", "degree_raw": "本科",
            "work_type": "实习",
            "duty": "负责开发", "requirement": "熟悉Python",
            "skills": "", "post_url": "https://...",
        },
        {
            "source": "bytedance", "original_id": "003",
            "title_raw": "", "company": "",
            "department": "", "category": "技术",
            "sub_category": "", "city_raw": "",
            "district": "", "salary_raw": "面议",
            "experience_raw": "3-5年", "degree_raw": "硕士及以上",
            "work_type": "全职",
            "duty": "", "requirement": "",
            "skills": "", "post_url": "",
        },
    ])
    cleaned = clean_job_data(sample_df)
    print(f"  清洗前: {len(sample_df)} 条")
    print(f"  清洗后: {len(cleaned)} 条")
    if not cleaned.empty:
        print(f"  id 示例: {cleaned['id'].tolist()}")
        print(f"  salary 示例: {list(zip(cleaned['salary_min'], cleaned['salary_max'], cleaned['salary_bonus']))}")
