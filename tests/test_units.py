# -*- coding: utf-8 -*-
"""
tests/test_units.py
====================
核心清洗函数单元测试。

覆盖目标
--------
    - parse_salary      : 11 种薪资格式（k 值/纯数字/日薪/年薪/面议/绩效/空值/边界）
    - normalize_city    : 去市/英文/括号/空值
    - normalize_education: 6 级枚举 + 含"及以上"后缀
    - parse_experience  : 区间/不限/应届/以上/以下/单值/空值
    - clean_text        : HTML/前缀/多余空格
    - clean_job_data    : 完整 DataFrame 清洗管道

框架
----
    Python unittest（内置，零额外依赖）

执行
----
    python -m pytest tests/test_units.py -v
    python -m unittest tests.test_units -v
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd

# 确保项目根目录在 sys.path 中
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from utils.data_cleaner import (  # noqa: E402
    clean_job_data,
    clean_text,
    get_city_tier,
    normalize_city,
    normalize_education,
    parse_experience,
    parse_salary,
)


# ======================================================================
# parse_salary 测试
# ======================================================================

class TestParseSalary(unittest.TestCase):
    """薪资解析单元测试 —— 覆盖设计文档中所有格式。"""

    def test_standard_k_range(self) -> None:
        """标准格式：'25k-45k'"""
        r = parse_salary("25k-45k")
        self.assertEqual(r, (25.0, 45.0, ""))

    def test_uppercase_k_range(self) -> None:
        """大写 K 格式：'25K-45K'"""
        r = parse_salary("25K-45K")
        self.assertEqual(r, (25.0, 45.0, ""))

    def test_no_space_k_range(self) -> None:
        """无空格：'20-35k'"""
        r = parse_salary("20-35k")
        self.assertEqual(r, (20.0, 35.0, ""))

    def test_tilde_separator(self) -> None:
        """波浪线分隔：'15k~25k'"""
        r = parse_salary("15k~25k")
        self.assertEqual(r, (15.0, 25.0, ""))

    def test_with_bonus_dot(self) -> None:
        """带绩效倍数（中间点）：'25k-40k·14薪'"""
        r = parse_salary("25k-40k·14薪")
        self.assertEqual(r, (25.0, 40.0, "14薪"))

    def test_with_bonus_star(self) -> None:
        """带绩效倍数（星号）：'30K-50K*16'"""
        r = parse_salary("30K-50K*16")
        self.assertEqual(r, (30.0, 50.0, "16薪"))

    def test_pure_yuan_range(self) -> None:
        """纯数字大数：'25000-45000'"""
        r = parse_salary("25000-45000")
        self.assertEqual(r, (25.0, 45.0, ""))

    def test_daily_salary_range(self) -> None:
        """日薪区间：'200-300元/天' → 月薪 (4.4, 6.6)"""
        r = parse_salary("200-300元/天")
        self.assertEqual(r, (4.4, 6.6, ""))

    def test_daily_salary_single(self) -> None:
        """日薪单值：'200元/天' → 月薪 (4.4, 4.4)"""
        r = parse_salary("200元/天")
        self.assertEqual(r, (4.4, 4.4, ""))

    def test_annual_salary_range(self) -> None:
        """年薪区间：'25万-45万/年' → 月薪 (20.8, 37.5)"""
        r = parse_salary("25万-45万/年")
        self.assertEqual(r, (20.8, 37.5, ""))

    def test_negotiable(self) -> None:
        """面议：返回 None"""
        r = parse_salary("面议")
        self.assertEqual(r, (None, None, ""))

    def test_negotiable_variants(self) -> None:
        """面议变体：'薪资面议'、'薪资面谈'"""
        for s in ("薪资面议", "薪资面谈", "工资面议", "待遇面议"):
            r = parse_salary(s)
            self.assertEqual(r, (None, None, ""), f"输入 {s} 应为面议")

    def test_empty_string(self) -> None:
        """空字符串 → None"""
        r = parse_salary("")
        self.assertEqual(r, (None, None, ""))

    def test_none_input(self) -> None:
        """None 输入 → None"""
        r = parse_salary(None)
        self.assertEqual(r, (None, None, ""))

    def test_single_k_value(self) -> None:
        """单值 K 格式：'30k'（仅边界测试，可能无法解析）"""
        r = parse_salary("30k")
        # 当前实现可能不匹配单值，应返回 None
        self.assertEqual(r, (None, None, ""))


# ======================================================================
# normalize_city 测试
# ======================================================================

class TestNormalizeCity(unittest.TestCase):
    """城市规范化单元测试。"""

    def test_remove_city_suffix(self) -> None:
        """去 '市' 后缀：'北京市' → '北京'"""
        self.assertEqual(normalize_city("北京市"), "北京")
        self.assertEqual(normalize_city("深圳市"), "深圳")
        self.assertEqual(normalize_city("杭州市"), "杭州")
        self.assertEqual(normalize_city("广州市"), "广州")

    def test_no_change(self) -> None:
        """已有标准名不变：'北京' → '北京'"""
        self.assertEqual(normalize_city("北京"), "北京")
        self.assertEqual(normalize_city("深圳"), "深圳")

    def test_english_to_chinese(self) -> None:
        """英文拼音映射"""
        self.assertEqual(normalize_city("Hangzhou"), "杭州")
        self.assertEqual(normalize_city("Shanghai"), "上海")
        self.assertEqual(normalize_city("Shenzhen"), "深圳")

    def test_remove_parentheses(self) -> None:
        """去除括号及内容：'深圳(南山区)' → '深圳'"""
        self.assertEqual(normalize_city("深圳(南山区)"), "深圳")
        self.assertEqual(normalize_city("北京（海淀）"), "北京")

    def test_empty_input(self) -> None:
        """空值 → '未知'"""
        self.assertEqual(normalize_city(""), "未知")
        self.assertEqual(normalize_city(None), "未知")


# ======================================================================
# normalize_education 测试
# ======================================================================

class TestNormalizeEducation(unittest.TestCase):
    """学历规范化单元测试。"""

    def test_exact_mapping(self) -> None:
        """精确映射"""
        self.assertEqual(normalize_education("博士"), "博士")
        self.assertEqual(normalize_education("硕士"), "硕士")
        self.assertEqual(normalize_education("本科"), "本科")
        self.assertEqual(normalize_education("大专"), "大专")
        self.assertEqual(normalize_education("高中"), "高中")

    def test_with_suffix_above(self) -> None:
        """含 '及以上' 后缀：'本科及以上' → '本科'"""
        self.assertEqual(normalize_education("本科及以上"), "本科")
        self.assertEqual(normalize_education("硕士及以上"), "硕士")
        self.assertEqual(normalize_education("大专及以上"), "大专")

    def test_fulltime_prefix(self) -> None:
        """全日制前缀：'全日制本科' → '本科'"""
        self.assertEqual(normalize_education("全日制本科"), "本科")
        self.assertEqual(normalize_education("统招本科"), "本科")

    def test_unlimited(self) -> None:
        """学历不限"""
        self.assertEqual(normalize_education("学历不限"), "不限")
        self.assertEqual(normalize_education("无要求"), "不限")

    def test_vocational(self) -> None:
        """中专/中技 → '中专及以下'"""
        self.assertEqual(normalize_education("中专"), "中专及以下")
        self.assertEqual(normalize_education("中技"), "中专及以下")

    def test_empty_input(self) -> None:
        """空值 → '不限'"""
        self.assertEqual(normalize_education(""), "不限")
        self.assertEqual(normalize_education(None), "不限")


# ======================================================================
# parse_experience 测试
# ======================================================================

class TestParseExperience(unittest.TestCase):
    """经验解析单元测试。"""

    def test_range(self) -> None:
        """区间格式"""
        self.assertEqual(parse_experience("1-3年"), (1, 3))
        self.assertEqual(parse_experience("3-5年"), (3, 5))
        self.assertEqual(parse_experience("5-10年"), (5, 10))

    def test_unlimited(self) -> None:
        """经验不限"""
        self.assertEqual(parse_experience("经验不限"), (0, 99))
        self.assertEqual(parse_experience("不限"), (0, 99))
        self.assertEqual(parse_experience("无要求"), (0, 99))

    def test_fresh_graduate(self) -> None:
        """应届生"""
        self.assertEqual(parse_experience("在校生/应届生"), (0, 0))
        self.assertEqual(parse_experience("应届生"), (0, 0))

    def test_above_n_years(self) -> None:
        """N 年以上"""
        self.assertEqual(parse_experience("3年以上"), (3, 99))
        self.assertEqual(parse_experience("5年以上"), (5, 99))

    def test_below_n_years(self) -> None:
        """N 年以下"""
        self.assertEqual(parse_experience("1年以下"), (0, 1))

    def test_single_year(self) -> None:
        """单值年数"""
        self.assertEqual(parse_experience("5年"), (5, 6))

    def test_empty_input(self) -> None:
        """空值 → 不限"""
        self.assertEqual(parse_experience(""), (0, 99))
        self.assertEqual(parse_experience(None), (0, 99))


# ======================================================================
# clean_text 测试
# ======================================================================

class TestCleanText(unittest.TestCase):
    """文本清洗单元测试。"""

    def test_remove_html(self) -> None:
        """去除 HTML 标签"""
        r = clean_text("<p>岗位职责：负责系统开发</p>")
        self.assertEqual(r, "负责系统开发")

    def test_remove_prefix(self) -> None:
        """去除岗位职责/任职要求前缀"""
        r = clean_text("岗位职责：1. 负责开发\n2. 维护系统")
        self.assertEqual(r, "1. 负责开发\n2. 维护系统")

    def test_collapse_whitespace(self) -> None:
        """压缩多余空格"""
        r = clean_text("1. 负责  开发   系统")
        self.assertEqual(r, "1. 负责 开发 系统")

    def test_empty(self) -> None:
        """空值"""
        self.assertEqual(clean_text(""), "")
        self.assertEqual(clean_text(None), "")


# ======================================================================
# clean_job_data 主清洗管道测试
# ======================================================================

class TestCleanJobData(unittest.TestCase):
    """主清洗管道单元测试。"""

    def setUp(self) -> None:
        """构造标准测试 DataFrame。"""
        self.df = pd.DataFrame([
            {
                "source": "bytedance", "original_id": "001",
                "title_raw": "Python后端", "company": "字节跳动",
                "department": "抖音研发部", "category": "技术",
                "sub_category": "后端", "city_raw": "深圳", "district": "南山区",
                "salary_raw": "25k-45k·15薪",
                "experience_raw": "3-5年", "degree_raw": "本科及以上",
                "work_type": "全职",
                "duty": "岗位职责：1. 负责开发\n2. 维护系统",
                "requirement": "<p>1. 本科以上\n2. Python</p>",
                "skills": "Python, Go, MySQL", "post_url": "https://.../001",
                "published_at": "2025-06-01", "updated_at": "2025-06-15",
            },
            {
                "source": "tencent", "original_id": "002",
                "title_raw": "Python后台开发", "company": "腾讯",
                "department": "CSIG", "category": "技术",
                "sub_category": "", "city_raw": "深圳市", "district": "",
                "salary_raw": "200-300元/天",
                "experience_raw": "在校生/应届生", "degree_raw": "本科",
                "work_type": "实习",
                "duty": "工作内容：负责开发",
                "requirement": "1. 熟悉Python",
                "skills": "Python, C++", "post_url": "https://.../002",
                "published_at": "", "updated_at": "2025-06-15",
            },
            # 脏数据：关键字段缺失，应被过滤
            {
                "source": "bytedance", "original_id": "003",
                "title_raw": "", "company": "",   # 缺失 title/company
                "department": "", "category": "技术",
                "sub_category": "", "city_raw": "",  # 缺失 city
                "district": "", "salary_raw": "面议",
                "experience_raw": "3-5年", "degree_raw": "硕士及以上",
                "work_type": "全职",
                "duty": "", "requirement": "",
                "skills": "", "post_url": "",  # 缺失 post_url
                "published_at": "", "updated_at": "",
            },
        ])

    def test_filter_dirty_rows(self) -> None:
        """脏数据过滤：第 3 行应被剔除"""
        cleaned = clean_job_data(self.df)
        self.assertEqual(len(cleaned), 2, "应有 2 条通过清洗")
        ids = cleaned["original_id"].tolist()
        self.assertIn("001", ids)
        self.assertIn("002", ids)
        self.assertNotIn("003", ids)

    def test_salary_parsing_in_pipeline(self) -> None:
        """薪资管道解析"""
        cleaned = clean_job_data(self.df)
        row = cleaned[cleaned["original_id"] == "001"].iloc[0]
        self.assertEqual(row["salary_min"], 25.0)
        self.assertEqual(row["salary_max"], 45.0)
        self.assertEqual(row["salary_bonus"], "15薪")

    def test_city_normalization_in_pipeline(self) -> None:
        """城市规范化管道"""
        cleaned = clean_job_data(self.df)
        row = cleaned[cleaned["original_id"] == "002"].iloc[0]
        self.assertEqual(row["city"], "深圳")  # "深圳市" → "深圳"

    def test_degree_normalization_in_pipeline(self) -> None:
        """学历规范化管道"""
        cleaned = clean_job_data(self.df)
        row = cleaned[cleaned["original_id"] == "001"].iloc[0]
        self.assertEqual(row["degree"], "本科")  # "本科及以上" → "本科"

    def test_experience_parsing_in_pipeline(self) -> None:
        """经验解析管道"""
        cleaned = clean_job_data(self.df)
        row1 = cleaned[cleaned["original_id"] == "001"].iloc[0]
        self.assertEqual(row1["experience_min"], 3)
        self.assertEqual(row1["experience_max"], 5)

        row2 = cleaned[cleaned["original_id"] == "002"].iloc[0]
        self.assertEqual(row2["experience_min"], 0)
        self.assertEqual(row2["experience_max"], 0)

    def test_text_cleaning_in_pipeline(self) -> None:
        """文本清洗管道"""
        cleaned = clean_job_data(self.df)
        row = cleaned[cleaned["original_id"] == "001"].iloc[0]
        self.assertNotIn("岗位职责", str(row["duty"]))
        self.assertNotIn("<p>", str(row["requirement"]))

    def test_id_generation(self) -> None:
        """ID 生成：source_original_id"""
        cleaned = clean_job_data(self.df)
        ids = cleaned["id"].tolist()
        self.assertIn("bytedance_001", ids)
        self.assertIn("tencent_002", ids)


# ======================================================================
# get_city_tier 测试
# ======================================================================

class TestGetCityTier(unittest.TestCase):
    """城市等级查询测试。"""

    def test_tier_1_cities(self) -> None:
        """一线城市"""
        self.assertEqual(get_city_tier("北京"), 1)
        self.assertEqual(get_city_tier("上海"), 1)
        self.assertEqual(get_city_tier("深圳"), 1)
        self.assertEqual(get_city_tier("广州"), 1)

    def test_tier_2_cities(self) -> None:
        """新一线/二线"""
        self.assertEqual(get_city_tier("杭州"), 2)
        self.assertEqual(get_city_tier("成都"), 2)

    def test_unknown_city(self) -> None:
        """未收录城市 → 3"""
        self.assertEqual(get_city_tier("拉萨"), 3)
        self.assertEqual(get_city_tier(""), 3)


# ======================================================================
# main
# ======================================================================

if __name__ == "__main__":
    unittest.main(verbosity=2)
