# -*- coding: utf-8 -*-
"""
tests/test_integration.py
==========================
全链路集成测试 —— 从异构原始数据到 Markdown 摘要的端到端验证。

测试覆盖
--------
    1. 异构 Mock 数据构造（字节格式 + 腾讯格式）
    2. Transformer 动态映射 → 标准中间字典
    3. DataFrame 清洗管道（data_cleaner.py）
    4. SQLite 事务级 Upsert 写入 + 联合唯一索引去重验证
    5. 从 SQLite 读取全量数据
    6. 四大维度统计指标计算
    7. AI Agent Markdown 摘要文本生成
    8. 条件筛选查询（city + degree + keyword）
    9. tearDown 自动清理临时数据库文件

沙盒隔离
--------
    使用 tempfile 动态创建临时 .db 文件，零污染生产库。

框架
----
    Python unittest

执行
----
    python -m pytest tests/test_integration.py -v
    python -m unittest tests.test_integration -v
"""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any

import pandas as pd

# 确保项目根目录在 sys.path 中
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from utils.data_cleaner import clean_job_data  # noqa: E402
from utils.data_persistence import (  # noqa: E402
    init_sqlite_db,
    load_from_sqlite,
    query_jobs_by_filters,
    save_to_sqlite,
)
from utils.data_stats import (  # noqa: E402
    calculate_market_metrics,
    generate_agent_prompt_summary,
)
from utils.data_transformer import transform_jobs  # noqa: E402


# ======================================================================
# Mock 数据工厂
# ======================================================================

def _make_bytedance_raw_jobs() -> list[dict[str, Any]]:
    """构造字节跳动格式的原始岗位字典列表。"""
    return [
        {
            "job_id": "72800001",
            "title": "Python后端开发工程师",
            "sub_title": "Data-抖音研发部",
            "job_category": {
                "id": "100001", "name": "后端",
                "parent": {"id": "100", "name": "技术"},
            },
            "city_info": {
                "code": "101280600", "name": "深圳",
                "country": "中国", "district": "南山区",
            },
            "salary": {
                "time_unit": "month",
                "show_salary": "25k-45k",
                "salary_left": "25000",
                "salary_right": "45000",
            },
            "experience": {"name": "3-5年", "id": "3"},
            "degree": {"name": "本科", "id": "3"},
            "recruit_type": {
                "name": "社会招聘", "id": "2",
                "parent": {"name": "全职", "id": "1"},
            },
            "job_hot_flag": 0, "job_subject": [],
            "publish_time": "1748832000000",
            "duty": "1. 负责抖音推荐系统后端架构设计与开发",
            "requirement": "1. 本科及以上学历\n2. 熟练掌握Python/Golang",
            "post_url": "https://jobs.bytedance.com/job/72800001",
            "tags": ["Python", "Go", "MySQL"],
            "department": {"name": "抖音研发部", "id": "20240101"},
        },
        {
            "job_id": "72800002",
            "title": "数据分析师",
            "sub_title": "Data-抖音电商",
            "job_category": {
                "id": "200001", "name": "数据分析",
                "parent": {"id": "200", "name": "数据"},
            },
            "city_info": {
                "code": "101010100", "name": "北京",
                "country": "中国", "district": "海淀区",
            },
            "salary": {
                "time_unit": "month",
                "show_salary": "20k-35k",
                "salary_left": "20000",
                "salary_right": "35000",
            },
            "experience": {"name": "1-3年", "id": "2"},
            "degree": {"name": "硕士", "id": "5"},
            "recruit_type": {
                "name": "社会招聘", "id": "2",
                "parent": {"name": "全职", "id": "1"},
            },
            "job_hot_flag": 0, "job_subject": [],
            "publish_time": "1748745600000",
            "duty": "1. 负责抖音电商数据仓库建设",
            "requirement": "1. 统计学/数学相关专业\n2. 熟练 SQL",
            "post_url": "https://jobs.bytedance.com/job/72800002",
            "tags": ["SQL", "Python", "Spark"],
            "department": {"name": "抖音电商", "id": "20240201"},
        },
        {
            "job_id": "72800003",
            "title": "前端开发实习生",
            "sub_title": "Data-TikTok",
            "job_category": {
                "id": "100003", "name": "前端",
                "parent": {"id": "100", "name": "技术"},
            },
            "city_info": {
                "code": "101280600", "name": "深圳",
                "country": "中国", "district": "",
            },
            "salary": {
                "time_unit": "day",
                "show_salary": "200-300元/天",
                "salary_left": "200",
                "salary_right": "300",
            },
            "experience": {"name": "在校生/应届生", "id": "1"},
            "degree": {"name": "本科及以上", "id": "3"},
            "recruit_type": {
                "name": "校园招聘", "id": "3",
                "parent": {"name": "实习", "id": "2"},
            },
            "job_hot_flag": 0, "job_subject": [],
            "publish_time": "1748668800000",
            "duty": "1. 参与 TikTok 前端页面开发",
            "requirement": "1. 熟悉 React/Vue\n2. 25 届在校生",
            "post_url": "https://jobs.bytedance.com/job/72800003",
            "tags": ["React", "Vue", "TypeScript"],
            "department": {"name": "TikTok", "id": "20300101"},
        },
    ]


def _make_tencent_raw_jobs() -> list[dict[str, Any]]:
    """构造腾讯格式的原始岗位字典列表（列表页 + 详情页已合并）。"""
    return [
        {
            "postId": "1272589455647055872",
            "RecruitPostName": "Python后台开发",
            "CountryName": "中国",
            "LocationName": "深圳",
            "LocationCode": "0755",
            "CategoryName": "技术",
            "Responsibility": "1. 负责腾讯云服务器后台系统开发\n2. 参与架构优化",
            "Requirement": "1. 计算机相关专业本科及以上，3年以上经验\n2. 掌握Python/C++/Java",
            "LastUpdateTime": "2025-06-15 10:30:00",
            "PostURL": "https://careers.tencent.com/job-desc.html?postId=1272589455647055872",
            "RequireWorkYearsName": "3-5年",
            "RequireWorkYears": "3",
            "Salary": "25k-40k*14",
            "DegreeName": "本科",
            "WorkType": "全职",
            "BGName": "CSIG",
            "Tags": ["Python", "C++", "Java"],
        },
        {
            "postId": "1272589455647055873",
            "RecruitPostName": "算法工程师",
            "CountryName": "中国",
            "LocationName": "北京",
            "LocationCode": "010",
            "CategoryName": "技术",
            "Responsibility": "1. 负责 NLP 算法模型训练与优化",
            "Requirement": "1. 硕士及以上学历\n2. 熟悉深度学习框架",
            "LastUpdateTime": "2025-06-20 14:00:00",
            "PostURL": "https://careers.tencent.com/job-desc.html?postId=1272589455647055873",
            "RequireWorkYearsName": "经验不限",
            "RequireWorkYears": "0",
            "Salary": "30k-60k*16",
            "DegreeName": "硕士",
            "WorkType": "全职",
            "BGName": "IEG",
            "Tags": ["NLP", "PyTorch", "Transformers"],
        },
    ]


# ======================================================================
# 集成测试类
# ======================================================================

class TestFullPipelineIntegration(unittest.TestCase):
    """全链路集成测试 —— 端到端数据生命周期验证。"""

    @classmethod
    def setUpClass(cls) -> None:
        """初始化模块级共享资源。"""
        cls.temp_dir = tempfile.TemporaryDirectory()
        cls.test_db_path = os.path.join(cls.temp_dir.name, "hireinsight_test.db")
        cls.raw_bytedance = _make_bytedance_raw_jobs()
        cls.raw_tencent = _make_tencent_raw_jobs()

    @classmethod
    def tearDownClass(cls) -> None:
        """清理模块级资源。"""
        cls.temp_dir.cleanup()

    def setUp(self) -> None:
        """每个测试前确保干净状态。"""
        # 确保测试库不存在（每个测试独立建库）
        if os.path.exists(self.test_db_path):
            os.remove(self.test_db_path)

    def tearDown(self) -> None:
        """每个测试后清理临时文件。"""
        if os.path.exists(self.test_db_path):
            try:
                os.remove(self.test_db_path)
            except PermissionError:
                pass

    # ------------------------------------------------------------------
    # Step 1: Transformer 转换
    # ------------------------------------------------------------------

    def test_01_transformer_bytedance(self) -> None:
        """字节原始数据 → 标准中间字典"""
        results = transform_jobs(self.raw_bytedance, source="bytedance")
        self.assertEqual(len(results), 3, "字节应有 3 条转换成功")

        r = results[0]
        self.assertEqual(r["source"], "bytedance")
        self.assertEqual(r["company"], "字节跳动")
        self.assertEqual(r["title_raw"], "Python后端开发工程师")
        self.assertEqual(r["category"], "技术")
        self.assertEqual(r["sub_category"], "后端")
        self.assertEqual(r["city_raw"], "深圳")
        self.assertEqual(r["salary_raw"], "25k-45k")
        self.assertEqual(r["experience_raw"], "3-5年")
        self.assertEqual(r["degree_raw"], "本科")
        self.assertEqual(r["skills"], "Python, Go, MySQL")

    def test_02_transformer_tencent(self) -> None:
        """腾讯原始数据 → 标准中间字典"""
        results = transform_jobs(self.raw_tencent, source="tencent")
        self.assertEqual(len(results), 2, "腾讯应有 2 条转换成功")

        r = results[0]
        self.assertEqual(r["source"], "tencent")
        self.assertEqual(r["company"], "腾讯")
        self.assertEqual(r["title_raw"], "Python后台开发")
        self.assertEqual(r["category"], "技术")
        self.assertEqual(r["city_raw"], "深圳")
        self.assertEqual(r["salary_raw"], "25k-40k*14")
        self.assertEqual(r["department"], "CSIG")

    def test_03_transformer_combined(self) -> None:
        """两源异构数据合并转换"""
        all_raw = self.raw_bytedance + [
            {k: v for k, v in d.items()} for d in self.raw_tencent
        ]
        # 分两次调用
        bd = transform_jobs(self.raw_bytedance, source="bytedance")
        tx = transform_jobs(self.raw_tencent, source="tencent")
        combined = bd + tx
        self.assertEqual(len(combined), 5, "合并应有 5 条")

    # ------------------------------------------------------------------
    # Step 2: 清洗管道
    # ------------------------------------------------------------------

    def test_04_clean_job_data(self) -> None:
        """转换后 DataFrame → 清洗管道"""
        bd = transform_jobs(self.raw_bytedance, source="bytedance")
        tx = transform_jobs(self.raw_tencent, source="tencent")
        all_records = bd + tx
        df = pd.DataFrame(all_records)

        cleaned = clean_job_data(df)
        self.assertGreaterEqual(len(cleaned), 4, "清洗后应有至少 4 条")

        # 验证关键清洗字段
        byte_row = cleaned[cleaned["original_id"] == "72800001"].iloc[0]
        self.assertEqual(byte_row["salary_min"], 25.0)
        self.assertEqual(byte_row["city"], "深圳")
        self.assertEqual(byte_row["degree"], "本科")
        self.assertEqual(byte_row["experience_min"], 3)

        # 日薪
        intern_row = cleaned[cleaned["original_id"] == "72800003"].iloc[0]
        self.assertEqual(intern_row["salary_min"], 4.4)
        self.assertEqual(intern_row["salary_max"], 6.6)
        self.assertEqual(intern_row["work_type"], "实习")

        # 腾讯
        tx_row = cleaned[cleaned["original_id"] == "1272589455647055872"].iloc[0]
        self.assertEqual(tx_row["salary_min"], 25.0)
        self.assertEqual(tx_row["salary_bonus"], "14薪")

    # ------------------------------------------------------------------
    # Step 3: SQLite 持久化 + 去重验证
    # ------------------------------------------------------------------

    def test_05_init_db(self) -> None:
        """数据库初始化"""
        db_path = init_sqlite_db(self.test_db_path)
        self.assertTrue(os.path.exists(db_path))
        self.assertEqual(os.path.abspath(db_path), os.path.abspath(self.test_db_path))

    def test_06_save_and_load(self) -> None:
        """写入 → 读取 验证"""
        init_sqlite_db(self.test_db_path)

        df = pd.DataFrame(self._get_cleaned_records())
        count = save_to_sqlite(df, db_path=self.test_db_path)
        self.assertEqual(count, 5, "应写入 5 条")

        loaded = load_from_sqlite(db_path=self.test_db_path)
        self.assertEqual(len(loaded), 5, "应读取 5 条")

    def test_07_upsert_dedup(self) -> None:
        """Upsert 去重验证：original_id + source 重复时更新而非新增"""
        init_sqlite_db(self.test_db_path)

        # 第一次写入
        df1 = pd.DataFrame(self._get_cleaned_records())
        save_to_sqlite(df1, db_path=self.test_db_path)

        # 第二次写入：修改 bytedance_72800001 的薪资
        df2 = pd.DataFrame(self._get_cleaned_records())
        # 修改第一条记录的薪资
        mask = (df2["original_id"] == "72800001") & (df2["source"] == "bytedance")
        df2.loc[mask, "salary_min"] = 30.0
        df2.loc[mask, "salary_max"] = 50.0
        df2.loc[mask, "title_raw"] = "Python后端开发工程师(updated)"
        save_to_sqlite(df2, db_path=self.test_db_path)

        # 验证总数不变
        loaded = load_from_sqlite(db_path=self.test_db_path)
        self.assertEqual(len(loaded), 5, "去重后仍为 5 条")

        # 验证更新值
        row = loaded[
            (loaded["original_id"] == "72800001") & (loaded["source"] == "bytedance")
        ].iloc[0]
        self.assertEqual(row["salary_min"], 30.0, "Upsert 应为新值 30.0")
        self.assertIn("updated", str(row["title_raw"]))

    # ------------------------------------------------------------------
    # Step 4: 统计指标计算
    # ------------------------------------------------------------------

    def test_08_market_metrics_from_dataframe(self) -> None:
        """从 DataFrame 计算四大维度指标"""
        bd = transform_jobs(self.raw_bytedance, source="bytedance")
        tx = transform_jobs(self.raw_tencent, source="tencent")
        df = pd.DataFrame(bd + tx)
        cleaned = clean_job_data(df)

        stats = calculate_market_metrics(source=cleaned)
        self.assertIn("meta", stats)
        self.assertIn("salary", stats)
        self.assertIn("city", stats)
        self.assertIn("degree", stats)
        self.assertIn("experience", stats)

        self.assertEqual(stats["meta"]["total_count"], len(cleaned))
        self.assertGreater(stats["salary"].get("count", 0), 0)

    def test_09_market_metrics_from_db(self) -> None:
        """从 SQLite 计算四大维度指标（DB 源）"""
        init_sqlite_db(self.test_db_path)
        df = pd.DataFrame(self._get_cleaned_records())
        save_to_sqlite(df, db_path=self.test_db_path)

        stats = calculate_market_metrics(source=self.test_db_path)
        self.assertGreater(stats["meta"]["total_count"], 0)
        self.assertIn("salary", stats)
        self.assertIn("city", stats)
        self.assertIn("degree", stats)
        self.assertIn("experience", stats)

        # 城市 Top 10
        top10 = stats["city"].get("top10", {})
        self.assertIn("深圳", top10)
        self.assertIn("北京", top10)

        # 学历分布
        deg_dist = stats["degree"].get("distribution", {})
        self.assertIn("本科", deg_dist)

        # 经验分布
        exp_dist = stats["experience"].get("distribution", {})
        self.assertTrue(any(v["count"] > 0 for v in exp_dist.values()))

    # ------------------------------------------------------------------
    # Step 5: AI Agent 摘要生成
    # ------------------------------------------------------------------

    def test_10_generate_agent_summary(self) -> None:
        """生成 Markdown 摘要文本"""
        bd = transform_jobs(self.raw_bytedance, source="bytedance")
        tx = transform_jobs(self.raw_tencent, source="tencent")
        df = pd.DataFrame(bd + tx)
        cleaned = clean_job_data(df)
        stats = calculate_market_metrics(source=cleaned)

        summary = generate_agent_prompt_summary(stats, target_position="Python 后端开发")

        # 文本完整性断言
        self.assertIsInstance(summary, str)
        self.assertGreater(len(summary), 200)
        self.assertIn("Python 后端开发", summary)
        self.assertIn("薪资水平", summary)
        self.assertIn("地区分布", summary)
        self.assertIn("学历要求", summary)
        self.assertIn("工作经验", summary)
        self.assertIn("AI Agent 指令", summary)
        self.assertIn("就业形势分析报告", summary)

    # ------------------------------------------------------------------
    # Step 6: 条件查询
    # ------------------------------------------------------------------

    def test_11_query_by_filters(self) -> None:
        """SQLite 条件筛选查询"""
        init_sqlite_db(self.test_db_path)
        df = pd.DataFrame(self._get_cleaned_records())
        save_to_sqlite(df, db_path=self.test_db_path)

        # 按城市
        sz = query_jobs_by_filters(self.test_db_path, filters={"city": "深圳"})
        self.assertGreater(len(sz), 0)
        self.assertTrue(all(sz["city"] == "深圳"))

        # 按学历
        degree_df = query_jobs_by_filters(self.test_db_path, filters={"degree": "本科"})
        self.assertGreater(len(degree_df), 0)

        # 按城市 + 学历 + 关键词
        combined = query_jobs_by_filters(
            self.test_db_path,
            filters={"city": "深圳", "degree": "本科", "keyword": "Python"},
        )
        self.assertGreater(len(combined), 0)

        # 按 source
        bd = query_jobs_by_filters(self.test_db_path, filters={"source": "bytedance"})
        self.assertGreater(len(bd), 0)

    # ------------------------------------------------------------------
    # 辅助
    # ------------------------------------------------------------------

    def _get_cleaned_records(self) -> list[dict[str, Any]]:
        """获取清洗后的标准记录字典列表。"""
        bd = transform_jobs(self.raw_bytedance, source="bytedance")
        tx = transform_jobs(self.raw_tencent, source="tencent")
        df = pd.DataFrame(bd + tx)
        cleaned = clean_job_data(df)
        return cleaned.to_dict("records")


# ======================================================================
# main
# ======================================================================

if __name__ == "__main__":
    unittest.main(verbosity=2)
