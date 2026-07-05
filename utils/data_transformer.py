# -*- coding: utf-8 -*-
"""
utils/data_transformer.py
==========================
多源异构数据转换层（Transformer）。

功能定位
--------
    本模块位于"原始爬虫层"与"数据清洗层"之间，负责将网易和腾讯
    两套异构的原始字典结构统一转换为标准中间字典。

设计模式：策略模式（Strategy Pattern）
    针对不同的 source 值，动态路由到对应的私有转换函数。
    新增大厂时只需添加一个新的 _transform_<source> 方法，无需修改主流程。

转换契约
--------
    1. 输入：原始爬虫返回的 list[dict]（网易 / 米哈游各自的结构）
    2. 输出：标准中间字典，包含以下固定 Key：
       source, original_id, title_raw, company, department,
       category, sub_category, city_raw, district,
       salary_raw, experience_raw, degree_raw, work_type,
       duty, requirement, skills, post_url
    3. 本层不做任何正则清洗 / 薪资换算 / 文本清理，仅做字段路径映射与提取。

安全提取原则
-----------
    严禁使用 raw["key"] 硬编码访问，必须全量使用 .get() 安全提取语法。
    每个转换函数都应该是"自防御"的，字段缺失时自动回退到空字符串 ""。

依赖
----
    无外部依赖（纯标准库）。

Author: HireInsight-Agent
"""

from __future__ import annotations

import logging
from typing import Any

# ---------------------------------------------------------------------------
# 模块日志
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 标准中间字典模板
# ---------------------------------------------------------------------------

# 标准输出键名常量（方便全局对齐，防止拼写错误）
_STANDARD_KEYS: list[str] = [
    "source",
    "original_id",
    "title_raw",
    "company",
    "department",
    "category",
    "sub_category",
    "city_raw",
    "district",
    "salary_raw",
    "experience_raw",
    "degree_raw",
    "work_type",
    "duty",
    "requirement",
    "skills",
    "post_url",
    "published_at",
    "updated_at",
]


def _empty_record(source: str) -> dict[str, Any]:
    """生成一条字段全为默认值的标准记录模板。"""
    record: dict[str, Any] = {key: "" for key in _STANDARD_KEYS}
    # skills 改用空列表方便拼接
    record["skills"] = []
    record["source"] = source
    return record


# ---------------------------------------------------------------------------
# 滴滴 → 标准映射
# ---------------------------------------------------------------------------

def _transform_didi(raw: dict[str, Any]) -> dict[str, Any]:
    """滴滴出行 → 标准中间格式。"""
    record = _empty_record(source="didi")
    record["original_id"] = str(raw.get("id") or raw.get("jdId", "")).strip()
    record["title_raw"] = str(raw.get("jobName", "")).strip()
    record["company"] = "滴滴"
    record["department"] = str(raw.get("deptName", "")).strip()
    record["category"] = str(raw.get("jobType", "")).strip()
    record["sub_category"] = str(raw.get("labelName", "")).strip()
    record["city_raw"] = str(raw.get("workArea", "")).strip()
    record["district"] = ""
    record["salary_raw"] = ""
    record["experience_raw"] = ""
    record["degree_raw"] = ""
    record["work_type"] = "全职"
    record["duty"] = ""
    record["requirement"] = ""
    record["skills"] = ""
    jd_id = record["original_id"]
    record["post_url"] = f"https://talent.didiglobal.com/social/job/{jd_id}" if jd_id else ""
    record["published_at"] = str(raw.get("createTime", ""))[:10] if raw.get("createTime") else ""
    record["updated_at"] = record["published_at"]
    return record


# ---------------------------------------------------------------------------
# 字节跳动 → 标准映射
# ---------------------------------------------------------------------------

def _transform_bytedance(raw: dict[str, Any]) -> dict[str, Any]:
    """字节跳动 → 标准中间格式。"""
    record = _empty_record(source="bytedance")
    record["original_id"] = str(raw.get("id", "")).strip()
    record["title_raw"] = str(raw.get("title", "")).strip()
    record["company"] = "字节跳动"
    record["department"] = str(raw.get("job_category", {}).get("name", "") if isinstance(raw.get("job_category"), dict) else raw.get("job_category", ""))
    record["category"] = str(raw.get("job_subject", {}).get("name", "") if isinstance(raw.get("job_subject"), dict) else raw.get("job_subject", ""))
    record["sub_category"] = str(raw.get("recruit_type", {}).get("name", "") if isinstance(raw.get("recruit_type"), dict) else "")
    city = raw.get("city_info", {})
    if isinstance(city, dict):
        record["city_raw"] = city.get("name", "")
    else:
        record["city_raw"] = str(city or "")
    record["district"] = ""
    record["salary_raw"] = ""
    record["experience_raw"] = ""
    record["degree_raw"] = ""
    record["work_type"] = "全职"
    record["duty"] = str(raw.get("description", ""))[:2000] if raw.get("description") else ""
    record["requirement"] = str(raw.get("requirement", ""))[:2000] if raw.get("requirement") else ""
    record["skills"] = ""
    jid = record["original_id"]
    record["post_url"] = f"https://jobs.bytedance.com/experienced/position/{jid}/detail" if jid else ""
    record["published_at"] = str(raw.get("publish_time", ""))[:10] if raw.get("publish_time") else ""
    record["updated_at"] = record["published_at"]
    return record


# ---------------------------------------------------------------------------
# 腾讯 → 标准映射
# ---------------------------------------------------------------------------

def _transform_tencent(raw: dict[str, Any]) -> dict[str, Any]:
    """
    将腾讯招聘原始岗位字典转换为标准中间格式。

    映射关系：
        source       → "tencent"
        original_id  → raw["PostId"]
        title_raw    → raw["RecruitPostName"]
        company      → "腾讯"（硬编码）
        department   → raw["BGName"]（事业群，如 CSIG/WXG/IEG 等）
        category     → raw["CategoryName"]
        city_raw     → raw["LocationName"]
        experience_raw → raw["RequireWorkYearsName"]
        duty         → raw["Responsibility"]
        post_url     → raw["PostURL"]

    Parameters
    ----------
    raw : dict
        腾讯招聘 API 返回的单条原始岗位字典（Data.Posts[] 元素）

    Returns
    -------
    dict
        标准中间字典
    """
    record = _empty_record(source="tencent")

    record["original_id"] = str(raw.get("PostId", "")).strip()
    record["title_raw"] = str(raw.get("RecruitPostName", "")).strip()
    record["company"] = "腾讯"
    record["department"] = str(raw.get("BGName", "")).strip()
    record["category"] = str(raw.get("CategoryName", "")).strip()
    record["sub_category"] = str(raw.get("ProductName", "")).strip()
    record["city_raw"] = str(raw.get("LocationName", "")).strip()
    record["district"] = ""
    record["salary_raw"] = ""
    record["experience_raw"] = str(raw.get("RequireWorkYearsName", "")).strip()
    record["degree_raw"] = ""
    record["work_type"] = "全职"
    record["duty"] = str(raw.get("Responsibility", "")).strip()
    record["requirement"] = ""
    record["skills"] = ""
    record["post_url"] = str(raw.get("PostURL", "")).strip()
    record["published_at"] = str(raw.get("LastUpdateTime", "")).strip()
    record["updated_at"] = record["published_at"]

    return record


# ---------------------------------------------------------------------------
# 莉莉丝 → 标准映射（已废弃，保留兼容）
# ---------------------------------------------------------------------------

def _transform_lilith(raw: dict[str, Any]) -> dict[str, Any]:
    """
    将莉莉丝游戏招聘原始岗位字典转换为标准中间格式。

    莉莉丝使用飞书（Feishu）ATS 招聘平台，字段结构与 Feishu Career API 一致。

    映射关系：
        source       → "lilith"
        original_id  → raw["id"]
        title_raw    → raw["title"] / raw["name"]
        company      → "莉莉丝游戏"（硬编码）
        department   → raw["department"] / raw["departmentName"]
        category     → raw["category"] / raw["jobCategory"]
        sub_category → ""（莉莉丝无细类）
        city_raw     → raw["city"] / raw["workCity"] / location 字段
        salary_raw   → raw["salary"] / raw["salaryDesc"]（飞书可能返回）
        experience_raw → raw["experience"] / raw["workYear"]
        degree_raw   → raw["education"] / raw["degree"]
        work_type    → raw["jobType"] / raw["employmentType"]
        duty         → raw["description"] / raw["responsibility"]
        requirement  → raw["requirement"] / raw["qualification"]
        skills       → ""（无标签字段）
        post_url     → raw 中的 apply_url 或拼接 Feishu 链接

    Parameters
    ----------
    raw : dict
        莉莉丝 Feishu 招聘 API 返回的单条原始岗位字典

    Returns
    -------
    dict
        标准中间字典（字段全部非 None）
    """
    record = _empty_record(source="lilith")

    # ---- 基础标识 ----
    job_id = raw.get("id")
    record["original_id"] = str(job_id).strip() if job_id is not None else ""

    # ---- 岗位名称 ----
    record["title_raw"] = str(raw.get("title") or raw.get("name") or "").strip()

    # ---- 公司（硬编码） ----
    record["company"] = "莉莉丝游戏"

    # ---- 部门 ----
    record["department"] = str(
        raw.get("department") or raw.get("departmentName") or ""
    ).strip()

    # ---- 职位类别 ----
    record["category"] = str(
        raw.get("category") or raw.get("jobCategory") or raw.get("job_category") or ""
    ).strip()
    record["sub_category"] = ""

    # ---- 城市 ----
    city = raw.get("city") or raw.get("workCity") or raw.get("work_city")
    if isinstance(city, list) and city:
        record["city_raw"] = ", ".join(str(c) for c in city if c)
    elif city:
        record["city_raw"] = str(city).strip()
    else:
        record["city_raw"] = ""
    record["district"] = ""

    # ---- 薪资 ----
    record["salary_raw"] = str(
        raw.get("salary") or raw.get("salaryDesc") or raw.get("salary_desc") or ""
    ).strip()

    # ---- 经验要求 ----
    record["experience_raw"] = str(
        raw.get("experience") or raw.get("workYear") or raw.get("work_year") or ""
    ).strip()

    # ---- 学历要求 ----
    record["degree_raw"] = str(
        raw.get("education") or raw.get("degree") or ""
    ).strip()

    # ---- 工作类型 ----
    jt = str(raw.get("jobType") or raw.get("employmentType") or raw.get("job_type") or "").strip()
    type_map = {
        "全职": "全职", "实习": "实习", "兼职": "兼职",
        "full-time": "全职", "intern": "实习", "part-time": "兼职",
        "Full-time": "全职", "Intern": "实习",
    }
    record["work_type"] = type_map.get(jt, jt)

    # ---- 文本内容 ----
    record["duty"] = str(
        raw.get("description") or raw.get("responsibility") or ""
    ).strip()
    record["requirement"] = str(
        raw.get("requirement") or raw.get("qualification") or ""
    ).strip()

    # ---- 技能标签 ----
    record["skills"] = ""

    # ---- 链接 ----
    apply_url = raw.get("applyUrl") or raw.get("apply_url") or raw.get("url")
    if apply_url:
        record["post_url"] = str(apply_url).strip()
    elif record["original_id"]:
        record["post_url"] = (
            f"https://lilithgames.jobs.feishu.cn/global/position/"
            f"{record['original_id']}"
        )

    # ---- 时间 ----
    publish_time = raw.get("publishTime") or raw.get("createTime") or raw.get("create_time")
    if publish_time and isinstance(publish_time, (int, float)):
        from datetime import datetime, timezone
        try:
            dt = datetime.fromtimestamp(publish_time / 1000, tz=timezone.utc)
            record["published_at"] = dt.strftime("%Y-%m-%d")
        except (OSError, ValueError):
            record["published_at"] = str(publish_time)
    elif publish_time and isinstance(publish_time, str):
        record["published_at"] = publish_time
    else:
        record["published_at"] = ""
    record["updated_at"] = record["published_at"]

    return record


# ---------------------------------------------------------------------------
# 米哈游 → 标准映射（已废弃，保留防止旧数据引用）
# ---------------------------------------------------------------------------

def _transform_mihoyo(raw: dict[str, Any]) -> dict[str, Any]:
    """
    将米哈游招聘原始岗位字典转换为标准中间格式。

    映射关系：
        source       → "mihoyo"
        original_id  → raw["id"]
        title_raw    → raw["name"]
        company      → "米哈游"（硬编码）
        department   → raw["firstDepName"]
        category     → raw["firstPostTypeName"]
        sub_category → ""（米哈游无细类）
        city_raw     → workPlaceNameList[0] 或逗号拼接
        district     → ""（米哈游接口无行政区）
        salary_raw   → raw.get("salaryRange", "")（若有薪资字段）
        experience_raw → raw["reqWorkYearsName"]
        degree_raw   → raw["reqEducationName"]
        work_type    → "全职" / "实习"（按 workType 映射）
        duty         → raw["description"]
        requirement  → raw["requirement"]
        skills       → ""（米哈游无标签字段）
        post_url     → f"https://jobs.mihoyo.com/job-detail/{id}"

    Parameters
    ----------
    raw : dict
        米哈游招聘 API 返回的单条原始岗位字典（data.list[] 元素）

    Returns
    -------
    dict
        标准中间字典（字段全部非 None）
    """
    from datetime import datetime, timezone

    record = _empty_record(source="mihoyo")

    # ---- 基础标识 ----
    job_id = raw.get("id")
    record["original_id"] = str(job_id).strip() if job_id is not None else ""

    # ---- 岗位名称 ----
    record["title_raw"] = str(raw.get("name", "")).strip()

    # ---- 公司（硬编码） ----
    record["company"] = "米哈游"

    # ---- 部门 ----
    record["department"] = str(raw.get("firstDepName", "")).strip()

    # ---- 职位类别 ----
    record["category"] = str(raw.get("firstPostTypeName", "")).strip()
    record["sub_category"] = ""           # 米哈游无细类

    # ---- 城市（从 workPlaceNameList 提取） ----
    wp_list = raw.get("workPlaceNameList", [])
    if isinstance(wp_list, list) and wp_list:
        record["city_raw"] = ", ".join(str(c) for c in wp_list if c)
        record["district"] = ""
    else:
        record["city_raw"] = ""
        record["district"] = ""

    # ---- 薪资（米哈游接口可能返回薪资范围） ----
    record["salary_raw"] = str(raw.get("salaryRange", "")).strip()

    # ---- 经验要求 ----
    record["experience_raw"] = str(raw.get("reqWorkYearsName", "")).strip()

    # ---- 学历要求 ----
    record["degree_raw"] = str(raw.get("reqEducationName", "")).strip()

    # ---- 工作类型（workType: "0"=全职, "1"=实习） ----
    wt = str(raw.get("workType", "")).strip()
    record["work_type"] = {"0": "全职", "1": "实习"}.get(wt, "")

    # ---- 文本内容 ----
    record["duty"] = str(raw.get("description", "")).strip()
    record["requirement"] = str(raw.get("requirement", "")).strip()

    # ---- 技能标签（米哈游无此字段） ----
    record["skills"] = ""

    # ---- 链接 ----
    if record["original_id"]:
        record["post_url"] = (
            f"https://jobs.mihoyo.com/job-detail/{record['original_id']}"
        )

    # ---- 时间（updateTime 为毫秒时间戳） ----
    ts_ms = raw.get("updateTime")
    if ts_ms is not None and isinstance(ts_ms, (int, float)):
        try:
            dt = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc)
            record["published_at"] = dt.strftime("%Y-%m-%d")
            record["updated_at"] = dt.strftime("%Y-%m-%d")
        except (OSError, ValueError):
            record["published_at"] = ""
            record["updated_at"] = ""
    else:
        record["published_at"] = ""
        record["updated_at"] = ""

    return record


# ---------------------------------------------------------------------------
# 网易 → 标准映射
# ---------------------------------------------------------------------------

def _transform_netease(raw: dict[str, Any]) -> dict[str, Any]:
    """
    将网易招聘原始岗位字典转换为标准中间格式。

    映射关系：
        source       → "netease"
        original_id  → raw["id"]
        title_raw    → raw["name"]
        company      → "网易"（硬编码）
        department   → raw["firstDepName"]
        category     → raw["firstPostTypeName"]
        sub_category → ""（网易无细类）
        city_raw     → workPlaceNameList[0] 或逗号拼接
        district     → ""（网易接口无行政区）
        salary_raw   → ""（网易接口不返回薪资）
        experience_raw → raw["reqWorkYearsName"]
        degree_raw   → raw["reqEducationName"]
        work_type    → "全职" / "实习"（按 workType 映射）
        duty         → raw["description"]
        requirement  → raw["requirement"]
        skills       → ""（网易无标签字段）
        post_url     → f"https://hr.163.com/job-detail.html?id={id}"

    Parameters
    ----------
    raw : dict
        网易招聘 API 返回的单条原始岗位字典（data.list[] 元素）

    Returns
    -------
    dict
        标准中间字典（字段全部非 None）
    """
    from datetime import datetime, timezone

    record = _empty_record(source="netease")

    # ---- 基础标识 ----
    job_id = raw.get("id")
    record["original_id"] = str(job_id).strip() if job_id is not None else ""

    # ---- 岗位名称 ----
    record["title_raw"] = str(raw.get("name", "")).strip()

    # ---- 公司（硬编码） ----
    record["company"] = "网易"

    # ---- 部门 ----
    record["department"] = str(raw.get("firstDepName", "")).strip()

    # ---- 职位类别 ----
    record["category"] = str(raw.get("firstPostTypeName", "")).strip()
    record["sub_category"] = ""           # 网易无细类

    # ---- 城市（从 workPlaceNameList 提取） ----
    wp_list = raw.get("workPlaceNameList", [])
    if isinstance(wp_list, list) and wp_list:
        # 优先取第一个城市，多个城市时逗号拼接
        record["city_raw"] = ", ".join(str(c) for c in wp_list if c)
        record["district"] = ""
    else:
        record["city_raw"] = ""
        record["district"] = ""

    # ---- 薪资（网易接口不返回薪资，留空给 Cleaner 处理） ----
    record["salary_raw"] = ""

    # ---- 经验要求 ----
    record["experience_raw"] = str(raw.get("reqWorkYearsName", "")).strip()

    # ---- 学历要求 ----
    record["degree_raw"] = str(raw.get("reqEducationName", "")).strip()

    # ---- 工作类型（workType: "0"=全职, "1"=实习） ----
    wt = str(raw.get("workType", "")).strip()
    record["work_type"] = {"0": "全职", "1": "实习"}.get(wt, "")

    # ---- 文本内容 ----
    record["duty"] = str(raw.get("description", "")).strip()
    record["requirement"] = str(raw.get("requirement", "")).strip()

    # ---- 技能标签（网易无此字段） ----
    record["skills"] = ""

    # ---- 链接 ----
    if record["original_id"]:
        record["post_url"] = (
            f"https://hr.163.com/job-detail.html?id={record['original_id']}"
        )

    # ---- 时间（updateTime 为毫秒时间戳） ----
    ts_ms = raw.get("updateTime")
    if ts_ms is not None and isinstance(ts_ms, (int, float)):
        try:
            dt = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc)
            record["published_at"] = dt.strftime("%Y-%m-%d")
            record["updated_at"] = dt.strftime("%Y-%m-%d")
        except (OSError, ValueError):
            record["published_at"] = ""
            record["updated_at"] = ""
    else:
        record["published_at"] = ""
        record["updated_at"] = ""

    return record


# ---------------------------------------------------------------------------
# 转换路由表（扩展大厂时只需在此注册）
# ---------------------------------------------------------------------------

_TRANSFORMERS: dict[str, Any] = {
    "netease": _transform_netease,
    "tencent": _transform_tencent,
    "bytedance": _transform_bytedance,
    "didi": _transform_didi,
    "lilith": _transform_lilith,   # [已废弃]
    "mihoyo": _transform_mihoyo,   # [已废弃]
}

_VALID_SOURCES: set[str] = set(_TRANSFORMERS.keys())


# ---------------------------------------------------------------------------
# 主入口：批量转换
# ---------------------------------------------------------------------------

def transform_jobs(raw_list: list[dict], source: str) -> list[dict]:
    """
    批量将原始岗位字典转换为标准中间格式。

    策略路由
    --------
        source="netease" → _transform_netease()
        source="tencent"  → _transform_tencent()
        source="lilith"   → _transform_lilith()（已废弃）
        source="mihoyo"   → _transform_mihoyo()（已废弃）

    容错机制
    --------
        遍历转换时，单条记录转换失败不会中断整个批次。
        try-except 捕获异常后，记录 error 日志的源、索引和异常信息，然后跳过该条。

    Parameters
    ----------
    raw_list : list[dict]
        原始爬虫返回的岗位字典列表（网易或米哈游格式）
    source : str
        数据来源标识，必须是 "netease" 或 "mihoyo"

    Returns
    -------
    list[dict]
        标准中间字典列表（失败记录已跳过，可能比 raw_list 短）

    Raises
    ------
    ValueError
        传入的 source 不在支持列表中
    """
    # ---- 参数校验 ----
    source = str(source).strip().lower()
    if source not in _VALID_SOURCES:
        raise ValueError(
            f"不支持的 source 参数: '{source}'，"
            f"允许的值: {sorted(_VALID_SOURCES)}"
        )

    transformer = _TRANSFORMERS[source]

    logger.info(
        "[transformer] 开始转换，source=%s，原始量=%d",
        source, len(raw_list)
    )

    transformed: list[dict] = []
    success_count = 0
    failure_count = 0

    for idx, raw in enumerate(raw_list):
        try:
            record = transformer(raw)
            # 只保留至少有 original_id 的记录
            if record["original_id"]:
                transformed.append(record)
                success_count += 1
            else:
                logger.warning(
                    "[transformer] 索引 %d 转换后 original_id 为空，已跳过。"
                    " 原始数据 keys: %s",
                    idx, list(raw.keys()) if isinstance(raw, dict) else type(raw)
                )
                failure_count += 1
        except Exception as e:
            failure_count += 1
            logger.error(
                "[transformer] 索引 %d 转换失败（已跳过）: %s",
                idx, e,
                exc_info=False,   # 不打完整堆栈，避免日志爆炸
            )

    logger.info(
        "[transformer] 转换完成 | 成功: %d | 失败/跳过: %d | 总计: %d",
        success_count, failure_count, len(transformed)
    )

    return transformed
