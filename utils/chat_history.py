# -*- coding: utf-8 -*-
"""
utils/chat_history.py
=====================
对话历史数据库层 —— 面试模拟和灯塔计划的交互记录持久化。

核心功能
--------
    1. init_history_tables()          —— 建表（interview_history + lighthouse_history）
    2. save_interview_record()        —— 保存面试模拟记录
    3. save_lighthouse_record()       —— 保存灯塔计划记录
    4. get_interview_history()        —— 获取用户面试历史列表
    5. get_lighthouse_history()       —— 获取用户灯塔历史列表
    6. get_interview_record()         —— 获取单条面试记录详情
    7. get_lighthouse_record()        —— 获取单条灯塔记录详情
    8. delete_interview_record()      —— 删除面试记录（仅限本人）
    9. delete_lighthouse_record()     —— 删除灯塔记录（仅限本人）

数据契约
--------
    - 复杂字段（List[str], List[dict], Dict）使用 JSON TEXT 序列化存储
    - 所有查询严格按 user_id 过滤，确保多用户数据隔离
    - 删除操作同时校验 record_id 和 user_id，防止越权

Author: HireInsight-Agent
"""

from __future__ import annotations

import json
import os
import secrets
import sqlite3
from datetime import datetime
from typing import Optional

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------

_DEFAULT_DB_PATH: str = "data/hireinsight.db"


# ---------------------------------------------------------------------------
# DDL 模板
# ---------------------------------------------------------------------------

_CREATE_INTERVIEW_TABLE_SQL: str = """
CREATE TABLE IF NOT EXISTS interview_history (
    id               TEXT PRIMARY KEY NOT NULL,
    user_id          TEXT NOT NULL,
    target_position  TEXT NOT NULL,
    target_company   TEXT,
    user_resume      TEXT,
    market_report    TEXT,
    gap_analysis     TEXT,
    interview_questions TEXT,
    created_at       TEXT NOT NULL
)
"""

_CREATE_INTERVIEW_INDEX_SQL: str = """
CREATE INDEX IF NOT EXISTS idx_interview_user_id
ON interview_history (user_id, created_at DESC)
"""

_CREATE_LIGHTHOUSE_TABLE_SQL: str = """
CREATE TABLE IF NOT EXISTS lighthouse_history (
    id               TEXT PRIMARY KEY NOT NULL,
    user_id          TEXT NOT NULL,
    target_position  TEXT NOT NULL,
    grade            TEXT NOT NULL,
    user_answers     TEXT,
    tech_tendency    TEXT,
    roadmap          TEXT,
    created_at       TEXT NOT NULL
)
"""

_CREATE_LIGHTHOUSE_INDEX_SQL: str = """
CREATE INDEX IF NOT EXISTS idx_lighthouse_user_id
ON lighthouse_history (user_id, created_at DESC)
"""


# ---------------------------------------------------------------------------
# 1. 建表（幂等）
# ---------------------------------------------------------------------------

def init_history_tables(db_path: str = _DEFAULT_DB_PATH) -> None:
    """
    初始化对话历史表。

    Parameters
    ----------
    db_path : str
        数据库文件路径，默认 data/hireinsight.db
    """
    db_dir = os.path.dirname(db_path)
    if db_dir and not os.path.exists(db_dir):
        os.makedirs(db_dir, exist_ok=True)

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    try:
        cursor.execute(_CREATE_INTERVIEW_TABLE_SQL)
        cursor.execute(_CREATE_INTERVIEW_INDEX_SQL)
        cursor.execute(_CREATE_LIGHTHOUSE_TABLE_SQL)
        cursor.execute(_CREATE_LIGHTHOUSE_INDEX_SQL)
        conn.commit()
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# 2. 保存面试模拟记录
# ---------------------------------------------------------------------------

def save_interview_record(
    user_id: str,
    target_position: str,
    target_company: str | None,
    user_resume: str,
    market_report: str | None,
    gap_analysis: str | None,
    interview_questions: list[str] | None,
    db_path: str = _DEFAULT_DB_PATH,
) -> str:
    """
    保存面试模拟记录。

    Parameters
    ----------
    user_id : str
        用户 ID
    target_position : str
        目标岗位
    target_company : str | None
        目标公司
    user_resume : str
        简历文本
    market_report : str | None
        市场趋势报告
    gap_analysis : str | None
        技能差距诊断
    interview_questions : list[str] | None
        面试题列表
    db_path : str
        数据库文件路径

    Returns
    -------
    str
        记录 ID
    """
    record_id = secrets.token_hex(16)
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    try:
        cursor.execute(
            """
            INSERT INTO interview_history
            (id, user_id, target_position, target_company, user_resume,
             market_report, gap_analysis, interview_questions, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record_id,
                user_id,
                target_position,
                target_company,
                user_resume,
                market_report,
                gap_analysis,
                json.dumps(interview_questions, ensure_ascii=False) if interview_questions else None,
                now,
            ),
        )
        conn.commit()
        return record_id
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# 3. 保存灯塔计划记录
# ---------------------------------------------------------------------------

def save_lighthouse_record(
    user_id: str,
    target_position: str,
    grade: str,
    user_answers: list[dict] | None,
    tech_tendency: dict | None,
    roadmap: str | None,
    db_path: str = _DEFAULT_DB_PATH,
) -> str:
    """
    保存灯塔计划记录。

    Parameters
    ----------
    user_id : str
        用户 ID
    target_position : str
        目标岗位
    grade : str
        年级
    user_answers : list[dict] | None
        用户答题记录
    tech_tendency : dict | None
        技术倾向分值
    roadmap : str | None
        Markdown 路线图
    db_path : str
        数据库文件路径

    Returns
    -------
    str
        记录 ID
    """
    record_id = secrets.token_hex(16)
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    try:
        cursor.execute(
            """
            INSERT INTO lighthouse_history
            (id, user_id, target_position, grade, user_answers,
             tech_tendency, roadmap, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record_id,
                user_id,
                target_position,
                grade,
                json.dumps(user_answers, ensure_ascii=False) if user_answers else None,
                json.dumps(tech_tendency, ensure_ascii=False) if tech_tendency else None,
                roadmap,
                now,
            ),
        )
        conn.commit()
        return record_id
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# 4. 获取面试历史列表
# ---------------------------------------------------------------------------

def get_interview_history(
    user_id: str,
    db_path: str = _DEFAULT_DB_PATH,
    limit: int = 20,
) -> list[dict]:
    """
    获取用户的面试模拟历史列表。

    Parameters
    ----------
    user_id : str
        用户 ID
    db_path : str
        数据库文件路径
    limit : int
        返回条数上限，默认 20

    Returns
    -------
    list[dict]
        历史记录列表，按时间倒序排列
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    try:
        cursor.execute(
            """
            SELECT id, target_position, target_company, created_at
            FROM interview_history
            WHERE user_id = ?
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (user_id, limit),
        )
        rows = cursor.fetchall()
        return [
            {
                "id": row[0],
                "target_position": row[1],
                "target_company": row[2],
                "created_at": row[3],
            }
            for row in rows
        ]
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# 5. 获取灯塔历史列表
# ---------------------------------------------------------------------------

def get_lighthouse_history(
    user_id: str,
    db_path: str = _DEFAULT_DB_PATH,
    limit: int = 20,
) -> list[dict]:
    """
    获取用户的灯塔计划历史列表。

    Parameters
    ----------
    user_id : str
        用户 ID
    db_path : str
        数据库文件路径
    limit : int
        返回条数上限，默认 20

    Returns
    -------
    list[dict]
        历史记录列表，按时间倒序排列
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    try:
        cursor.execute(
            """
            SELECT id, target_position, grade, created_at
            FROM lighthouse_history
            WHERE user_id = ?
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (user_id, limit),
        )
        rows = cursor.fetchall()
        return [
            {
                "id": row[0],
                "target_position": row[1],
                "grade": row[2],
                "created_at": row[3],
            }
            for row in rows
        ]
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# 6. 获取单条面试记录详情
# ---------------------------------------------------------------------------

def get_interview_record(
    record_id: str,
    user_id: str,
    db_path: str = _DEFAULT_DB_PATH,
) -> dict | None:
    """
    获取单条面试记录详情（含完整内容）。

    Parameters
    ----------
    record_id : str
        记录 ID
    user_id : str
        用户 ID（用于权限校验）
    db_path : str
        数据库文件路径

    Returns
    -------
    dict | None
        记录详情字典，不存在或无权访问返回 None
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    try:
        cursor.execute(
            """
            SELECT id, user_id, target_position, target_company, user_resume,
                   market_report, gap_analysis, interview_questions, created_at
            FROM interview_history
            WHERE id = ? AND user_id = ?
            """,
            (record_id, user_id),
        )
        row = cursor.fetchone()
        if row is None:
            return None

        questions = row[7]
        return {
            "id": row[0],
            "user_id": row[1],
            "target_position": row[2],
            "target_company": row[3],
            "user_resume": row[4],
            "market_report": row[5],
            "gap_analysis": row[6],
            "interview_questions": json.loads(questions) if questions else None,
            "created_at": row[8],
        }
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# 7. 获取单条灯塔记录详情
# ---------------------------------------------------------------------------

def get_lighthouse_record(
    record_id: str,
    user_id: str,
    db_path: str = _DEFAULT_DB_PATH,
) -> dict | None:
    """
    获取单条灯塔计划记录详情（含完整内容）。

    Parameters
    ----------
    record_id : str
        记录 ID
    user_id : str
        用户 ID（用于权限校验）
    db_path : str
        数据库文件路径

    Returns
    -------
    dict | None
        记录详情字典，不存在或无权访问返回 None
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    try:
        cursor.execute(
            """
            SELECT id, user_id, target_position, grade, user_answers,
                   tech_tendency, roadmap, created_at
            FROM lighthouse_history
            WHERE id = ? AND user_id = ?
            """,
            (record_id, user_id),
        )
        row = cursor.fetchone()
        if row is None:
            return None

        answers = row[4]
        tendency = row[5]
        return {
            "id": row[0],
            "user_id": row[1],
            "target_position": row[2],
            "grade": row[3],
            "user_answers": json.loads(answers) if answers else None,
            "tech_tendency": json.loads(tendency) if tendency else None,
            "roadmap": row[6],
            "created_at": row[7],
        }
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# 8. 删除面试记录
# ---------------------------------------------------------------------------

def delete_interview_record(
    record_id: str,
    user_id: str,
    db_path: str = _DEFAULT_DB_PATH,
) -> bool:
    """
    删除指定面试记录（仅限本人）。

    Parameters
    ----------
    record_id : str
        记录 ID
    user_id : str
        用户 ID
    db_path : str
        数据库文件路径

    Returns
    -------
    bool
        是否删除成功
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    try:
        cursor.execute(
            "DELETE FROM interview_history WHERE id = ? AND user_id = ?",
            (record_id, user_id),
        )
        conn.commit()
        return cursor.rowcount > 0
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# 9. 删除灯塔记录
# ---------------------------------------------------------------------------

def delete_lighthouse_record(
    record_id: str,
    user_id: str,
    db_path: str = _DEFAULT_DB_PATH,
) -> bool:
    """
    删除指定灯塔记录（仅限本人）。

    Parameters
    ----------
    record_id : str
        记录 ID
    user_id : str
        用户 ID
    db_path : str
        数据库文件路径

    Returns
    -------
    bool
        是否删除成功
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    try:
        cursor.execute(
            "DELETE FROM lighthouse_history WHERE id = ? AND user_id = ?",
            (record_id, user_id),
        )
        conn.commit()
        return cursor.rowcount > 0
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# 本地测试
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import tempfile

    test_db = os.path.join(tempfile.gettempdir(), "history_test.db")

    print("=" * 60)
    print("1. 初始化历史表")
    print("=" * 60)
    init_history_tables(test_db)
    print("   表已创建")

    print("\n" + "=" * 60)
    print("2. 保存面试记录")
    print("=" * 60)
    rid = save_interview_record(
        user_id="u001",
        target_position="Python后端开发",
        target_company="字节跳动",
        user_resume="5年Python经验...",
        market_report="市场趋势报告...",
        gap_analysis="技能差距分析...",
        interview_questions=["请介绍Python的GIL", "Python装饰器原理"],
        db_path=test_db,
    )
    print(f"   记录 ID: {rid}")

    print("\n" + "=" * 60)
    print("3. 保存灯塔记录")
    print("=" * 60)
    rid2 = save_lighthouse_record(
        user_id="u001",
        target_position="Java后端开发",
        grade="大三",
        user_answers=[{"question_id": 1, "tendency": "backend"}],
        tech_tendency={"backend": 80, "frontend": 20},
        roadmap="## 学习路线图...",
        db_path=test_db,
    )
    print(f"   记录 ID: {rid2}")

    print("\n" + "=" * 60)
    print("4. 查询历史列表")
    print("=" * 60)
    ih = get_interview_history("u001", test_db)
    print(f"   面试历史: {len(ih)} 条")
    lh = get_lighthouse_history("u001", test_db)
    print(f"   灯塔历史: {len(lh)} 条")

    print("\n" + "=" * 60)
    print("5. 查询详情")
    print("=" * 60)
    if ih:
        detail = get_interview_record(ih[0]["id"], "u001", test_db)
        print(f"   面试详情 questions: {len(detail['interview_questions']) if detail and detail['interview_questions'] else 0}")

    print("\n" + "=" * 60)
    print("6. 删除记录")
    print("=" * 60)
    if ih:
        ok = delete_interview_record(ih[0]["id"], "u001", test_db)
        print(f"   删除面试记录: {ok}")
    ok2 = delete_lighthouse_record(rid2, "u001", test_db)
    print(f"   删除灯塔记录: {ok2}")

    os.remove(test_db)
    print("\n测试完成，临时库已清理。")
