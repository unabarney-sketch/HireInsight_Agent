# -*- coding: utf-8 -*-
"""
utils/session_manager.py
=======================
会话管理器 —— 封装 Streamlit session_state 中的登录态管理。

核心功能
--------
    1. init_auth()           —— 初始化认证表（应用启动时调用一次）
    2. init_history()        —— 初始化历史表（应用启动时调用一次）
    3. is_logged_in()        —— 检查当前是否已登录
    4. get_current_user()    —— 获取当前登录用户信息
    5. login_user()          —— 写入登录态到 session_state
    6. logout_user()         —— 清除登录态

依赖
----
    - utils.auth_db
    - utils.chat_history

Author: HireInsight-Agent
"""

from __future__ import annotations

import streamlit as st

from .auth_db import init_auth_tables
from .chat_history import init_history_tables


# ---------------------------------------------------------------------------
# 1. 初始化认证和历史表（应用启动时调用）
# ---------------------------------------------------------------------------

def init_auth(db_path: str = "data/hireinsight.db") -> None:
    """
    初始化认证表。

    Parameters
    ----------
    db_path : str
        数据库文件路径
    """
    init_auth_tables(db_path)


def init_history(db_path: str = "data/hireinsight.db") -> None:
    """
    初始化历史记录表。

    Parameters
    ----------
    db_path : str
        数据库文件路径
    """
    init_history_tables(db_path)


# ---------------------------------------------------------------------------
# 2. 登录态检查
# ---------------------------------------------------------------------------

def is_logged_in() -> bool:
    """
    检查当前用户是否已登录。

    Returns
    -------
    bool
    """
    return st.session_state.get("user_id") is not None


# ---------------------------------------------------------------------------
# 3. 获取当前用户信息
# ---------------------------------------------------------------------------

def get_current_user() -> dict | None:
    """
    获取当前登录用户信息。

    Returns
    -------
    dict | None
        {"id": ..., "username": ..., "display_name": ..., "avatar_base64": ..., "grade": ...} 或 None
    """
    user_id = st.session_state.get("user_id")
    if not user_id:
        return None
    return {
        "id": user_id,
        "username": st.session_state.get("username", ""),
        "display_name": st.session_state.get("display_name", ""),
        "avatar_base64": st.session_state.get("avatar_base64"),
        "grade": st.session_state.get("grade"),
    }


# ---------------------------------------------------------------------------
# 4. 写入登录态
# ---------------------------------------------------------------------------

def login_user(user_info: dict) -> None:
    """
    写入登录态到 session_state。

    Parameters
    ----------
    user_info : dict
        包含 id, username, display_name, avatar_base64, grade 的用户信息字典
    """
    st.session_state["user_id"] = user_info["id"]
    st.session_state["username"] = user_info["username"]
    st.session_state["display_name"] = user_info["display_name"]
    st.session_state["avatar_base64"] = user_info.get("avatar_base64")
    st.session_state["grade"] = user_info.get("grade")


# ---------------------------------------------------------------------------
# 4. 刷新当前用户信息（资料修改后调用）
# ---------------------------------------------------------------------------

def refresh_user_session(db_path: str = "data/hireinsight.db") -> None:
    """
    从数据库重新拉取当前用户信息并写入 session_state。
    用于资料保存后即时刷新名片。

    Parameters
    ----------
    db_path : str
        数据库文件路径
    """
    from .auth_db import get_user_by_id as _get_user

    user_id = st.session_state.get("user_id")
    if not user_id:
        return
    user = _get_user(user_id, db_path)
    if user:
        login_user(user)


# ---------------------------------------------------------------------------
# 5. 清除登录态
# ---------------------------------------------------------------------------

def logout_user() -> None:
    """
    清除登录态，完全重置所有页面相关的 session_state。

    注意：函数末尾强制调用 st.rerun()，确保页面瞬间跳转回登录墙，
    避免用户点击注销后需要再点一次按钮才会刷新的卡顿感。
    """
    # 清除登录态
    for key in ["user_id", "username", "display_name", "avatar_base64", "grade"]:
        if key in st.session_state:
            del st.session_state[key]

    # 清除页面相关状态
    for key in [
        "current_page",
        "sidebar_nav",
        "dashboard_filters",
        "lighthouse_filtered_questions",
        "lighthouse_results",
        "lighthouse_quiz_started",
        "lighthouse_target_position",
        "lighthouse_grade",
        "interview_history_view",
        "lighthouse_history_view",
        "interview_record_detail",
        "lighthouse_record_detail",
        "profile_interview_detail",
        "profile_lighthouse_detail",
        "profile_avatar_processed",
    ]:
        if key in st.session_state:
            del st.session_state[key]

    # 强制刷新页面，瞬间跳转到登录墙
    st.rerun()
