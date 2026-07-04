# -*- coding: utf-8 -*-
"""
utils/auth_db.py
================
用户认证数据库层 —— 建表、注册、登录验证。

核心功能
--------
    1. init_auth_tables()      —— 建表（users）
    2. register_user()         —— 用户注册（PBKDF2-SHA256 哈希）
    3. verify_login()          —— 验证登录
    4. get_user_by_id()        —— 按 ID 查询用户

安全
----
    - 密码使用 PBKDF2-SHA256 + 随机 Salt 存储，迭代 100000 次
    - 所有 SQL 使用参数化查询，防注入
    - 永不明文存储密码

Author: HireInsight-Agent
"""

from __future__ import annotations

import hashlib
import os
import secrets
import sqlite3
from datetime import datetime
from typing import Optional

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------

_DEFAULT_DB_PATH: str = "data/hireinsight.db"

_ITERATIONS: int = 100_000

# ---------------------------------------------------------------------------
# DDL 模板
# ---------------------------------------------------------------------------

_CREATE_USERS_TABLE_SQL: str = """
CREATE TABLE IF NOT EXISTS users (
    id           TEXT PRIMARY KEY NOT NULL,
    username     TEXT UNIQUE NOT NULL,
    password_hash TEXT NOT NULL,
    salt         TEXT NOT NULL,
    display_name TEXT,
    avatar_base64 TEXT,
    grade        TEXT,
    created_at   TEXT NOT NULL
)
"""

_CREATE_USERS_USERNAME_INDEX_SQL: str = """
CREATE UNIQUE INDEX IF NOT EXISTS idx_users_username
ON users (username)
"""


# ---------------------------------------------------------------------------
# 1. 建表（幂等）
# ---------------------------------------------------------------------------

def init_auth_tables(db_path: str = _DEFAULT_DB_PATH) -> None:
    """
    初始化用户认证表（含自动迁移）。

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
        cursor.execute(_CREATE_USERS_TABLE_SQL)
        cursor.execute(_CREATE_USERS_USERNAME_INDEX_SQL)
        conn.commit()

        # ---- 自动迁移：为旧表补全 avatar_base64 / grade 列 ----
        _migrate_add_column(cursor, conn, "users", "avatar_base64", "TEXT")
        _migrate_add_column(cursor, conn, "users", "grade", "TEXT")
    finally:
        conn.close()


def _migrate_add_column(
    cursor: sqlite3.Cursor, conn: sqlite3.Connection,
    table: str, column: str, col_type: str,
) -> None:
    """安全添加列（幂等）；若列已存在则静默跳过。"""
    try:
        cursor.execute(f"ALTER TABLE {table} ADD COLUMN {column} {col_type}")
        conn.commit()
    except sqlite3.OperationalError:
        pass  # 列已存在


# ---------------------------------------------------------------------------
# 2. 密码哈希
# ---------------------------------------------------------------------------

def _hash_password(password: str, salt: bytes | None = None) -> tuple[str, bytes]:
    """
    PBKDF2-SHA256 哈希密码。

    Parameters
    ----------
    password : str
        明文密码
    salt : bytes | None
        可选的 Salt，不传则随机生成

    Returns
    -------
    tuple[str, bytes]
        (hex_hash, salt)
    """
    if salt is None:
        salt = secrets.token_bytes(16)
    pwd_bytes = password.encode("utf-8")
    hash_bytes = hashlib.pbkdf2_hmac("sha256", pwd_bytes, salt, _ITERATIONS)
    hex_hash = hash_bytes.hex()
    return hex_hash, salt


# ---------------------------------------------------------------------------
# 3. 用户注册
# ---------------------------------------------------------------------------

def register_user(
    username: str,
    password: str,
    display_name: str,
    db_path: str = _DEFAULT_DB_PATH,
) -> tuple[bool, str]:
    """
    注册用户。

    Parameters
    ----------
    username : str
        用户名（3-32 字符，字母数字下划线）
    password : str
        密码（至少 6 位）
    display_name : str
        显示名称
    db_path : str
        数据库文件路径

    Returns
    -------
    tuple[bool, str]
        (是否成功, 提示信息)。成功时返回 (True, user_id)，失败返回 (False, 错误信息)
    """
    # ---- 校验 ----
    username = username.strip().lower()
    display_name = display_name.strip()

    if not username or len(username) < 3 or len(username) > 32:
        return False, "用户名长度需在 3-32 字符之间"
    if not all(c.isalnum() or c == "_" for c in username):
        return False, "用户名只能包含字母、数字和下划线"
    if not password or len(password) < 6:
        return False, "密码至少 6 位"
    if not display_name:
        return False, "请输入显示名称"

    # ---- 哈希 ----
    hex_hash, salt = _hash_password(password)

    # ---- 写入 ----
    user_id = secrets.token_hex(16)
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    try:
        cursor.execute(
            """
            INSERT INTO users (id, username, password_hash, salt, display_name, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (user_id, username, hex_hash, salt.hex(), display_name, now),
        )
        conn.commit()
        return True, user_id
    except sqlite3.IntegrityError:
        conn.rollback()
        return False, "用户名已存在"
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# 4. 登录验证
# ---------------------------------------------------------------------------

def verify_login(
    username: str,
    password: str,
    db_path: str = _DEFAULT_DB_PATH,
) -> dict | None:
    """
    验证登录。

    Parameters
    ----------
    username : str
        用户名
    password : str
        明文密码
    db_path : str
        数据库文件路径

    Returns
    -------
    dict | None
        成功返回用户信息字典 {"id": ..., "username": ..., "display_name": ...}，失败返回 None
    """
    username = username.strip().lower()

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    try:
        cursor.execute(
            "SELECT id, username, password_hash, salt, display_name, avatar_base64, grade "
            "FROM users WHERE username = ?",
            (username,),
        )
        row = cursor.fetchone()
        if row is None:
            return None

        user_id, db_username, stored_hash, salt_hex, display_name, avatar_base64, grade = row

        # 验证密码
        salt = bytes.fromhex(salt_hex)
        hex_hash, _ = _hash_password(password, salt)

        if hex_hash != stored_hash:
            return None

        return {
            "id": user_id,
            "username": db_username,
            "display_name": display_name,
            "avatar_base64": avatar_base64,
            "grade": grade,
        }
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# 5. 按 ID 查询用户
# ---------------------------------------------------------------------------

def get_user_by_id(user_id: str, db_path: str = _DEFAULT_DB_PATH) -> dict | None:
    """
    按用户 ID 查询用户信息。

    Parameters
    ----------
    user_id : str
        用户 ID
    db_path : str
        数据库文件路径

    Returns
    -------
    dict | None
        用户信息字典，不存在返回 None
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    try:
        cursor.execute(
            "SELECT id, username, display_name, avatar_base64, grade FROM users WHERE id = ?",
            (user_id,),
        )
        row = cursor.fetchone()
        if row is None:
            return None
        return {
            "id": row[0],
            "username": row[1],
            "display_name": row[2],
            "avatar_base64": row[3],
            "grade": row[4],
        }
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# 6. 更新用户资料
# ---------------------------------------------------------------------------

def update_user_profile(
    user_id: str,
    display_name: str | None = None,
    grade: str | None = None,
    avatar_base64: str | None = None,
    db_path: str = _DEFAULT_DB_PATH,
) -> tuple[bool, str]:
    """
    更新用户资料（显示名称、年级、头像）。
    只更新传入的非 None 字段。

    Parameters
    ----------
    user_id : str
        用户 ID
    display_name : str | None
        新的显示名称，None 表示不更新
    grade : str | None
        年级，None 表示不更新
    avatar_base64 : str | None
        头像 Base64，None 表示不更新
    db_path : str
        数据库文件路径

    Returns
    -------
    tuple[bool, str]
        (是否成功, 提示信息)
    """
    set_clauses: list[str] = []
    params: list[str] = []

    if display_name is not None:
        display_name = display_name.strip()
        if not display_name:
            return False, "显示名称不能为空"
        set_clauses.append("display_name = ?")
        params.append(display_name)

    if grade is not None:
        set_clauses.append("grade = ?")
        params.append(grade)

    if avatar_base64 is not None:
        set_clauses.append("avatar_base64 = ?")
        params.append(avatar_base64)

    if not set_clauses:
        return True, "没有需要更新的字段"

    params.append(user_id)

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    try:
        cursor.execute(
            f"UPDATE users SET {', '.join(set_clauses)} WHERE id = ?",
            params,
        )
        conn.commit()
        return cursor.rowcount > 0, "更新成功" if cursor.rowcount > 0 else "用户不存在"
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# 本地测试
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import tempfile

    test_db = os.path.join(tempfile.gettempdir(), "auth_test.db")

    print("=" * 60)
    print("1. 初始化认证表")
    print("=" * 60)
    init_auth_tables(test_db)
    print("   表已创建")

    print("\n" + "=" * 60)
    print("2. 注册测试用户")
    print("=" * 60)
    ok, msg = register_user("testuser", "password123", "测试用户", test_db)
    print(f"   注册: {ok}, info: {msg}")

    # 重复注册
    ok2, msg2 = register_user("testuser", "password123", "测试用户", test_db)
    print(f"   重复注册: {ok2}, info: {msg2}")

    print("\n" + "=" * 60)
    print("3. 登录验证")
    print("=" * 60)
    user = verify_login("testuser", "password123", test_db)
    print(f"   正确密码: {user is not None}")

    user2 = verify_login("testuser", "wrongpass", test_db)
    print(f"   错误密码: {user2 is not None}")

    user3 = verify_login("nonexistent", "password123", test_db)
    print(f"   不存在的用户: {user3 is not None}")

    print("\n" + "=" * 60)
    print("4. 按 ID 查询")
    print("=" * 60)
    if user:
        found = get_user_by_id(user["id"], test_db)
        print(f"   查询结果: {found}")

    os.remove(test_db)
    print("\n测试完成，临时库已清理。")
