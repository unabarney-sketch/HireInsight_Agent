"""
HireInsight-Agent - Streamlit 主程序入口

智能求职辅助系统，面向校招/社招场景

三大核心模块：
1. 灯塔计划（低年级定向）- 职业规划
2. 数据大屏（岗位数据分析）
3. 面试模拟（LangGraph + RAG）

技术栈：Streamlit + LangGraph + DeepSeek API + RAG (TF-IDF)
"""
import os
import sys
import base64
import streamlit as st
from dotenv import load_dotenv
import plotly.graph_objects as go
from utils.data_persistence import (
    get_distinct_values,
    query_jobs_by_filters,
    load_from_sqlite,
    init_sqlite_db,
    save_to_sqlite
)
from utils.data_stats import calculate_market_metrics
from utils.data_transformer import transform_jobs
from utils.data_cleaner import clean_job_data
from utils.pdf_parser import extract_text_from_pdf_bytes
import pandas as pd

# LangGraph 工作流导入
from graphs.interview_graph import run_interview_workflow
from graphs.state import InterviewState, LighthousePlanState
from graphs.lighthouse_nodes import question_filter_node, assessment_node
from utils.question_bank import get_full_question_bank, get_valid_tendencies

# 用户认证与历史记录
from utils.auth_db import register_user, verify_login, update_user_profile
from utils.chat_history import (
    save_interview_record,
    save_lighthouse_record,
    get_interview_history,
    get_lighthouse_history,
    get_interview_record,
    get_lighthouse_record,
    delete_interview_record,
    delete_lighthouse_record,
)
from utils.session_manager import (
    init_auth,
    init_history,
    is_logged_in,
    get_current_user,
    login_user,
    logout_user,
    refresh_user_session,
)

# RAG 向量存储单例导入
try:
    from utils.rag_loader import get_or_init_collection as _get_or_init_collection
except ImportError:
    _get_or_init_collection = None

# 加载环境变量
load_dotenv()

# ============================================================
# Banner 图片绝对路径（防 Streamlit 相对路径上下文对齐失败）
# ============================================================
_CURRENT_DIR: str = os.path.dirname(os.path.abspath(__file__))

def _resolve_banner_path(filename: str) -> str | None:
    """在 assets/ 下自动匹配 .jpg/.png 后缀，返回绝对路径或 None"""
    for ext in (".jpg", ".png"):
        path = os.path.join(_CURRENT_DIR, "assets", f"{filename}{ext}")
        if os.path.isfile(path):
            return path
    return None

INTERVIEW_BANNER_PATH: str | None = _resolve_banner_path("interview_banner")
LIGHTHOUSE_BANNER_PATH: str | None = _resolve_banner_path("lighthouse_banner")
HOME_BANNER_PATH: str | None = _resolve_banner_path("home_banner")
DASHBOARD_BANNER_PATH: str | None = _resolve_banner_path("dashboard_banner")

# ============================================================
# 数据库路径
# ============================================================
_DB_PATH: str = os.path.join("data", "hireinsight.db")

# ============================================================
# 全局页面路由映射（统一短名 ↔ Emoji 全名双向索引）
# ============================================================
PAGE_MAP: dict[str, str] = {
    "首页": "🏠 首页",
    "数据大屏": "📊 数据大屏",
    "灯塔计划": "🧭 灯塔计划",
    "面试模拟": "🎤 面试模拟",
    "我的": "👤 我的",
    "系统设置": "⚙️ 系统设置",
}
# 反向索引：Emoji 全名 → 短名
PAGE_REVERSE_MAP: dict[str, str] = {v: k for k, v in PAGE_MAP.items()}
# 有序列表（供 st.sidebar.radio 使用）
PAGE_OPTIONS: list[str] = list(PAGE_MAP.values())

# ============================================================
# 大屏缓存层（防重复查询 SQLite）
# ============================================================
@st.cache_data(ttl=300)
def _get_distinct_values_for_dashboard(column: str) -> list[str]:
    """获取去重值列表（5 分钟 TTL 缓存，供大屏侧边栏筛选用）。

    Parameters
    ----------
    column : str
        列名（city / degree / source），透传给 get_distinct_values。

    Returns
    -------
    list[str]
        去重值列表，数据库不存在或为空时返回空列表。
    """
    if not os.path.exists(_DB_PATH):
        return []
    return get_distinct_values(_DB_PATH, column)


# ============================================================
# 面试模拟缓存层（防高频 SQLite I/O）
# ============================================================
@st.cache_data(ttl=3600)
def _load_market_stats_for_interview(
    target_position: str
) -> dict | None:
    """加载市场数据统计字典（1 小时 TTL 缓存）。

    用于面试模拟模块的 Market_Agent，避免用户反复上传简历时
    每次都重读 SQLite 并执行 Pandas 分组聚合。

    Parameters
    ----------
    target_position : str
        目标岗位（当前未用于过滤，仅传递给下游）

    Returns
    -------
    dict | None
        calculate_market_metrics 返回的完整统计字典，数据库为空时返回 None
    """
    _ = target_position  # 保留接口，未来可按岗位筛选
    if not os.path.exists(_DB_PATH):
        return None
    try:
        df = load_from_sqlite(_DB_PATH)
        if df is None or df.empty:
            return None
        return calculate_market_metrics(source=df)
    except Exception:
        return None


# ============================================================
# 页面配置
# ============================================================
st.set_page_config(
    page_title="HireInsight Agent - 智能求职辅助系统",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# 全局 Creamy Clean 奶油风 CSS
# ============================================================
st.markdown("""
<style>
/* ---- 全局背景：温暖奶茶色 ---- */
.stApp {
    background-color: #F6F5F2;
}
section[data-testid="stSidebar"] > div {
    background-color: #F6F5F2;
}
header[data-testid="stHeader"] {
    background-color: #F6F5F2;
}

/* ---- 宽屏容器 ---- */
div[data-testid="stAppViewBlockContainer"] {
    max-width: 95% !important;
    padding-left: 2rem !important;
    padding-right: 2rem !important;
}

/* ---- 字体：PingFang 人文无衬线 ---- */
html, body, .stApp, section[data-testid="stSidebar"],
.stMarkdown, .stButton, .stTextInput, .stSelectbox, .stTextArea {
    font-family: "PingFang SC", "Helvetica Neue", "Microsoft YaHei", sans-serif !important;
}

/* ---- 卡片容器：纯白大圆角 ---- */
div[data-testid="stVerticalBlockBorderWrapper"] {
    border: 1px solid #EFEFEF !important;
    border-radius: 18px !important;
    background-color: #FFFFFF !important;
    padding: 24px 32px !important;
}

/* ---- 按钮：巧克力褐 + 悬停微浮 ---- */
.stButton > button,
.stDownloadButton > button {
    background-color: #2D2722 !important;
    border: none !important;
    color: #FFFFFF !important;
    border-radius: 18px !important;
    font-weight: 500 !important;
    padding: 10px 28px !important;
    transition: transform 0.25s ease, box-shadow 0.25s ease !important;
}
.stButton > button:hover,
.stDownloadButton > button:hover {
    transform: translateY(-3px) !important;
    box-shadow: 0 6px 18px rgba(45, 39, 34, 0.15) !important;
    background-color: #3E352E !important;
}

/* ---- 强制覆盖 Streamlit 内置 button[kind] 文字色（含内部 p 标签） ---- */
.stButton > button[kind="primary"],
.stButton > button[kind="primary"]:hover,
.stButton > button[kind="primary"]:focus,
.stButton > button[kind="primary"]:active,
.stButton > button[kind="secondary"],
.stButton > button[kind="secondary"]:hover,
.stButton > button[kind="secondary"]:focus,
.stButton > button[kind="secondary"]:active,
.stDownloadButton > button[kind="primary"],
.stDownloadButton > button[kind="primary"]:hover,
.stDownloadButton > button[kind="secondary"],
.stDownloadButton > button[kind="secondary"]:hover,
.stButton > button p,
.stButton > button span,
.stButton > button div p,
.stButton > button div span,
.stDownloadButton > button p,
.stDownloadButton > button span,
.stDownloadButton > button div p,
.stDownloadButton > button div span {
    color: #FFFFFF !important;
}

/* ---- 表单提交按钮 ---- */
.stFormSubmitButton > button {
    background-color: #2D2722 !important;
    border: none !important;
    color: #FFFFFF !important;
    border-radius: 18px !important;
    font-weight: 500 !important;
    transition: transform 0.25s ease, box-shadow 0.25s ease !important;
}
.stFormSubmitButton > button:hover {
    transform: translateY(-3px) !important;
    box-shadow: 0 6px 18px rgba(45, 39, 34, 0.15) !important;
    background-color: #3E352E !important;
}

/* ---- 输入框 ---- */
.stTextInput input,
.stTextArea textarea {
    background-color: #FFFFFF !important;
    border: 1px solid #EFEFEF !important;
    color: #4A4A4A !important;
    border-radius: 18px !important;
}
.stTextInput input:focus,
.stTextArea textarea:focus {
    border-color: #D3D3D3 !important;
    box-shadow: 0 0 0 3px rgba(45, 39, 34, 0.04) !important;
}

/* ---- Selectbox ---- */
.stSelectbox > div > div {
    background-color: #FFFFFF !important;
    border: 1px solid #EFEFEF !important;
    color: #4A4A4A !important;
    border-radius: 18px !important;
}

/* ---- File Uploader ---- */
.stFileUploader section {
    border: 1px dashed #D3D3D3 !important;
    background-color: #FFFFFF !important;
    border-radius: 18px !important;
}

/* ---- Expander ---- */
[data-testid="stExpander"] details {
    border: 1px solid #EFEFEF !important;
    background-color: #FFFFFF !important;
    border-radius: 18px !important;
}

/* ---- 标题：巧克力褐 ---- */
h1, h2, h3, h4, h5, h6 {
    color: #2D2722 !important;
    font-weight: 600 !important;
}
h1 {
    font-size: 2rem !important;
    letter-spacing: -0.02em !important;
}

/* ---- 普通文本：深炭灰 ---- */
p, span, label, .stMarkdown, .stTextInput label, .stSelectbox label, .stTextArea label {
    color: #4A4A4A !important;
}

/* ---- Sidebar ---- */
section[data-testid="stSidebar"] label {
    color: #7A7A7A !important;
}
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h3 {
    color: #2D2722 !important;
}
section[data-testid="stSidebar"] hr {
    border-color: #EFEFEF !important;
}

/* ---- Metrics ---- */
[data-testid="stMetricValue"] {
    color: #2D2722 !important;
    font-weight: 600 !important;
}
[data-testid="stMetricLabel"] {
    color: #7A7A7A !important;
}
[data-testid="stMetricDelta"] {
    color: #34C759 !important;
}

/* ---- Links ---- */
a {
    color: #5A8FC7 !important;
}

/* ---- Dividers ---- */
hr {
    border-color: #EFEFEF !important;
}

/* ---- Captions ---- */
.stCaption, small {
    color: #7A7A7A !important;
}

/* ---- Multiselect Tags ---- */
.stMultiSelect [data-baseweb="tag"] {
    border-radius: 10px !important;
    background-color: #F5F5F2 !important;
    border: 1px solid #EFEFEF !important;
    color: #4A4A4A !important;
}

/* ---- Slider ---- */
.stSlider [data-baseweb="slider"] div[role="slider"] {
    background-color: #2D2722 !important;
}

/* ---- Tabs ---- */
.stTabs [data-baseweb="tab-list"] {
    border-bottom: 1px solid #EFEFEF !important;
}

/* ---- Radio ---- */
.stRadio [data-baseweb="radio"] label span {
    color: #4A4A4A !important;
}

/* ---- Status Container ---- */
.stStatusWidget {
    border: 1px solid #EFEFEF !important;
    border-radius: 18px !important;
}

/* ---- Toast ---- */
div[data-testid="stToast"] {
    background-color: #FFFFFF !important;
    border: 1px solid #EFEFEF !important;
    border-radius: 18px !important;
}

/* ---- Code ---- */
code {
    color: #2D2722 !important;
    background-color: #F5F5F2 !important;
    border-radius: 6px !important;
}

/* ---- Alert boxes ---- */
div[data-testid="stAlert"] {
    border-radius: 18px !important;
}

/* ---- DataFrame / Table ---- */
.stDataFrame, .stDataFrame th, .stDataFrame td {
    border-color: #EFEFEF !important;
}

/* ---- 看板与侧边栏：绝对不跳动 ---- */
section[data-testid="stSidebar"], div[data-testid="stSidebarContent"] {
    transition: none !important;
}
section[data-testid="stSidebar"] * {
    transition: none !important;
}

/* ---- 组件间距 ---- */
.block-container {
    gap: 20px !important;
}

/* ---- 大屏内部虚线卡片（数据展示细格） ---- */
.creamy-inner-card {
    border: 1px dashed #D3D3D3 !important;
    border-radius: 14px !important;
    padding: 16px 20px !important;
    background-color: #FFFFFF !important;
}
</style>
""", unsafe_allow_html=True)

# ============================================================
# 环境变量检查
# ============================================================
def check_api_key():
    """检查 DeepSeek API Key 是否配置"""
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key or api_key == "your_api_key_here":
        return False
    return True


def render_api_key_warning():
    """渲染 API Key 未配置警告"""
    st.warning("⚠️ 请先配置 DeepSeek API Key", icon="🔑")
    st.code("复制 .env.example 为 .env，填入您的 API Key")
    st.markdown("""
    获取方式：
    1. 访问 [DeepSeek Platform](https://platform.deepseek.com/)
    2. 注册并获取 API Key
    3. 创建项目根目录的 `.env` 文件
    """)
    st.divider()


# ============================================================
# 登录/注册页面
# ============================================================
def render_login_page():
    """渲染登录/注册页面 —— Creamy Clean 奶油风"""
    # 登录页居中
    st.markdown("""
    <style>
    .login-wrapper {
        display: flex;
        justify-content: center;
        align-items: center;
        min-height: 80vh;
    }
    .login-card {
        background: #FFFFFF;
        border: 1px solid #EFEFEF;
        border-radius: 24px;
        padding: 48px 40px;
        max-width: 420px;
        width: 100%;
        box-shadow: 0 4px 24px rgba(45, 39, 34, 0.06);
    }
    </style>
    """, unsafe_allow_html=True)

    col_left, col_center, col_right = st.columns([1, 3, 1])
    with col_center:
        st.markdown(
            "<h1 style='text-align:center;font-size:2.4rem;font-weight:700;color:#2D2722;margin-bottom:4px;'>"
            "HireInsight</h1>",
            unsafe_allow_html=True,
        )
        st.markdown(
            "<p style='text-align:center;color:#7A7A7A;font-size:0.95rem;margin-bottom:32px;'>"
            "智能求职辅助系统</p>",
            unsafe_allow_html=True,
        )

        # 登录/注册切换标签
        tabs = st.tabs(["登录", "注册"])

        # ---------- 登录 ----------
        with tabs[0]:
            with st.container(border=False):
                login_username = st.text_input("用户名", placeholder="请输入用户名", key="login_username")
                login_password = st.text_input("密码", type="password", placeholder="请输入密码", key="login_password")
                st.markdown("")
                if st.button("登录", use_container_width=True, key="btn_login"):
                    if not login_username or not login_password:
                        st.error("请输入用户名和密码")
                    else:
                        user = verify_login(login_username, login_password, _DB_PATH)
                        if user:
                            login_user(user)
                            st.toast(f"欢迎回来，{user['display_name']}！", icon="🎉")
                            st.rerun()
                        else:
                            st.error("用户名或密码错误")

        # ---------- 注册 ----------
        with tabs[1]:
            with st.container(border=False):
                reg_username = st.text_input("用户名", placeholder="3-32 位字母/数字/下划线", key="reg_username")
                reg_display = st.text_input("显示名称", placeholder="你的昵称", key="reg_display")
                reg_password = st.text_input("密码", type="password", placeholder="至少 6 位", key="reg_password")
                reg_confirm = st.text_input("确认密码", type="password", placeholder="再次输入密码", key="reg_confirm")
                st.markdown("")
                if st.button("注册", use_container_width=True, key="btn_register"):
                    if not reg_username or not reg_password or not reg_display:
                        st.error("请填写所有必填项")
                    elif reg_password != reg_confirm:
                        st.error("两次输入的密码不一致")
                    else:
                        ok, msg = register_user(reg_username, reg_password, reg_display, _DB_PATH)
                        if ok:
                            st.success(f"注册成功！请登录")
                        else:
                            st.error(msg)


# ============================================================
# 侧边栏导航
# ============================================================
def render_sidebar():
    """渲染侧边栏导航（与 st.session_state 双向联动）"""
    st.sidebar.title("🎯 HireInsight")
    st.sidebar.markdown("---")

    # ---- 用户信息名片（奶油风大厂级）----
    user = get_current_user()
    if user:
        avatar_b64 = user.get("avatar_base64")
        # 头像 HTML
        if avatar_b64:
            avatar_html = (
                f'<img src="data:image/png;base64,{avatar_b64}" '
                f'style="width:60px;height:60px;border-radius:50%;object-fit:cover;'
                f'border:2px solid #EFEFEF;" />'
            )
        else:
            avatar_html = (
                '<div style="width:60px;height:60px;border-radius:50%;'
                'background:linear-gradient(135deg, #2D2722 0%, #5C4D42 100%);'
                'display:flex;align-items:center;justify-content:center;'
                'font-size:1.6rem;color:#FFFFFF;">👤</div>'
            )

        # 获取最近意向岗位
        recent_pos = "尚未设置"
        try:
            ihist = get_interview_history(user["id"], _DB_PATH, limit=1)
            if ihist:
                recent_pos = ihist[0]["target_position"]
            else:
                lhist = get_lighthouse_history(user["id"], _DB_PATH, limit=1)
                if lhist:
                    recent_pos = lhist[0]["target_position"]
        except Exception:
            pass

        user_grade = user.get("grade") or "尚未设置"

        st.sidebar.markdown(
            f"""<div style='background-color:#FFFFFF;border:1px solid #EFEFEF;
            border-radius:18px;padding:20px 16px;margin-bottom:16px;text-align:center;'>
            <div style='display:flex;justify-content:center;margin-bottom:12px;'>
                {avatar_html}
            </div>
            <div style='color:#2D2722;font-size:1rem;font-weight:600;'>
                {user['display_name'] or user['username']}
            </div>
            <div style='color:#A0A0A0;font-size:0.73rem;margin-bottom:8px;'>
                @{user['username']}
            </div>
            <div style='color:#7A7A7A;font-size:0.76rem;line-height:1.8;'>
                <div>🎯 {recent_pos}</div>
                <div>🎓 {user_grade}</div>
            </div>
            </div>""",
            unsafe_allow_html=True,
        )
    else:
        st.sidebar.markdown(
            """<div style='background-color:#FFFFFF;border:1px solid #EFEFEF;border-radius:18px;
            padding:16px 20px;margin-bottom:16px;font-size:0.88rem;'>
            <div style='color:#7A7A7A;font-size:0.8rem;'>未登录</div>
            </div>""",
            unsafe_allow_html=True,
        )

    # 根据 session_state 计算当前激活的 radio index，使侧边栏与按钮联动
    current_short = st.session_state.get("current_page", "首页")
    # 通过 PAGE_MAP 找到对应 Emoji 全名，若未匹配到则 fallback 到 0
    active_emoji = PAGE_MAP.get(current_short, PAGE_OPTIONS[0])
    try:
        active_index = PAGE_OPTIONS.index(active_emoji)
    except ValueError:
        active_index = 0

    # 功能模块选择（显式绑定 key 防止 Streamlit 内部自动重置 index）
    page = st.sidebar.radio(
        "选择功能模块",
        PAGE_OPTIONS,
        index=active_index,
        key="sidebar_nav",
    )

    st.sidebar.markdown("---")

    st.sidebar.markdown(
        "<div style='border:1px solid #EFEFEF;border-radius:18px;padding:14px 16px;"
        "font-size:0.85rem;line-height:2.0;color:#7A7A7A;background-color:#FFFFFF;'>"
        "<span style='color:#2D2722;font-weight:600;'>开发进度</span><br>"
        "<span style='color:#34C759;'>●</span> 数据大屏<br>"
        "<span style='color:#34C759;'>●</span> 灯塔计划<br>"
        "<span style='color:#34C759;'>●</span> 面试模拟"
        "</div>",
        unsafe_allow_html=True,
    )

    return page


# ============================================================
# 首页
# ============================================================
def render_home():
    """渲染首页 —— Creamy Clean 奶油风"""
    user = get_current_user()

    # ---- 顶部 Banner 大图（存在则视觉做减法，剔除所有文本标题与副标题）----
    if HOME_BANNER_PATH and os.path.isfile(HOME_BANNER_PATH):
        st.image(HOME_BANNER_PATH, use_container_width=True)
    else:
        st.markdown(
            f"<h1 style='font-weight:700;font-size:2.3rem;color:#2D2722;margin-bottom:6px;"
            f"letter-spacing:-0.03em;'>HireInsight Agent</h1>",
            unsafe_allow_html=True,
        )
        st.markdown(
            "<p style='color:#7A7A7A;font-size:1rem;font-weight:400;margin-bottom:6px;'>"
            "智能求职辅助系统 — 让每一次面试更有准备</p>",
            unsafe_allow_html=True,
        )
        if user:
            st.markdown(
                f"<p style='color:#5A8FC7;font-size:0.9rem;font-weight:400;margin-bottom:28px;'>"
                f"👋 欢迎，{user['display_name']}</p>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown("<p style='margin-bottom:28px;'>&nbsp;</p>", unsafe_allow_html=True)

    st.markdown("---")

    # ============================================================
    # 功能模块卡片
    # ============================================================
    col1, col2, col3 = st.columns(3)

    with col1:
        with st.container(border=True):
            st.markdown(
                "<h3 style='color:#2D2722;font-weight:600;font-size:1.1rem;margin-top:0;'>"
                "📊 数据大屏</h3>",
                unsafe_allow_html=True,
            )
            st.markdown(
                "<p style='color:#7A7A7A;font-size:0.88rem;line-height:1.8;margin-bottom:18px;'>"
                "实时岗位数据分析<br>"
                "薪资分布可视化<br>"
                "城市与学历统计</p>",
                unsafe_allow_html=True,
            )
            if st.button("进入模块", key="goto_dashboard", use_container_width=True):
                st.session_state["current_page"] = "数据大屏"
                st.rerun()

    with col2:
        with st.container(border=True):
            st.markdown(
                "<h3 style='color:#2D2722;font-weight:600;font-size:1.1rem;margin-top:0;'>"
                "🧭 灯塔计划</h3>",
                unsafe_allow_html=True,
            )
            st.markdown(
                "<p style='color:#7A7A7A;font-size:0.88rem;line-height:1.8;margin-bottom:18px;'>"
                "技术倾向测评<br>"
                "专属学习路径<br>"
                "低年级定向规划</p>",
                unsafe_allow_html=True,
            )
            if st.button("进入模块", key="goto_lighthouse", use_container_width=True):
                st.session_state["current_page"] = "灯塔计划"
                st.rerun()

    with col3:
        with st.container(border=True):
            st.markdown(
                "<h3 style='color:#2D2722;font-weight:600;font-size:1.1rem;margin-top:0;'>"
                "🎤 面试模拟</h3>",
                unsafe_allow_html=True,
            )
            st.markdown(
                "<p style='color:#7A7A7A;font-size:0.88rem;line-height:1.8;margin-bottom:18px;'>"
                "简历 Gap 诊断<br>"
                "AI 生成面试题<br>"
                "企业面经 RAG</p>",
                unsafe_allow_html=True,
            )
            if st.button("进入模块", key="goto_interview", use_container_width=True):
                st.session_state["current_page"] = "面试模拟"
                st.rerun()

    st.markdown("---")

    # ============================================================
    # 系统状态优雅面板
    # ============================================================
    st.markdown(
        "<p style='color:#7A7A7A;font-size:0.9rem;font-weight:500;margin-bottom:10px;'>"
        "系统状态</p>",
        unsafe_allow_html=True,
    )

    # 收集状态信息
    api_configured = check_api_key()
    api_dot = "#34C759" if api_configured else "#FF3B30"
    api_text = "已就绪" if api_configured else "未配置"

    rag_dir = os.getenv("CHROMA_PERSIST_DIR", "./data/rag_store")
    rag_exists = os.path.exists(rag_dir)
    rag_dot = "#34C759" if rag_exists else "#FF9F0A"
    rag_text = "已初始化" if rag_exists else "待初始化"

    db_exists = os.path.exists(_DB_PATH)
    db_record_count = 0
    if db_exists:
        try:
            import sqlite3
            conn = sqlite3.connect(f"file:{_DB_PATH}?mode=ro", uri=True)
            cursor = conn.execute("SELECT COUNT(*) FROM jobs")
            db_record_count = cursor.fetchone()[0]
            conn.close()
        except Exception:
            db_record_count = 0
    if db_exists and db_record_count > 0:
        db_dot = "#34C759"
        db_text = f"已连接 · {db_record_count} 条记录"
    elif db_exists:
        db_dot = "#FF9F0A"
        db_text = "已连接 · 暂无数据"
    else:
        db_dot = "#FF3B30"
        db_text = "未连接"

    # 渲染状态面板
    st.markdown(
        f"""<div style='background-color:#FFFFFF;border:1px solid #EFEFEF;border-radius:18px;
        padding:22px 26px;font-size:0.9rem;line-height:2.2;color:#2D2722;'>
        <span style='color:{api_dot};font-size:1.1rem;'>●</span>
        &nbsp;DeepSeek 智能模型&nbsp;&nbsp;&nbsp;&nbsp;
        <span style='color:#7A7A7A;'>{api_text}</span><br>
        <span style='color:{rag_dot};font-size:1.1rem;'>●</span>
        &nbsp;RAG 本地向量存储&nbsp;&nbsp;
        <span style='color:#7A7A7A;'>{rag_text}</span><br>
        <span style='color:{db_dot};font-size:1.1rem;'>●</span>
        &nbsp;离线数仓数据库&nbsp;&nbsp;&nbsp;&nbsp;
        <span style='color:#7A7A7A;'>{db_text}</span>
        </div>""",
        unsafe_allow_html=True,
    )


# ============================================================
# 数据大屏模块
# ============================================================
def render_dashboard():
    """渲染数据大屏页面"""
    # ---- 顶部 Banner 大图（存在则替代文本标题与副标题）----
    if DASHBOARD_BANNER_PATH and os.path.isfile(DASHBOARD_BANNER_PATH):
        st.image(DASHBOARD_BANNER_PATH, use_container_width=True)
    else:
        st.title("📊 岗位数据大屏")
        st.markdown("**实时招聘信息分析看板**")

    # ========================================================
    # Sidebar: 大屏专用筛选区
    # ========================================================
    st.sidebar.markdown("---")
    st.sidebar.subheader("🔍 数据筛选")

    # 初始化 session_state 中的筛选状态
    if "dashboard_filters" not in st.session_state:
        st.session_state["dashboard_filters"] = {}

    # 数据库存在性检查
    db_exists = os.path.exists(_DB_PATH)

    # ---- 1. 城市（st.multiselect） ----
    cities = _get_distinct_values_for_dashboard("city")
    if not cities and not db_exists:
        st.sidebar.info("暂无岗位数据，请先使用底部控制台采集数据")
    selected_cities = st.sidebar.multiselect(
        "城市",
        options=cities,
        default=[],
        help="选择一个或多个城市，留空表示全部"
    )

    # ---- 2. 学历（st.multiselect） ----
    degrees = _get_distinct_values_for_dashboard("degree")
    selected_degrees = st.sidebar.multiselect(
        "学历",
        options=degrees,
        default=[],
        help="选择学历要求，留空表示全部"
    )

    # ---- 3. 公司来源（st.multiselect，奶油风中文本地化） ----
    SOURCE_OPTIONS = ["netease", "tencent"]
    SOURCE_LABELS = {
        "netease": "网易官方",
        "tencent": "腾讯官方",
    }
    selected_sources = st.sidebar.multiselect(
        "公司来源",
        options=SOURCE_OPTIONS,
        default=[],
        format_func=lambda s: SOURCE_LABELS.get(s, s),
        help="选择数据来源，留空表示全部"
    )

    # ---- 4. 薪资范围（st.slider 双滑块） ----
    salary_range = st.sidebar.slider(
        "薪资范围 (k/月)",
        min_value=0,
        max_value=100,
        value=(0, 100),
        step=5,
        help="滑动选择月薪区间（单位：千元）"
    )

    # ---- 5. 经验要求（st.slider 双滑块） ----
    exp_range = st.sidebar.slider(
        "经验要求 (年)",
        min_value=0,
        max_value=20,
        value=(0, 20),
        step=1,
        help="滑动选择经验年限区间"
    )

    # ---- 6. 关键词搜索 ----
    keyword = st.sidebar.text_input(
        "关键词搜索",
        value="",
        placeholder="如 Python、后端、实习",
        help="模糊搜索岗位标题（LIKE 匹配）"
    )

    # ========================================================
    # filters 字典组装
    # ========================================================
    filters: dict = {}

    if selected_cities:
        filters["city"] = selected_cities
    if selected_degrees:
        filters["degree"] = selected_degrees
    if selected_sources:
        filters["source"] = selected_sources
    if salary_range != (0, 100):
        filters["salary_min"] = salary_range[0]
        filters["salary_max"] = salary_range[1]
    if exp_range != (0, 20):
        filters["experience_min"] = exp_range[0]
        filters["experience_max"] = exp_range[1]
    if keyword.strip():
        filters["keyword"] = keyword.strip()

    st.session_state["dashboard_filters"] = filters

    # ---- 重置筛选按钮 ----
    st.sidebar.markdown("---")
    col_act1, col_act2 = st.sidebar.columns(2)
    with col_act1:
        if st.button("🔄 重置筛选", use_container_width=True):
            st.session_state["dashboard_filters"] = {}
            st.rerun()
    with col_act2:
        active_count = len(filters)
        if active_count > 0:
            st.caption(f"当前 {active_count} 项筛选生效中")
        else:
            st.caption("显示全部数据")

    # ---- 离线数仓状态指示器 ----
    st.sidebar.markdown("---")
    if db_exists:
        try:
            import sqlite3
            conn = sqlite3.connect(f"file:{_DB_PATH}?mode=ro", uri=True)
            cursor = conn.execute("SELECT COUNT(*) FROM jobs")
            dw_count = cursor.fetchone()[0]
            conn.close()
        except Exception:
            dw_count = 0
    else:
        dw_count = 0

    if dw_count > 0:
        st.sidebar.markdown(
            f"<div style='font-size:0.82rem;color:#7A7A7A;line-height:1.6'>"
            f"📁 <b>数仓状态</b>：离线优先模式<br>"
            f"&nbsp;&nbsp;&nbsp;&nbsp;当前已同步 <b style='color:#34C759'>{dw_count}</b> 条岗位"
            f"</div>",
            unsafe_allow_html=True,
        )
    else:
        st.sidebar.caption("📁 数仓状态：增量同步待激活")

    # ========================================================
    # ⚙️ 数仓同步配置（侧边栏滑块控流面板）
    # ========================================================
    st.sidebar.markdown("---")
    st.sidebar.subheader("⚙️ 数仓同步配置")

    crawl_pages = st.sidebar.slider(
        "📬 单家公司批量爬取页数",
        min_value=1,
        max_value=30,
        value=15,
        step=1,
        help="左右拖拽控制每次向网易/腾讯招聘 API 请求的分页深度"
    )
    st.session_state["crawl_pages"] = crawl_pages

    # 将原本底部控制台的触发按钮迁移到侧边栏滑块正下方
    pipeline_clicked = st.sidebar.button(
        "🔄 开始大批量异步同步数据",
        type="primary",
        use_container_width=True,
    )

    # ========================================================
    # 主区域：数据接入 + KPI 指标栏 + Plotly 图表看板
    # ========================================================
    _MULTI_VALUE_KEYS = {"city", "degree", "source"}
    single_filters = {k: v for k, v in filters.items() if k not in _MULTI_VALUE_KEYS}

    stats = None
    filtered_df = None
    show_charts = False

    if db_exists:
        try:
            filtered_df = query_jobs_by_filters(
                _DB_PATH,
                filters=single_filters if single_filters else None,
                limit=None,
            )
            for mkey in _MULTI_VALUE_KEYS:
                if mkey in filters:
                    fval = filters[mkey]
                    if isinstance(fval, list) and len(fval) > 0:
                        filtered_df = filtered_df[filtered_df[mkey].isin(fval)]
            stats = calculate_market_metrics(source=filtered_df)
            if not filtered_df.empty and stats.get("meta", {}).get("total_count", 0) > 0:
                show_charts = True
        except FileNotFoundError:
            db_exists = False
            show_charts = False

    if not db_exists:
        st.info(
            "💡 离线数仓为空，请使用左侧【⚙️ 数仓同步配置】面板同步初始岗位数据。"
        )
    elif not show_charts:
        st.info(
            "🔍 当前筛选条件下暂无岗位数据，"
            "请调整侧边栏筛选条件或前往底部控制台采集数据。"
        )
    else:
        st.markdown("---")

        total_count = stats["meta"]["total_count"]
        salary_count = stats["salary"].get("count", 0)
        city_count = stats["city"].get("unique_cities", 0)
        degree_count = stats["degree"].get("unique_degrees", 0)

        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        with kpi1:
            st.metric("📋 总岗位数", total_count)
        with kpi2:
            st.metric("💰 有效薪资数", salary_count)
        with kpi3:
            st.metric("🌆 城市覆盖数", city_count)
        with kpi4:
            st.metric("🎓 学历种类数", degree_count)

        st.markdown("---")

        left_col, right_col = st.columns([2, 3])

        with left_col:
            salary_dist = stats["salary"].get("salary_distribution", {})
            if salary_dist:
                fig_salary = go.Figure(data=[go.Bar(
                    x=list(salary_dist.keys()),
                    y=list(salary_dist.values()),
                    text=list(salary_dist.values()),
                    textposition="outside",
                    textfont=dict(size=13, color="#FFFFFF"),
                    marker=dict(
                        color=list(salary_dist.values()),
                        colorscale="Blues",
                        showscale=False,
                        line=dict(width=1, color="#2D2722"),
                    ),
                    hovertemplate="薪资段: %{x}<br>岗位数: %{y}<extra></extra>",
                )])
                fig_salary.update_layout(
                    title=dict(text="💰 薪资分布", font=dict(size=16, color="#FFFFFF")),
                    xaxis_title="薪资区间",
                    yaxis_title="岗位数",
                    template="plotly_dark",
                    height=400,
                    margin=dict(l=20, r=20, t=40, b=20),
                    font=dict(color="#FFFFFF", family="Arial"),
                    paper_bgcolor="rgba(45,39,34,1)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    legend=dict(font=dict(color="#FFFFFF")),
                )
                st.plotly_chart(fig_salary, use_container_width=True)
            else:
                st.caption("暂无有效薪资数据")

            st.markdown("---")

            degree_dist = stats["degree"].get("distribution", {})
            if degree_dist:
                # 🔴 零值过滤：剔除 count==0 的分类，消除 0% 凌乱引线
                degree_dist = {k: v for k, v in degree_dist.items() if v.get("count", 0) > 0}
                degree_labels = list(degree_dist.keys())
                degree_values = [v["count"] for v in degree_dist.values()]
                fig_degree = go.Figure(data=[go.Pie(
                    labels=degree_labels,
                    values=degree_values,
                    hole=0.5,
                    textinfo="label+percent",
                    textfont=dict(size=12, color="#FFFFFF"),
                    marker=dict(
                        colors=["#636efa", "#00cc96", "#ab63fa", "#ffa15a",
                                "#19d3f3", "#ff6692", "#b6e880"],
                        line=dict(width=1, color="#2D2722"),
                    ),
                    hovertemplate="学历: %{label}<br>岗位数: %{value}<br>占比: %{percent}<extra></extra>",
                )])
                fig_degree.update_layout(
                    title=dict(text="🎓 学历占比", font=dict(size=16, color="#FFFFFF")),
                    template="plotly_dark",
                    height=400,
                    margin=dict(l=20, r=20, t=40, b=20),
                    font=dict(color="#FFFFFF", family="Arial"),
                    paper_bgcolor="rgba(45,39,34,1)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    legend=dict(font=dict(color="#FFFFFF")),
                )
                st.plotly_chart(fig_degree, use_container_width=True)

        with right_col:
            city_top10 = stats["city"].get("top10", {})
            if city_top10:
                sorted_cities = sorted(city_top10.items(), key=lambda x: x[1])
                city_names = [c[0] for c in sorted_cities]
                city_counts = [c[1] for c in sorted_cities]
                fig_city = go.Figure(data=[go.Bar(
                    x=city_counts,
                    y=city_names,
                    orientation="h",
                    text=city_counts,
                    textposition="outside",
                    textfont=dict(size=13, color="#FFFFFF"),
                    marker=dict(
                        color=city_counts,
                        colorscale="Viridis",
                        showscale=False,
                        line=dict(width=1, color="#2D2722"),
                    ),
                    hovertemplate="城市: %{y}<br>岗位数: %{x}<extra></extra>",
                )])
                fig_city.update_layout(
                    title=dict(text="🌆 城市需求 Top 10", font=dict(size=16, color="#FFFFFF")),
                    xaxis_title="岗位数",
                    yaxis_title=None,
                    template="plotly_dark",
                    height=400,
                    margin=dict(l=20, r=20, t=40, b=20),
                    font=dict(color="#FFFFFF", family="Arial"),
                    paper_bgcolor="rgba(45,39,34,1)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    legend=dict(font=dict(color="#FFFFFF")),
                )
                st.plotly_chart(fig_city, use_container_width=True)

            st.markdown("---")

            exp_dist = stats["experience"].get("distribution", {})
            if exp_dist:
                # 🔴 零值过滤：剔除 count==0 的分类，消除 0% 凌乱引线
                exp_dist = {k: v for k, v in exp_dist.items() if v.get("count", 0) > 0}
                exp_labels = list(exp_dist.keys())
                exp_values = [v["count"] for v in exp_dist.values()]
                fig_exp = go.Figure(data=[go.Pie(
                    labels=exp_labels,
                    values=exp_values,
                    hole=0.5,
                    textinfo="label+percent",
                    textfont=dict(size=12, color="#FFFFFF"),
                    marker=dict(
                        colors=["#ffa15a", "#19d3f3", "#ab63fa", "#00cc96",
                                "#636efa", "#ff6692", "#b6e880"],
                        line=dict(width=1, color="#2D2722"),
                    ),
                    hovertemplate="经验: %{label}<br>岗位数: %{value}<br>占比: %{percent}<extra></extra>",
                )])
                fig_exp.update_layout(
                    title=dict(text="⏳ 经验要求占比", font=dict(size=16, color="#FFFFFF")),
                    template="plotly_dark",
                    height=400,
                    margin=dict(l=20, r=20, t=40, b=20),
                    font=dict(color="#FFFFFF", family="Arial"),
                    paper_bgcolor="rgba(45,39,34,1)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    legend=dict(font=dict(color="#FFFFFF")),
                )
                st.plotly_chart(fig_exp, use_container_width=True)

        # ========================================================
        # 实时岗位底仓明细表单（宏观图表 → 微观数据闭环）
        # ========================================================
        if filtered_df is not None and not filtered_df.empty:
            st.markdown("---")
            with st.container(border=True):
                st.subheader("📋 实时岗位底仓明细")

                # 字段精选 + 中文映射（含公司名称和岗位链接）
                display_cols: dict[str, str] = {
                    "title": "岗位名称",
                    "company": "公司名称",
                    "salary_raw": "薪资待遇",
                    "city": "工作城市",
                    "degree": "学历要求",
                    "experience_raw": "经验要求",
                    "source": "公司来源",
                    "post_url": "岗位链接",
                }
                # 仅取当前筛选中存在的列
                available_cols: dict[str, str] = {
                    k: v for k, v in display_cols.items() if k in filtered_df.columns
                }
                df_display = filtered_df[list(available_cols.keys())].rename(
                    columns=available_cols
                )
                # 公司来源中文本地化
                if "公司来源" in df_display.columns:
                    source_map = {"netease": "网易官方", "tencent": "腾讯官方"}
                    df_display["公司来源"] = df_display["公司来源"].map(
                        lambda s: source_map.get(s, s)
                    )

                st.dataframe(
                    df_display,
                    use_container_width=True,
                    height=min(38 * max(len(df_display), 1) + 38, 600),
                    column_config={
                        "岗位链接": st.column_config.LinkColumn(
                            "🔗 岗位详情",
                            help="点击直接跳转到网易/腾讯官方招聘官网查看原公告",
                            display_text="点击跳转",
                        ),
                        "公司来源": st.column_config.TextColumn("数据通道"),
                    },
                    hide_index=True,
                )

    # ========================================================
    # 离线数仓控制台（信息面板，触发按钮已迁移至侧边栏）
    # ========================================================
    st.markdown("---")
    with st.expander("🛠️ 离线数仓控制台", expanded=False):
        st.info(
            "📋 **离线数仓控制台**\n\n"
            "数据同步触发已迁移至左侧边栏 **⚙️ 数仓同步配置** 面板。\n\n"
            "• 拖拽滑块控制每次爬取的页数（1-30 页）\n"
            "• 点击「🔄 开始大批量异步同步数据」一键同步\n\n"
            "同步过程通过 Upsert 合并入本地离线数仓，已存在岗位自动更新。"
        )

    if pipeline_clicked:
        pages = st.session_state.get("crawl_pages", 15)

        with st.spinner(f"正在为您深度抓取双子星厂商前 {pages} 页实时岗位..."):
            scraper_available = False
            try:
                from utils.scraper.netease_scraper import NetEaseScraper  # noqa: F811
                from utils.scraper.tencent_scraper import TencentScraper  # noqa: F811
                scraper_available = True
            except ImportError:
                pass

            with st.status("🔄 增量同步进行中...", expanded=True) as status:
                status.update(
                    label="[1/6] 初始化数仓表结构...", state="running", expanded=True
                )
                init_sqlite_db(_DB_PATH)

                raw_jobs: list[dict] = []

                if scraper_available:
                    status.update(
                        label=f"[2/6] 增量采集 | 实时抓取中（NetEase + Tencent，各 {pages} 页）...",
                        state="running",
                        expanded=True,
                    )
                    try:
                        with NetEaseScraper(max_pages=pages) as netease_scraper:
                            netease_raw = netease_scraper.crawl()
                            st.write(f"✅ 网易：抓取 {len(netease_raw)} 条")
                            raw_jobs.extend(netease_raw)
                    except Exception as e:
                        st.warning(f"⚠️ 网易爬虫异常：{e}")

                    try:
                        tencent_scraper = TencentScraper(max_pages=pages)
                        tencent_raw = tencent_scraper.crawl()
                        st.write(f"✅ 腾讯：抓取 {len(tencent_raw)} 条")
                        raw_jobs.extend(tencent_raw)
                    except Exception as e:
                        st.warning(f"⚠️ 腾讯爬虫异常：{e}")


                status.update(
                    label="[3/6] 增量同步 | 多源数据转换（Transformer）...",
                    state="running",
                    expanded=True,
                )
                transformed_jobs: list[dict] = []
                if scraper_available:
                    netease_raw = [j for j in raw_jobs if j.get("source") == "netease"]
                    tencent_raw = [j for j in raw_jobs if j.get("source") == "tencent"]
                    other_raw = [j for j in raw_jobs
                                 if j not in netease_raw and j not in tencent_raw]
                    if netease_raw:
                        transformed_jobs.extend(transform_jobs(netease_raw, source="netease"))
                    if tencent_raw:
                        transformed_jobs.extend(transform_jobs(tencent_raw, source="tencent"))
                    if other_raw:
                        transformed_jobs.extend(other_raw)
                else:
                    transformed_jobs = raw_jobs
                st.write(f"✅ 转换完成：{len(transformed_jobs)} 条标准记录")

                status.update(
                    label="[4/6] 增量同步 | 正则清洗与标准化（Cleaner）...",
                    state="running",
                    expanded=True,
                )
                df_cleaned = clean_job_data(pd.DataFrame(transformed_jobs))
                st.write(f"✅ 清洗完成：{len(df_cleaned)} 条有效记录")

                status.update(
                    label="[5/6] 增量同步 | 写入离线数仓（Upsert）...",
                    state="running",
                    expanded=True,
                )
                n_written = save_to_sqlite(df_cleaned, _DB_PATH)
                st.write(f"✅ 数仓写入：{n_written} 条已持久化")

                status.update(
                    label="[6/6] 增量同步完成 ✅",
                    state="complete",
                    expanded=False,
                )

        st.cache_data.clear()
        st.toast(f"✨ 成功控流抓取 {pages} 页数据，数仓底仓已灌满！", icon="🎉")
        st.rerun()


# ============================================================
# 灯塔计划模块
# ============================================================
def render_lighthouse():
    """渲染灯塔计划页面 —— 两阶段 UI：情景测评 → 技术倾向雷达图 + Roadmap 静态看板"""
    # ---- 顶部 Banner 大图（绝对路径；不存在时文本兜底）----
    if LIGHTHOUSE_BANNER_PATH and os.path.isfile(LIGHTHOUSE_BANNER_PATH):
        st.image(LIGHTHOUSE_BANNER_PATH, use_container_width=True)
    else:
        st.title("🧭 灯塔计划")

    # ---- 历史记录入口（Banner 正下方，不占横向空间）----
    user = get_current_user()
    if user:
        if st.button("📜 历史记录", key="lighthouse_history_btn"):
            st.session_state["lighthouse_history_view"] = True
            st.session_state["lighthouse_record_detail"] = None
            st.rerun()

    # ============================================================
    # 初始化 session_state 控制变量（防跨页面切换报错）
    # ============================================================
    if "lighthouse_filtered_questions" not in st.session_state:
        st.session_state.lighthouse_filtered_questions = None
    if "lighthouse_results" not in st.session_state:
        st.session_state.lighthouse_results = None
    if "lighthouse_quiz_started" not in st.session_state:
        st.session_state.lighthouse_quiz_started = False
    if "lighthouse_target_position" not in st.session_state:
        st.session_state.lighthouse_target_position = ""
    if "lighthouse_grade" not in st.session_state:
        st.session_state.lighthouse_grade = ""
    if "lighthouse_history_view" not in st.session_state:
        st.session_state.lighthouse_history_view = False
    if "lighthouse_record_detail" not in st.session_state:
        st.session_state.lighthouse_record_detail = None

    # ---- 历史记录查看模式 ----
    if st.session_state.get("lighthouse_history_view"):
        _render_lighthouse_history()
        return

    # ---- 记录详情查看模式 ----
    if st.session_state.get("lighthouse_record_detail"):
        _render_lighthouse_record_detail(st.session_state["lighthouse_record_detail"])
        return

    # ============================================================
    # 阶段二：结果看板渲染
    # ============================================================
    if st.session_state.lighthouse_results is not None:
        results = st.session_state.lighthouse_results
        _render_lighthouse_result(
            results=results,
            target_position=st.session_state.lighthouse_target_position,
            grade=st.session_state.lighthouse_grade,
        )

        # 保存历史记录
        if user and not st.session_state.get("lighthouse_saved", False):
            try:
                save_lighthouse_record(
                    user_id=user["id"],
                    target_position=st.session_state.lighthouse_target_position,
                    grade=st.session_state.lighthouse_grade,
                    user_answers=results.get("user_answers"),
                    tech_tendency=results.get("tech_tendency"),
                    roadmap=results.get("roadmap"),
                    db_path=_DB_PATH,
                )
                st.session_state["lighthouse_saved"] = True
                st.toast("✅ 测评结果已保存到历史记录", icon="💾")
            except Exception as e:
                st.warning(f"⚠️ 保存历史记录失败: {e}")

        # 重置按钮
        st.markdown("---")
        col_reset, _ = st.columns([1, 4])
        with col_reset:
            if st.button("🔄 重新测评", type="secondary", use_container_width=True):
                st.session_state.lighthouse_filtered_questions = None
                st.session_state.lighthouse_results = None
                st.session_state.lighthouse_quiz_started = False
                st.session_state.lighthouse_target_position = ""
                st.session_state.lighthouse_grade = ""
                st.session_state["lighthouse_saved"] = False
                st.rerun()
        return

    # ============================================================
    # 阶段一-步骤 1：用户信息输入
    # ============================================================
    if not st.session_state.lighthouse_quiz_started:
        with st.container():
            st.subheader("📝 基本信息")

            col1, col2 = st.columns(2)
            with col1:
                target_position = st.text_input(
                    "目标岗位",
                    placeholder="例如：Java后端开发、AI算法工程师、前端开发",
                    help="输入你想从事的技术岗位方向",
                )
            with col2:
                grade = st.selectbox(
                    "当前年级",
                    options=["大一", "大二", "大三", "大四", "研一", "研二", "研三", "博士", "已毕业/社招"],
                    help="选择你当前所在的年级阶段",
                )

            st.markdown("")

            if st.button("🚀 开始测评", type="primary", use_container_width=True):
                if not target_position.strip():
                    st.error("请输入目标岗位")
                    return

                all_questions = get_full_question_bank()

                initial_state: LighthousePlanState = {
                    "target_position": target_position.strip(),
                    "grade": grade,
                    "all_questions": all_questions,
                    "filtered_questions": [],
                    "user_answers": [],
                    "user_choices": [],
                    "tech_tendency": None,
                    "roadmap_json": None,
                    "roadmap": None,
                    "current_step": "init",
                    "execution_error": None,
                    "is_completed": False,
                }

                with st.status("🧠 正在分析最适合你的测评题目...", expanded=True) as status:
                    try:
                        filter_result = question_filter_node(initial_state)
                    except Exception as e:
                        filter_result = {
                            "filtered_questions": all_questions,
                            "current_step": "question_filter_failed",
                            "execution_error": f"题目筛选失败，已使用完整题库: {str(e)}",
                        }
                    status.update(label="✅ 选题完成", state="complete")

                st.session_state.lighthouse_filtered_questions = filter_result.get(
                    "filtered_questions", all_questions
                )
                st.session_state.lighthouse_quiz_started = True
                st.session_state.lighthouse_target_position = target_position.strip()
                st.session_state.lighthouse_grade = grade

                if filter_result.get("execution_error"):
                    st.warning(f"⚠️ {filter_result['execution_error']}")

                st.rerun()

    # ============================================================
    # 阶段一-步骤 2：答题表单
    # ============================================================
    if (
        st.session_state.lighthouse_quiz_started
        and st.session_state.lighthouse_filtered_questions
    ):
        questions = st.session_state.lighthouse_filtered_questions
        st.subheader(f"📋 技术倾向测评（共 {len(questions)} 道情景选择题）")
        st.caption("请仔细阅读每道题的情景描述，选择最符合你真实偏好的选项。没有对错之分，请诚实作答。")

        with st.form("lighthouse_quiz_form", clear_on_submit=False):
            user_selections: dict = {}

            for idx, q in enumerate(questions):
                qid = q["id"]
                st.markdown(f"### 题目 {idx + 1}")
                st.markdown(f"**场景：** {q['scenario']}")
                st.markdown("")

                option_labels = []
                for opt_idx, opt in enumerate(q["options"]):
                    label = f"{chr(65 + opt_idx)}. {opt['text']}"
                    option_labels.append(label)

                choice = st.radio(
                    "请选择：",
                    options=range(len(option_labels)),
                    format_func=lambda i: option_labels[i],
                    key=f"lighthouse_q_{qid}",
                    index=None,
                )
                user_selections[qid] = choice
                st.markdown("---")

            submitted = st.form_submit_button(
                "📊 生成我的技术路线图",
                type="primary",
                use_container_width=True,
            )

            if submitted:
                unanswered = [qid for qid, sel in user_selections.items() if sel is None]
                if unanswered:
                    st.error(f"还有 {len(unanswered)} 道题目未作答，请完成所有题目后再提交")
                    return

                user_answers = []
                for q in questions:
                    qid = q["id"]
                    sel_idx = user_selections[qid]
                    opt = q["options"][sel_idx]
                    user_answers.append({
                        "question_id": qid,
                        "scenario": q["scenario"],
                        "selected_option": sel_idx,
                        "option_text": opt["text"],
                        "tendency": opt["tendency"],
                    })

                with st.status("🧠 AI 正在评估你的技术倾向并生成学习路线图...", expanded=True) as status:
                    assess_state: LighthousePlanState = {
                        "target_position": st.session_state.lighthouse_target_position,
                        "grade": st.session_state.lighthouse_grade,
                        "user_answers": user_answers,
                        "all_questions": questions,
                        "filtered_questions": questions,
                        "tech_tendency": None,
                        "roadmap_json": None,
                        "roadmap": None,
                        "current_step": "assessment_pending",
                        "execution_error": None,
                        "is_completed": False,
                    }

                    try:
                        assess_result = assessment_node(assess_state)
                    except Exception as e:
                        assess_result = {
                            "execution_error": f"AssessmentNode 执行异常: {str(e)}",
                            "is_completed": True,
                            "roadmap": None,
                        }

                    status.update(label="✅ 评估完成", state="complete")

                st.session_state.lighthouse_results = assess_result
                st.session_state["lighthouse_saved"] = False
                st.rerun()


# ============================================================
# 灯塔计划：历史记录查看
# ============================================================
def _render_lighthouse_history():
    """渲染灯塔计划历史记录列表"""
    st.subheader("📜 灯塔计划历史记录")
    user = get_current_user()
    if not user:
        st.warning("请先登录")
        return

    history = get_lighthouse_history(user["id"], _DB_PATH, limit=20)
    if not history:
        st.info("暂无历史记录，快去完成一次测评吧！")
    else:
        for record in history:
            with st.container(border=True):
                col1, col2, col3 = st.columns([4, 1, 1])
                with col1:
                    st.markdown(
                        f"**{record['target_position']}** · {record['grade']}"
                    )
                    st.caption(f"{record['created_at']}")
                with col2:
                    if st.button("查看", key=f"lighthouse_view_{record['id']}"):
                        st.session_state["lighthouse_record_detail"] = record["id"]
                        st.session_state["lighthouse_history_view"] = False
                        st.rerun()
                with col3:
                    with st.popover("🗑️", key=f"lighthouse_popover_{record['id']}"):
                        st.warning("确定要永久删除这条记录吗？")
                        if st.button("确认删除", key=f"lighthouse_del_{record['id']}"):
                            if delete_lighthouse_record(record["id"], user["id"], _DB_PATH):
                                st.toast("已删除", icon="🗑️")
                                st.rerun()

    if st.button("⬅️ 返回测评", key="lighthouse_back"):
        st.session_state["lighthouse_history_view"] = False
        st.rerun()


def _render_lighthouse_record_detail(record_id: str):
    """渲染单条灯塔记录详情"""
    user = get_current_user()
    if not user:
        st.warning("请先登录")
        return

    record = get_lighthouse_record(record_id, user["id"], _DB_PATH)
    if not record:
        st.error("记录不存在或无权访问")
        return

    st.subheader(f"🎯 {record['target_position']} · {record['grade']} · 历史测评")
    st.caption(f"测评时间：{record['created_at']}")

    _render_lighthouse_result(
        results={
            "tech_tendency": record.get("tech_tendency"),
            "roadmap": record.get("roadmap"),
            "execution_error": None,
        },
        target_position=record["target_position"],
        grade=record["grade"],
    )

    if st.button("⬅️ 返回列表", key="lighthouse_detail_back"):
        st.session_state["lighthouse_record_detail"] = None
        st.session_state["lighthouse_history_view"] = True
        st.rerun()


# ============================================================
# 辅助函数：渲染灯塔计划结果看板
# ============================================================
def _render_lighthouse_result(
    results: dict,
    target_position: str,
    grade: str,
):
    """渲染阶段二结果看板：雷达图 + Markdown 路线图（含降级错误页）"""

    execution_error = results.get("execution_error")
    tech_tendency = results.get("tech_tendency")
    roadmap = results.get("roadmap")

    # ---- 降级错误页 ----
    if execution_error and not tech_tendency and not roadmap:
        st.error(f"⚠️ 评估过程中遇到问题：{execution_error}")
        st.info(
            "建议重新测评或稍后再试。如果问题持续出现，"
            "请检查 DeepSeek API Key 是否有效以及网络连接是否正常。"
        )
        return

    # ---- 正常结果渲染 ----
    st.subheader(f"🎯 {target_position} · {grade} · 技术倾向评估报告")

    # ---- Plotly 雷达图 ----
    _render_radar_chart(
        tech_tendency=tech_tendency,
        execution_error=execution_error,
    )

    # ---- 纯文本降级 ----
    if execution_error and not tech_tendency and roadmap:
        st.warning(f"⚠️ {execution_error}")
        st.markdown("### 📝 降级文本路线图")
        st.markdown(roadmap)
        return

    # ---- 静态 Markdown 路线图 ----
    if roadmap:
        st.markdown(roadmap)


def _render_radar_chart(
    tech_tendency: dict | None,
    execution_error: str | None,
):
    """绘制 Plotly 雷达图（温暖蓝色系半透明填充）"""
    directions = get_valid_tendencies()

    # 提取分值，缺失方向补 0
    values = []
    for d in directions:
        if tech_tendency and isinstance(tech_tendency, dict):
            values.append(tech_tendency.get(d, 0))
        else:
            values.append(0)

    # 闭合雷达图（首尾同值）
    display_directions = directions + [directions[0]]
    display_values = values + [values[0]]

    fig = go.Figure(
        data=go.Scatterpolar(
            r=display_values,
            theta=display_directions,
            fill="toself",
            fillcolor="rgba(99, 110, 250, 0.25)",
            line=dict(color="#636efa", width=2.5),
            marker=dict(
                color="#636efa",
                size=8,
                line=dict(color="#ffffff", width=1.5),
            ),
            name="技术倾向",
            hovertemplate="<b>%{theta}</b><br>倾向分值: %{r}<extra></extra>",
        )
    )

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                tickfont=dict(size=11, color="#555"),
                gridcolor="rgba(99, 110, 250, 0.15)",
                linecolor="rgba(99, 110, 250, 0.3)",
            ),
            angularaxis=dict(
                tickfont=dict(size=13, color="#333"),
                gridcolor="rgba(99, 110, 250, 0.2)",
                linecolor="rgba(99, 110, 250, 0.3)",
            ),
            bgcolor="rgba(245, 247, 255, 0.5)",
        ),
        title=dict(
            text="六维度技术倾向评估" + (" (降级数据)" if execution_error else ""),
            font=dict(size=16, color="#333"),
        ),
        height=500,
        margin=dict(l=40, r=40, t=60, b=20),
        paper_bgcolor="rgba(255,255,255,0)",
        plot_bgcolor="rgba(255,255,255,0)",
    )

    st.plotly_chart(fig, use_container_width=True)


# ============================================================
# 面试模拟模块（LangGraph + RAG 全链路）
# ============================================================
def render_interview():
    """渲染面试模拟页面 —— LangGraph 拓扑工作流 + 本地 RAG 全链路"""
    # ---- 顶部 Banner 大图（绝对路径；不存在时文本兜底）----
    if INTERVIEW_BANNER_PATH and os.path.isfile(INTERVIEW_BANNER_PATH):
        st.image(INTERVIEW_BANNER_PATH, use_container_width=True)
    else:
        st.title("🎤 面试模拟")

    # ---- 历史对话入口（Banner 正下方，不占横向空间）----
    user = get_current_user()
    if user:
        if st.button("📜 历史对话", key="interview_history_btn"):
            st.session_state["interview_history_view"] = True
            st.session_state["interview_record_detail"] = None
            st.rerun()

    if not check_api_key():
        render_api_key_warning()
        return

    st.success("✅ DeepSeek API 已配置，可以开始面试模拟")

    # ---- 历史记录查看模式 ----
    if st.session_state.get("interview_history_view"):
        _render_interview_history()
        return

    # ---- 记录详情查看模式 ----
    if st.session_state.get("interview_record_detail"):
        _render_interview_record_detail(st.session_state["interview_record_detail"])
        return

    # ---- 向量库单例：确保 RAG 存储已初始化 ----
    if _get_or_init_collection is not None:
        try:
            _get_or_init_collection()
        except Exception:
            pass  # 初始化失败不阻断流程

    # ---- 用户输入区域 ----
    st.subheader("📝 输入您的求职信息")

    uploaded_file = st.file_uploader(
        "上传简历（PDF 格式）",
        type=["pdf"],
        help="支持 PDF 格式简历",
    )

    target_position = st.text_input(
        "目标岗位",
        placeholder="例如：Python后端工程师",
        help="输入您想要申请的岗位",
    )

    target_company = st.text_input(
        "目标公司（可选）",
        placeholder="例如：字节跳动",
        help="输入您想要申请的公司",
    )

    # ---- 开始分析按钮 ----
    if st.button("🚀 开始面试模拟", type="primary", use_container_width=True):
        if not target_position:
            st.error("请输入目标岗位")
            return

        # =====================================================
        # Step 1: 简历解析与市场数据加载
        # =====================================================
        user_resume = ""
        if uploaded_file is not None:
            try:
                pdf_bytes = uploaded_file.read()
                user_resume = extract_text_from_pdf_bytes(pdf_bytes)
                if not user_resume.strip():
                    st.warning("⚠️ 未能从 PDF 中提取到文本内容，将仅基于市场数据进行分析")
            except Exception as e:
                st.warning(f"⚠️ PDF 解析失败: {e}")
        if not user_resume:
            st.info("ℹ️ 未上传简历，将仅基于市场数据进行岗位分析")

        market_summary = _load_market_stats_for_interview(target_position)

        # =====================================================
        # Step 2: 构造 InterviewState 初始状态
        # =====================================================
        initial_state: InterviewState = {
            "user_resume": user_resume,
            "target_position": target_position,
            "target_company": target_company or None,
            "market_summary": market_summary,
            "market_data": None,
            "market_report": None,
            "gap_analysis": None,
            "interview_questions": None,
            "rag_context": None,
            "current_step": "init",
            "execution_error": None,
            "is_completed": False,
        }

        # =====================================================
        # Step 3: st.status 容器驱动 LangGraph 工作流
        # =====================================================
        with st.status("🤖 AI 面试官正在工作...", expanded=True) as status:
            result = run_interview_workflow(initial_state)
            status.update(label="✅ 分析完成", state="complete")

        # =====================================================
        # Step 4: 静态 Markdown 看板渲染
        # =====================================================
        st.markdown("---")

        # 市场趋势报告
        market_report = result.get("market_report")
        if market_report:
            st.markdown(market_report)
            st.markdown("---")

        # 技能差距诊断
        gap_analysis = result.get("gap_analysis")
        if gap_analysis:
            st.markdown(gap_analysis)
            st.markdown("---")

        # 面试题（核心交付物，直展不折叠）
        questions = result.get("interview_questions")
        if questions:
            st.subheader("🎯 定制化面试变形题")
            for i, q in enumerate(questions, 1):
                st.markdown(
                    f"> **题目 {i}**\n>\n> {q}\n"
                )
            st.markdown(
                "> 💡 **提示**：针对以上题目进行模拟练习，"
                "重点关注差距诊断报告中标记为 🟠 或 🔴 的维度。"
            )

        # =====================================================
        # Step 5: 保存历史记录
        # =====================================================
        if user:
            try:
                save_interview_record(
                    user_id=user["id"],
                    target_position=target_position,
                    target_company=target_company or None,
                    user_resume=user_resume,
                    market_report=market_report,
                    gap_analysis=gap_analysis,
                    interview_questions=questions,
                    db_path=_DB_PATH,
                )
                st.toast("✅ 面试分析已保存到历史记录", icon="💾")
            except Exception as e:
                st.warning(f"⚠️ 保存历史记录失败: {e}")


# ============================================================
# 面试模拟：历史记录查看
# ============================================================
def _render_interview_history():
    """渲染面试模拟历史记录列表"""
    st.subheader("📜 面试模拟历史记录")
    user = get_current_user()
    if not user:
        st.warning("请先登录")
        return

    history = get_interview_history(user["id"], _DB_PATH, limit=20)
    if not history:
        st.info("暂无历史记录，快去进行一次面试模拟吧！")
    else:
        for record in history:
            with st.container(border=True):
                col1, col2, col3 = st.columns([4, 1, 1])
                with col1:
                    company_text = f" · {record['target_company']}" if record.get('target_company') else ""
                    st.markdown(
                        f"**{record['target_position']}**{company_text}"
                    )
                    st.caption(f"{record['created_at']}")
                with col2:
                    if st.button("查看", key=f"interview_view_{record['id']}"):
                        st.session_state["interview_record_detail"] = record["id"]
                        st.session_state["interview_history_view"] = False
                        st.rerun()
                with col3:
                    with st.popover("🗑️", key=f"interview_popover_{record['id']}"):
                        st.warning("确定要永久删除这条记录吗？")
                        if st.button("确认删除", key=f"interview_del_{record['id']}"):
                            if delete_interview_record(record["id"], user["id"], _DB_PATH):
                                st.toast("已删除", icon="🗑️")
                                st.rerun()

    if st.button("⬅️ 返回模拟", key="interview_back"):
        st.session_state["interview_history_view"] = False
        st.rerun()


def _render_interview_record_detail(record_id: str):
    """渲染单条面试记录详情"""
    user = get_current_user()
    if not user:
        st.warning("请先登录")
        return

    record = get_interview_record(record_id, user["id"], _DB_PATH)
    if not record:
        st.error("记录不存在或无权访问")
        return

    st.subheader(f"🎤 {record['target_position']} · 历史面试分析")
    if record.get('target_company'):
        st.caption(f"目标公司：{record['target_company']}")
    st.caption(f"分析时间：{record['created_at']}")

    st.markdown("---")

    if record.get("market_report"):
        st.markdown(record["market_report"])
        st.markdown("---")

    if record.get("gap_analysis"):
        st.markdown(record["gap_analysis"])
        st.markdown("---")

    questions = record.get("interview_questions")
    if questions:
        st.subheader("🎯 定制化面试变形题")
        for i, q in enumerate(questions, 1):
            st.markdown(f"> **题目 {i}**\n>\n> {q}\n")
        st.markdown(
            "> 💡 **提示**：针对以上题目进行模拟练习，"
            "重点关注差距诊断报告中标记为 🟠 或 🔴 的维度。"
        )

    if st.button("⬅️ 返回列表", key="interview_detail_back"):
        st.session_state["interview_record_detail"] = None
        st.session_state["interview_history_view"] = True
        st.rerun()


# ============================================================
# 系统设置页面
# ============================================================
def render_settings():
    """渲染系统设置页面"""
    st.title("⚙️ 系统设置")
    
    st.subheader("API 配置")
    
    api_key = os.getenv("DEEPSEEK_API_KEY", "")
    if api_key and api_key != "your_api_key_here":
        masked_key = api_key[:8] + "..." + api_key[-4:]
        st.success(f"✅ DeepSeek API Key: `{masked_key}`")
    else:
        st.warning("⚠️ API Key 未配置")
    
    st.text_input("DEEPSEEK_MODEL", value=os.getenv("DEEPSEEK_MODEL", "deepseek-chat"), disabled=True)
    st.text_input("CHROMA_PERSIST_DIR", value=os.getenv("CHROMA_PERSIST_DIR", "./data/rag_store"), disabled=True)
    st.text_input("SQLITE_DB_PATH", value=os.getenv("SQLITE_DB_PATH", "./data/jobs.db"), disabled=True)
    
    st.markdown("---")
    st.subheader("数据管理")
    
    if st.button("🔄 重新加载环境变量"):
        load_dotenv(override=True)
        st.rerun()


# ============================================================
# "👤 我的" —— 多用户资产管理中心
# ============================================================

def render_my_profile():
    """渲染"我的"页面：左侧名片 + 右侧修改表单 + 历史资产管家"""
    st.title("👤 我的")

    user = get_current_user()
    if not user:
        st.warning("请先登录")
        return

    # ============================================================
    # 详情查看模式（优先级最高）
    # ============================================================
    if "profile_interview_detail" not in st.session_state:
        st.session_state["profile_interview_detail"] = None
    if "profile_lighthouse_detail" not in st.session_state:
        st.session_state["profile_lighthouse_detail"] = None

    if st.session_state["profile_interview_detail"]:
        _render_my_interview_detail(st.session_state["profile_interview_detail"])
        return
    if st.session_state["profile_lighthouse_detail"]:
        _render_my_lighthouse_detail(st.session_state["profile_lighthouse_detail"])
        return

    # ============================================================
    # 上半部分：三栏横向排版 —— 名片 | 资产 | 修改表单
    # ============================================================
    col1, col2, col3 = st.columns([1.2, 1.8, 2], gap="large")

    # ---- 加载统计 ----
    interview_history = get_interview_history(user["id"], _DB_PATH, limit=1)
    lighthouse_history = get_lighthouse_history(user["id"], _DB_PATH, limit=1)
    recent_position = "尚未设置"
    if interview_history:
        recent_position = interview_history[0]["target_position"]
    elif lighthouse_history:
        recent_position = lighthouse_history[0]["target_position"]
    total_interview = len(get_interview_history(user["id"], _DB_PATH, limit=1000))
    total_lighthouse = len(get_lighthouse_history(user["id"], _DB_PATH, limit=1000))

    # ============================================================
    # 左栏：个人名片 + 极简上传 + 注销
    # ============================================================
    with col1:
        avatar_b64 = user.get("avatar_base64")
        if avatar_b64:
            st.markdown(
                f"""<div style='text-align:center;'>
                <img src="data:image/png;base64,{avatar_b64}"
                style='width:90px;height:90px;border-radius:50%;object-fit:cover;
                border:2px solid #EFEFEF;' /></div>""",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                """<div style='width:90px;height:90px;border-radius:50%;
                background:linear-gradient(135deg, #2D2722 0%, #5C4D42 100%);
                display:flex;align-items:center;justify-content:center;
                font-size:2.3rem;color:#FFFFFF;margin:0 auto;'>👤</div>""",
                unsafe_allow_html=True,
            )

        st.markdown(
            f"""<div style='text-align:center;margin-top:10px;'>
            <div style='color:#2D2722;font-size:1.1rem;font-weight:600;'>{user['display_name']}</div>
            <div style='color:#7A7A7A;font-size:0.8rem;'>🎓 {user.get('grade') or '未设置'}</div>
            </div>""",
            unsafe_allow_html=True,
        )

        st.markdown("<div style='margin-top:14px;'></div>", unsafe_allow_html=True)

        # Popover 极简上传按钮
        if "profile_avatar_processed" not in st.session_state:
            st.session_state["profile_avatar_processed"] = False

        with st.popover("📷 上传头像", use_container_width=True):
            uploaded = st.file_uploader(
                "选择图片", type=["png", "jpg", "jpeg"],
                label_visibility="collapsed", key="profile_avatar_uploader",
            )
            if uploaded is not None and not st.session_state["profile_avatar_processed"]:
                st.session_state["profile_avatar_processed"] = True
                raw = uploaded.read()
                avatar_b64_new = base64.b64encode(raw).decode("utf-8")
                ok, _ = update_user_profile(
                    user["id"], avatar_base64=avatar_b64_new, db_path=_DB_PATH,
                )
                if ok:
                    refresh_user_session(_DB_PATH)
                    st.toast("头像已更新 ✅", icon="🖼️")
                    st.rerun()

        st.markdown("<div style='margin-top:10px;'></div>", unsafe_allow_html=True)
        if st.button("🚪 注销", key="my_profile_logout"):
            logout_user()
            st.rerun()

    # ============================================================
    # 中栏：核心资产数据置顶
    # ============================================================
    with col2:
        st.markdown(
            f"""<div style='background-color:#FFFFFF;border:1px solid #EFEFEF;
            border-radius:18px;padding:24px 28px;height:100%;'>
            <div style='color:#7A7A7A;font-size:0.78rem;margin-bottom:16px;
            font-weight:500;letter-spacing:0.04em;'>📊 核心资产总览</div>
            <div style='margin-bottom:18px;'>
                <div style='color:#7A7A7A;font-size:0.75rem;margin-bottom:3px;'>🎯 最近目标岗位</div>
                <div style='color:#2D2722;font-size:1.05rem;font-weight:600;'>{recent_position}</div>
            </div>
            <div style='margin-bottom:18px;'>
                <div style='color:#7A7A7A;font-size:0.75rem;margin-bottom:3px;'>📜 面试模拟存档</div>
                <div style='color:#2D2722;font-size:1.6rem;font-weight:700;line-height:1;'>{total_interview} <span style='font-size:0.85rem;font-weight:400;color:#7A7A7A;'>条</span></div>
            </div>
            <div>
                <div style='color:#7A7A7A;font-size:0.75rem;margin-bottom:3px;'>🧭 灯塔计划存档</div>
                <div style='color:#2D2722;font-size:1.6rem;font-weight:700;line-height:1;'>{total_lighthouse} <span style='font-size:0.85rem;font-weight:400;color:#7A7A7A;'>条</span></div>
            </div>
            </div>""",
            unsafe_allow_html=True,
        )

    # ============================================================
    # 右栏：紧凑版资料修改表单
    # ============================================================
    with col3:
        grade_options = ["大一", "大二", "大三", "大四", "研一", "研二", "研三", "博士", "已毕业"]
        current_grade = user.get("grade") or "未设置"
        try:
            grade_index = grade_options.index(current_grade)
        except ValueError:
            grade_index = 0

        with st.form("profile_edit_form", clear_on_submit=False):
            st.caption("✏️ 修改资料")
            new_display_name = st.text_input(
                "显示名称", value=user["display_name"],
            )
            new_grade = st.selectbox(
                "当前年级", options=grade_options, index=grade_index,
            )
            submitted = st.form_submit_button("💾 保存修改", use_container_width=True)
            if submitted:
                ok, msg = update_user_profile(
                    user["id"],
                    display_name=new_display_name,
                    grade=new_grade,
                    db_path=_DB_PATH,
                )
                if ok:
                    refresh_user_session(_DB_PATH)
                    st.toast("资料已保存 ✅", icon="💾")
                    st.rerun()
                else:
                    st.error(msg)

    st.markdown("---")

    # ============================================================
    # 下半部分：历史资产管家（双 Tab）
    # ============================================================
    tab1, tab2 = st.tabs(["📜 面试模拟历史", "🧭 灯塔计划历史"])

    with tab1:
        _render_my_interview_tab(user)

    with tab2:
        _render_my_lighthouse_tab(user)


def _render_my_interview_tab(user: dict):
    """渲染「我的」页面面试历史 Tab"""
    history = get_interview_history(user["id"], _DB_PATH, limit=50)
    if not history:
        st.markdown(
            "<div style='text-align:center;padding:48px 24px;color:#7A7A7A;'>"
            "<div style='font-size:3rem;margin-bottom:12px;'>✨</div>"
            "<div style='font-size:1.05rem;'>还没有保存过面试模拟记录呢</div>"
            "<div style='font-size:0.9rem;margin-top:6px;'>快去"
            "<span style='color:#2D2722;font-weight:600;'>【面试模拟】</span>"
            "开启你的第一步吧！</div>"
            "</div>",
            unsafe_allow_html=True,
        )
        return

    for record in history:
        with st.container(border=True):
            col1, col2, col3 = st.columns([4, 1, 1])
            with col1:
                company_text = f" · {record['target_company']}" if record.get('target_company') else ""
                st.markdown(f"**{record['target_position']}**{company_text}")
                st.caption(record["created_at"])
            with col2:
                if st.button("查看", key=f"my_interview_view_{record['id']}"):
                    st.session_state["profile_interview_detail"] = record["id"]
                    st.rerun()
            with col3:
                with st.popover("🗑️", key=f"my_interview_popover_{record['id']}"):
                    st.warning("确定要永久删除这条记录吗？")
                    if st.button("确认删除", key=f"my_interview_del_{record['id']}"):
                        if delete_interview_record(record["id"], user["id"], _DB_PATH):
                            st.toast("已删除", icon="🗑️")
                            st.rerun()


def _render_my_lighthouse_tab(user: dict):
    """渲染「我的」页面灯塔历史 Tab"""
    history = get_lighthouse_history(user["id"], _DB_PATH, limit=50)
    if not history:
        st.markdown(
            "<div style='text-align:center;padding:48px 24px;color:#7A7A7A;'>"
            "<div style='font-size:3rem;margin-bottom:12px;'>✨</div>"
            "<div style='font-size:1.05rem;'>还没有保存过灯塔计划记录呢</div>"
            "<div style='font-size:0.9rem;margin-top:6px;'>快去"
            "<span style='color:#2D2722;font-weight:600;'>【灯塔计划】</span>"
            "开启你的第一步吧！</div>"
            "</div>",
            unsafe_allow_html=True,
        )
        return

    for record in history:
        with st.container(border=True):
            col1, col2, col3 = st.columns([4, 1, 1])
            with col1:
                st.markdown(f"**{record['target_position']}** · {record['grade']}")
                st.caption(record["created_at"])
            with col2:
                if st.button("查看", key=f"my_lighthouse_view_{record['id']}"):
                    st.session_state["profile_lighthouse_detail"] = record["id"]
                    st.rerun()
            with col3:
                with st.popover("🗑️", key=f"my_lighthouse_popover_{record['id']}"):
                    st.warning("确定要永久删除这条记录吗？")
                    if st.button("确认删除", key=f"my_lighthouse_del_{record['id']}"):
                        if delete_lighthouse_record(record["id"], user["id"], _DB_PATH):
                            st.toast("已删除", icon="🗑️")
                            st.rerun()


def _render_my_interview_detail(record_id: str):
    """渲染「我的」页面面试记录详情"""
    user = get_current_user()
    if not user:
        st.warning("请先登录")
        return

    record = get_interview_record(record_id, user["id"], _DB_PATH)
    if not record:
        st.error("记录不存在或无权访问")
        st.session_state["profile_interview_detail"] = None
        return

    st.subheader(f"🎤 {record['target_position']} · 历史面试分析")
    if record.get('target_company'):
        st.caption(f"目标公司：{record['target_company']}")
    st.caption(f"分析时间：{record['created_at']}")
    st.markdown("---")

    if record.get("market_report"):
        st.markdown(record["market_report"])
        st.markdown("---")
    if record.get("gap_analysis"):
        st.markdown(record["gap_analysis"])
        st.markdown("---")

    questions = record.get("interview_questions")
    if questions:
        st.subheader("🎯 定制化面试变形题")
        for i, q in enumerate(questions, 1):
            st.markdown(f"> **题目 {i}**\n>\n> {q}\n")

    if st.button("⬅️ 返回我的", key="my_interview_detail_back"):
        st.session_state["profile_interview_detail"] = None
        st.rerun()


def _render_my_lighthouse_detail(record_id: str):
    """渲染「我的」页面灯塔记录详情"""
    user = get_current_user()
    if not user:
        st.warning("请先登录")
        return

    record = get_lighthouse_record(record_id, user["id"], _DB_PATH)
    if not record:
        st.error("记录不存在或无权访问")
        st.session_state["profile_lighthouse_detail"] = None
        return

    st.subheader(f"🎯 {record['target_position']} · {record['grade']} · 历史测评")
    st.caption(f"测评时间：{record['created_at']}")

    _render_lighthouse_result(
        results={
            "tech_tendency": record.get("tech_tendency"),
            "roadmap": record.get("roadmap"),
            "execution_error": None,
        },
        target_position=record["target_position"],
        grade=record["grade"],
    )

    if st.button("⬅️ 返回我的", key="my_lighthouse_detail_back"):
        st.session_state["profile_lighthouse_detail"] = None
        st.rerun()


# ============================================================
# 主函数
# ============================================================
def main():
    """主函数 —— 双向路由 + 登录保护"""
    # 初始化认证表和历史表（幂等，只执行一次）
    init_auth(_DB_PATH)
    init_history(_DB_PATH)

    # 未登录 → 展示登录/注册页
    if not is_logged_in():
        render_login_page()
        return

    # 初始化 session_state
    if "current_page" not in st.session_state:
        st.session_state["current_page"] = "首页"

    # 渲染侧边栏并获取当前页面（带 Emoji 的全名）
    page = render_sidebar()

    # --- 双向同步 ---
    # 1. 侧边栏点击 → 同步到 session_state（按钮下次能拿到正确值）
    short_name = PAGE_REVERSE_MAP.get(page, "首页")
    if st.session_state["current_page"] != short_name:
        st.session_state["current_page"] = short_name

    # 2. 按钮点击 → 计算对应的 Emoji 全名用于路由查找
    route_key = PAGE_MAP.get(st.session_state["current_page"], "🏠 首页")

    # 页面渲染映射
    page_mapping = {
        "🏠 首页": render_home,
        "📊 数据大屏": render_dashboard,
        "🧭 灯塔计划": render_lighthouse,
        "🎤 面试模拟": render_interview,
        "👤 我的": render_my_profile,
        "⚙️ 系统设置": render_settings,
    }

    # 执行页面渲染函数
    render_func = page_mapping.get(route_key, render_home)
    render_func()


if __name__ == "__main__":
    main()
