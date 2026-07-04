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

# RAG 向量存储单例导入
try:
    from utils.rag_loader import get_or_init_collection as _get_or_init_collection
except ImportError:
    _get_or_init_collection = None

# 加载环境变量
load_dotenv()

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
# 侧边栏导航
# ============================================================
def render_sidebar():
    """渲染侧边栏导航（与 st.session_state 双向联动）"""
    st.sidebar.title("🎯 HireInsight")
    st.sidebar.markdown("---")

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
    st.sidebar.info(
        "**开发进度**\n"
        "• 数据大屏：✅ 已完成\n"
        "• 灯塔计划：✅ 已完成\n"
        "• 面试模拟：✅ 已完成\n"
    )

    return page


# ============================================================
# 首页
# ============================================================
def render_home():
    """渲染首页"""
    st.title("🎯 HireInsight Agent")
    st.markdown("**智能求职辅助系统 - 让每一次面试更有准备**")
    
    st.markdown("---")
    
    # 功能卡片
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### 📊 数据大屏")
        st.markdown("实时岗位数据分析\n薪资分布可视化\n城市/学历统计")
        if st.button("进入", key="goto_dashboard"):
            st.session_state["current_page"] = "数据大屏"
            st.rerun()
    
    with col2:
        st.markdown("### 🧭 灯塔计划")
        st.markdown("技术倾向测评\n专属学习路径\n低年级定向规划")
        if st.button("进入", key="goto_lighthouse"):
            st.session_state["current_page"] = "灯塔计划"
            st.rerun()
    
    with col3:
        st.markdown("### 🎤 面试模拟")
        st.markdown("简历 Gap 诊断\nAI 生成面试题\n企业面经 RAG")
        if st.button("进入", key="goto_interview"):
            st.session_state["current_page"] = "面试模拟"
            st.rerun()
    
    st.markdown("---")
    
    # 系统状态
    st.subheader("📋 系统状态")
    
    status_col1, status_col2, status_col3 = st.columns(3)
    
    with status_col1:
        api_configured = check_api_key()
        if api_configured:
            st.success("✅ DeepSeek API 已配置")
        else:
            st.error("❌ DeepSeek API 未配置")
    
    with status_col2:
        rag_dir = os.getenv("CHROMA_PERSIST_DIR", "./data/rag_store")
        if os.path.exists(rag_dir):
            st.success(f"✅ RAG 向量存储已初始化")
        else:
            st.info(f"📁 RAG 向量存储待初始化")
    
    with status_col3:
        db_path = os.getenv("SQLITE_DB_PATH", "./data/jobs.db")
        if os.path.exists(db_path.replace("./", os.getcwd() + "/")):
            st.success(f"✅ 岗位数据库已就绪")
        else:
            st.info(f"📁 数据库待创建")


# ============================================================
# 数据大屏模块（占位符）
# ============================================================
def render_dashboard():
    """渲染数据大屏页面"""
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

    # ---- 3. 公司来源（st.multiselect） ----
    sources = _get_distinct_values_for_dashboard("source")
    selected_sources = st.sidebar.multiselect(
        "公司来源",
        options=sources,
        default=[],
        help="选择数据来源，留空表示全部"
    )

    # ---- 4. 薪资范围（st.slider 双滑块） ----
    # 🔴 关键契约：手动解包元组 → filters["salary_min"] / filters["salary_max"]
    salary_range = st.sidebar.slider(
        "薪资范围 (k/月)",
        min_value=0,
        max_value=100,
        value=(0, 100),
        step=5,
        help="滑动选择月薪区间（单位：千元）"
    )

    # ---- 5. 经验要求（st.slider 双滑块） ----
    # 🔴 关键契约：手动解包元组 → filters["experience_min"] / filters["experience_max"]
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
    # 🔴 Step 3: filters 字典组装（Slider 元组解包契约）
    # ========================================================
    filters: dict = {}

    # 城市（multiselect → list，支持单值或多值）
    if selected_cities:
        filters["city"] = selected_cities

    # 学历（multiselect → list）
    if selected_degrees:
        filters["degree"] = selected_degrees

    # 公司来源（multiselect → list）
    if selected_sources:
        filters["source"] = selected_sources

    # 🔴 薪资 Slider 元组解包：仅当用户修改默认值时添加独立标量键
    if salary_range != (0, 100):
        filters["salary_min"] = salary_range[0]
        filters["salary_max"] = salary_range[1]

    # 🔴 经验 Slider 元组解包：仅当用户修改默认值时添加独立标量键
    if exp_range != (0, 20):
        filters["experience_min"] = exp_range[0]
        filters["experience_max"] = exp_range[1]

    # 关键词
    if keyword.strip():
        filters["keyword"] = keyword.strip()

    # ---- 存入 session_state ----
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
            f"<div style='font-size:0.82rem;color:#a0a0a0;line-height:1.6'>"
            f"📁 <b>数仓状态</b>：离线优先模式<br>"
            f"&nbsp;&nbsp;&nbsp;&nbsp;当前已同步 <b style='color:#00cc96'>{dw_count}</b> 条岗位"
            f"</div>",
            unsafe_allow_html=True,
        )
    else:
        st.sidebar.caption("📁 数仓状态：增量同步待激活")

    # ========================================================
    # 主区域：数据接入 + KPI 指标栏 + Plotly 图表看板
    # ========================================================

    # --------------------------------------------------------
    # 数据接入（严格调用中台接口，零 SQL / 零 Pandas 聚合）
    # 使用标志位驱动，避免早期 return 阻断裂空状态下的控制台渲染
    # --------------------------------------------------------
    _MULTI_VALUE_KEYS = {"city", "degree", "source"}
    single_filters = {k: v for k, v in filters.items() if k not in _MULTI_VALUE_KEYS}

    stats = None
    filtered_df = None
    show_charts = False

    if db_exists:
        try:
            # 调用中台筛选接口（仅传入标量筛选键）
            filtered_df = query_jobs_by_filters(
                _DB_PATH,
                filters=single_filters if single_filters else None,
                limit=None,
            )

            # 后过滤多值键（pandas .isin()，零 SQL）
            for mkey in _MULTI_VALUE_KEYS:
                if mkey in filters:
                    fval = filters[mkey]
                    if isinstance(fval, list) and len(fval) > 0:
                        filtered_df = filtered_df[filtered_df[mkey].isin(fval)]

            # 调用中台统计接口
            stats = calculate_market_metrics(source=filtered_df)

            if not filtered_df.empty and stats.get("meta", {}).get("total_count", 0) > 0:
                show_charts = True

        except FileNotFoundError:
            db_exists = False
            show_charts = False

    # --------------------------------------------------------
    # Step 7: 熔断降级 —— 根据数据库 / 筛选结果分级提示
    # --------------------------------------------------------
    if not db_exists:
        st.info(
            "💡 离线数仓为空，请展开下方【🛠️ 离线数仓控制台】同步初始岗位数据。"
        )
    elif not show_charts:
        st.info(
            "🔍 当前筛选条件下暂无岗位数据，"
            "请调整侧边栏筛选条件或前往底部控制台采集数据。"
        )
    else:
        st.markdown("---")

        # ========================================================
        # Step 4: 顶部 KPI 指标栏（4 列横向卡片）
        # ========================================================
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

        # ========================================================
        # Step 5: 四大 Plotly 图表（2:3 双栏布局，暗色科技风）
        # ========================================================
        left_col, right_col = st.columns([2, 3])

        # -------------------- 左侧（2/5 宽度）--------------------
        with left_col:
            # --- 薪资分布柱状图 ---
            salary_dist = stats["salary"].get("salary_distribution", {})
            if salary_dist:
                fig_salary = go.Figure(data=[go.Bar(
                    x=list(salary_dist.keys()),
                    y=list(salary_dist.values()),
                    text=list(salary_dist.values()),
                    textposition="outside",
                    textfont=dict(size=13, color="#e0e0e0"),
                    marker=dict(
                        color=list(salary_dist.values()),
                        colorscale="Blues",
                        showscale=False,
                        line=dict(width=1, color="#1a1a2e"),
                    ),
                    hovertemplate="薪资段: %{x}<br>岗位数: %{y}<extra></extra>",
                )])
                fig_salary.update_layout(
                    title=dict(text="💰 薪资分布", font=dict(size=16, color="#e0e0e0")),
                    xaxis_title="薪资区间",
                    yaxis_title="岗位数",
                    template="plotly_dark",
                    height=400,
                    margin=dict(l=20, r=20, t=40, b=20),
                    plot_bgcolor="#0e1117",
                    paper_bgcolor="#0e1117",
                )
                st.plotly_chart(fig_salary, use_container_width=True)
            else:
                st.caption("暂无有效薪资数据")

            st.markdown("---")

            # --- 学历占比环形图 ---
            degree_dist = stats["degree"].get("distribution", {})
            if degree_dist:
                degree_labels = list(degree_dist.keys())
                degree_values = [v["count"] for v in degree_dist.values()]
                fig_degree = go.Figure(data=[go.Pie(
                    labels=degree_labels,
                    values=degree_values,
                    hole=0.5,
                    textinfo="label+percent",
                    textfont=dict(size=12, color="#e0e0e0"),
                    marker=dict(
                        colors=["#636efa", "#00cc96", "#ab63fa", "#ffa15a",
                                "#19d3f3", "#ff6692", "#b6e880"],
                        line=dict(width=1, color="#0e1117"),
                    ),
                    hovertemplate="学历: %{label}<br>岗位数: %{value}<br>占比: %{percent}<extra></extra>",
                )])
                fig_degree.update_layout(
                    title=dict(text="🎓 学历占比", font=dict(size=16, color="#e0e0e0")),
                    template="plotly_dark",
                    height=400,
                    margin=dict(l=20, r=20, t=40, b=20),
                    paper_bgcolor="#0e1117",
                )
                st.plotly_chart(fig_degree, use_container_width=True)

        # -------------------- 右侧（3/5 宽度）--------------------
        with right_col:
            # --- 城市需求 Top10 水平条形图 ---
            city_top10 = stats["city"].get("top10", {})
            if city_top10:
                # 按岗位数升序排列（Plotly 从下往上画）
                sorted_cities = sorted(city_top10.items(), key=lambda x: x[1])
                city_names = [c[0] for c in sorted_cities]
                city_counts = [c[1] for c in sorted_cities]
                fig_city = go.Figure(data=[go.Bar(
                    x=city_counts,
                    y=city_names,
                    orientation="h",
                    text=city_counts,
                    textposition="outside",
                    textfont=dict(size=13, color="#e0e0e0"),
                    marker=dict(
                        color=city_counts,
                        colorscale="Viridis",
                        showscale=False,
                        line=dict(width=1, color="#1a1a2e"),
                    ),
                    hovertemplate="城市: %{y}<br>岗位数: %{x}<extra></extra>",
                )])
                fig_city.update_layout(
                    title=dict(text="🌆 城市需求 Top 10", font=dict(size=16, color="#e0e0e0")),
                    xaxis_title="岗位数",
                    yaxis_title=None,
                    template="plotly_dark",
                    height=400,
                    margin=dict(l=20, r=20, t=40, b=20),
                    plot_bgcolor="#0e1117",
                    paper_bgcolor="#0e1117",
                )
                st.plotly_chart(fig_city, use_container_width=True)

            st.markdown("---")

            # --- 经验占比环形图 ---
            exp_dist = stats["experience"].get("distribution", {})
            if exp_dist:
                exp_labels = list(exp_dist.keys())
                exp_values = [v["count"] for v in exp_dist.values()]
                fig_exp = go.Figure(data=[go.Pie(
                    labels=exp_labels,
                    values=exp_values,
                    hole=0.5,
                    textinfo="label+percent",
                    textfont=dict(size=12, color="#e0e0e0"),
                    marker=dict(
                        colors=["#ffa15a", "#19d3f3", "#ab63fa", "#00cc96",
                                "#636efa", "#ff6692", "#b6e880"],
                        line=dict(width=1, color="#0e1117"),
                    ),
                    hovertemplate="经验: %{label}<br>岗位数: %{value}<br>占比: %{percent}<extra></extra>",
                )])
                fig_exp.update_layout(
                    title=dict(text="⏳ 经验要求占比", font=dict(size=16, color="#e0e0e0")),
                    template="plotly_dark",
                    height=400,
                    margin=dict(l=20, r=20, t=40, b=20),
                    paper_bgcolor="#0e1117",
                )
                st.plotly_chart(fig_exp, use_container_width=True)

    # ========================================================
    # Step 6: 一键采集控制台（始终可见，不受空数据库影响）
    # ========================================================
    st.markdown("---")
    with st.expander("🛠️ 离线数仓控制台", expanded=False):
        st.warning(
            "⚠️ 增量更新：从大厂招聘官网抓取最新岗位数据，通过 Upsert 合并入本地离线数仓。"
            "抓取+清洗约需 1-3 分钟，请耐心等待"
        )

        # 🔴 只捕获按钮状态，不做任何 st.status 操作（避免嵌套违规）
        pipeline_clicked = st.button("🔄 一键获取并更新最新数据", type="primary", use_container_width=True)

    # ⚠️ st.status 必须放在 st.expander 外部，否则触发 StreamlitAPIException
    if pipeline_clicked:
        # ---- 尝试加载爬虫模块 ----
        scraper_available = False
        try:
            from utils.scraper.netease_scraper import NetEaseScraper  # noqa: F811
            from utils.scraper.tencent_scraper import TencentScraper  # noqa: F811
            scraper_available = True
        except ImportError:
            pass

        with st.status("🔄 增量同步进行中...", expanded=True) as status:
            # === [1/6] 初始化数仓表结构 ===
            status.update(
                label="[1/6] 初始化数仓表结构...", state="running", expanded=True
            )
            init_sqlite_db(_DB_PATH)

            # === [2/6] 爬虫抓取 ===
            raw_jobs: list[dict] = []
            used_real_scraper = False  # 标记是否尝试了真机爬虫

            if scraper_available:
                status.update(
                    label="[2/6] 增量采集 | 实时抓取中（NetEase + Tencent）...",
                    state="running",
                    expanded=True,
                )
                try:
                    with NetEaseScraper(max_pages=5) as netease_scraper:
                        netease_raw = netease_scraper.crawl()
                        st.write(f"✅ 网易：抓取 {len(netease_raw)} 条")
                        raw_jobs.extend(netease_raw)
                except Exception as e:
                    st.warning(f"⚠️ 网易爬虫异常：{e}")

                try:
                    with TencentScraper(max_pages=5) as tencent_scraper:
                        tencent_raw = tencent_scraper.crawl()
                        st.write(f"✅ 腾讯：抓取 {len(tencent_raw)} 条")
                        raw_jobs.extend(tencent_raw)
                except Exception as e:
                    st.warning(f"⚠️ 腾讯爬虫异常：{e}")

                used_real_scraper = True

            # 🔄 智能容灾降级：真机爬虫失败或返回空数据 → 自动切换到本地高拟真模拟通道
            if not raw_jobs:
                if used_real_scraper:
                    status.update(
                        label="[2/6] 增量采集 | 实时接口异常，智能降级至模拟通道...",
                        state="running",
                        expanded=True,
                    )
                    st.info("📡 实时接口遭遇网络反爬，已自动切入本地高拟真数据通道保障演示。")
                else:
                    status.update(
                        label="[2/6] 增量采集 | 爬虫模块未安装，生成模拟数据...",
                        state="running",
                        expanded=True,
                    )

                # 高拟真模拟岗位样本：7 城市 × 5 学历档次 = 35 条
                mock_cities = ["深圳", "北京", "上海", "杭州", "成都", "广州", "武汉"]
                mock_degrees = ["本科", "硕士", "博士", "大专", "不限"]
                mock_companies = ["字节跳动", "腾讯", "阿里巴巴", "美团", "百度"]
                for i, city in enumerate(mock_cities):
                    for j, degree in enumerate(mock_degrees):
                        raw_jobs.append({
                            "id": f"mock_job_{i}_{j}",
                            "original_id": f"mock_job_{i}_{j}",
                            "title_raw": f"Python后端开发工程师-{city}",
                            "company": mock_companies[(i + j) % len(mock_companies)],
                            "city_raw": city,
                            "salary_raw": f"{15 + j*5}k-{30 + j*10}k",
                            "experience_raw": f"{j}-{j+3}年" if j > 0 else "应届生",
                            "degree_raw": degree,
                            "category": "技术",
                            "department": "研发部",
                            "duty": f"负责{city}区域后台系统开发",
                            "requirement": "熟悉Python/Go",
                            "skills": "Python, MySQL, Redis",
                            "post_url": f"https://example.com/job/{i}_{j}",
                            "work_type": "全职",
                            "source": "mock_bytedance",
                            "published_at": "2026-06-15",
                            "updated_at": "2026-06-20",
                        })
                st.write(f"✅ 模拟数据：生成 {len(raw_jobs)} 条")
                scraper_available = False

            # === [3/6] 增量同步 | 数据转换 ===
            status.update(
                label="[3/6] 增量同步 | 多源数据转换（Transformer）...",
                state="running",
                expanded=True,
            )
            transformed_jobs: list[dict] = []
            if scraper_available:
                # 按来源分离转换（source 由各爬虫在去重阶段注入）
                netease_raw = [j for j in raw_jobs if j.get("source") == "netease"]
                tencent_raw = [j for j in raw_jobs if j.get("source") == "tencent"]
                other_raw = [j for j in raw_jobs
                             if j not in netease_raw and j not in tencent_raw]

                if netease_raw:
                    transformed_jobs.extend(transform_jobs(netease_raw, source="netease"))
                if tencent_raw:
                    transformed_jobs.extend(transform_jobs(tencent_raw, source="tencent"))
                if other_raw:
                    transformed_jobs.extend(other_raw)  # 模拟数据已是标准格式
            else:
                transformed_jobs = raw_jobs  # 模拟数据直接使用

            st.write(f"✅ 转换完成：{len(transformed_jobs)} 条标准记录")

            # === [4/6] 增量同步 | 数据清洗 ===
            status.update(
                label="[4/6] 增量同步 | 正则清洗与标准化（Cleaner）...",
                state="running",
                expanded=True,
            )
            df_cleaned = clean_job_data(pd.DataFrame(transformed_jobs))
            st.write(f"✅ 清洗完成：{len(df_cleaned)} 条有效记录")

            # === [5/6] 增量同步 | 数仓写入 ===
            status.update(
                label="[5/6] 增量同步 | 写入离线数仓（Upsert）...",
                state="running",
                expanded=True,
            )
            n_written = save_to_sqlite(df_cleaned, _DB_PATH)
            st.write(f"✅ 数仓写入：{n_written} 条已持久化")

            # === [6/6] 增量同步完成 ===
            status.update(
                label="[6/6] 增量同步完成 ✅",
                state="complete",
                expanded=False,
            )

        # 🔴 关键：清空全量缓存，防止 st.rerun() 后命中旧缓存
        st.cache_data.clear()
        st.toast("🎉 增量更新完成！本地数仓已成功合并最新岗位。", icon="🎉")
        st.rerun()


# ============================================================
# 灯塔计划模块（占位符）
# ============================================================
def render_lighthouse():
    """渲染灯塔计划页面 —— 两阶段 UI：情景测评 → 技术倾向雷达图 + Roadmap 静态看板"""
    st.title("🧭 灯塔计划")
    st.markdown("**为你的技术生涯点亮第一座灯塔**")
    st.markdown("---")

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

                # ---- 加载题库 ----
                all_questions = get_full_question_bank()

                # ---- 构造初始 State ----
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

                # ---- 调用 QuestionFilterNode（st.status 包裹） ----
                with st.status("🧠 正在分析最适合你的测评题目...", expanded=True) as status:
                    try:
                        filter_result = question_filter_node(initial_state)
                    except Exception as e:
                        # 降级：LLM 选题失败 → 使用全部 10 题
                        filter_result = {
                            "filtered_questions": all_questions,
                            "current_step": "question_filter_failed",
                            "execution_error": f"题目筛选失败，已使用完整题库: {str(e)}",
                        }
                    status.update(label="✅ 选题完成", state="complete")

                # ---- 写入 session_state 缓存（防 re-run 重复调用 LLM） ----
                st.session_state.lighthouse_filtered_questions = filter_result.get(
                    "filtered_questions", all_questions
                )
                st.session_state.lighthouse_quiz_started = True
                st.session_state.lighthouse_target_position = target_position.strip()
                st.session_state.lighthouse_grade = grade

                # 如果选题阶段产生了 execution_error，记录但不阻断答题
                if filter_result.get("execution_error"):
                    st.warning(f"⚠️ {filter_result['execution_error']}")

                st.rerun()

    # ============================================================
    # 阶段一-步骤 2：答题表单（st.form 包裹，杜绝 radio 每次 click 重运行）
    # ============================================================
    if (
        st.session_state.lighthouse_quiz_started
        and st.session_state.lighthouse_filtered_questions
    ):
        questions = st.session_state.lighthouse_filtered_questions
        st.subheader(f"📋 技术倾向测评（共 {len(questions)} 道情景选择题）")
        st.caption("请仔细阅读每道题的情景描述，选择最符合你真实偏好的选项。没有对错之分，请诚实作答。")

        with st.form("lighthouse_quiz_form", clear_on_submit=False):
            # ---- 渲染每道题 ----
            user_selections: dict = {}  # {question_id: selected_option_index}

            for idx, q in enumerate(questions):
                qid = q["id"]
                st.markdown(f"### 题目 {idx + 1}")
                st.markdown(f"**场景：** {q['scenario']}")
                st.markdown("")

                option_labels = []
                for opt_idx, opt in enumerate(q["options"]):
                    label = f"{chr(65 + opt_idx)}. {opt['text']}"  # A. B. C. D.
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

            # ---- 提交按钮 ----
            submitted = st.form_submit_button(
                "📊 生成我的技术路线图",
                type="primary",
                use_container_width=True,
            )

            if submitted:
                # 校验：所有题目必须回答
                unanswered = [qid for qid, sel in user_selections.items() if sel is None]
                if unanswered:
                    st.error(f"还有 {len(unanswered)} 道题目未作答，请完成所有题目后再提交")
                    return

                # ---- 构造 user_answers ----
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

                # ---- 调用 AssessmentNode（st.status 包裹） ----
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

                # ---- 存入 session_state → 触发阶段二渲染 ----
                st.session_state.lighthouse_results = assess_result
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
    st.title("🎤 面试模拟")
    st.markdown("**AI 驱动的简历诊断 + 定制化面试题生成**")

    if not check_api_key():
        render_api_key_warning()
        return

    st.success("✅ DeepSeek API 已配置，可以开始面试模拟")

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
# 主函数
# ============================================================
def main():
    """主函数 —— 双向路由：侧边栏 radio ↔ 首页按钮联动"""
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
        "⚙️ 系统设置": render_settings,
    }

    # 执行页面渲染函数
    render_func = page_mapping.get(route_key, render_home)
    render_func()


if __name__ == "__main__":
    main()
