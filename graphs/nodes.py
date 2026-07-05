"""
HireInsight-Agent - LangGraph Agent 节点实现

三个核心 Agent 节点：
1. Market_Agent - 市场趋势分析
2. Critic_Agent - 简历差距诊断
3. Interviewer_Agent - 面试题生成

所有节点遵循同步阻塞模式，不引入 asyncio
"""
from typing import Dict, Any, List, Optional
import os
import pandas as pd
import streamlit as st
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from .state import InterviewState

# 尝试导入工具层函数（从项目根目录运行的绝对导入）
try:
    from utils.data_stats import generate_agent_prompt_summary as _generate_prompt_summary
except ImportError:
    _generate_prompt_summary = None

try:
    from utils.rag_loader import query_experiences as _query_experiences
except ImportError:
    _query_experiences = None


# ============================================================
# LLM 客户端初始化（模块级单例 + 硬超时保护）
# ============================================================
_LLM_CLIENT = None


def get_llm_client():
    """
    获取 DeepSeek LLM 客户端（模块级单例 + 硬超时保护）

    关键设计：
    - 使用 ChatOpenAI 兼容接口（非 ChatDeepSeek，避免老版本不支持新模型参数）
    - 模型：deepseek-reasoner（具备 Reasoning 思考链能力）
    - timeout=120：锁死 2 分钟硬超时，Windows 下唯一的"逃生通道"
    - max_retries=0：不做自动重试，让异常快速冒泡到外层 try...except
    """
    global _LLM_CLIENT
    if _LLM_CLIENT is None:
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            raise ValueError("DEEPSEEK_API_KEY 环境变量未设置，请在 .env 文件中配置")
        _LLM_CLIENT = ChatOpenAI(
            model="deepseek-reasoner",
            openai_api_base="https://api.deepseek.com/v1",
            openai_api_key=api_key,
            timeout=120,          # 2 分钟硬超时
            max_retries=0,        # 不做自动重试，异常快速冒泡
        )
    return _LLM_CLIENT


def call_deepseek(system_prompt: str, user_prompt: str) -> str:
    """
    调用 DeepSeek API 生成文本

    Args:
        system_prompt: 系统提示词（角色定义）
        user_prompt:  用户提示词（输入数据）

    Returns:
        LLM 生成的文本内容
    """
    llm = get_llm_client()
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt),
    ]
    response = llm.invoke(messages)
    return response.content


# ============================================================
# Node 1: Market_Agent - 市场趋势分析
# ============================================================
def market_agent_node(state: InterviewState) -> InterviewState:
    """
    Market_Agent 节点

    输入：
        - target_position: 目标岗位
        - market_data:    市场数据 DataFrame
        - market_summary: 市场数据统计字典（优先使用）

    输出：
        - market_report: 市场趋势与竞争分析报告

    flows -> 进度日志通过 st.write() 上报到 st.status 容器
    """
    st.write("🔄 **[1/3] Market_Agent** 正在分析市场趋势与竞争格局...")

    target_position = state.get("target_position", "")
    market_data: Optional[pd.DataFrame] = state.get("market_data")
    market_summary: Optional[Dict[str, Any]] = state.get("market_summary")

    # ---- 构造市场数据摘要文本 ----
    market_summary_text = ""

    # 优先使用 statistics dict（由 data_stats 预计算好的完整统计）生成摘要
    if market_summary and _generate_prompt_summary is not None:
        market_summary_text = _generate_prompt_summary(market_summary, target_position)
    elif market_data is not None:
        # 降级：从 DataFrame 现场计算基本统计
        total_jobs = len(market_data)
        position_count = (
            market_data["title"].nunique()
            if "title" in market_data.columns
            else 0
        )

        parts = [
            f"# 📊 {target_position or '全量'} 岗位市场数据看板",
            f"> 总岗位数：**{total_jobs}** 条 | 涉及岗位数：**{position_count}** 个",
            "",
        ]

        # 薪资统计
        if "salary_min" in market_data.columns and "salary_max" in market_data.columns:
            avg_min = market_data["salary_min"].mean()
            avg_max = market_data["salary_max"].mean()
            median_min = market_data["salary_min"].median()
            parts.append("## 💰 一、薪资水平")
            parts.append(f"- 平均薪资范围：{avg_min:.1f}K - {avg_max:.1f}K/月")
            parts.append(f"- 中位数最低薪资：{median_min:.1f}K/月")
            parts.append("")

        # 城市分布
        if "city" in market_data.columns:
            top_cities = market_data["city"].value_counts().head(5)
            parts.append("## 🏙️ 二、城市分布 (Top 5)")
            for city, count in top_cities.items():
                parts.append(f"- {city}：{count} 个岗位")
            parts.append("")

        # 学历分布
        if "education_cleaned" in market_data.columns:
            edu_dist = market_data["education_cleaned"].value_counts()
            parts.append("## 🎓 三、学历要求分布")
            for edu, count in edu_dist.items():
                parts.append(f"- {edu}：{count} 个岗位 ({count / total_jobs * 100:.1f}%)")
            parts.append("")

        market_summary_text = "\n".join(parts)
    else:
        market_summary_text = (
            f"目标岗位：{target_position}\n"
            "（暂无市场数据，请先运行数据大屏模块加载岗位数据）"
        )

    # ---- 构造 Prompt 并调用 DeepSeek ----
    system_prompt = (
        "你是一位资深的人力资源市场分析师，专注于互联网/科技行业的招聘趋势分析。\n\n"
        "请基于提供的市场数据，为求职者撰写一份专业的《市场趋势与竞争分析报告》。\n\n"
        "报告结构要求：\n"
        "1. **市场概况**：目标岗位的整体市场需求量、地域分布特征\n"
        "2. **薪资竞争力**：基于数据分析薪资水平、范围与竞争力\n"
        "3. **学历与经验门槛**：市场对学历和经验的普遍要求\n"
        "4. **求职建议**：针对该岗位的求职策略和技能提升方向\n\n"
        "请使用具体的数据支撑你的结论，语言专业但不晦涩，结论要有可操作性。\n"
        '输出格式为结构化的 Markdown，从 "## 市场趋势与竞争分析报告" 开始。'
    )

    user_prompt = (
        f'以下是为 "{target_position}" 岗位汇总的最新市场数据，请据此撰写分析报告：\n\n'
        f"{market_summary_text}"
    )

    market_report = call_deepseek(system_prompt, user_prompt)
    st.write("✨ 市场报告生成完毕，正在注入上下文...")
    st.write("✅ **[1/3] Market_Agent** 分析完成")

    return {
        "market_report": market_report,
        "current_step": "market_analysis_completed",
    }


# ============================================================
# Node 2: Critic_Agent - 简历差距诊断
# ============================================================
def critic_agent_node(state: InterviewState) -> InterviewState:
    """
    Critic_Agent 节点（简历"找茬"诊断）

    输入：
        - user_resume:     简历文本
        - market_report:   市场报告（来自 Market_Agent）
        - target_position: 目标岗位（用于 RAG 检索）

    输出：
        - gap_analysis: 技能差距诊断结果
        - rag_context:  RAG 检索结果（面经片段）

    flows -> 进度日志通过 st.write() 上报到 st.status 容器
    """
    st.write("🔄 **[2/3] Critic_Agent** 正在检索企业面经并进行简历差距诊断...")

    user_resume = state.get("user_resume", "")
    market_report = state.get("market_report", "")
    target_position = state.get("target_position", "")

    # ---- RAG 检索：获取 Top-2 企业面经片段 ----
    rag_context: List[Dict] = []
    rag_context_str = ""
    if _query_experiences is not None and target_position:
        try:
            rag_context = _query_experiences(query=target_position, n_results=2)
        except Exception:
            pass  # RAG 检索失败不阻断流程，静默降级

    if rag_context:
        parts = ["\n\n### 🔍 企业面经参考\n"]
        for i, item in enumerate(rag_context, 1):
            doc = item.get("document", "")
            meta = item.get("metadata", {})
            company = meta.get("company", meta.get("source", "未知企业"))
            parts.append(f"**面经片段 {i}**（{company}）:\n> {doc[:300]}...\n")
        rag_context_str = "\n".join(parts)

    # ---- 构造 Prompt 并调用 DeepSeek ----
    system_prompt = (
        "你是一位资深的招聘面试官和技术评审专家，擅长对照市场 JD 与候选人简历进行差距分析。\n\n"
        "请基于候选人简历、市场岗位数据和真实企业面经，进行系统性差距诊断。\n\n"
        "诊断维度（至少 5 个）：\n"
        "1. **技术栈匹配度**：简历中的技能是否覆盖市场主流要求\n"
        "2. **项目经验深度**：项目复杂度是否达到行业标准\n"
        "3. **行业/业务理解**：是否具备目标岗位所需的业务领域知识\n"
        "4. **学历与证书**：是否符合市场普遍门槛\n"
        "5. **软技能**：沟通、管理、协作等方面的竞争力\n\n"
        "输出要求：\n"
        '- 报告标题为 "## 技能差距诊断报告"\n'
        "- 使用 Markdown 表格展示诊断结果，表头为："
        "| 诊断维度 | 当前水平评估 | 市场要求标准 | 差距等级 | 提升建议 |\n"
        "- 差距等级分为：🟢 无差距 / 🟡 小差距 / 🟠 中等差距 / 🔴 显著差距\n"
        "- 表格后附一段 100-200 字的综合评语\n"
        "- 请客观专业，不要过度否定或夸大，结论要有依据。"
    )

    user_prompt = (
        f"## 候选人简历\n{user_resume[:2000]}\n\n"
        f"---\n\n"
        f"## 市场趋势与竞争分析报告\n{market_report[:2000]}\n\n"
        f"{rag_context_str}\n"
        f"---\n\n"
        f"请对上述候选人进行全面的技能差距诊断，按要求的 Markdown 格式输出。"
    )

    gap_analysis = call_deepseek(system_prompt, user_prompt)
    st.write("✅ **[2/3] Critic_Agent** 差距诊断完成")

    return {
        "gap_analysis": gap_analysis,
        "rag_context": [item.get("document", "") for item in rag_context],
        "current_step": "gap_analysis_completed",
    }


# ============================================================
# Node 3: Interviewer_Agent - 面试题生成
# ============================================================
def interviewer_agent_node(state: InterviewState) -> InterviewState:
    """
    Interviewer_Agent 节点（面试题生成器）

    输入：
        - gap_analysis:    技能差距诊断结果
        - rag_context:     RAG 检索结果（面经片段）
        - target_position: 目标岗位

    输出：
        - interview_questions: 3-5 道定制面试变形题 (List[str])
        - market_report:       面试题 Markdown 看板（复用字段供前端统一渲染）

    flows -> 进度日志通过 st.write() 上报到 st.status 容器
    """
    st.write("🔄 **[3/3] Interviewer_Agent** 正在根据技能盲区定制面试变形题...")

    gap_analysis = state.get("gap_analysis", "")
    target_position = state.get("target_position", "")
    rag_context: Optional[List[str]] = state.get("rag_context", [])

    # ---- 面经参考拼接 ----
    rag_context_str = ""
    if rag_context:
        rag_context_str = (
            "\n\n### 🔍 真实面经参考\n"
            + "\n".join([f"> {ctx[:200]}..." for ctx in rag_context[:2]])
        )

    # ---- 构造 Prompt 并调用 DeepSeek ----
    system_prompt = (
        "你是一位资深的大厂面试官，专攻技术岗位的深度面试。\n\n"
        "请基于候选人的技能差距诊断报告，动态生成 3-5 道定制化面试变形题。\n\n"
        "出题原则：\n"
        "1. **针对弱项**：每道题必须精准打击差距报告中的技能盲区\n"
        "2. **层层递进**：从基础概念 → 实战应用 → 架构设计，逐步加深\n"
        "3. **结合真实面经**：参考企业真实面试风格出题，避免模板化\n"
        "4. **场景化**：用具体业务场景包装问题，考察候选人的实际问题解决能力\n\n"
        "输出格式：\n"
        '- 报告标题为 "## 🎯 定制化面试变形题"\n'
        "- 每道题使用以下格式：\n"
        "```\n"
        "### 题目 N：{简短标题}\n"
        "**考察方向**：{该题针对的技能盲区}\n"
        "**题目内容**：{详细的场景化面试题描述}\n"
        "**追问方向**：{面试官可能追问的 1-2 个方向}\n"
        "```\n"
        "- 3-5 道题，不可多也不可少\n"
        "- 报告末尾加一行 > 💡 提示信息\n\n"
        "- 在提示信息之后，以 '## 🔮 建议追问方向' 为标题，列出 3 条针对候选人弱项的深挖追问标签。"
        "每条以 '- ' 开头，必须是口语化、有行动感的具体问题或指令。\n"
        "示例：\n"
        "```\n"
        "## 🔮 建议追问方向\n"
        "- 帮我针对「分布式一致性」这个盲区生成 3 道阶梯式面试题\n"
        "- 推荐 2 个能用业余时间补齐消息队列短板的开源项目\n"
        "- 对比一下我在简历中缺少的 Docker/K8s 技能与市场要求的 Gap 有多大\n"
        "```\n"
    )

    user_prompt = (
        f"## 技能差距诊断报告\n{gap_analysis}\n\n"
        f"---\n\n"
        f"## 目标岗位\n{target_position}\n\n"
        f"{rag_context_str}\n"
        f"---\n\n"
        f"请基于以上诊断报告，为候选人动态生成 3-5 道定制化面试题，按要求的 Markdown 格式输出。"
    )

    interview_output = call_deepseek(system_prompt, user_prompt)

    # ---- 从 LLM 输出中解析题目列表 ----
    import re
    question_blocks = re.split(r'###\s*题目\s*\d+[：:]', interview_output)
    interview_questions: List[str] = []
    for block in question_blocks[1:]:  # 跳过第一个空段（标题之前的内容）
        match = re.search(
            r'\*\*题目内容\*\*[：:]\s*(.+?)(?=\n\*\*追问|\Z)', block, re.DOTALL
        )
        if match:
            interview_questions.append(match.group(1).strip())

    if not interview_questions:
        # 降级：解析失败时用整个输出作为单条题目
        interview_questions = [interview_output]

    # ---- 解析建议追问方向 ----
    suggested_follow_ups: List[str] = []
    follow_up_match = re.search(
        r'##\s*🔮\s*建议追问方向\s*\n((?:\s*-\s*.+\n?)+)',
        interview_output
    )
    if follow_up_match:
        follow_up_lines = follow_up_match.group(1).strip().split('\n')
        for line in follow_up_lines:
            cleaned = re.sub(r'^\s*-\s*', '', line).strip()
            if cleaned:
                suggested_follow_ups.append(cleaned)

    # ---- 从 market_report（静态看板）中移除建议追问方向块，保持看板干净 ----
    clean_report = re.sub(
        r'\n*##\s*🔮\s*建议追问方向\s*\n(?:\s*-\s*.+\n?)+', '', interview_output
    ).rstrip()

    st.write("✅ **[3/3] Interviewer_Agent** 面试题生成完毕")

    return {
        "interview_questions": interview_questions,
        "market_report": clean_report,  # 干净的静态看板（不含追问标签）
        "suggested_follow_ups": suggested_follow_ups,  # 追问标签单独传递
        "current_step": "interview_generation_completed",
        "is_completed": True,
    }


# ============================================================
# 节点注册表（供图构建使用）
# ============================================================
AGENT_NODES = {
    "market_agent": market_agent_node,
    "critic_agent": critic_agent_node,
    "interviewer_agent": interviewer_agent_node,
}
