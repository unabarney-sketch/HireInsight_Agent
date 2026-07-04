"""
HireInsight-Agent - 灯塔计划 Agent 节点实现

两个核心节点：
1. QuestionFilterNode - LLM 动态选题（从 10 道题库筛选 5-7 题）
2. AssessmentNode   - 技术倾向评估 + Roadmap 生成

所有节点遵循同步阻塞模式，不引入 asyncio。
"""
from typing import Dict, Any, Optional, List
import json
import re
import logging
import streamlit as st

from .state import LighthousePlanState
from .nodes import get_llm_client, call_deepseek

logger = logging.getLogger(__name__)

# ============================================================
# JSON 解析工具函数（防范 DeepSeek Reasoner 思考链污染）
# ============================================================

def _extract_json_from_reasoner_response(text: str) -> Optional[Dict[str, Any]]:
    """
    从 DeepSeek Reasoner 的响应中提取 JSON 对象。

    问题背景：deepseek-reasoner 会在输出最终结果前先输出大段思考链（CoT），
    直接用 json.loads() 必然失败。本函数实现多层 Fallback 解析。

    Args:
        text: LLM 原始响应文本

    Returns:
        解析成功的 dict，或 None（需要降级处理）
    """
    if not text or not text.strip():
        return None

    original_text = text

    # ---- 第 1 层：精准提取 ```json ... ``` fenced code block ----
    match = re.search(r'```json\s*(\{.*?\})\s*```', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            # 第 1 层失败，继续尝试
            pass

    # ---- 第 2 层：粗暴截取首尾 { 和 } 之间的内容 ----
    start = text.find('{')
    end = text.rfind('}')
    if start != -1 and end != -1 and end > start:
        candidate = text[start:end + 1]
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            # 第 2 层失败，继续尝试
            pass

    # ---- 第 3 层（灾难降级）：返回 None，由调用方将原始文本写入 roadmap ----
    logger.warning(
        "JSON 解析全部失败，原始响应前 500 字符:\n%s",
        original_text[:500]
    )
    return None


# ============================================================
# Node A: QuestionFilterNode - 动态选题
# ============================================================

def question_filter_node(state: LighthousePlanState) -> Dict[str, Any]:
    """
    QuestionFilterNode：根据目标岗位和年级，从题库中筛选 5-7 道最相关的题目。

    职责：
    - 读取 state 中的 target_position、grade、all_questions
    - 调用 LLM 进行语义匹配筛选
    - 将筛选结果写入 filtered_questions
    - 通过 st.write() 向 st.status 容器上报进度

    异常处理：
    - LLM 调用超时/异常 → 降级为返回全部 10 道题（兜底策略）
    - 降级时设置 execution_error 但 is_completed=False，允许用户继续答题
    """
    if not state.get("execution_error"):
        st.write("🔄 **[1/2] QuestionFilterNode** 正在根据你的目标岗位筛选最合适的测评题目...")

    target_position = state.get("target_position", "")
    grade = state.get("grade", "")
    all_questions = state.get("all_questions", [])

    if not all_questions:
        # 无题库数据时直接放行（前端会在 st.session_state 中兜底）
        return {
            "filtered_questions": [],
            "current_step": "question_filter_completed",
        }

    # ---- 构造题库文本 ----
    # 只传 tags + id + scenario 给 LLM，不传完整 options（减少 token 消耗 + 避免倾向泄露）
    questions_brief = []
    for q in all_questions:
        tags_str = ", ".join(q.get("tags", []))
        questions_brief.append(
            f"题目 {q['id']} [标签: {tags_str}] [分类: {q['category']}]\n"
            f"场景: {q['scenario'][:120]}..."
        )
    questions_text = "\n\n".join(questions_brief)

    # ---- 构造 Prompt ----
    system_prompt = (
        "你是一位资深的技术职业规划师，专注于为低年级学生设计技术倾向测评。\n\n"
        "你的任务：从 10 道情景多选题中，筛选出 5-7 道与用户的目标岗位最相关的题目。\n\n"
        "筛选原则：\n"
        "1. 题目的 tags 与目标岗位所需技术栈语义匹配（如 'Java后端开发' 优先选含 'JVM'、'并发'、'Spring' 标签的题）\n"
        "2. 考虑用户年级：低年级（大一大二）优先选 'medium' 难度的题，高年级（大三大四/研）可包含 'hard' 难度\n"
        "3. 覆盖广度：尽量从不同 category 中选题，避免集中在同一类别\n"
        "4. 最少选 5 题，最多选 7 题\n\n"
        "输出格式：仅输出一个 JSON 数组，包含被选中题目的 id。\n"
        '示例输出：```json\\n[1, 3, 5, 7, 8, 10]\\n```\n'
        "不要输出任何解释性文字，只输出 JSON 数组。"
    )

    user_prompt = (
        f"用户目标岗位：{target_position}\n"
        f"用户年级：{grade}\n\n"
        f"以下是题库（共 10 题）：\n\n{questions_text}\n\n"
        f"请根据上述筛选原则，选出 5-7 道最适合该用户的题目 ID，仅输出 JSON 数组。"
    )

    # ---- 调用 LLM + 超时保护 ----
    try:
        response = call_deepseek(system_prompt, user_prompt)
    except Exception as e:
        logger.error("QuestionFilterNode LLM 调用失败: %s", e)
        # 降级：返回全部 10 题，不阻断流程
        return {
            "filtered_questions": all_questions,
            "current_step": "question_filter_failed",
            "execution_error": f"QuestionFilterNode 超时或异常: {str(e)}",
        }

    # ---- 解析 LLM 返回的题目 ID ----
    try:
        # 提取可能的 JSON 数组（可能被 ```json 包裹或裸数组）
        match = re.search(r'```(?:json)?\s*(\[.*?\])\s*```', response, re.DOTALL)
        if match:
            ids_raw = json.loads(match.group(1))
        else:
            # 尝试直接匹配方括号内容
            bracket_match = re.search(r'\[([0-9,\s]+)\]', response)
            if bracket_match:
                ids_raw = json.loads(bracket_match.group(0))
            else:
                raise ValueError("无法从 LLM 响应中提取题目 ID 列表")

        selected_ids = [int(x) for x in ids_raw if isinstance(x, (int, float))]
    except Exception as e:
        logger.warning("QuestionFilterNode 题目 ID 解析失败: %s，降级为全部 10 题", e)
        return {
            "filtered_questions": all_questions,
            "current_step": "question_filter_failed",
            "execution_error": f"题目筛选失败，已使用完整题库: {str(e)}",
        }

    # ---- 根据 ID 过滤题目 ----
    filtered = [q for q in all_questions if q["id"] in selected_ids]

    # 兜底：如果筛选结果不在 5-7 范围内，回退为全部题目
    if len(filtered) < 5 or len(filtered) > 7:
        logger.warning(
            "LLM 筛选结果 %d 题（不在 5-7 范围），回退为全部 10 题", len(filtered)
        )
        filtered = all_questions

    if not state.get("execution_error"):
        st.write(f"✅ 已从题库中筛选出 **{len(filtered)}** 道最适合你的情景题")
        st.write("✅ **[1/2] QuestionFilterNode** 选题完成")

    return {
        "filtered_questions": filtered,
        "current_step": "question_filter_completed",
    }


# ============================================================
# Node B: AssessmentNode - 技术倾向评估 + Roadmap 生成
# ============================================================

def assessment_node(state: LighthousePlanState) -> Dict[str, Any]:
    """
    AssessmentNode：评估用户技术倾向 + 生成结构化学习路线图。

    职责：
    - 读取 user_answers、target_position、grade
    - 调用 LLM 计算六维度技术倾向（0-100）+ 生成 Roadmap JSON
    - 解析 LLM 输出的 JSON（多层 Fallback）
    - 超时/异常时写入 execution_error 并标记 is_completed=True

    关键防范：
    - DeepSeek Reasoner 的 CoT 思考链污染 → _extract_json_from_reasoner_response()
    - LLM 超时/网络异常 → try-except 包裹全节点，返回降级状态
    """
    if not state.get("execution_error"):
        st.write("🔄 **[2/2] AssessmentNode** 正在评估你的技术倾向并生成个性化学习路线图...")

    target_position = state.get("target_position", "")
    grade = state.get("grade", "")
    user_answers = state.get("user_answers", [])

    # ---- 构造答题摘要 ----
    if not user_answers:
        answers_text = "（用户未作答任何题目）"
    else:
        lines = []
        for ans in user_answers:
            qid = ans.get("question_id", "?")
            tendency = ans.get("tendency", "未知")
            option_text = ans.get("option_text", "")
            scenario = ans.get("scenario", "")
            lines.append(
                f"题目 {qid}:\n"
                f"  场景: {scenario[:80]}...\n"
                f"  选择倾向: {tendency}\n"
                f"  选项内容: {option_text[:100]}...\n"
            )
        answers_text = "\n".join(lines)

    # ---- 构造 Prompt ----
    system_prompt = (
        "你是一位资深的技术职业规划专家和人才评估顾问，\n"
        "专注于通过情景选择题分析求职者的技术倾向并输出个性化学习路线图。\n\n"
        "## 任务一：技术倾向评估\n"
        "基于用户的答题选择，量化其在六大技术方向上的倾向分值（0-100 分）：\n"
        '- "前端": 前端开发（React/Vue/Webpack/浏览器渲染/状态管理等）\n'
        '- "后端": 后端开发（Java/Spring/数据库/消息队列/分布式架构等）\n'
        '- "AI数据": AI与数据（机器学习/深度学习/数据分析/数学建模/LLM等）\n'
        '- "测试运维": 测试与运维（CI/CD/自动化测试/容器化/可观测性/混沌工程等）\n'
        '- "产品": 产品管理（用户需求/商业价值/数据分析/项目管理/UX等）\n'
        '- "客户端": 客户端与底层（OS/网络协议/渲染引擎/GPU编程/嵌入式/Rust/C++等）\n\n'
        "评分原则：\n"
        "1. 每题选中的 tendency 方向获得加分（基础分 70-85）\n"
        "2. 未被选中的方向给 20-45 分（展现区分度）\n"
        "3. 结合目标岗位和年级做微调（如目标为后端，后端分可略微上浮）\n"
        "4. 所有分值必须确保六个方向有明显差异而非平均分布\n\n"
        "## 任务二：学习路线图生成\n"
        "根据技术倾向分析结果，生成一份个性化学习路线图 JSON，包含：\n"
        '- "stages": 阶段列表，每个阶段含 name（如"入门"、"进阶"、"实战"、"专精"）、duration（预计时间）、'
        'milestones（里程碑列表）、resources（推荐资源列表，每项含 title、type、url 或 description）\n'
        '- "summary": 100-200 字综合评语，总结用户的技术画像和学习建议\n\n'
        "## 输出格式要求\n"
        "你必须输出一个完整的 JSON 对象，包裹在 ```json 代码块中。JSON 结构如下：\n"
        "```json\n"
        "{\n"
        '  "tech_tendency": {\n'
        '    "前端": 整数,\n'
        '    "后端": 整数,\n'
        '    "AI数据": 整数,\n'
        '    "测试运维": 整数,\n'
        '    "产品": 整数,\n'
        '    "客户端": 整数\n'
        "  },\n"
        '  "roadmap_json": {\n'
        '    "stages": [\n'
        '      {"name": "...", "duration": "...", "milestones": ["...", "..."], '
        '"resources": [{"title": "...", "type": "课程/书籍/项目/文档", "url": "..."}]}\n'
        "    ],\n"
        '    "summary": "...综合评语..."\n'
        "  }\n"
        "}\n"
        "```\n"
        "重要：不要输出任何 JSON 以外的解释性文字。"
    )

    user_prompt = (
        f"用户信息：\n"
        f"- 目标岗位：{target_position}\n"
        f"- 当前年级：{grade}\n\n"
        f"用户答题记录：\n{answers_text}\n\n"
        f"请基于以上答题记录，评估技术倾向并生成 Roadmap JSON。"
    )

    # ---- 调用 LLM + 超时/异常捕获 ----
    try:
        response = call_deepseek(system_prompt, user_prompt)
    except Exception as e:
        logger.error("AssessmentNode LLM 调用失败: %s", e)
        return {
            "execution_error": f"AssessmentNode 超时或异常: {str(e)}",
            "is_completed": True,
            "current_step": "assessment_failed",
        }

    # ---- 多层 Fallback JSON 解析 ----
    parsed = _extract_json_from_reasoner_response(response)

    if parsed is None:
        # 灾难降级：JSON 全量解析失败 → 原始文本写入 roadmap
        logger.warning("AssessmentNode JSON 解析全部失败，降级为纯文本路线图")
        return {
            "roadmap": response,                    # 原始 LLM 输出作为 Markdown 路线图
            "roadmap_json": None,
            "tech_tendency": None,
            "is_completed": True,
            "current_step": "assessment_completed_with_fallback",
            "execution_error": "JSON 解析失败，已降级为纯文本路线图",
        }

    # ---- 提取 tech_tendency 和 roadmap_json ----
    tech_tendency = parsed.get("tech_tendency")
    roadmap_json = parsed.get("roadmap_json")

    # ---- 构造 Markdown 版 roadmap ----
    roadmap_md = _build_roadmap_markdown(
        tech_tendency=tech_tendency,
        roadmap_json=roadmap_json,
        target_position=target_position,
        grade=grade,
    )

    if not state.get("execution_error"):
        st.write("✅ **[2/2] AssessmentNode** 技术倾向评估与路线图生成完毕")

    return {
        "tech_tendency": tech_tendency,
        "roadmap_json": roadmap_json,
        "roadmap": roadmap_md,
        "is_completed": True,
        "current_step": "assessment_completed",
    }


# ============================================================
# 辅助函数：将评估结果转化为 Markdown 路线图
# ============================================================

def _build_roadmap_markdown(
    tech_tendency: Optional[Dict[str, Any]],
    roadmap_json: Optional[Dict[str, Any]],
    target_position: str,
    grade: str,
) -> str:
    """
    将 JSON 结构化路线图转换为可读的 Markdown 格式。

    此函数在 LLM 返回的 roadmap_json 基础上进行 Markdown 渲染编排，
    确保即使 LLM 返回的 JSON 中存在微小格式异常，前端仍有可用的展示内容。
    """
    lines = [
        f"## 🧭 {target_position} 技术学习路线图",
        f"*用户年级：{grade}*",
        "",
    ]

    # ---- 技术倾向摘要 ----
    if tech_tendency and isinstance(tech_tendency, dict):
        lines.append("### 📊 技术倾向评估结果")
        lines.append("")
        lines.append("| 技术方向 | 倾向分值 (0-100) |")
        lines.append("|---|---|")
        for direction in ["前端", "后端", "AI数据", "测试运维", "产品", "客户端"]:
            score = tech_tendency.get(direction, 0)
            bar = "█" * int(score / 10) + "░" * (10 - int(score / 10))
            lines.append(f"| {direction} | {score} |")
        lines.append("")

    # ---- 综合评语 ----
    if roadmap_json and isinstance(roadmap_json, dict):
        summary = roadmap_json.get("summary", "")
        if summary:
            lines.append(f"> 💡 {summary}")
            lines.append("")

    # ---- 阶段路线 ----
    if roadmap_json and isinstance(roadmap_json, dict):
        stages = roadmap_json.get("stages", [])
        if stages:
            lines.append("### 📅 学习阶段规划")
            lines.append("")
            for i, stage in enumerate(stages, 1):
                name = stage.get("name", f"阶段 {i}")
                duration = stage.get("duration", "待定")
                milestones = stage.get("milestones", [])
                resources = stage.get("resources", [])

                lines.append(f"#### 阶段 {i}：{name}")
                lines.append(f"⏱️ 预计时间：{duration}")
                lines.append("")

                if milestones:
                    lines.append("**🎯 里程碑：**")
                    for m in milestones:
                        lines.append(f"- {m}")
                    lines.append("")

                if resources:
                    lines.append("**📚 推荐资源：**")
                    for res in resources:
                        title = res.get("title", "未命名")
                        res_type = res.get("type", "")
                        url = res.get("url", res.get("description", ""))
                        type_badge = f"`{res_type}` " if res_type else ""
                        if url and url.startswith("http"):
                            lines.append(f"- {type_badge}[{title}]({url})")
                        else:
                            lines.append(f"- {type_badge}{title}" + (f" — {url}" if url else ""))
                    lines.append("")

    return "\n".join(lines)


# ============================================================
# 节点注册表（供图构建使用）
# ============================================================

AGENT_NODES = {
    "question_filter": question_filter_node,
    "assessment": assessment_node,
}
