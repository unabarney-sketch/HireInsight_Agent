"""
HireInsight-Agent - LangGraph State Schema 定义
定义整个工作流的共享状态结构
"""
from typing import Optional, List, Dict, Any, TypedDict
import pandas as pd


class InterviewState(TypedDict, total=False):
    """
    LangGraph 工作流共享状态
    
    所有字段需支持序列化，配合 Streamlit st.session_state 做检查点恢复
    """
    
    # === 用户输入 ===
    user_resume: str                          # 简历文本（PDF解析后）
    target_position: str                       # 目标岗位
    target_company: Optional[str]             # 目标公司（可选）
    
    # === 市场数据 ===
    market_data: Optional[pd.DataFrame]       # 岗位数据 DataFrame
    market_summary: Optional[Dict[str, Any]]  # 市场数据统计摘要
    
    # === Agent 输出 ===
    market_report: Optional[str]              # Market_Agent 输出：市场趋势报告
    gap_analysis: Optional[str]               # Critic_Agent 输出：技能差距诊断
    interview_questions: Optional[List[str]]   # Interviewer_Agent 输出：面试题列表
    
    # === RAG 上下文 ===
    rag_context: Optional[List[str]]          # RAG 检索结果
    top_company_experiences: Optional[List[Dict]]  # Top 企业面经片段
    
    # === 执行状态 ===
    current_step: str                          # 当前执行阶段
    execution_error: Optional[str]            # 执行错误信息
    is_completed: bool                         # 是否完成
    
    # === 面试反馈（可选）===
    selected_question: Optional[int]          # 用户选中的问题索引
    ai_feedback: Optional[str]                 # AI 对选中问题的反馈


class LighthousePlanState(TypedDict, total=False):
    """
    灯塔计划（职业规划模块）状态

    关键约定：
    - execution_error 非空 + is_completed=True → 前端渲染降级错误页
    - JSON 解析失败不算 execution_error（降级为纯文本 roadmap），
      只有 LLM 调用超时/网络异常才写入 execution_error
    """
    # === 用户输入 ===
    target_position: str                    # 目标岗位（如 "Java后端开发"、"AI算法工程师"）
    grade: str                              # 当前年级（如 "大三"、"研一"）

    # === 题目数据 ===
    all_questions: List[dict]               # 完整题库（10 题），来自 utils/question_bank.py
    filtered_questions: List[dict]          # LLM 筛选后的题目（5-7 题），须写入 st.session_state 缓存

    # === 用户答题 ===
    user_answers: List[dict]                # 答题详情 [{question_id, selected_option, tendency}]
    user_choices: List[int]                  # 保留兼容：各题所选选项索引

    # === Agent 输出 ===
    tech_tendency: Optional[Dict[str, float]]   # 六维度技术倾向分值（前端/后端/AI数据/测试运维/产品/客户端，0-100）
    roadmap_json: Optional[Dict[str, Any]]      # JSON 结构化路线图（含阶段划分、里程碑、推荐资源）
    roadmap: Optional[str]                      # Markdown 可读路线图

    # === 执行状态 ===
    current_step: str                           # 当前执行阶段
    execution_error: Optional[str]              # 执行错误信息（非空时前端跳过正常结果，渲染降级页）
    is_completed: bool                          # 是否完成（含异常完成）


class MarketDataState(TypedDict, total=False):
    """
    数据大屏模块状态
    """
    # 原始/清洗后的岗位数据
    raw_data: Optional[pd.DataFrame]
    cleaned_data: Optional[pd.DataFrame]
    
    # 统计结果
    salary_stats: Optional[Dict[str, Any]]
    region_stats: Optional[Dict[str, Any]]
    education_stats: Optional[Dict[str, Any]]
    experience_stats: Optional[Dict[str, Any]]
    
    # 筛选条件
    filters: Dict[str, Any]
