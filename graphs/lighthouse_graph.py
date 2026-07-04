"""
HireInsight-Agent - 灯塔计划 LangGraph 图构建与编译

采用直线 DAG 拓扑：
QuestionFilterNode → AssessmentNode → END

执行契约：
- recursion_limit = 10（全局步数预算）
- 纯同步阻塞模式（Streamlit 同步执行）
- 超时由 LLM 客户端层 ChatOpenAI(timeout=120) 保证，Windows 兼容
- st.status 容器提供视觉反馈
"""
from langgraph.graph import StateGraph, END

from .state import LighthousePlanState
from .lighthouse_nodes import (
    question_filter_node,
    assessment_node,
)


def create_lighthouse_graph() -> StateGraph:
    """
    构建灯塔计划工作流图

    拓扑结构（纯同步直线 DAG，无循环、无条件边）：
    [START] → [QuestionFilterNode] → [AssessmentNode] → [END]

    Returns:
        StateGraph: 编译好的 LangGraph 图对象
    """
    workflow = StateGraph(LighthousePlanState)

    # 注册两个 Agent 节点
    workflow.add_node("question_filter", question_filter_node)
    workflow.add_node("assessment", assessment_node)

    # 直线串联：入口 → 无条件顺序边 → 出口
    workflow.set_entry_point("question_filter")
    workflow.add_edge("question_filter", "assessment")
    workflow.add_edge("assessment", END)

    return workflow


def compile_lighthouse_graph():
    """
    编译灯塔计划工作流图

    Returns:
        编译后的可执行图对象
    """
    graph = create_lighthouse_graph()
    return graph.compile()


def run_lighthouse_workflow(initial_state: LighthousePlanState) -> LighthousePlanState:
    """
    执行灯塔计划工作流（同步模式）

    参数：
        initial_state: 初始状态（包含 target_position, grade, all_questions 等）

    返回：
        最终状态（包含 filtered_questions, tech_tendency, roadmap 等输出）

    超时降级：
        - 硬超时锁死在 LLM 客户端层：ChatOpenAI(timeout=120)
        - API 超时/网络异常 → except 捕获 → fallback 顺序链式执行
        - 每个节点内部有独立的 try-except，单个节点失败不导致全流程崩溃
    """
    app = compile_lighthouse_graph()

    config = {
        "recursion_limit": 10,  # 全局步数预算
    }

    try:
        final_state = app.invoke(initial_state, config)
        return final_state
    except Exception:
        # LangGraph 执行异常（含 LLM 超时）→ 降级为顺序链式执行
        return run_lighthouse_workflow_fallback(initial_state)


# ============================================================
# 逃生通道：顺序链式调用（降级方案）
# ============================================================
def run_lighthouse_workflow_fallback(
    initial_state: LighthousePlanState,
) -> LighthousePlanState:
    """
    降级执行方案：移除图编排，改为普通 Python 函数顺序链式调用

    触发条件：
        - LangGraph 内部执行抛出异常
        - DeepSeek API 超时（超过 120 秒）
        - 网络中断或其他执行期错误

    执行链路：
        question_filter_node → assessment_node
    """
    # 阶段一：选题
    filter_result = question_filter_node(initial_state)
    state = {**initial_state, **filter_result}

    # 如果选题节点已经报错，跳过评估直接返回
    if state.get("execution_error"):
        state["is_completed"] = True
        return state

    # 阶段二：评估
    assess_result = assessment_node(state)
    state = {**state, **assess_result}
    state["is_completed"] = True
    return state


# ============================================================
# 导出接口
# ============================================================
__all__ = [
    "create_lighthouse_graph",
    "compile_lighthouse_graph",
    "run_lighthouse_workflow",
    "run_lighthouse_workflow_fallback",
]
