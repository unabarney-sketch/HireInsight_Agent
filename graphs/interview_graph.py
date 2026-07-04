"""
HireInsight-Agent - LangGraph 图构建与编译

采用直线 DAG 拓扑：
Market_Agent → Critic_Agent → Interviewer_Agent → END

执行契约：
- recursion_limit = 10（全局步数预算）
- 纯同步阻塞模式（Streamlit 同步执行）
- 超时由 LLM 客户端层 ChatOpenAI(timeout=120) 保证，Windows 兼容
- st.status 容器提供视觉反馈
"""
from langgraph.graph import StateGraph, END

from .state import InterviewState
from .nodes import (
    market_agent_node,
    critic_agent_node,
    interviewer_agent_node,
)


def create_interview_graph() -> StateGraph:
    """
    构建面试模拟工作流图

    拓扑结构（纯同步直线 DAG，无循环、无条件边）：
    [START] → [Market_Agent] → [Critic_Agent] → [Interviewer_Agent] → [END]

    Returns:
        StateGraph: 编译好的 LangGraph 图对象
    """
    workflow = StateGraph(InterviewState)

    # 注册三个 Agent 节点
    workflow.add_node("market_agent", market_agent_node)
    workflow.add_node("critic_agent", critic_agent_node)
    workflow.add_node("interviewer_agent", interviewer_agent_node)

    # 直线串联：入口 → 无条件顺序边 → 出口
    workflow.set_entry_point("market_agent")
    workflow.add_edge("market_agent", "critic_agent")
    workflow.add_edge("critic_agent", "interviewer_agent")
    workflow.add_edge("interviewer_agent", END)

    return workflow


def compile_interview_graph():
    """
    编译面试工作流图

    Returns:
        编译后的可执行图对象
    """
    graph = create_interview_graph()
    return graph.compile()


def run_interview_workflow(initial_state: InterviewState) -> InterviewState:
    """
    执行面试工作流（同步模式）

    参数：
        initial_state: 初始状态（包含 user_resume, target_position 等）

    返回：
        最终状态（包含 market_report, gap_analysis, interview_questions 等输出）

    超时降级：
        - 超时不依赖 Windows 不支持的 signal.SIGALRM
        - 硬超时锁死在 LLM 客户端层：ChatOpenAI(timeout=120)
        - API 超时/网络异常 → except 捕获 → 自动触发 fallback 顺序链式执行
    """
    app = compile_interview_graph()

    config = {
        "recursion_limit": 10,  # 全局步数预算
    }

    try:
        final_state = app.invoke(initial_state, config)
        return final_state
    except Exception:
        # LangGraph 执行异常（含 LLM 超时）→ 降级为顺序链式执行
        return run_interview_workflow_fallback(initial_state)


# ============================================================
# 逃生通道：顺序链式调用（降级方案）
# ============================================================
def run_interview_workflow_fallback(initial_state: InterviewState) -> InterviewState:
    """
    降级执行方案：移除图编排，改为普通 Python 函数顺序链式调用

    触发条件：
        - LangGraph 内部执行抛出异常
        - DeepSeek API 超时（超过 120 秒）
        - 网络中断或其他执行期错误

    执行链路：
        market_agent_node → critic_agent_node → interviewer_agent_node
    """
    state = market_agent_node(initial_state)
    state = critic_agent_node(state)
    state = interviewer_agent_node(state)
    state["is_completed"] = True
    return state


# ============================================================
# 导出接口
# ============================================================
__all__ = [
    "create_interview_graph",
    "compile_interview_graph",
    "run_interview_workflow",
    "run_interview_workflow_fallback",
]
