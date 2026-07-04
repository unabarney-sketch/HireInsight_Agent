
## 项目概述

**HireInsight-Agent** 是一个基于 LangGraph + RAG 的智能求职辅助系统，面向校招/社招场景，提供职能测评、岗位数据大屏、AI 面试模拟三大核心能力。

## 技术栈

| 层级 | 技术 | 约束 |
|------|------|------|
| 前端交互 | Streamlit | 同步阻塞模式 + `st.session_state` 状态缓存 |
| AI 编排 | LangGraph | 基于 State 的有向图 (DG)，支持条件边，`recursion_limit=10` |
| 数据处理 | Pandas + SQLite | 本地持久化 |
| 向量检索 | ChromaDB | **嵌入式模式**，禁止 Docker 部署 |
| LLM | DeepSeek API | 开启 Reasoning 思考链 |

## 目录结构

```
HireInsight-Agent/
├── data/               # SQLite 岗位数据 + ChromaDB 向量仓
├── graphs/             # LangGraph 节点 (Nodes) 与条件边 (Edges) 控制逻辑
│   ├── state.py        # StateSchema 定义（支持序列化）
│   ├── nodes.py        # 各 Agent 节点实现
│   └── interview_graph.py  # 图构建与编译
├── utils/              # PDF 解析、正则清洗、RAG 加载脚本
├── app.py              # Streamlit 主程序入口
└── requirements.txt     # 锁死版本的依赖声明
```

## 三大核心模块

### 模块一：灯塔计划（低年级定向）
- 前端渲染 5-10 道动态情景多选题
- LLM 结构化评估用户技术倾向
- JSON 规范化输出技术演进闭环 Roadmap

### 模块二：数据感知与大屏（全流程指标）
- 逆向大厂公开招聘 API，获取标准化 JSON 数据
- 正则清洗：统一薪资换算（千元/月），规范化城市字段
- Pandas 分组统计 + Streamlit 实时图表渲染

### 模块三：LangGraph 拓扑工作流 + 本地 RAG（核心亮点）

> **架构决策**：采用纯同步执行模式，页面阻塞期间由 `st.status` 容器提供实时视觉反馈。

#### 执行流程（三节点线性链）

```
[入口] → Market_Agent → Critic_Agent → Interviewer_Agent → [结束]
```

1. **`Market_Agent`**
   - 输入：`target_position`、`market_data`（Pandas DataFrame 概要）
   - 输出：`market_report`（市场趋势与竞争分析）

2. **`Critic_Agent`**
   - 输入：`user_resume`、`market_report`、RAG 检索结果
   - 输出：`gap_analysis`（技能差距诊断）
   - RAG 增强：ChromaDB 检索 Top-2 企业面经片段

3. **`Interviewer_Agent`**
   - 输入：`gap_analysis`、`rag_context`
   - 输出：`interview_questions`（3-5 道定制面试变形题）

#### 执行契约

| 参数 | 值 | 说明 |
|------|-----|------|
| `recursion_limit` | 10 | 全局步数预算，覆盖整图一次完整执行 |
| 执行模式 | 同步阻塞 | Streamlit 脚本重运行期间页面无响应 |
| 视觉反馈 | `st.status` 容器 | 实时展示当前执行阶段与完成状态 |
| 输出渲染 | 静态 Markdown 看板 | 执行完成后一次性渲染，不依赖轮询 |

## 开发边界（熔断机制）

- **禁止**独立多轮对话聊天框：所有 Agent 结果使用静态 Markdown 看板一键生成
- **禁止**外置型 Docker 向量库：只允许嵌入式 ChromaDB 或纯 NumPy 矩阵
- **禁止**异步/多线程方案：Streamlit 同步模型下不引入 `asyncio`、`threading`
- **禁止**轮询刷新：`time.sleep` + `st.rerun()` 会造成死循环，勿用
- **逃生通道**：若 LangGraph 单次执行超时 2 分钟，降级为顺序链式调用（移除图编排）

## 开发命令

```bash
# 创建虚拟环境
python -m venv venv
venv\Scripts\activate      # Windows

# 安装依赖
pip install -r requirements.txt

# 启动应用
streamlit run app.py

# 运行测试（如有）
pytest tests/
```

## 关键约束

- 所有 AI 输出使用 **静态 Markdown 看板** 渲染，避免 Streamlit 页面刷新导致状态丢失
- ChromaDB 使用 `PersistentClient` 本地模式，持久化至 `data/` 目录
- LangGraph State 设计需支持序列化，配合 Streamlit `st.session_state` 做检查点恢复
- DeepSeek API Key 通过环境变量 `DEEPSEEK_API_KEY` 注入，不硬编码
```

---

**主要变更说明**：
1. **第 14 行**：DAG → 有向图 (DG)，明确 `recursion_limit=10`
2. **第 42-62 行**：完全重写模块三，采用纯同步 + `st.status` 方案
3. **第 53-56 行**：新增三条禁止规则（异步/轮询/多线程）
4. **第 54 行**：逃生通道从 2 小时改为 2 分钟，更务实