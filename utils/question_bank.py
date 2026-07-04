"""
HireInsight-Agent 模块一「灯塔计划」—— 预设情景多选题库

设计原则：
- 每道题 4 个选项，tendency 分别指向 4 个不同技术方向（互不相同）
- 题干贴近真实工程痛点，拒绝空泛行业黑话
- 选项必须包含该领域的具体技术术语，让选择直接映射到技术倾向
- 10 题覆盖：性能优化(2) / 架构选型(2) / Bug排查(1) / 技术债治理(1) / 数据处理(2) / 项目规划(1) / 安全攻防(1)

tendency 取值范围：前端 / 后端 / AI数据 / 测试运维 / 产品 / 客户端
"""

from typing import List

QUESTION_BANK: List[dict] = [
    # ============================================================
    # Q1：性能优化 — 高并发系统瓶颈（Java 后端 / OS / 数学建模 / 可观测性）
    # ============================================================
    {
        "id": 1,
        "scenario": (
            "你负责的一个核心服务在流量高峰期出现严重性能退化："
            "单机 QPS 从 8000 骤降至 1200，P99 延迟从 80ms 飙升至 3.2s。"
            "JVM GC 日志显示 Full GC 频率从每小时 2 次上升到每分钟 3 次，"
            "但堆内存配置和业务代码近期均未变更。你更愿意从哪个维度入手根治这个问题？"
        ),
        "options": [
            {
                "text": (
                    "深入分析线程池的 corePoolSize/maxPoolSize 配比与任务队列的背压策略，"
                    "检查 Spring Boot 异步模型下 @Async 方法的事务传播边界条件，"
                    "并重新设计数据库连接池的 min-idle 与 max-lifetime 参数"
                ),
                "tendency": "后端",
            },
            {
                "text": (
                    "绕过 JVM 层直接使用 eBPF 在内核态采集 TCP 重传率与 epoll 事件循环的唤醒频率，"
                    "分析 OS 级别的内存页换入换出（pgmajfault）和 NUMA 节点的跨片访问延迟，"
                    "从内核调度器 cgroup 层面重新划分 CPU 时间片"
                ),
                "tendency": "客户端",
            },
            {
                "text": (
                    "将请求依赖图建模为偏序集（Poset），利用哈斯图的拓扑分层识别出关键路径中的最大反链，"
                    "通过组合优化中的 Dilworth 定理将并发请求拆分为最少数量的无依赖子集，"
                    "从群论视角重新设计分布式锁的状态迁移函数"
                ),
                "tendency": "AI数据",
            },
            {
                "text": (
                    "搭建全链路压测体系，在预发环境注入 10 倍常态流量并采集 JFR（Java Flight Recorder）"
                    "热点方法火焰图，建立自动化故障注入平台并通过可观测性三支柱"
                    "（Metrics/Tracing/Logging）关联定位瓶颈的系统性成因"
                ),
                "tendency": "测试运维",
            },
        ],
        "difficulty": "hard",
        "category": "性能优化",
        "tags": ["并发", "JVM", "eBPF", "哈斯图", "全链路压测"],
    },

    # ============================================================
    # Q2：架构选型 — 数据密集型系统设计
    # ============================================================
    {
        "id": 2,
        "scenario": (
            "你需要设计一个日增 500GB 日志数据的实时分析系统。"
            "上游每秒写入约 12 万条异构日志（JSON/Protobuf/明文混合），"
            "下游需要支持两类查询：(a) 近 5 分钟数据的低延迟聚合（< 200ms），"
            "(b) 近 30 天数据的多维 OLAP 分析。团队只有 3 个后端开发 + 1 个前端。"
            "你最倾向于承担哪个角色？"
        ),
        "options": [
            {
                "text": (
                    "使用 React 的虚拟滚动 + Web Worker 在前端完成大数据量的增量渲染，"
                    "设计基于 Proxy 的状态管理中间层对 OLAP 结果集做 O(1) 缓存命中，"
                    "并利用 WebAssembly 把核心聚合计算下沉到浏览器端"
                ),
                "tendency": "前端",
            },
            {
                "text": (
                    "设计 Kafka 多分区键路由策略保证写入顺序，"
                    "用 Flink SQL 做实时 ETL 将异构日志归一化为 Apache Iceberg 表格式，"
                    "对 (a) 类查询用 Redis Sorted Set + Lua 脚本做滑动窗口聚合，"
                    "对 (b) 类查询在 ClickHouse 上建立物化视图和预聚合立方体"
                ),
                "tendency": "后端",
            },
            {
                "text": (
                    "对历史日志做 PageRank 异常检测，识别哪些日志模式是故障的先行指标，"
                    "用 LSTM-Autoencoder 对时序指标做无监督异常检测，"
                    "为 OLAP 查询引入贝叶斯优化自动选择最优物化视图组合"
                ),
                "tendency": "AI数据",
            },
            {
                "text": (
                    "从用户视角定义查询场景的优先级矩阵（紧急度 × 频次），"
                    "输出最小可行产品（MVP）范围文档：明确第一批只支持 Top 3 查询模式，"
                    "用 A/B 实验数据验证方案后驱动迭代，制定从 5 分钟延迟到 < 200ms 的分阶段 OKR"
                ),
                "tendency": "产品",
            },
        ],
        "difficulty": "hard",
        "category": "架构选型",
        "tags": ["Kafka", "Flink", "ClickHouse", "LSTM", "Web Worker"],
    },

    # ============================================================
    # Q3：数据处理 — YOLO 模型训练与部署落地
    # ============================================================
    {
        "id": 3,
        "scenario": (
            "团队要将 YOLOv8 目标检测模型部署到边缘设备（NVIDIA Jetson Orin）上，"
            "用于产线缺陷实时检测。当前 FP32 模型推理延迟为 180ms，"
            "但产线要求严格控制在 50ms 以内且准确率（mAP@0.5）不低于 92%。"
            "同时，训练数据中 30% 的标注存在类别歧义（'划痕' 与 '裂纹' 难以区分）。"
            "你会优先从哪个方向切入？"
        ),
        "options": [
            {
                "text": (
                    "使用 TensorRT 对模型进行 INT8 量化并做层融合（Layer Fusion）优化计算图，"
                    "将预处理和后处理逻辑重写为 CUDA kernel 以减少 Host-to-Device 拷贝，"
                    "在 ONNX Runtime 中开启 graph optimization level 为 ORT_ENABLE_ALL"
                ),
                "tendency": "客户端",
            },
            {
                "text": (
                    "搭建 FastAPI 异步服务封装模型推理，用 Celery + Redis 做任务队列分发，"
                    "设计 gRPC 接口协议并在后端实现模型版本灰度发布策略，"
                    "通过 Spring Cloud Gateway 做统一的 API 鉴权与限流"
                ),
                "tendency": "后端",
            },
            {
                "text": (
                    "重新审视标注数据：对歧义类别做标签平滑（Label Smoothing）和置信度校准，"
                    "设计主动学习（Active Learning）策略筛选边界样本优先重标注，"
                    "用 AutoAugment 搜索最优数据增强策略组合并使用 Focal Loss 缓解类别不均衡"
                ),
                "tendency": "AI数据",
            },
            {
                "text": (
                    "搭建 CI/CD Pipeline 将模型评估自动化：每次 PR 触发 mAP 回归测试，"
                    "用 Docker 多阶段构建将模型镜像体积从 4.2GB 优化到 800MB，"
                    "编写混沌工程实验验证推理服务在 GPU 宕机时的自动 failover 行为"
                ),
                "tendency": "测试运维",
            },
        ],
        "difficulty": "hard",
        "category": "数据处理",
        "tags": ["YOLOv8", "TensorRT", "主动学习", "CUDA", "CI/CD"],
    },

    # ============================================================
    # Q4：Bug 排查 — 内存泄漏定位
    # ============================================================
    {
        "id": 4,
        "scenario": (
            "生产环境一台 32GB 内存的 Java 应用在运行 72 小时后必定 OOM 崩溃，"
            "但重启后内存使用率又正常地从 4GB 缓慢爬升。Heap dump 分析发现："
            "byte[] 占用 78% 堆内存但 GC Root 路径指向了多个看似无关的 Spring Bean。"
            "同事怀疑是第三方 SDK 的 native memory leak，你认为最有效的排查路径是？"
        ),
        "options": [
            {
                "text": (
                    "使用 MAT（Memory Analyzer Tool）的 OQL 查询所有 byte[] 的支配树，"
                    "定位持有最大 retained heap 的 Spring Bean，追溯其 @Scope 注解和 BeanFactory 生命周期，"
                    "用 Arthas 的 monitor/watch 命令实时监控可疑方法的调用频率与参数值分布"
                ),
                "tendency": "后端",
            },
            {
                "text": (
                    "切换到 OS 维度：开启 jemalloc 的 profiling 模式跟踪 native memory 分配栈，"
                    "用 pmap -x 观察进程的 RSS 与虚拟内存映射关系，"
                    "分析 /proc/PID/smaps 中匿名映射区的增长趋势判断是否存在 glibc malloc 碎片化"
                ),
                "tendency": "客户端",
            },
            {
                "text": (
                    "构建内存分配的时间序列模型：每 10 分钟采样一次堆中各类型对象数量，"
                    "用 ARIMA 模型预测未来 2 小时的堆增长曲线，"
                    "通过 Isolation Forest 自动检测哪些对象类型的增长率是异常的离群值"
                ),
                "tendency": "AI数据",
            },
            {
                "text": (
                    "首先测量问题的业务影响：统计 OOM 影响的用户请求比例和收入损失金额，"
                    "评估是加内存（纵向扩容）还是加实例（横向扩容）的 ROI 更高，"
                    "输出一份包含技术方案和业务影响的决策建议文档供领导层审批"
                ),
                "tendency": "产品",
            },
        ],
        "difficulty": "medium",
        "category": "Bug排查",
        "tags": ["OOM", "MAT", "Arthas", "jemalloc", "时间序列"],
    },

    # ============================================================
    # Q5：数据处理 — 大规模数据管道效率优化
    # ============================================================
    {
        "id": 5,
        "scenario": (
            "你维护的一个 ETL 管道需要每天处理 2TB 的 Parquet 文件，"
            "当前 Spark 作业耗时 4.5 小时，但 SLA 要求 1.5 小时内完成。"
            "Spark UI 显示 Stage 3 存在严重的数据倾斜："
            "一个 Task 处理 800GB 数据耗时 2.1 小时，其他 199 个 Task 各处理不到 5GB。"
            "你会优先从哪个方向优化？"
        ),
        "options": [
            {
                "text": (
                    "分析倾斜 Key 的分布特征，对热点 Key 加随机前缀打散（Salting），"
                    "同时优化 Spark SQL 中的 JOIN 策略：用 Broadcast Hash Join 替代 Sort Merge Join，"
                    "调整 spark.sql.shuffle.partitions 和 spark.sql.adaptive.enabled 参数"
                ),
                "tendency": "后端",
            },
            {
                "text": (
                    "放弃 Spark 转而使用 Rust 重写核心 ETL 逻辑："
                    "利用 Arrow 列式内存格式 + DataFusion 查询引擎实现零拷贝数据传输，"
                    "在编译期消除所有虚函数调用的运行时开销，"
                    "对 Parquet 的 Row Group 级别做 SIMD 向量化解码"
                ),
                "tendency": "客户端",
            },
            {
                "text": (
                    "用统计方法定位根因：对倾斜 Key 做卡方检验判断分布不均是否统计显著，"
                    "利用信息熵（Shannon Entropy）量化分区方案的均衡度，"
                    "构建成本模型（Cost Model）基于数据直方图自动推荐最优分区键组合"
                ),
                "tendency": "AI数据",
            },
            {
                "text": (
                    "从用户体验视角重新定义 SLA：调研下游 BI 看板用户的实际查询模式，"
                    "发现 92% 的查询只涉及最近 3 天数据，将管道拆分为热数据（小时级增量）"
                    "+ 冷数据（天级全量），将 1.5 小时的要求放宽到差异化承诺"
                ),
                "tendency": "产品",
            },
        ],
        "difficulty": "hard",
        "category": "数据处理",
        "tags": ["Spark", "数据倾斜", "Rust", "信息熵", "SLA"],
    },

    # ============================================================
    # Q6：技术债治理 — 遗留系统现代化改造
    # ============================================================
    {
        "id": 6,
        "scenario": (
            "你接手了一个 8 年前用 Struts2 + JSP + JDBC 直接拼接 SQL 构建的订单系统。"
            "代码库约 42 万行，无单元测试，每次修改一个模块都会无意间影响到其他 3 个模块。"
            "业务方要求在不中断线上服务的前提下，分阶段提升系统可维护性。"
            "你最愿意从哪个切入点开始？"
        ),
        "options": [
            {
                "text": (
                    "引入 Strangler Fig 模式：用 Spring Boot + MyBatis-Plus 重写核心订单域，"
                    "通过 Nginx 按 URL 前缀灰度切流，新老系统并行运行期间用 Canal 做 MySQL binlog"
                    "双向同步保证数据一致性，逐步下掉旧代码"
                ),
                "tendency": "后端",
            },
            {
                "text": (
                    "优先建立安全网：对订单域的核心接口编写 Characterization Test（特征测试），"
                    "不关心代码实现正确性、只锁定当前行为，配合 JaCoCo 度量覆盖率的增长曲线，"
                    "搭建 CI 流水线确保任何 PR 的行为变异都会被测试捕获"
                ),
                "tendency": "测试运维",
            },
            {
                "text": (
                    "用静态分析方法做摸底：扫描全量代码生成模块依赖的 DSM 矩阵"
                    "（Design Structure Matrix），用社区检测算法聚类出高内聚低耦合的边界上下文集，"
                    "输出一份量化的技术债热力图作为改造优先级排序依据"
                ),
                "tendency": "AI数据",
            },
            {
                "text": (
                    "先用现代前端框架（React/Vue）重构 JSP 页面层，剥离前后端耦合，"
                    "通过 BFF（Backend for Frontend）层逐步替代 Struts2 Action，"
                    "并在前端建立基于 Storybook 的组件库实现 UI 的跨模块复用"
                ),
                "tendency": "前端",
            },
        ],
        "difficulty": "medium",
        "category": "技术债治理",
        "tags": ["Strangler Fig", "Canal", "DSM", "Storybook", "Characterization Test"],
    },

    # ============================================================
    # Q7：性能优化 — OS 底层与运行时开销
    # ============================================================
    {
        "id": 7,
        "scenario": (
            "你的 Go 服务在 16 核机器上只跑出 4 核的有效利用率，"
            "perf top 显示 runtime.futex 和 sync.runtime_Semacquire 占比合计 34%，"
            "CPU 的 sys% 远高于 usr%。业务逻辑看起来是 CPU-bound 的纯计算，"
            "但实际瓶颈却出在内核态的系统调用开销上。你会怎么深挖？"
        ),
        "options": [
            {
                "text": (
                    "用 strace -c 统计进程的系统调用频率与耗时分布，定位到高频的 futex(FUTEX_WAIT) "
                    "来自多个 goroutine 争抢同一把 sync.Mutex，将锁粒度从整个数据结构拆分为"
                    "分段锁（Sharded Lock），并用 atomic.CompareAndSwap 替代互斥锁实现无锁结构"
                ),
                "tendency": "后端",
            },
            {
                "text": (
                    "从 CPU 缓存一致性协议（MESI）的角度分析：False Sharing 导致不同 goroutine 写入"
                    "同一 Cache Line 的不同字段时频繁触发 RFO（Request for Ownership），"
                    "通过调整结构体内存布局（Padding 到 64 字节对齐）消除伪共享"
                ),
                "tendency": "客户端",
            },
            {
                "text": (
                    "将并行计算抽象为代数数据类型（ADT）：用半环（Semiring）结构表达计算的结合律"
                    "与分配律，将原串行算法改写为可并行的幺半群（Monoid）fold 操作，"
                    "利用抽象代数的不变量在编译期消除不必要的同步点"
                ),
                "tendency": "AI数据",
            },
            {
                "text": (
                    "量化不同锁策略的 trade-off：绘制并发度-吞吐量曲线对比 Mutex/Spinlock/"
                    "Lock-Free 三种方案在不同 goroutine 数量下的性能表现，"
                    "输出技术决策文档，在延迟和 CPU 利用率之间找到业务可接受的最优平衡点"
                ),
                "tendency": "产品",
            },
        ],
        "difficulty": "medium",
        "category": "性能优化",
        "tags": ["futex", "False Sharing", "MESI", "Monoid", "CAS"],
    },

    # ============================================================
    # Q8：安全攻防 — 供应链安全与依赖管理
    # ============================================================
    {
        "id": 8,
        "scenario": (
            "安全团队扫描发现你的项目依赖了一款 npm 包的特定版本（1.3.7），"
            "该版本存在 CVE-2024 高危反序列化漏洞（CVSS 9.8）。"
            "但升级到安全版本 1.4.0 后，你的单元测试中有 23 个 case 失败——"
            "因为 1.4.0 的 API 签名做了 breaking change。"
            "发布截止日期是明天中午。你会如何处理？"
        ),
        "options": [
            {
                "text": (
                    "在入口网关层（Nginx/Kong）部署 WAF 规则，"
                    "对命中反序列化特征（如 JSON 嵌套深度 > 20 层或包含 $type 字段）"
                    "的请求直接返回 403，作为临时缓解措施。同时启动 1.4.0 迁移分支，"
                    "用适配器模式封装新旧 API 差异，通过 feature flag 灰度切换"
                ),
                "tendency": "后端",
            },
            {
                "text": (
                    "使用 npm audit fix --force 批量升级所有受影响依赖，"
                    "将失败的 23 个 case 分类：属于断言格式变化的更新快照，属于行为变化的"
                    "添加回归保护后修复。用 Renovate 配置自动 PR 规则避免未来类似断更问题"
                ),
                "tendency": "测试运维",
            },
            {
                "text": (
                    "分析漏洞利用链在 1.3.7 中的具体调用路径，"
                    "如果入口点在项目中未被使用（reachability analysis），单独 patch 该函数而非全量升级，"
                    "将 patch 发布到内部 npm registry 并在 package.json 锁定该定制版本"
                ),
                "tendency": "客户端",
            },
            {
                "text": (
                    "首先评估风险敞口：确认该漏洞在项目中的攻击面（Exploitability × Impact），"
                    "如果利用条件苛刻（如需要内网访问），向安全团队申请 3 天豁免期，"
                    "将 1.4.0 迁移纳入下个 sprint 的 plan，当前版本配合告警规则正常上线"
                ),
                "tendency": "产品",
            },
        ],
        "difficulty": "medium",
        "category": "安全攻防",
        "tags": ["CVE", "反序列化", "WAF", "renovate", "reachability"],
    },

    # ============================================================
    # Q9：架构选型 — 技术栈决策
    # ============================================================
    {
        "id": 9,
        "scenario": (
            "新项目需要实现一个跨平台（iOS/Android/Web/桌面）的富文本协作编辑器，"
            "核心交互类似 Google Docs：多人实时编辑、OT/CRDT 冲突解决、版本历史回放。"
            "团队规模 6 人，人均 2-3 年经验，对原生开发和 Web 均有基础但不深入。"
            "你会在技术选型讨论中力推哪个方案？"
        ),
        "options": [
            {
                "text": (
                    "基于 Flutter 的 Impeller 渲染引擎统一移动端和桌面端，"
                    "Web 端用 Flutter Web + CanvasKit 保证渲染一致性，"
                    "协作层用 gRPC-Web + Protobuf 序列化操作元数据，"
                    "在客户端层实现 CRDT 的 RGA（Replicated Growable Array）算法"
                ),
                "tendency": "客户端",
            },
            {
                "text": (
                    "全平台统一用 Web 技术栈：React + Tiptap（基于 ProseMirror）实现编辑器核心，"
                    "用 Yjs（YATA CRDT）处理冲突解决，通过 WebSocket + Redis Pub/Sub 做协作同步，"
                    "桌面端用 Tauri 打包，移动端用 Capacitor 封装为原生 App"
                ),
                "tendency": "前端",
            },
            {
                "text": (
                    "将冲突解决从客户端抽到服务端：客户端只做乐观更新，"
                    "服务端维护一个基于 Lamport 逻辑时钟的 OT 转换引擎，"
                    "所有操作经过后端并串行化后再广播，利用 MongoDB Change Stream 实现版本历史的增量备份"
                ),
                "tendency": "后端",
            },
            {
                "text": (
                    "不急于做技术选型：先用 UX 原型（Figma/Sketch）做可交互设计稿进行用户测试，"
                    "通过观察真实用户在协作场景中的心智模型来定义核心功能边界，"
                    "根据用户反馈决定是否真的需要 CRDT 级别的实时协作还是异步评论模式即可"
                ),
                "tendency": "产品",
            },
        ],
        "difficulty": "medium",
        "category": "架构选型",
        "tags": ["CRDT", "OT", "Flutter", "Yjs", "ProseMirror"],
    },

    # ============================================================
    # Q10：项目规划 — 从零到一的启动策略
    # ============================================================
    {
        "id": 10,
        "scenario": (
            "你作为技术负责人需要在一个季度内从零交付一个 AI 简历解析与岗位匹配的 SaaS 产品。"
            "核心链路：上传 PDF 简历 → 解析关键信息（学历/技能/经历）→ 与 JD 做语义匹配 → "
            "生成排序推荐列表。目前只有你一人 + 两个即将入职的实习生。"
            "面临的最大张力是：产品和投资人期望在 Demo Day 看到完整的交互，"
            "但你清楚内部技术债如果全堆在第一个版本上，第二个月就会寸步难行。"
            "你的第一个 Sprint 会集中火力在哪里？"
        ),
        "options": [
            {
                "text": (
                    "用 Streamlit 快速搭建端到端原型：调用 DeepSeek API 做 PDF 解析和语义匹配，"
                    "所有逻辑写在单文件中，牺牲架构优雅换取 Demo Day 的可演示性。"
                    "在 Demo 后第二个 Sprint 再逐步拆分为微服务并引入消息队列解耦"
                ),
                "tendency": "后端",
            },
            {
                "text": (
                    "第一天就建立工程地基：配置 CI/CD（GitHub Actions + Docker Compose），"
                    "编写 OpenAPI 3.0 接口规范并生成 Server/Client Stub，"
                    "集成 Sentry + Prometheus + Grafana 做全链路监控，"
                    "先搭好 API Gateway、鉴权、日志系统再开始写业务代码"
                ),
                "tendency": "测试运维",
            },
            {
                "text": (
                    "核心聚焦匹配模型的冷启动问题：在没有标注数据的情况下，"
                    "用 LLM（DeepSeek）做 few-shot prompt 生成初步匹配结果，"
                    "同时设计一个隐式反馈收集机制（用户在推荐列表中的点击行为），"
                    "用 Click-Through Rate 作为弱监督信号训练一个轻量级的 BERT-based reranker"
                ),
                "tendency": "AI数据",
            },
            {
                "text": (
                    "先做问题验证而非工程实现：在第一周与 5-8 位目标用户做深度访谈，"
                    "用纸笔原型验证核心假设——用户真的需要自动匹配还是只需要更好的搜索过滤器，"
                    "根据访谈结论重新定义 MVP 功能边界后再写第一行代码"
                ),
                "tendency": "产品",
            },
        ],
        "difficulty": "medium",
        "category": "项目规划",
        "tags": ["SaaS", "MVP", "LLM", "Sentry", "用户研究"],
    },
]


def get_full_question_bank() -> List[dict]:
    """返回完整的 10 道预设情景多选题库"""
    return QUESTION_BANK


def format_questions_for_llm(questions: List[dict]) -> str:
    """
    将题目列表格式化为适合传入 LLM prompt 的字符串

    Args:
        questions: 题目列表

    Returns:
        格式化的字符串，每道题包含 id、scenario、选项文本
    """
    lines = []
    for q in questions:
        lines.append(f"## 题目 {q['id']} [{q['category']}]")
        lines.append(f"场景：{q['scenario']}")
        lines.append("选项：")
        for i, opt in enumerate(q["options"]):
            label = chr(65 + i)  # A, B, C, D
            lines.append(f"  {label}. {opt['text']}")
        lines.append("")
    return "\n".join(lines)


def get_question_by_id(question_id: int) -> dict | None:
    """按 id 获取单道题目"""
    for q in QUESTION_BANK:
        if q["id"] == question_id:
            return q
    return None


def get_questions_by_category(category: str) -> List[dict]:
    """按 category 筛选题目"""
    return [q for q in QUESTION_BANK if q["category"] == category]


def get_valid_tendencies() -> List[str]:
    """返回合法的 tendency 取值列表"""
    return ["前端", "后端", "AI数据", "测试运维", "产品", "客户端"]
