## 用户需求

处于"后中期精细化开发阶段"，要求以高级系统架构师角色，参考高保真视觉设计蓝图（`assets/参考首页图.png`）进行系统建模，为首页重构输出一份详细的技术落地方案与步骤规划文档。仅输出规划文档 `docs/homepage_rebuild_plan.md`，**不修改任何 .py 业务代码**。

### 资产说明

- **参考视觉蓝图**：`assets/参考首页图.png`，展示最终首页的完整模块布局（分栏排版、卡片圆角、文字排布）和高保真奶油风视觉档次
- **可用静态底图**：`assets/pure1.png`，已擦除左侧文字的纯净插画 Banner，将作为顶部欢迎 Banner 的 CSS 背景图渲染
- **当前已有资产**：`pure_login_layout.png`、`home_banner.png`、`dashboard_banner.png` 等（home_banner.png 将被 pure1.png 替代）

### 数据底座

离线数仓 573 条真实岗位资产：腾讯 299、滴滴 208、美团 42、网易 14、字节 10

### 四大核心板块需求

1. **顶部欢迎 Banner**：用 `st.container` 结合 CSS `background-image` 渲染 `pure1.png`，左侧空白区用 HTML/CSS 渲染动态文本"你好，Ethan 👋"、副标题"探索大厂机会，从这里开始"、职位搜索输入框 + 搜索按钮 + 热门推荐标签行
2. **中层三大数据看板**（`st.columns([1,1,1])`）：

- Card 1 真实大厂职位直推：SQL `SELECT * FROM job_positions ORDER BY crawled_at DESC LIMIT 8` 捞取最新大厂岗位，高仿列表渲染（公司 Logo 圆标 + 公司名 + 岗位名 + 薪资标签 + 右箭头）
- Card 2 我的手动投递 CRM：联动 `applications` 表（待创建），展示投递节点状态（已投递/一面中/Offer/感谢信），含状态高亮标签色块
- Card 3 573条大厂底仓数仓实时监控：用 Streamlit 原生极简横向条形图或定制 CSS 进度条复现五大厂数据量级分布大盘

3. **底层工具与资讯链**：左侧 `st.columns(4)` 四工具卡片入口（数据大屏/灯塔测评/全真面试/简历诊断）注入 Hover 放大微动效 CSS；右侧技术资讯 Markdown 列表占位符（面经快讯）

## 技术方案

### 技术栈

- **前端框架**：Streamlit 1.x（已使用 `layout="wide"`）
- **样式注入**：`st.markdown(..., unsafe_allow_html=True)` 内联 CSS
- **数据层**：SQLite（`data/hireinsight.db`），通过 `utils/data_persistence.py` 提供的 `query_jobs_by_filters()` 和 `load_from_sqlite()` 查询
- **图片资源**：`base64` 编码静态图片为 Data URI，或通过 `st.image()` 渲染
- **可视化**：原生 HTML/CSS 进度条（Card 3 数据大盘），不引入额外 JS 图表库

### 实现策略

#### CSS 样式定制方案

在现有约 430 行 Creamy Clean CSS 体系之上增量扩展，遵循以下原则：

- **不破坏现有全局样式**：所有新增样式使用 `.home-*` 命名空间前缀
- **色调继承**：延续 #F6F5F2 背景、#2D2722 巧克力褐、18px 圆角、1px #EFEFEF 边框
- **Banner 区域**：新增 `.home-banner-container` 类，设置 `background-image: url(...)` + `background-size: cover` + `background-position: center` + `border-radius: 20px` + `position: relative` + `min-height: 380px`。内部文字叠加层用 `.home-banner-overlay` 绝对定位在左侧
- **数据卡片增强**：在现有 `div[data-testid="stVerticalBlockBorderWrapper"]` 基础上，通过 `.home-dashboard-card` 添加柔和阴影 `box-shadow: 0 2px 12px rgba(45,39,34,0.04)`，悬停时 `box-shadow: 0 6px 24px rgba(45,39,34,0.08)` + `transform: translateY(-2px)` + `transition: all 0.3s ease`
- **条形图进度条**：用纯 HTML `div` 嵌套实现横向进度条，外层 `width:100%; background:#F5F5F2; border-radius:8px; height:12px`，内层按数据比例设置 `width:X%; background:渐变; border-radius:8px; transition: width 0.8s ease`
- **Hover 动效**：`.home-tool-card:hover { transform: scale(1.03); box-shadow: 0 8px 28px rgba(45,39,34,0.1); }` + `transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1)`
- **状态标签色块**：`.status-tag` 基础类 `display:inline-block; padding:2px 10px; border-radius:10px; font-size:0.78rem; font-weight:500`，四种状态变体 `status-applied`(绿色 #E8F5E9/#2E7D32)、`status-interview`(橙色 #FFF3E0/#E65100)、`status-offer`(金色 #FFFDE7/#F9A825)、`status-rejected`(蓝色 #E3F2FD/#1565C0)

#### Banner 纯 CSS 背景图方案

由于 Streamlit 不支持直接在 `st.container` 上设 `background-image`，采用 HTML `div` 层方案：

1. 用 `base64` 读取 `assets/pure1.png` 编码为 Data URI
2. 通过 `st.markdown(f'<div class="home-banner-container" style="background-image:url(data:image/png;base64,{b64})">...{左侧文字HTML}...</div>', unsafe_allow_html=True)` 渲染
3. 左侧文字层使用 `position:absolute; left:48px; top:50%; transform:translateY(-50%); max-width:45%` 定位在 Banner 左侧空白区
4. 搜索框使用纯 HTML `<input type="text" placeholder="搜索职位/公司/技能..."/>` + `<button>搜索职位</button>`，键盘事件需通过 `st.text_input` 桥接（HTML input 的 onChange 无法直接与 Streamlit session_state 通信）。实际落地时搜索框用 `st.text_input` 组件放在 Banner 下方独立行，而非嵌入 CSS 背景容器内，以保持 Streamlit 交互能力

#### 容器分栏嵌套树

```
顶级: st.markdown(CSS注入)                              # ~550 行全局+首页专属 CSS
│
├── Layer 0: st.container()                             # 顶级容器（非必须，可省略）
│   │
│   ├── Block A: st.markdown()                          # Banner HTML div 层
│   │   └── <div class="home-banner-container">
│   │       ├── 背景图: pure1.png (CSS background-image)
│   │       └── <div class="home-banner-overlay">
│   │           ├── <h1>你好, {user['display_name']} 👋</h1>
│   │           ├── <p>探索大厂机会，从这里开始</p>
│   │           ├── <div class="home-search-row">
│   │           │   ├── <input placeholder="搜索职位/公司/技能">
│   │           │   └── <button>搜索职位</button>
│   │           │   └── (Fallback: st.text_input 独立行)
│   │           └── <div class="hot-tags">
│   │               └── <span>热门: 后端开发 算法 AI 产品经理 数据分析</span>
│   │
│   ├── Block B: st.columns([1, 1, 1])                  # 三栏数据看板
│   │   ├── col[0]: st.container(border=True)
│   │   │   └── Card 1: 真实大厂职位直推
│   │   │       ├── st.markdown("🎯 真实大厂职位直推")
│   │   │       ├── for job in jobs[:6]:
│   │   │       │   └── st.markdown(HTML列表项)
│   │   │       │       ├── 公司Logo圆标(首字母)
│   │   │       │       ├── 公司名 + 岗位名
│   │   │       │       ├── 薪资标签 <span class="salary-tag">20k-40k</span>
│   │   │       │       └── 直达链接 > 
│   │   │       └── st.markdown("查看全部职位 →")
│   │   │
│   │   ├── col[1]: st.container(border=True)
│   │   │   └── Card 2: 我的手动投递 CRM
│   │   │       ├── st.markdown("📋 我的手动投递 CRM")
│   │   │       ├── if applications exists:
│   │   │       │   └── for app in applications:
│   │   │       │       └── st.markdown(HTML列表项)
│   │   │       │           ├── 图标 + 公司名 + 岗位名
│   │   │       │           ├── 投递时间
│   │   │       │           └── <span class="status-tag status-xxx">状态</span>
│   │   │       └── else: 空状态占位符
│   │   │
│   │   └── col[2]: st.container(border=True)
│   │       └── Card 3: 573条大厂底仓数仓实时监控
│   │           ├── st.markdown("🔥 573条大厂底仓数仓实时监控")
│   │           ├── for company, count in data_distribution:
│   │           │   └── st.markdown(HTML横向进度条)
│   │           │       ├── ● 彩色圆点
│   │           │       ├── 公司名
│   │           │       ├── <div class="progress-bar-bg"><div class="progress-bar-fill" style="width:X%"></div></div>
│   │           │       └── 数字 badage
│   │           └── st.markdown("查看数据大盘 →")
│   │
│   └── Block C: st.columns([3, 2])                     # 底部工具与资讯
│       ├── col_left: 
│       │   ├── st.markdown("🛠️ 求职工具")
│       │   └── st.columns(4)
│       │       └── for tool in tools:                    # 四个工具卡片
│       │           └── <div class="home-tool-card">
│       │               ├── 图标
│       │               ├── 标题
│       │               └── 副标题
│       │
│       └── col_right:
│           ├── st.markdown("📰 技术资讯 · 每日更新")
│           └── st.markdown(静态MD列表)
│               ├── 07-04 · React 19 正式发布 · 前端之巅 · 2.3k
│               ├── 07-03 · 2026 大厂校招薪资汇总 · 牛客网 · 5.1k
│               └── ...
```

#### 数据查询方案

- **Card 1 职位列表**：`SELECT title, company, salary_min, salary_max, city, url FROM job_positions ORDER BY crawled_at DESC LIMIT 8`，通过 `load_from_sqlite()` 或直接 `sqlite3.connect` 查询
- **Card 3 数据大盘**：`SELECT source, COUNT(*) as cnt FROM job_positions GROUP BY source`，得到分布数据后硬编码为列表渲染。考虑到 573 条是静态快照，使用 `st.cache_data(ttl=3600)` 缓存查询结果
- **Card 2 投递 CRM**：applications 表尚未创建（在 Phase 3 实现），当前用空状态占位符 `st.info("投递记录功能即将上线，敬请期待")`
- **公司名称映射**：`{"tencent":"腾讯","didi":"滴滴","meituan":"美团","netease":"网易","bytedance":"字节"}`

### 性能考量

- Banner 图片 `pure1.png` 使用 base64 Data URI 会增加首次渲染大小（预估 100-300KB），需控制图片分辨率。流程中应先检测图片尺寸，若过大则用 PIL 压缩至宽度 1200px 以内
- 数据面板 SQL 查询使用 `st.cache_data(ttl=3600)` 缓存，避免每次 rerun 都查询 SQLite
- 五大厂数据分布可直接在 CSS 中硬编码（573 条为离线快照），Card 3 渲染零 SQL 开销
- HTML 列表项循环渲染避免使用 `st.dataframe`（全量重渲染），改用 `st.markdown` 拼接 HTML 字符串批量输出

### 向后兼容

- 新 CSS 全部使用 `.home-*` 前缀，不与现有全局样式冲突
- `render_home()` 函数保持向后兼容：若 `pure1.png` 不存在或用户未登录，fallback 到原有 Banner 逻辑
- applications 表不存在时 Card 2 降级显示占位文本，不抛异常

## 设计风格

延续项目已确立的 **Creamy Clean 奶油风** 设计体系，参考 `assets/参考首页图.png` 的高保真布局进行精细化复现。

### 整体氛围

温暖、柔和、人文关怀的求职助手调性。浅米色背景 #F6F5F2 打底，纯白卡片悬浮其上，圆角统一 18-20px，阴影极浅（rgba 透明度 0.04-0.08），营造"浮在奶油上"的视觉感受。

### Banner 区域

左侧文字叠加层采用三段式排版：大标题 2.2rem 700 字重巧克力褐、副标题 0.95rem 400 字重暖灰色、搜索框 48px 高白色圆角输入框 + 深色按钮。右侧 pure1.png 插画人物自然占据视觉重心。热门标签行用小号圆角 pill 标签横向排列。

### 三栏数据卡片

三等分列，每列一张白色卡片。Card 1 职位列表项：左侧公司首字母圆形色块（32px） + 中间公司名/岗位双行文字 + 右侧橙色薪资标签 + 箭头图标。Card 2 投递列表：左侧图标 + 内容区（公司/岗位/时间）+ 右侧状态标签（四色区分）。Card 3 数据大盘：五行横向进度条，每行左侧彩色圆点 + 公司名 + 填充条（渐变色）+ 右侧数字 badge。

### 底部工具区

四宫格工具卡片：大图标居中 + 标题 0.95rem + 副标题 0.78rem 暖灰色，Hover 时卡片整体放大 1.03 倍 + 阴影加深。资讯区用简洁的 Markdown 列表，日期/标题/来源/阅读量四列对齐。

### 动效

- Banner 区域无动效（静态图片背景）
- 卡片 hover：`transform: translateY(-2px)` + 阴影加深，transition 0.3s ease
- 工具卡片 hover：`transform: scale(1.03)` + 阴影加深，transition 0.25s cubic-bezier
- 进度条加载：`width` 从 0 过渡到目标值，transition 0.8s ease