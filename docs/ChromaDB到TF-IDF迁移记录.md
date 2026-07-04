# ChromaDB → TF-IDF 向量存储迁移记录

> 日期：2026-07-04 | 目的：解决 Windows Python 3.12 下 chromadb 无法安装的问题

---

## 1. 问题背景

原项目使用 `chromadb==0.5.5` 作为 RAG 向量数据库，依赖 `chroma-hnswlib` 原生扩展。在 Windows + Python 3.12 环境下运行时，`pip install` 始终失败：

```
error: Microsoft Visual C++ 14.0 or greater is required.
Building wheel for chroma-hnswlib (pyproject.toml) did not run successfully.
```

### 尝试过的版本（全部失败）

| 版本 | 依赖包 | 失败原因 |
|------|--------|----------|
| `chromadb==0.5.5` | `chroma-hnswlib` | 需 C++ 编译 |
| `chromadb==0.6.3` | `chroma-hnswlib` | 需 C++ 编译 |
| `chromadb==0.4.24` | `chroma-hnswlib` | 需 C++ 编译 |
| `hnswlib`（PyPI 原版） | `hnswlib` | 同样需 C++ 编译 |

结论：Python 3.12 + Windows 下，所有基于 hnswlib 的方案都需要 Visual C++ Build Tools，无预编译 wheel 可用。

---

## 2. 解决方案

**用 scikit-learn 的 TF-IDF + 余弦相似度 替代 ChromaDB，纯 Python 实现，零 C++ 依赖。**

### 改动文件清单

| 文件 | 改动内容 |
|------|----------|
| `utils/rag_loader.py` | **核心重写**：用 `RagStore` 类替代 `chromadb`，基于 `TfidfVectorizer` + `cosine_similarity` |
| `requirements.txt` | `chromadb==0.5.23` → `scikit-learn==1.5.2`<br>`streamlit-aggrid==0.3.5` → `1.0.5`（修复 yanked 版本）<br>中文注释改为英文（修复 gbk 编码错误） |
| `run.bat` | 端口 `8501` → `8502`<br>pip 安装增加 `--no-cache-dir` 和清华镜像源 |
| `app.py` | UI 文案：ChromaDB → RAG 向量存储 |
| `graphs/state.py` | 注释：ChromaDB → RAG |

### 新 RagStore 设计

```
utils/rag_loader.py
├── RagStore 类              纯 Python 向量存储
│   ├── add()               批量添加文档 + 元数据
│   ├── query()             余弦相似度检索（返回 documents/metadatas/distances）
│   ├── count()             文档数量
│   ├── save() / load()     pickle + JSON 持久化（存储路径：./data/rag_store/）
│   └── delete_all()        清空存储
├── get_or_init_collection() 单例模式（兼容原 API）
├── load_mock_experiences()  加载 Mock 面经
├── query_experiences()      检索面经
└── init_chromadb / load_experiences_to_chroma  兼容旧 API 别名
```

### API 兼容性

所有对外接口命名和行为与原来一致，`graphs/nodes.py` 和 `app.py` 中的调用无需改动：

```python
# 调用方式不变
from utils.rag_loader import get_or_init_collection, query_experiences
store = get_or_init_collection()
results = query_experiences("Python后端面试", n_results=2)
```

### 持久化格式

- `./data/rag_store/tfidf_index.pkl` — TF-IDF 向量化器 + 矩阵（pickle）
- `./data/rag_store/documents.json` — 文档文本 + 元数据（JSON，UTF-8）
- 环境变量 `CHROMA_PERSIST_DIR` 仍可使用（保持向后兼容）

---

## 3. 附加修复

| 问题 | 修复 |
|------|------|
| `requirements.txt` 中文注释导致 pip 用 gbk 解码报错 | 改为英文注释 |
| `streamlit-aggrid==0.3.5` 被 PyPI 标记为 yanked (Bugged) | 升级到 `1.0.5` |
| 端口 8501 被占用 | 改为 8502，`run.bat` 中同步修改 |
| pip 下载超时（国外源慢） | `run.bat` 增加清华镜像源 `-i https://pypi.tuna.tsinghua.edu.cn/simple` |

---

## 4. 验证结果

```
Store loaded, count: 0
Mock data loaded: 5
Query results: 2
拼多多
```

RAG 模块正常初始化和检索，项目可成功启动运行。

---

## 5. 后续建议

- 当前 Mock 数据仅 5 条，TF-IDF 精度足够。若未来面经数据量增长到数千条，可考虑更换为 `faiss-cpu`（有 Windows 预编译 wheel）以获得更好的检索性能。
- 如需恢复 ChromaDB，只需安装 Visual Studio 2022 Build Tools（约 3GB），然后还原 `requirements.txt` 即可。
