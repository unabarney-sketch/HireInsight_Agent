"""
RAG 检索模块

使用 scikit-learn TF-IDF + 余弦相似度进行向量检索。
纯 Python 实现，无需任何 C++ 编译环境，跨平台兼容。
"""

import os
import pickle
import json
from typing import List, Dict, Optional

import numpy as np
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()

# ============================================================
# 持久化路径
# ============================================================
_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./data/rag_store")
_INDEX_FILE = os.path.join(_PERSIST_DIR, "tfidf_index.pkl")
_DATA_FILE = os.path.join(_PERSIST_DIR, "documents.json")


def _ensure_dir() -> None:
    """确保持久化目录存在。"""
    os.makedirs(_PERSIST_DIR, exist_ok=True)


# ============================================================
# RagStore: 兼容原 ChromaDB API 的向量存储
# ============================================================
class RagStore:
    """基于 TF-IDF 的向量存储，API 兼容 ChromaDB Collection。"""

    def __init__(self):
        self.vectorizer: TfidfVectorizer = TfidfVectorizer(max_features=5000)
        self.tfidf_matrix = None
        self.documents: List[str] = []
        self.metadatas: List[Dict] = []
        self.ids: List[str] = []

    # ------ 数据操作 ------
    def add(self, ids: List[str], documents: List[str],
            metadatas: List[Dict]) -> None:
        """批量添加文档。"""
        self.ids.extend(ids)
        self.documents.extend(documents)
        self.metadatas.extend(metadatas)
        self._rebuild_index()

    def count(self) -> int:
        """返回文档数量。"""
        return len(self.ids)

    def query(self, query_texts: List[str], n_results: int = 2) -> Dict:
        """相似度检索。"""
        result = {
            "documents": [[]],
            "metadatas": [[]],
            "distances": [[]],
        }

        if self.tfidf_matrix is None or not self.documents:
            return result

        query_vec = self.vectorizer.transform(query_texts)
        similarities = cosine_similarity(query_vec, self.tfidf_matrix)[0]

        top_k = min(n_results, len(similarities))
        top_indices = np.argsort(similarities)[::-1][:top_k]

        return {
            "documents": [[self.documents[i] for i in top_indices]],
            "metadatas": [[self.metadatas[i] for i in top_indices]],
            "distances": [[float(1.0 - similarities[i]) for i in top_indices]],
        }

    # ------ 索引重建 ------
    def _rebuild_index(self) -> None:
        """全量重建 TF-IDF 索引。"""
        if self.documents:
            self.tfidf_matrix = self.vectorizer.fit_transform(self.documents)
        else:
            self.tfidf_matrix = None

    # ------ 持久化 ------
    def save(self) -> None:
        """保存到磁盘。"""
        _ensure_dir()
        with open(_INDEX_FILE, "wb") as f:
            pickle.dump({
                "vectorizer": self.vectorizer,
                "tfidf_matrix": self.tfidf_matrix,
            }, f)
        with open(_DATA_FILE, "w", encoding="utf-8") as f:
            json.dump({
                "ids": self.ids,
                "documents": self.documents,
                "metadatas": self.metadatas,
            }, f, ensure_ascii=False)

    @classmethod
    def load(cls) -> "RagStore":
        """从磁盘加载。"""
        _ensure_dir()
        store = cls()

        if os.path.exists(_INDEX_FILE) and os.path.exists(_DATA_FILE):
            with open(_INDEX_FILE, "rb") as f:
                data = pickle.load(f)
                store.vectorizer = data["vectorizer"]
                store.tfidf_matrix = data["tfidf_matrix"]
            with open(_DATA_FILE, "r", encoding="utf-8") as f:
                doc_data = json.load(f)
                store.ids = doc_data["ids"]
                store.documents = doc_data["documents"]
                store.metadatas = doc_data["metadatas"]

        return store

    def delete_all(self) -> None:
        """清空存储并删除持久化文件。"""
        self.ids.clear()
        self.documents.clear()
        self.metadatas.clear()
        self.tfidf_matrix = None
        self.vectorizer = TfidfVectorizer(max_features=5000)
        # 删除磁盘文件
        for path in (_INDEX_FILE, _DATA_FILE):
            if os.path.exists(path):
                os.remove(path)


# ============================================================
# 对外 API（与原 chromadb 版本兼容）
# ============================================================

def init_store() -> RagStore:
    """
    初始化向量存储。

    返回：
        RagStore 实例
    """
    store = RagStore.load()
    return store


def create_interview_collection(store: Optional[RagStore] = None) -> RagStore:
    """
    获取/初始化面经存储。

    参数：
        store: RagStore 实例，若为 None 则初始化新实例

    返回：
        RagStore 实例
    """
    if store is None:
        store = init_store()
    return store


# ============================================================
# 单例模式：防 Streamlit 多线程重复实例化
# ============================================================
_store_instance: Optional[RagStore] = None


def get_or_init_collection() -> RagStore:
    """
    单例获取/初始化向量存储。

    初始化逻辑：目录检测 → 返回 store。
    此单例设计可防止 Streamlit 多线程重复实例化。
    """
    global _store_instance
    if _store_instance is not None:
        return _store_instance

    _store_instance = init_store()
    return _store_instance


def load_experiences_to_rag(
    experiences: List[Dict[str, str]],
    store: Optional[RagStore] = None,
) -> int:
    """
    批量加载面经数据到向量存储。

    参数：
        experiences: 面经列表，每项包含：
            - company: 公司名
            - position: 岗位名
            - question: 面试题
            - answer: 参考回答（可选）
            - difficulty: 难度（easy/medium/hard）
        store: RagStore 实例

    返回：
        成功加载的数据条数
    """
    st = create_interview_collection(store)

    ids = []
    documents = []
    metadatas = []

    for i, exp in enumerate(experiences):
        exp_id = f"exp_{i}"
        doc = (
            f"公司：{exp.get('company', '未知')}\n"
            f"岗位：{exp.get('position', '未知')}\n"
            f"面试题：{exp.get('question', '')}\n"
            f"参考回答：{exp.get('answer', '')}"
        )

        ids.append(exp_id)
        documents.append(doc)
        metadatas.append({
            "company": exp.get("company", "未知"),
            "position": exp.get("position", "未知"),
            "difficulty": exp.get("difficulty", "medium"),
        })

    if ids:
        st.add(ids=ids, documents=documents, metadatas=metadatas)
        st.save()

    return len(ids)


def query_experiences(
    query: str,
    n_results: int = 2,
    store: Optional[RagStore] = None,
) -> List[Dict]:
    """
    检索面经数据。

    参数：
        query: 检索查询文本
        n_results: 返回条数
        store: RagStore 实例

    返回：
        检索结果列表
    """
    st = create_interview_collection(store)

    results = st.query(query_texts=[query], n_results=n_results)

    if not results or not results.get("documents"):
        return []

    return [
        {
            "document": doc,
            "metadata": meta,
            "distance": dist,
        }
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        )
    ]


def clear_collection(store: Optional[RagStore] = None) -> None:
    """
    清空面经集合（用于测试重置）。
    """
    if store is None:
        store = init_store()
    store.delete_all()


# ============================================================
# 兼容旧 API 名称（chromadb 时代沿用）
# ============================================================
init_chromadb = init_store
load_experiences_to_chroma = load_experiences_to_rag


# ============================================================
# 独立运行测试
# ============================================================
if __name__ == "__main__":
    print("=== RAG 向量存储初始化测试 ===")
    print(f"存储目录：{_PERSIST_DIR}")
    print()

    # 初始化
    store = init_store()
    print(f"向量存储已加载，当前文档数：{store.count()}")

    # 测试检索
    results = query_experiences("Python后端面试", n_results=2, store=store)
    print(f"\n检索结果：{len(results)} 条")
    for r in results:
        preview = r["document"][:100].replace("\n", " | ")
        print(f"  - [{r['metadata']['company']} | {r['metadata']['position']}] dist={r['distance']:.4f}")
        print(f"    {preview}...")
