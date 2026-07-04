"""
HireInsight-Agent - 工具模块

包含：
1. PDF 解析工具
2. 正则清洗工具
3. RAG 加载脚本
4. 数据获取工具

注意
----
本包使用 __getattr__ 延迟导入机制，避免在未安装 pdfplumber / chromadb
等可选依赖时整个 utils 包无法加载。只有在显式通过包级接口访问
（如 ``from utils import extract_text_from_pdf``）时才会触发对应子模块的导入。
直接引用子模块（如 ``from utils.data_cleaner import clean_job_data``）不受影响。
"""

__all__ = [
    "extract_text_from_pdf",
    "clean_job_data",
    "init_chromadb",
    "load_experiences_to_chroma",
]

_LAZY_IMPORT_MAP: dict[str, tuple[str, str | tuple[str, ...]]] = {
    "extract_text_from_pdf":      (".pdf_parser",  "extract_text_from_pdf"),
    "clean_job_data":             (".data_cleaner", "clean_job_data"),
    "init_chromadb":              (".rag_loader",   ("init_chromadb", "load_experiences_to_chroma")),
    "load_experiences_to_chroma": (".rag_loader",   ("init_chromadb", "load_experiences_to_chroma")),
}


def __getattr__(name: str):
    if name in _LAZY_IMPORT_MAP:
        mod_path, attrs = _LAZY_IMPORT_MAP[name]
        mod = __import__(mod_path, fromlist=[attrs] if isinstance(attrs, str) else list(attrs), globals=globals(), level=1)
        if isinstance(attrs, str):
            obj = getattr(mod, attrs)
        else:
            obj = getattr(mod, name)
        # 缓存到模块命名空间，下次直接命中
        globals()[name] = obj
        return obj
    raise AttributeError(f"module 'utils' has no attribute '{name}'")
