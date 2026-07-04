"""
PDF 简历解析工具

提供简历文本提取功能，支持中英文简历
"""
import pdfplumber
from typing import Optional
import io


def extract_text_from_pdf(pdf_path: str) -> str:
    """
    从 PDF 文件提取文本内容
    
    参数：
        pdf_path: PDF 文件路径
    
    返回：
        提取的文本内容（按页拼接）
    
    示例：
        >>> text = extract_text_from_pdf("resume.pdf")
        >>> print(text[:500])
    """
    try:
        with pdfplumber.open(pdf_path) as pdf:
            pages_text = []
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    pages_text.append(text)
            return "\n\n".join(pages_text)
    except Exception as e:
        raise RuntimeError(f"PDF 解析失败: {e}")


def extract_text_from_pdf_bytes(pdf_bytes: bytes) -> str:
    """
    从 PDF 字节流提取文本内容
    
    适用于 Streamlit file_uploader 返回的 BytesIO 对象
    
    参数：
        pdf_bytes: PDF 文件字节流
    
    返回：
        提取的文本内容
    """
    try:
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            pages_text = []
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    pages_text.append(text)
            return "\n\n".join(pages_text)
    except Exception as e:
        raise RuntimeError(f"PDF 解析失败: {e}")


def extract_resume_structured_info(text: str) -> dict:
    """
    从简历文本中提取结构化信息（占位符）
    
    TODO: 后续可接入 LLM 进行结构化提取
    
    返回：
        包含姓名、邮箱、学历、技能等字段的字典
    """
    return {
        "name": "待提取",
        "email": "待提取",
        "phone": "待提取",
        "education": "待提取",
        "skills": [],
        "raw_text": text
    }


if __name__ == "__main__":
    # 测试用例
    print("PDF 解析工具测试")
    print("=" * 50)
    print("请使用 extract_text_from_pdf() 或 extract_text_from_pdf_bytes() 函数")
