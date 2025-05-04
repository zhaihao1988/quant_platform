# utils/vector_store.py

from sentence_transformers import SentenceTransformer

# 全局加载一次，避免重复初始化
_EMBED_MODEL = SentenceTransformer("shibing624/text2vec-base-chinese-paraphrase")

def embed_text(text: str) -> list[float]:
    """
    将输入文本编码为 768 维向量。
    对于超长文本，可考虑分段拼接或取平均。
    """
    # 若文本长度过大，可先手动分句、分段；此处简化处理
    vec = _EMBED_MODEL.encode([text], show_progress_bar=False)[0]
    return vec.tolist()
