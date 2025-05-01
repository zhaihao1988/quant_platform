# rag/retriever.py

from chromadb import Client, PersistentClient
from chromadb.config import Settings
from langchain_community.vectorstores import Chroma
from rag.embeddings import Embedder

# 本地持久化目录
PERSIST_DIR = "./chroma_db"

# 初始化 Client 与集合
_chroma_client = PersistentClient(path=PERSIST_DIR, settings=Settings())

def get_vectorstore():
    embedder = Embedder()
    return Chroma(
        client=_chroma_client,
        collection_name="announcements",
        embedding_function=embedder.embed,
        persist_directory=PERSIST_DIR,
    )

def add_documents(symbol: str, docs: list[str]):
    vs = get_vectorstore()
    metadatas = [{"symbol": symbol, "source_rank": i} for i in range(len(docs))]
    vs.add_documents(documents=docs, metadatas=metadatas)
    vs.persist()

def retrieve_context(query: str, k: int = 3) -> list[str]:
    vs = get_vectorstore()
    results = vs.similarity_search(query, k=k)
    return [r.page_content for r in results]
