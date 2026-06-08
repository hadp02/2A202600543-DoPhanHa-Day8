from pathlib import Path
import chromadb
from rank_bm25 import BM25Okapi
import numpy as np

CHROMA_PERSIST_DIR = str(Path(__file__).parent.parent / "data" / "chromadb")
COLLECTION_NAME = "drug_law_docs"

def get_all_chunks() -> list[dict]:
    try:
        client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
        collection = client.get_collection(COLLECTION_NAME)
        data = collection.get()
        if not data or not data["documents"]:
            return []
        return [{"content": doc, "metadata": meta} for doc, meta in zip(data["documents"], data["metadatas"])]
    except Exception:
        return []

CORPUS: list[dict] = get_all_chunks()

def build_bm25_index(corpus: list[dict]):
    """
    Xây dựng BM25 index từ corpus.
    """
    if not corpus: return None
    tokenized_corpus = [doc["content"].lower().split() for doc in corpus]
    bm25 = BM25Okapi(tokenized_corpus)
    return bm25

bm25_index = build_bm25_index(CORPUS)

def lexical_search(query: str, top_k: int = 10) -> list[dict]:
    """
    Tìm kiếm từ khóa sử dụng BM25.
    """
    if not bm25_index or not CORPUS:
        # Fallback for testing when corpus is empty
        return [{"content": f"dummy lexical {i}", "score": 1.0 - i*0.1, "metadata": {}} for i in range(top_k)]
        
    tokenized_query = query.lower().split()
    scores = bm25_index.get_scores(tokenized_query)
    
    # Get top_k indices
    top_indices = np.argsort(scores)[::-1][:top_k]
    
    results = []
    for idx in top_indices:
        if scores[idx] > 0:
            results.append({
                "content": CORPUS[idx]["content"],
                "score": float(scores[idx]),
                "metadata": CORPUS[idx]["metadata"]
            })
    return results

if __name__ == "__main__":
    # Test
    results = lexical_search("Điều 248 tàng trữ trái phép chất ma tuý", top_k=5)
    for r in results:
        print(f"[{r['score']:.3f}] {r['content'][:100]}...")
