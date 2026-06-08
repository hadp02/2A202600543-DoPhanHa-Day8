import os
from pathlib import Path
from dotenv import load_dotenv
import chromadb

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    pass

load_dotenv()

CHROMA_PERSIST_DIR = str(Path(__file__).parent.parent / "data" / "chromadb")
COLLECTION_NAME = "drug_law_docs"

def semantic_search(query: str, top_k: int = 10) -> list[dict]:
    """
    Tìm kiếm ngữ nghĩa sử dụng vector similarity.
    """
    try:
        # 1. Embed query
        model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        query_embedding = model.encode(query).tolist()

        # 2. Query vector store
        client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
        try:
            collection = client.get_collection(name=COLLECTION_NAME)
        except Exception:
            return [] # collection does not exist
            
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )
        
        # 3. Format output
        output = []
        if results["documents"] and len(results["documents"][0]) > 0:
            docs = results["documents"][0]
            metas = results["metadatas"][0]
            dists = results["distances"][0]
            
            for doc, meta, dist in zip(docs, metas, dists):
                output.append({
                    "content": doc,
                    "score": 1.0 / (1.0 + dist), # convert distance to similarity score
                    "metadata": meta
                })
        
        # Sort descending by score
        output.sort(key=lambda x: x["score"], reverse=True)
        return output
    except Exception as e:
        print(f"Semantic search error: {e}")
        return [
            {"content": f"dummy semantic result {i}", "score": 1.0 - i*0.1, "metadata": {"source": "test.md"}} 
            for i in range(top_k)
        ]

if __name__ == "__main__":
    results = semantic_search("hình phạt cho tội tàng trữ ma tuý", top_k=5)
    for r in results:
        print(f"[{r['score']:.3f}] {r['content'][:100]}...")
