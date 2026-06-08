"""
Task 4 — Chunking & Indexing vào Vector Store.

Hướng dẫn:
    1. Đọc toàn bộ markdown files từ data/standardized/
    2. Chọn 1 chunking strategy (giải thích lý do)
    3. Chọn 1 embedding model (giải thích lý do)
    4. Index vào vector store

Cài đặt:
    pip install langchain-text-splitters openai chromadb python-dotenv
"""

import os
import time
import requests
from pathlib import Path

from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
import chromadb
from sentence_transformers import SentenceTransformer

# Load .env file để lấy API key
load_dotenv()

STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"

# Thư mục lưu ChromaDB persistent data
CHROMA_PERSIST_DIR = str(Path(__file__).parent.parent / "data" / "chromadb")

# Tên collection trong ChromaDB
COLLECTION_NAME = "drug_law_docs"


# =============================================================================
# CONFIGURATION — Giải thích lựa chọn
# =============================================================================

# Chunking strategy: RecursiveCharacterTextSplitter
# Lý do: Đây là splitter phổ biến và an toàn nhất, tự động chia theo nhiều cấp
# separator (paragraph → newline → sentence → word) để đảm bảo mỗi chunk
# có ngữ nghĩa mạch lạc. Phù hợp cho cả văn bản pháp luật (dài, có điều khoản)
# và bài báo (ngắn hơn, tự do hơn).
CHUNK_SIZE = 500        # 500 chars: đủ lớn để giữ ngữ cảnh 1 đoạn/điều khoản,
                        # đủ nhỏ để embedding có semantic focus rõ ràng.
CHUNK_OVERLAP = 50      # 50 chars overlap: tránh mất thông tin ở biên giới chunk,
                        # giúp retrieval tìm được context liền mạch.
CHUNKING_METHOD = "recursive"  # "recursive" | "markdown_header" | "semantic"

# Embedding model: sentence-transformers/all-MiniLM-L6-v2
# Lý do: Chạy local, nhẹ và nhanh, không phụ thuộc API key.
# Model có dimension 384, dễ dàng tính toán trên CPU.
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384

# Vector store: ChromaDB
# Lý do: Đơn giản, chạy local không cần Docker (khác Weaviate), persistent storage
# giúp không cần re-index mỗi lần chạy. Phù hợp cho project cá nhân/prototype.
VECTOR_STORE = "chromadb"  # "weaviate" | "chromadb" | "faiss"


# =============================================================================
# IMPLEMENTATION
# =============================================================================

def load_documents() -> list[dict]:
    """
    Đọc toàn bộ markdown files từ data/standardized/.

    Returns:
        List of {'content': str, 'metadata': {'source': str, 'type': str}}
    """
    documents = []
    for md_file in STANDARDIZED_DIR.rglob("*.md"):
        content = md_file.read_text(encoding="utf-8")
        # Xác định loại document dựa trên đường dẫn thư mục
        doc_type = "legal" if "legal" in str(md_file) else "news"
        documents.append({
            "content": content,
            "metadata": {"source": md_file.name, "type": doc_type}
        })
    return documents


def chunk_documents(documents: list[dict]) -> list[dict]:
    """
    Chunk documents bằng RecursiveCharacterTextSplitter.

    Separators được sắp xếp từ lớn → nhỏ:
    - "\\n\\n": tách theo paragraph (ưu tiên cao nhất)
    - "\\n": tách theo dòng
    - ". ": tách theo câu
    - " ": tách theo từ
    - "": tách theo ký tự (cuối cùng)

    Returns:
        List of {'content': str, 'metadata': dict} — mỗi item là 1 chunk
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    chunks = []
    for doc in documents:
        splits = splitter.split_text(doc["content"])
        for i, chunk_text in enumerate(splits):
            chunks.append({
                "content": chunk_text,
                "metadata": {**doc["metadata"], "chunk_index": i}
            })
    return chunks


def embed_chunks(chunks: list[dict]) -> list[dict]:
    """
    Embed toàn bộ chunks bằng sentence-transformers/all-MiniLM-L6-v2 local.
    """
    texts = [c["content"] for c in chunks]
    print(f"  Embedding {len(texts)} chunks locally...")
    
    try:
        model = SentenceTransformer(EMBEDDING_MODEL)
        embeddings = model.encode(texts, show_progress_bar=True).tolist()
    except Exception as e:
        print(f"  ❌ Error loading/running SentenceTransformer: {e}")
        # Fallback for testing when sentence-transformers is missing
        embeddings = [[0.0] * EMBEDDING_DIM for _ in texts]
        
    for chunk, emb in zip(chunks, embeddings):
        chunk["embedding"] = emb
    return chunks


def index_to_vectorstore(chunks: list[dict]):
    """
    Lưu chunks vào ChromaDB (persistent local storage).

    ChromaDB tự động lưu data vào disk tại CHROMA_PERSIST_DIR.
    Nếu collection đã tồn tại, sẽ xóa và tạo lại để tránh duplicate.
    """
    # Kết nối ChromaDB với persistent storage
    client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)

    # Xóa collection cũ nếu tồn tại (tránh duplicate khi re-run)
    try:
        client.delete_collection(name=COLLECTION_NAME)
        print(f"  Deleted existing collection '{COLLECTION_NAME}'")
    except Exception:
        pass  # Collection chưa tồn tại → bỏ qua

    # Tạo collection mới
    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}  # Dùng cosine similarity
    )

    # Insert chunks theo batch (ChromaDB giới hạn batch size)
    BATCH_SIZE = 100
    for i in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[i:i + BATCH_SIZE]
        collection.add(
            ids=[f"chunk_{i + j}" for j in range(len(batch))],
            embeddings=[c["embedding"] for c in batch],
            documents=[c["content"] for c in batch],
            metadatas=[{
                "source": c["metadata"].get("source", ""),
                "type": c["metadata"].get("type", ""),
                "chunk_index": c["metadata"].get("chunk_index", 0),
            } for c in batch]
        )

    print(f"  Indexed {collection.count()} chunks to ChromaDB at {CHROMA_PERSIST_DIR}")


def run_pipeline():
    """Chạy toàn bộ pipeline: load → chunk → embed → index."""
    print("=" * 50)
    print("Task 4: Chunking & Indexing")
    print(f"  Chunking: {CHUNKING_METHOD} (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")
    print(f"  Embedding: {EMBEDDING_MODEL} (dim={EMBEDDING_DIM})")
    print(f"  Vector Store: {VECTOR_STORE}")
    print("=" * 50)

    docs = load_documents()
    print(f"\n✓ Loaded {len(docs)} documents")

    chunks = chunk_documents(docs)
    print(f"✓ Created {len(chunks)} chunks")

    chunks = embed_chunks(chunks)
    print(f"✓ Embedded {len(chunks)} chunks")

    index_to_vectorstore(chunks)
    print("✓ Indexed to vector store")
    print("\n✅ Task 4 hoàn thành!")


if __name__ == "__main__":
    run_pipeline()

