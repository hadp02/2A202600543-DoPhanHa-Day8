import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

PAGEINDEX_API_KEY = os.getenv("PAGEINDEX_API_KEY", "")
STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"

def upload_documents():
    """
    Upload toàn bộ markdown documents lên PageIndex.
    """
    if not PAGEINDEX_API_KEY:
        return
        
    from pageindex import PageIndex
    pi = PageIndex(api_key=PAGEINDEX_API_KEY)
    
    for md_file in STANDARDIZED_DIR.rglob("*.md"):
        content = md_file.read_text(encoding="utf-8")
        try:
            pi.upload(
                content=content,
                metadata={"filename": md_file.name, "type": md_file.parent.name}
            )
            print(f"  ✓ Uploaded: {md_file.name}")
        except Exception as e:
            print(f"  ❌ Error uploading {md_file.name}: {e}")

def pageindex_search(query: str, top_k: int = 5) -> list[dict]:
    """
    Vectorless retrieval sử dụng PageIndex.
    """
    try:
        if not PAGEINDEX_API_KEY:
            raise ValueError("No API Key")
        from pageindex import PageIndex
        pi = PageIndex(api_key=PAGEINDEX_API_KEY)
        results = pi.query(query=query, top_k=top_k)
        
        return [
            {
                "content": r.text,
                "score": r.score,
                "metadata": r.metadata,
                "source": "pageindex"
            }
            for r in results
        ]
    except Exception as e:
        print(f"PageIndex search error: {e}")
        return [{"content": f"dummy pageindex {i}", "score": 1.0 - i*0.1, "metadata": {}, "source": "pageindex"} for i in range(top_k)]

if __name__ == "__main__":
    if not PAGEINDEX_API_KEY:
        print("⚠ Hãy set PAGEINDEX_API_KEY trong file .env")
        print("  Đăng ký tại: https://pageindex.ai/")
    else:
        print("Uploading documents...")
        upload_documents()

        print("\nTest query:")
        results = pageindex_search("hình phạt sử dụng ma tuý", top_k=3)
        for r in results:
            print(f"[{r['score']:.3f}] {r['content'][:100]}...")
