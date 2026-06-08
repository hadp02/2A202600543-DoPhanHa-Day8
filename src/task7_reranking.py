import os
import requests
from typing import Optional

def rerank_cross_encoder(query: str, candidates: list[dict], top_k: int = 5) -> list[dict]:
    jina_api_key = os.getenv("JINA_API_KEY")
    if not jina_api_key:
        # Fallback without reranking if no API key
        for i, c in enumerate(candidates):
            if "score" not in c:
                c["score"] = 1.0 - i*0.1
        candidates_sorted = sorted(candidates, key=lambda x: x.get("score", 0), reverse=True)
        return candidates_sorted[:top_k]
    
    try:
        response = requests.post(
            "https://api.jina.ai/v1/rerank",
            headers={"Authorization": f"Bearer {jina_api_key}"},
            json={
                "model": "jina-reranker-v2-base-multilingual",
                "query": query,
                "documents": [c["content"] for c in candidates],
                "top_n": top_k
            },
            timeout=30
        )
        response.raise_for_status()
        reranked = response.json().get("results", [])
        return [
            {**candidates[r["index"]], "score": r["relevance_score"]}
            for r in reranked
        ]
    except Exception as e:
        print(f"Cross encoder rerank error: {e}")
        candidates_sorted = sorted(candidates, key=lambda x: x.get("score", 0), reverse=True)
        return candidates_sorted[:top_k]

def rerank_mmr(query_embedding: list[float], candidates: list[dict], top_k: int = 5, lambda_param: float = 0.7) -> list[dict]:
    # Placeholder implementation if MMR is chosen
    return candidates[:top_k]

def rerank_rrf(ranked_lists: list[list[dict]], top_k: int = 5, k: int = 60) -> list[dict]:
    rrf_scores = {}
    content_map = {}
    
    for ranked_list in ranked_lists:
        for rank, item in enumerate(ranked_list, 1):
            key = item["content"]
            rrf_scores[key] = rrf_scores.get(key, 0) + 1 / (k + rank)
            if key not in content_map:
                content_map[key] = item.copy()
                
    sorted_items = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    
    results = []
    for content, score in sorted_items[:top_k]:
        item = content_map[content]
        item["score"] = score
        results.append(item)
        
    return results

def rerank(
    query: str,
    candidates: list[dict],
    top_k: int = 5,
    method: str = "cross_encoder",
) -> list[dict]:
    if method == "cross_encoder":
        return rerank_cross_encoder(query, candidates, top_k)
    elif method == "mmr":
        return candidates[:top_k]
    elif method == "rrf":
        return candidates[:top_k]
    else:
        raise ValueError(f"Unknown rerank method: {method}")

if __name__ == "__main__":
    dummy_candidates = [
        {"content": "Điều 248: Tội tàng trữ trái phép chất ma tuý", "score": 0.8, "metadata": {}},
        {"content": "Nghệ sĩ X bị bắt vì sử dụng ma tuý", "score": 0.7, "metadata": {}},
        {"content": "Hình phạt tù từ 2-7 năm cho tội tàng trữ", "score": 0.6, "metadata": {}},
    ]
    results = rerank("hình phạt tàng trữ ma tuý", dummy_candidates, top_k=2)
    for r in results:
        print(f"[{r.get('score', 0):.3f}] {r['content']}")
