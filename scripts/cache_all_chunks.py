"""并发跑 22 份手册的 LLM 切块,缓存到 data/chunks_llm/{manual_name}.json

每份手册的 JSON 结构:
  {
    "manual_name": "电钻手册",
    "parts": [
      {
        "sub_index": 0,
        "sections": [{"title": "...", "text": "..."}, ...],
        "metrics": {...},
        "usage": {...}
      },
      ...
    ]
  }

并发用 ThreadPoolExecutor,默认 6 路(qwen API RPM 60 够用)。
"""
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src.llm_chunker import chunk_manual_file

CACHE_DIR = os.path.join(config.DATA_DIR, "chunks_llm")
MAX_WORKERS = 6


def process_one(filename: str) -> dict:
    """处理一份手册,把结果写到磁盘。返回简报。"""
    filepath = os.path.join(config.KB_DIR, filename)
    cache_path = os.path.join(CACHE_DIR, filename.replace(".txt", ".json"))

    # 跳过已经缓存的(支持断点续跑)
    if os.path.exists(cache_path):
        with open(cache_path, "r", encoding="utf-8") as f:
            cached = json.load(f)
        return {
            "filename": filename,
            "skipped": True,
            "n_sections": cached["total_sections"],
            "n_pics": cached["total_pics"],
            "cost_yuan": 0.0,
        }

    t0 = time.time()
    try:
        result = chunk_manual_file(filepath)
    except Exception as e:
        return {"filename": filename, "error": str(e)}

    # 写缓存(去掉 original_text,节省空间;校验完用不到)
    cached = {
        "manual_name": result["manual_name"],
        "n_parts": result["n_parts"],
        "parts": [
            {
                "sub_index": p["sub_index"],
                "sections": p["sections"],
                "metrics": p["metrics"],
                "usage": p["usage"],
            }
            for p in result["parts"]
        ],
        "total_sections": result["total_sections"],
        "total_pics": result["total_pics"],
        "total_cost_yuan": result["total_cost_yuan"],
    }
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(cached, f, ensure_ascii=False, indent=2)

    return {
        "filename": filename,
        "skipped": False,
        "n_sections": result["total_sections"],
        "n_pics": result["total_pics"],
        "cost_yuan": result["total_cost_yuan"],
        "elapsed_s": time.time() - t0,
        "issues": [issue for p in result["parts"] for issue in p["metrics"]["issues"]],
    }


def main():
    os.makedirs(CACHE_DIR, exist_ok=True)

    manuals = sorted(f for f in os.listdir(config.KB_DIR) if f.endswith(".txt"))
    print(f"待处理 {len(manuals)} 份手册, 并发 {MAX_WORKERS} 路")
    print(f"缓存目录: {CACHE_DIR}")
    print()

    t_start = time.time()
    results = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(process_one, m): m for m in manuals}
        done = 0
        for fut in as_completed(futures):
            r = fut.result()
            done += 1
            if "error" in r:
                print(f"[{done}/{len(manuals)}] ❌ {r['filename']}: {r['error']}")
            else:
                tag = "缓存" if r["skipped"] else f"{r['elapsed_s']:.0f}s ¥{r['cost_yuan']:.4f}"
                issue_str = f" ⚠️ {r.get('issues', [])}" if r.get("issues") else ""
                print(f"[{done}/{len(manuals)}] ✅ {r['filename']:30s} {r['n_sections']:3d} 节 / {r['n_pics']:3d} 图  ({tag}){issue_str}")
            results.append(r)

    total_elapsed = time.time() - t_start
    total_cost = sum(r.get("cost_yuan", 0) for r in results)
    total_sections = sum(r.get("n_sections", 0) for r in results)
    total_pics = sum(r.get("n_pics", 0) for r in results)
    errors = [r for r in results if "error" in r]

    print()
    print("=" * 60)
    print(f"总耗时: {total_elapsed:.1f}s")
    print(f"总花费: ¥{total_cost:.4f}")
    print(f"总切节: {total_sections}")
    print(f"总图片: {total_pics}")
    if errors:
        print(f"❌ 失败: {len(errors)}")
        for e in errors:
            print(f"   {e['filename']}: {e['error']}")
    else:
        print("✅ 全部成功")


if __name__ == "__main__":
    main()
