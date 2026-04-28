"""重跑质量不达标的手册（摩托艇 + 水泵），用 chunk_with_retry 加强约束 + 重试

使用方式:
  python scripts/rerun_quality_failed.py 摩托艇手册.txt 水泵手册.txt

不传文件名时默认重跑这两份。
"""
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src.knowledge_base import parse_manual, replace_pic_with_placeholder
from src.llm_chunker import chunk_with_retry, validate_sections


CACHE_DIR = os.path.join(config.DATA_DIR, "chunks_llm")
DEFAULT_TARGETS = ["摩托艇手册.txt", "水泵手册.txt"]


def process_one(filename: str) -> dict:
    """重跑一份手册（覆盖原 cache）"""
    filepath = os.path.join(config.KB_DIR, filename)
    cache_path = os.path.join(CACHE_DIR, filename.replace(".txt", ".json"))

    # 读旧缓存做对比
    old_metrics = None
    if os.path.exists(cache_path):
        with open(cache_path, "r", encoding="utf-8") as f:
            old = json.load(f)
        old_metrics = old["parts"][0]["metrics"]

    parsed = parse_manual(filepath)
    if not parsed:
        return {"filename": filename, "error": "parse 为空"}

    t0 = time.time()
    parts_out = []
    total_cost = 0.0
    for idx, (text, image_refs) in enumerate(parsed):
        text_with_pics = replace_pic_with_placeholder(text, image_refs)
        sections, usage, metrics = chunk_with_retry(
            text_with_pics,
            filename.replace(".txt", ""),
            max_retries=3,
            threshold_coverage=0.99,
        )
        parts_out.append({
            "sub_index": idx,
            "sections": sections,
            "metrics": metrics,
            "usage": usage,
        })
        total_cost += usage["cost_yuan"]

    cached = {
        "manual_name": filename.replace(".txt", ""),
        "n_parts": len(parts_out),
        "parts": parts_out,
        "total_sections": sum(len(p["sections"]) for p in parts_out),
        "total_pics": sum(p["metrics"]["n_pics_orig"] for p in parts_out),
        "total_cost_yuan": total_cost,
    }

    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(cached, f, ensure_ascii=False, indent=2)

    return {
        "filename": filename,
        "elapsed_s": time.time() - t0,
        "cost_yuan": total_cost,
        "old_metrics": old_metrics,
        "new_metrics": parts_out[0]["metrics"] if parts_out else {},
        "n_sections": cached["total_sections"],
        "n_pics": cached["total_pics"],
    }


def main():
    targets = sys.argv[1:] if len(sys.argv) > 1 else DEFAULT_TARGETS
    print(f"重跑 {len(targets)} 份: {targets}")
    print()

    t_start = time.time()
    with ThreadPoolExecutor(max_workers=len(targets)) as pool:
        futures = {pool.submit(process_one, t): t for t in targets}
        results = []
        for fut in as_completed(futures):
            r = fut.result()
            results.append(r)
            if "error" in r:
                print(f"❌ {r['filename']}: {r['error']}")
                continue

            old, new = r.get("old_metrics") or {}, r["new_metrics"]
            print(f"✅ {r['filename']}")
            print(f"   耗时 {r['elapsed_s']:.0f}s  花费 ¥{r['cost_yuan']:.4f}")
            print(f"   尝试次数: {new.get('attempts', '?')}  |  "
                  f"达标: {'是' if not new.get('unmet_threshold') else '否(取最佳)'}")
            if old:
                print(f"   字符覆盖: {old['char_coverage'] * 100:.2f}% → "
                      f"{new['char_coverage'] * 100:.2f}%")
                print(f"   PIC: {old['n_pics_new']}/{old['n_pics_orig']} → "
                      f"{new['n_pics_new']}/{new['n_pics_orig']}")
            else:
                print(f"   字符覆盖: {new['char_coverage'] * 100:.2f}%")
                print(f"   PIC: {new['n_pics_new']}/{new['n_pics_orig']}")
            print(f"   切节: {r['n_sections']}, 总 PIC: {r['n_pics']}")
            if new.get("issues"):
                print(f"   issues: {new['issues']}")
            print()

    print(f"总耗时: {time.time() - t_start:.0f}s")
    print(f"总花费: ¥{sum(r.get('cost_yuan', 0) for r in results):.4f}")


if __name__ == "__main__":
    main()
