"""单独重跑汇总英文手册的某个 sub_index 段,合并进现有 JSON

用法:
  python scripts/rerun_english_part.py 3
  python scripts/rerun_english_part.py 3 8 12  # 多个 sub_index
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src.knowledge_base import parse_manual, replace_pic_with_placeholder
from scripts.process_english_manual import chunk_one_part, MANUAL_FILENAME


def main():
    if len(sys.argv) < 2:
        print("用法: python scripts/rerun_english_part.py <sub_index> [<sub_index> ...]")
        sys.exit(1)

    target_indices = [int(x) for x in sys.argv[1:]]
    print(f"重跑 sub_index: {target_indices}")

    fp = os.path.join(config.KB_DIR, MANUAL_FILENAME)
    cache_path = os.path.join(config.DATA_DIR, "chunks_llm", MANUAL_FILENAME.replace(".txt", ".json"))

    # 读现有缓存
    if os.path.exists(cache_path):
        with open(cache_path, "r", encoding="utf-8") as f:
            cached = json.load(f)
        existing_parts = {p["sub_index"]: p for p in cached["parts"]}
        print(f"现有缓存: {len(existing_parts)} 段")
    else:
        cached = {"manual_name": "汇总英文手册", "parts": []}
        existing_parts = {}

    # 重跑指定段
    parsed = parse_manual(fp)
    t_start = time.time()
    new_results = {}
    for idx in target_indices:
        text, image_refs = parsed[idx]
        text_with_pics = replace_pic_with_placeholder(text, image_refs)
        print(f"\n处理段 {idx}: {len(text_with_pics)} 字, {len(image_refs)} 图")
        t0 = time.time()
        try:
            r = chunk_one_part(text_with_pics, "汇总英文手册", idx)
            m = r["metrics"]
            flag = "✅" if not m.get("unmet_threshold") else "⚠️"
            print(f"  {flag} {len(r['sections'])} 节, "
                  f"覆盖 {m['char_coverage']*100:.2f}%, "
                  f"PIC {m['n_pics_new']}/{m['n_pics_orig']}, "
                  f"粗切 {m['n_coarse_chunks']} 块, "
                  f"耗时 {time.time()-t0:.0f}s, "
                  f"¥{r['usage']['cost_yuan']:.4f}")
            new_results[idx] = r
        except Exception as e:
            print(f"  ❌ {e}")

    # 合并(替换原有)
    for idx, r in new_results.items():
        existing_parts[idx] = r

    parts_out = sorted(existing_parts.values(), key=lambda p: p["sub_index"])
    cached_new = {
        "manual_name": "汇总英文手册",
        "n_parts": len(parts_out),
        "parts": parts_out,
        "total_sections": sum(len(p["sections"]) for p in parts_out),
        "total_pics": sum(p["metrics"]["n_pics_orig"] for p in parts_out),
        "total_cost_yuan": sum(p["usage"]["cost_yuan"] for p in parts_out),
    }

    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(cached_new, f, ensure_ascii=False, indent=2)

    print(f"\n总耗时: {time.time() - t_start:.0f}s")
    print(f"缓存已更新: 现有 {len(parts_out)} 段")


if __name__ == "__main__":
    main()
