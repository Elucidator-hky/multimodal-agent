"""处理汇总英文手册:20 段并发,大段先按行首 # 粗切,缓存到 data/chunks_llm/汇总英文手册.json

并发策略:
  - 段级:20 段并发跑(8 路)
  - 粗切块级:大段内部 ~10k 字粗切块也并发(4 路)
  - 校验+重试:每个 LLM 调用都过 chunk_with_retry,字符覆盖 < 99% 自动重试
"""
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src.knowledge_base import parse_manual, replace_pic_with_placeholder
from src.llm_chunker import chunk_with_retry, validate_sections


MAX_INPUT_CHARS = 20000          # 单次 LLM 调用最大输入字符
COARSE_CHUNK_CHARS = 7000        # 粗切单块字数(留 output buffer 防截断)
PART_LEVEL_WORKERS = 20          # 段级并发(20 段全并发)
COARSE_LEVEL_WORKERS = 20        # 粗切块级并发(大段内 ~16-19 块全并发)
MANUAL_FILENAME = "汇总英文手册.txt"


def coarse_split_by_sharp(text: str, max_chunk_chars: int = COARSE_CHUNK_CHARS) -> list[str]:
    """按行首 `# ` 切,然后累积到 max_chunk_chars 一块。

    不破坏 [[PIC:xxx]] 占位符(都是行内字符,跨不到行首)。
    fallback: 如果整段没有行首 `#`,退化为按 max_chunk_chars 直接切。
    """
    parts = re.split(r'(?m)(?=^# )', text)
    parts = [p for p in parts if p]

    if not parts or len(parts) == 1:
        # 没行首 # 或全在一段,只好按字符硬切(尽量不切断 [[PIC:xxx]])
        chunks = []
        i = 0
        while i < len(text):
            end = min(i + max_chunk_chars, len(text))
            # 避开切断 [[PIC:xxx]]
            if end < len(text):
                # 后退到最近的 [[ 之前
                next_pic = text.rfind('[[PIC:', i, end)
                if next_pic > 0 and end - next_pic < 100:
                    end = next_pic
            chunks.append(text[i:end])
            i = end
        return chunks

    chunks = []
    buffer = []
    buffer_size = 0
    for p in parts:
        if buffer_size + len(p) > max_chunk_chars and buffer:
            chunks.append("".join(buffer))
            buffer = [p]
            buffer_size = len(p)
        else:
            buffer.append(p)
            buffer_size += len(p)
    if buffer:
        chunks.append("".join(buffer))
    return chunks


def chunk_one_part(text: str, manual_name: str, sub_index: int) -> dict:
    """处理一个 JSON part(段),返回 {sub_index, sections, metrics, usage, n_coarse_chunks}"""
    if len(text) <= MAX_INPUT_CHARS:
        sections, usage, metrics = chunk_with_retry(text, f"{manual_name} part {sub_index}")
        metrics["n_coarse_chunks"] = 1
        return {
            "sub_index": sub_index,
            "sections": sections,
            "metrics": metrics,
            "usage": usage,
        }

    # 大段:粗切
    coarse_chunks = coarse_split_by_sharp(text)

    # 并发处理每个粗切块,保留顺序
    with ThreadPoolExecutor(max_workers=COARSE_LEVEL_WORKERS) as pool:
        futures = [
            pool.submit(chunk_with_retry, c, f"{manual_name} part {sub_index}.{i}")
            for i, c in enumerate(coarse_chunks)
        ]
        results = [f.result() for f in futures]

    # 按粗切顺序合并 sections(因为 futures list 顺序就是 coarse_chunks 顺序)
    all_sections = []
    total_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "cost_yuan": 0.0}
    sub_attempts = []
    for sections, usage, metrics in results:
        all_sections.extend(sections)
        for k in ("prompt_tokens", "completion_tokens", "total_tokens"):
            total_usage[k] += usage[k]
        total_usage["cost_yuan"] += usage["cost_yuan"]
        sub_attempts.append(metrics["attempts"])

    # 整段二次校验(把所有 chunks sections 合一起跟原段比)
    final_metrics = validate_sections(all_sections, text)
    final_metrics["attempts"] = max(sub_attempts)
    final_metrics["n_coarse_chunks"] = len(coarse_chunks)
    final_metrics["unmet_threshold"] = (
        final_metrics["char_coverage"] < 0.99 or not final_metrics["pic_count_match"]
    )
    return {
        "sub_index": sub_index,
        "sections": all_sections,
        "metrics": final_metrics,
        "usage": total_usage,
    }


def main():
    fp = os.path.join(config.KB_DIR, MANUAL_FILENAME)
    cache_path = os.path.join(config.DATA_DIR, "chunks_llm", MANUAL_FILENAME.replace(".txt", ".json"))

    parsed = parse_manual(fp)
    print(f"{MANUAL_FILENAME}: {len(parsed)} 段")

    parts_input = []
    for idx, (text, image_refs) in enumerate(parsed):
        text_with_pics = replace_pic_with_placeholder(text, image_refs)
        parts_input.append((idx, text_with_pics))
        flag = " (粗切)" if len(text_with_pics) > MAX_INPUT_CHARS else ""
        print(f"  段 {idx:2d}: {len(text_with_pics):>7} 字, {len(image_refs):>3} 图{flag}")

    print()
    print(f"段级并发 {PART_LEVEL_WORKERS} 路, 粗切块级并发 {COARSE_LEVEL_WORKERS} 路")
    print()

    t_start = time.time()
    results: list = [None] * len(parts_input)

    with ThreadPoolExecutor(max_workers=PART_LEVEL_WORKERS) as pool:
        futures = {
            pool.submit(chunk_one_part, text, "汇总英文手册", idx): idx
            for idx, text in parts_input
        }
        done = 0
        for fut in as_completed(futures):
            idx = futures[fut]
            try:
                r = fut.result()
                done += 1
                results[idx] = r
                m = r["metrics"]
                flag = "✅" if not m.get("unmet_threshold") else "⚠️"
                print(f"[{done}/{len(parts_input)}] {flag} 段 {idx:2d}: "
                      f"{len(r['sections']):3d} 节, "
                      f"覆盖 {m['char_coverage']*100:.2f}%, "
                      f"PIC {m['n_pics_new']}/{m['n_pics_orig']}, "
                      f"粗切 {m['n_coarse_chunks']} 块, "
                      f"¥{r['usage']['cost_yuan']:.4f}")
            except Exception as e:
                done += 1
                print(f"[{done}/{len(parts_input)}] ❌ 段 {idx}: {e}")

    # 写缓存(按 sub_index 排序)
    parts_out = [r for r in results if r is not None]
    parts_out.sort(key=lambda r: r["sub_index"])

    cached = {
        "manual_name": "汇总英文手册",
        "n_parts": len(parts_out),
        "parts": parts_out,
        "total_sections": sum(len(p["sections"]) for p in parts_out),
        "total_pics": sum(p["metrics"]["n_pics_orig"] for p in parts_out),
        "total_cost_yuan": sum(p["usage"]["cost_yuan"] for p in parts_out),
    }

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(cached, f, ensure_ascii=False, indent=2)

    print()
    print("=" * 60)
    print(f"总耗时: {time.time() - t_start:.0f}s")
    print(f"总花费: ¥{cached['total_cost_yuan']:.4f}")
    print(f"总切节: {cached['total_sections']}, 总 PIC: {cached['total_pics']}")
    print(f"缓存: {cache_path}")
    fail = [p for p in parts_out if p["metrics"].get("unmet_threshold")]
    if fail:
        print(f"⚠️  {len(fail)} 段未达阈值(已保留最佳结果),sub_index: {[p['sub_index'] for p in fail]}")


if __name__ == "__main__":
    main()
