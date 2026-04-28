"""段 3(船 Boat 166k 字)LLM 反复失败,改用纯结构切:按行首 # 切成多个 sections,每节当一个 chunk,不调 LLM。

PIC 100% 保留,字符 100% 还原。每节 = 一个 # 章节(平均 ~2000 字)。

合并进 data/chunks_llm/汇总英文手册.json。
"""
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src.knowledge_base import parse_manual, replace_pic_with_placeholder
from src.llm_chunker import validate_sections


SUB_INDEX = 3
MANUAL_FILENAME = "汇总英文手册.txt"


def split_by_sharp_only(text: str) -> list[dict]:
    """按行首 # 切,每节直接当一个 section,不调 LLM。"""
    chunks = re.split(r'(?m)(?=^# )', text)
    sections = []
    for c in chunks:
        if not c.strip():
            continue
        # 第一行 # 后面的文字作为标题
        first_line = c.split("\n", 1)[0].lstrip("# ").strip()
        title = first_line[:80] if first_line else f"Section {len(sections) + 1}"
        sections.append({"title": title, "text": c})
    return sections


def main():
    fp = os.path.join(config.KB_DIR, MANUAL_FILENAME)
    cache_path = os.path.join(config.DATA_DIR, "chunks_llm", MANUAL_FILENAME.replace(".txt", ".json"))

    parsed = parse_manual(fp)
    text, image_refs = parsed[SUB_INDEX]
    text_with_pics = replace_pic_with_placeholder(text, image_refs)

    print(f"段 {SUB_INDEX}: {len(text_with_pics)} 字, {len(image_refs)} 图")

    sections = split_by_sharp_only(text_with_pics)
    print(f"按 # 切出 {len(sections)} 节")

    metrics = validate_sections(sections, text_with_pics)
    print(f"字符覆盖: {metrics['char_coverage'] * 100:.2f}%")
    print(f"PIC: {metrics['n_pics_new']}/{metrics['n_pics_orig']}")

    if not metrics["pic_count_match"] or metrics["char_coverage"] < 0.99:
        print("⚠️ 还有问题,但继续保存")

    # 读现有 cache
    with open(cache_path, "r", encoding="utf-8") as f:
        cached = json.load(f)
    existing_parts = {p["sub_index"]: p for p in cached["parts"]}

    # 加段 3
    metrics["attempts"] = 0  # 标记没用 LLM
    metrics["method"] = "split_by_sharp_only"
    existing_parts[SUB_INDEX] = {
        "sub_index": SUB_INDEX,
        "sections": sections,
        "metrics": metrics,
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "cost_yuan": 0.0},
    }

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

    print(f"\n✅ 已合并到 cache: {cached_new['n_parts']} 段, "
          f"{cached_new['total_sections']} 节, {cached_new['total_pics']} 图")


if __name__ == "__main__":
    main()
