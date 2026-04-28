"""用 difflib 找出 sections 没覆盖的原文片段,补进去当新 section。

适用于 LLM 漏抄了一段(包括其中的 PIC)的情况。
"""
import difflib
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src.knowledge_base import parse_manual, replace_pic_with_placeholder
from src.llm_chunker import validate_sections


CACHE_DIR = os.path.join(config.DATA_DIR, "chunks_llm")
MIN_GAP_SIZE = 30  # 间隙至少 30 字才补(避开空白噪声)


def find_gaps(original: str, sections_text: str) -> list[tuple[int, int, str]]:
    """用 SequenceMatcher 找原文中 sections 没覆盖的片段,返回 [(start, end, text), ...]"""
    matcher = difflib.SequenceMatcher(a=original, b=sections_text, autojunk=False)
    blocks = matcher.get_matching_blocks()

    gaps = []
    last_end_a = 0
    for block in blocks:
        if block.a > last_end_a:
            gap_start = last_end_a
            gap_end = block.a
            gap_text = original[gap_start:gap_end]
            if len(gap_text.strip()) >= MIN_GAP_SIZE or '[[PIC:' in gap_text:
                gaps.append((gap_start, gap_end, gap_text))
        last_end_a = block.a + block.size

    if last_end_a < len(original):
        tail = original[last_end_a:]
        if len(tail.strip()) >= MIN_GAP_SIZE or '[[PIC:' in tail:
            gaps.append((last_end_a, len(original), tail))

    return gaps


def patch_one_manual(filename: str) -> dict:
    filepath = os.path.join(config.KB_DIR, filename)
    cache_path = os.path.join(CACHE_DIR, filename.replace(".txt", ".json"))

    with open(cache_path, "r", encoding="utf-8") as f:
        cached = json.load(f)

    parsed = parse_manual(filepath)

    total_gaps_inserted = 0
    total_pics_recovered = 0
    total_chars_recovered = 0

    for part in cached["parts"]:
        text, image_refs = parsed[part["sub_index"]]
        original = replace_pic_with_placeholder(text, image_refs)

        sections_text = "".join(s["text"] for s in part["sections"])
        gaps = find_gaps(original, sections_text)

        for start, end, gap_text in gaps:
            n_pics_in_gap = len(re.findall(r'\[\[PIC:[^\]]+\]\]', gap_text))
            # 给 gap 一个标题(取前 30 个非空白字符)
            preview = re.sub(r'\s+', '', gap_text)[:30]
            title = f"[补] {preview}" if preview else "[补]"
            part["sections"].append({
                "title": title,
                "text": gap_text,
            })
            total_gaps_inserted += 1
            total_pics_recovered += n_pics_in_gap
            total_chars_recovered += len(gap_text)

        new_metrics = validate_sections(part["sections"], original)
        new_metrics["attempts"] = part["metrics"].get("attempts", 1)
        new_metrics["patched"] = True
        part["metrics"] = new_metrics

    cached["total_sections"] = sum(len(p["sections"]) for p in cached["parts"])
    cached["total_pics"] = sum(p["metrics"]["n_pics_orig"] for p in cached["parts"])

    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(cached, f, ensure_ascii=False, indent=2)

    return {
        "filename": filename,
        "gaps_inserted": total_gaps_inserted,
        "pics_recovered": total_pics_recovered,
        "chars_recovered": total_chars_recovered,
        "after_metrics": cached["parts"][0]["metrics"],
    }


def main():
    if len(sys.argv) > 1:
        targets = sys.argv[1:]
    else:
        targets = ["功能键盘手册.txt", "吹风机手册.txt", "洗碗机手册.txt"]

    print(f"补 gap: {targets}\n")
    for filename in targets:
        r = patch_one_manual(filename)
        m = r["after_metrics"]
        flag = "✅" if m["pic_count_match"] and m["char_coverage"] >= 0.99 else "⚠️"
        print(f"{flag} {r['filename']}")
        print(f"   补 gaps: +{r['gaps_inserted']}  含 PIC: +{r['pics_recovered']}  字符: +{r['chars_recovered']}")
        print(f"   补后: 字符覆盖 {m['char_coverage']*100:.2f}%, PIC {m['n_pics_new']}/{m['n_pics_orig']}")
        print()


if __name__ == "__main__":
    main()
