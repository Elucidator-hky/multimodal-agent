"""手动补回 sections 中缺失的 [[PIC:xxx]] 占位符。

策略 v2:每张缺失 PIC,用它在原文的上下文(前 25 字去空白)在 sections 里找,
        精准插入到对应位置(在上下文之后的位置)。

用法:
  python scripts/patch_missing_pics.py 功能键盘手册.txt 吹风机手册.txt 洗碗机手册.txt
"""
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src.knowledge_base import parse_manual, replace_pic_with_placeholder
from src.llm_chunker import validate_sections


CACHE_DIR = os.path.join(config.DATA_DIR, "chunks_llm")
CTX_LEN = 25  # 用 PIC 前多少字符(去空白)做上下文匹配


def _clean_with_index(text: str):
    """返回 (去空白后的字符串, 去空白索引→原索引 的映射)"""
    chars, idx = [], []
    for i, c in enumerate(text):
        if not c.isspace():
            chars.append(c)
            idx.append(i)
    return "".join(chars), idx


def patch_one_manual(filename: str) -> dict:
    filepath = os.path.join(config.KB_DIR, filename)
    cache_path = os.path.join(CACHE_DIR, filename.replace(".txt", ".json"))

    if not os.path.exists(cache_path):
        return {"filename": filename, "error": "cache 不存在"}

    with open(cache_path, "r", encoding="utf-8") as f:
        cached = json.load(f)

    parsed = parse_manual(filepath)

    pics_inserted = 0
    pics_failed = []

    for part in cached["parts"]:
        text, image_refs = parsed[part["sub_index"]]
        original = replace_pic_with_placeholder(text, image_refs)

        # 找原文所有 PIC 位置
        pic_matches = list(re.finditer(r'\[\[PIC:[^\]]+\]\]', original))

        # 当前 sections 中已有的 PIC(按 (pic_name, occurrence_count) 计数)
        existing_pics = {}
        for s in part["sections"]:
            for p in re.findall(r'\[\[PIC:[^\]]+\]\]', s["text"]):
                existing_pics[p] = existing_pics.get(p, 0) + 1

        # 遍历原文 PIC,找缺失的
        for m in pic_matches:
            pic = m.group(0)
            if existing_pics.get(pic, 0) > 0:
                existing_pics[pic] -= 1
                continue

            # 缺失!取上下文(前 50 字符,去空白后取末 CTX_LEN 字)
            ctx_window = original[max(0, m.start() - 80):m.start()]
            ctx_clean = re.sub(r'\s+', '', ctx_window)[-CTX_LEN:]
            if len(ctx_clean) < 12:
                pics_failed.append((pic, "context too short"))
                continue

            # 在 sections 中找包含 ctx_clean 的(去空白匹配)
            inserted = False
            for s in part["sections"]:
                s_clean, s_idx_map = _clean_with_index(s["text"])
                pos = s_clean.find(ctx_clean)
                if pos < 0:
                    continue

                # 找到了。在 s["text"] 里 ctx_clean 末尾对应的原始位置
                end_clean_idx = pos + len(ctx_clean) - 1
                if end_clean_idx >= len(s_idx_map):
                    continue
                insert_pos = s_idx_map[end_clean_idx] + 1
                s["text"] = s["text"][:insert_pos] + pic + s["text"][insert_pos:]
                pics_inserted += 1
                inserted = True
                break

            if not inserted:
                pics_failed.append((pic, "ctx not found in any section"))

        # 重新校验
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
        "pics_inserted": pics_inserted,
        "pics_failed": pics_failed,
        "after_metrics": cached["parts"][0]["metrics"],
    }


def main():
    if len(sys.argv) > 1:
        targets = sys.argv[1:]
    else:
        targets = ["功能键盘手册.txt", "吹风机手册.txt", "洗碗机手册.txt"]

    print(f"补 PIC: {targets}\n")
    for filename in targets:
        r = patch_one_manual(filename)
        if "error" in r:
            print(f"❌ {r['filename']}: {r['error']}")
            continue
        m = r["after_metrics"]
        flag = "✅" if m["pic_count_match"] else "⚠️"
        print(f"{flag} {r['filename']}")
        print(f"   补回 PIC: +{r['pics_inserted']}")
        if r["pics_failed"]:
            print(f"   未补回: {len(r['pics_failed'])} 张")
            for pic, reason in r["pics_failed"][:3]:
                print(f"     - {pic}: {reason}")
        print(f"   补后: 字符覆盖 {m['char_coverage']*100:.2f}%, "
              f"PIC {m['n_pics_new']}/{m['n_pics_orig']}")
        print()


if __name__ == "__main__":
    main()
