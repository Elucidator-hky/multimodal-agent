"""LLM 语义切块：用 qwen-plus 把手册切成"语义连贯的小节"

流程：
  原始 .txt → parse_manual → replace_pic_with_placeholder → chunk_with_llm → list[Section]

每个 Section: {"title": "...", "text": "原文连续片段含 [[PIC:xxx]]"}
"""
import json
import os
import re
import sys
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from openai import OpenAI

client = OpenAI(api_key=config.QWEN_API_KEY, base_url=config.QWEN_BASE_URL)

CHUNKER_MODEL = "qwen-plus"

# qwen-plus 单价(元/千 token)
PRICE_INPUT = 0.0008
PRICE_OUTPUT = 0.002


PROMPT_TEMPLATE = """你是文档结构分析助手。下面是一份《{manual_name}》的内容，作者用 # 表示章节标题，但很多 # 嵌在段落中没换行，导致按行切块困难。

任务：把这份手册切成"语义连贯的小节"，每个小节是一个独立可回答的知识单元（比如"充电步骤"、"过热延迟原理"、"DCB101 指示灯含义"等）。

严格要求：
1. 输出 JSON 对象 {{"sections": [{{"title": "...", "text": "..."}}, ...]}}
2. text 必须是**原文连续片段**，不能修改任何字符（包括标点、空格、换行）
3. 切完所有小节的 text 顺序拼接应**完全覆盖**原文（中间允许少量空白被丢弃）
4. 必须**完整保留**所有 [[PIC:xxx.png]] 占位符，不能漏一张
5. 每节目标 200-1500 字，避免过短或过长
6. 标题用 8-15 字一句话概括，反映小节核心内容

原文：
'''
{text}
'''
"""


# ─────── 英文版 prompt(自动检测语言后切换)───────

PROMPT_TEMPLATE_EN = """You are a document structure analyst. Below is content from an English product manual ({manual_name}). The author uses `# ` for chapter headings, but many `# ` markers are inline within paragraphs without line breaks, making line-based splitting hard.

Task: Split this content into semantically coherent sections, where each section is an independently-answerable knowledge unit (e.g., "Charging Procedure", "Engine Start Steps", "Safety Warnings").

STRICT REQUIREMENTS:
1. Output JSON: {{"sections": [{{"title": "...", "text": "..."}}, ...]}}
2. `text` must be a **verbatim contiguous substring of the original** — no character modification (preserve punctuation, spacing, line breaks)
3. Concatenating all sections' `text` in order should fully cover the original (minor whitespace loss acceptable)
4. **ALL [[PIC:xxx.png]] placeholders must be preserved** — every original placeholder must appear in some section, in original order
5. Target 200-1500 chars per section
6. Title: 5-10 English words summarizing section's core content

Original text:
'''
{text}
'''
"""


STRICT_PROMPT_TEMPLATE_EN = """You are a document structure analyst. Split this English manual ({manual_name}) into semantically coherent sections.

## 🔴 CRITICAL RULES (any violation will be rejected)

1. **Verbatim copy** of ALL original text — NEVER omit, summarize, skip, or paraphrase ANY paragraph, sentence, character, or [[PIC:xxx.png]] placeholder
2. Even if a paragraph looks unimportant (copyright notice, table of contents, repeated warnings), it MUST be preserved as-is in some section
3. **ALL [[PIC:xxx.png]] placeholders must appear in some section**, preserving original count AND order
4. `text` must be a CONTIGUOUS substring of the original, in original sequence

## Output Format

- JSON: `{{"sections": [{{"title": "...", "text": "..."}}, ...]}}`
- 200-1500 chars per section
- Title: 5-10 English words summarizing core content

## Auto-Validation (I will check)

- Concatenated character coverage must be ≥ 99%
- PIC placeholder count must EXACTLY equal the original

Previous attempts on similar manuals dropped paragraphs. Be EXTRA careful to preserve every character.

Original text:
'''
{text}
'''
"""


# ─────── 中文版严格 prompt(原有,留作中文重试) ───────

STRICT_PROMPT_TEMPLATE = """你是文档结构分析助手，需要把一份《{manual_name}》切成"语义连贯的小节"。

## 🔴 关键规则（违反任一条会被拒绝）

1. **逐字复制**所有原文 —— 严禁省略、概括、跳过任何段落、句子、字符或 [[PIC:xxx.png]] 占位符
2. 即使某段看起来**无关紧要**（如版权声明、目录页、重复警告），也必须**原样保留**到对应小节
3. **所有 [[PIC:xxx.png]] 占位符必须出现在某个小节里**，原文有多少张就必须有多少张
4. text 必须是**原文的连续片段**，顺序与原文完全一致，不能跳跃式抄录

## 输出要求

- JSON 对象 `{{"sections": [{{"title": "...", "text": "..."}}, ...]}}`
- 每节目标 200-1500 字
- 标题 8-15 字一句话概括小节核心内容

## 我会做的校验（自动检测）

- 拼接后字符覆盖率必须 ≥ 99%（去空白后逐字符比较）
- PIC 占位符数量必须与原文**完全一致**

之前你处理类似手册时漏过段落，这次请**特别注意逐字保留**。

原文：
'''
{text}
'''
"""


def _safe_json_loads(content: str) -> dict:
    """容错 JSON 解析:LLM 偶尔吐非法 \\uXXXX 转义,先清洗再 parse"""
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        # 修复非法 \\uXXXX(不足 4 位 hex 或非 hex)→ 替换为字面 \\\\u
        cleaned = re.sub(r'\\u(?![0-9a-fA-F]{4})', r'\\\\u', content)
        return json.loads(cleaned)


def _call_llm(prompt: str) -> tuple[list[dict], dict]:
    """单次 LLM 调用，返回 (sections, usage)"""
    resp = client.chat.completions.create(
        model=CHUNKER_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        response_format={"type": "json_object"},
        max_tokens=8192,  # qwen-plus 输出上限,显式设防默认偏小
    )
    content = resp.choices[0].message.content
    data = _safe_json_loads(content)
    sections = data.get("sections", [])
    u = resp.usage
    usage = {
        "prompt_tokens": u.prompt_tokens,
        "completion_tokens": u.completion_tokens,
        "total_tokens": u.total_tokens,
        "cost_yuan": (u.prompt_tokens * PRICE_INPUT + u.completion_tokens * PRICE_OUTPUT) / 1000,
    }
    return sections, usage


def is_english(text: str, threshold: float = 0.85) -> bool:
    """检测文本主要是英文(前 800 字符 ASCII 字母占比 > threshold)"""
    sample = text[:800]
    if not sample:
        return False
    # 只看字母字符的比例(中文字符 ord >= 128)
    letters = [c for c in sample if c.isalpha()]
    if not letters:
        return False
    ascii_letters = sum(1 for c in letters if ord(c) < 128)
    return ascii_letters / len(letters) > threshold


def chunk_with_retry(
    text: str,
    manual_name: str,
    max_retries: int = 3,
    threshold_coverage: float = 0.99,
) -> tuple[list[dict], dict, dict]:
    """带校验和重试的切块,自动按文本语言选中/英文 prompt。

    返回 (sections, usage, validation_metrics)。
    validation_metrics 含 attempts(实际尝试次数) 和 unmet_threshold(是否超出重试还不达标)。
    """
    use_en = is_english(text)
    base_template = PROMPT_TEMPLATE_EN if use_en else PROMPT_TEMPLATE
    strict_template = STRICT_PROMPT_TEMPLATE_EN if use_en else STRICT_PROMPT_TEMPLATE

    best = None  # 保留覆盖率最高的那次结果作为兜底
    total_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "cost_yuan": 0.0}
    last_error = None

    for attempt in range(1, max_retries + 1):
        template = base_template if attempt == 1 else strict_template
        prompt = template.format(manual_name=manual_name, text=text)

        try:
            sections, usage = _call_llm(prompt)
        except Exception as e:
            # JSON 解析失败 / API 错误 → 当作一次失败 attempt,继续重试
            last_error = e
            continue

        # 累计 token 花费
        for k in ("prompt_tokens", "completion_tokens", "total_tokens"):
            total_usage[k] += usage[k]
        total_usage["cost_yuan"] += usage["cost_yuan"]

        metrics = validate_sections(sections, text)
        metrics["attempts"] = attempt

        # 达标 → 立即返回
        if metrics["char_coverage"] >= threshold_coverage and metrics["pic_count_match"]:
            metrics["unmet_threshold"] = False
            return sections, total_usage, metrics

        # 不达标 → 留作 best 候选
        if best is None or metrics["char_coverage"] > best[2]["char_coverage"]:
            best = (sections, dict(total_usage), metrics)

    # 重试用完仍未达标
    if best is None:
        # 所有 attempt 都抛异常,无任何 best
        raise RuntimeError(
            f"All {max_retries} attempts failed for {manual_name}. Last error: {last_error}"
        )

    best_sections, best_usage_snapshot, best_metrics = best
    best_metrics["unmet_threshold"] = True
    best_metrics["attempts"] = max_retries
    return best_sections, total_usage, best_metrics


def chunk_with_llm(text: str, manual_name: str) -> tuple[list[dict], dict]:
    """调 qwen-plus 切块。返回 (sections, usage_info)。

    usage_info 含 prompt_tokens / completion_tokens / cost_yuan。
    """
    prompt = PROMPT_TEMPLATE.format(manual_name=manual_name, text=text)
    resp = client.chat.completions.create(
        model=CHUNKER_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        response_format={"type": "json_object"},
    )
    content = resp.choices[0].message.content
    data = json.loads(content)
    sections = data.get("sections", [])

    u = resp.usage
    usage = {
        "prompt_tokens": u.prompt_tokens,
        "completion_tokens": u.completion_tokens,
        "total_tokens": u.total_tokens,
        "cost_yuan": (u.prompt_tokens * PRICE_INPUT + u.completion_tokens * PRICE_OUTPUT) / 1000,
    }
    return sections, usage


def validate_sections(sections: list[dict], original_text: str) -> dict:
    """校验切块质量。返回 metrics dict + issues list。"""
    issues = []

    # 字符覆盖率(去空白后比较)
    def normalize(s: str) -> str:
        return re.sub(r'\s+', '', s)

    concat = "".join(s["text"] for s in sections)
    norm_orig = normalize(original_text)
    norm_new = normalize(concat)

    char_coverage = len(norm_new) / max(len(norm_orig), 1)
    char_exact = (norm_orig == norm_new)
    if not char_exact:
        issues.append(f"字符覆盖 {char_coverage * 100:.2f}%(差 {len(norm_orig) - len(norm_new)} 字)")

    # PIC 完整性 + 顺序
    orig_pics = re.findall(r'\[\[PIC:[^\]]+\]\]', original_text)
    new_pics = re.findall(r'\[\[PIC:[^\]]+\]\]', concat)
    pic_count_match = len(orig_pics) == len(new_pics)
    pic_order_match = orig_pics == new_pics

    if not pic_count_match:
        issues.append(f"PIC 数量不一致: {len(orig_pics)} → {len(new_pics)}")
    elif not pic_order_match:
        issues.append("PIC 顺序错乱")

    # 长度分布
    lengths = [len(s["text"]) for s in sections]
    too_short = sum(1 for l in lengths if l < 50)
    too_long = sum(1 for l in lengths if l > 2000)
    if too_short:
        issues.append(f"{too_short} 节过短 (<50 字)")
    if too_long:
        issues.append(f"{too_long} 节过长 (>2000 字)")

    return {
        "n_sections": len(sections),
        "char_coverage": char_coverage,
        "char_exact": char_exact,
        "pic_count_match": pic_count_match,
        "pic_order_match": pic_order_match,
        "n_pics_orig": len(orig_pics),
        "n_pics_new": len(new_pics),
        "length_min": min(lengths) if lengths else 0,
        "length_max": max(lengths) if lengths else 0,
        "length_avg": sum(lengths) // max(len(lengths), 1),
        "issues": issues,
    }


def chunk_manual_file(filepath: str) -> dict:
    """完整处理一份手册:解析 → 切块 → 校验。

    返回 dict:
      {"manual_name": ..., "parts": [{"text": ..., "sections": [...], "metrics": {...}}],
       "total_sections": int, "total_pics": int, "total_cost": float}

    一份手册可能有多个 JSON 段(汇总英文手册 20 段),每段独立切块。
    """
    from src.knowledge_base import parse_manual, replace_pic_with_placeholder

    manual_name = os.path.basename(filepath).replace(".txt", "")
    parsed = parse_manual(filepath)

    parts = []
    total_cost = 0.0
    for idx, (text, image_refs) in enumerate(parsed):
        text_with_pics = replace_pic_with_placeholder(text, image_refs)
        sections, usage = chunk_with_llm(text_with_pics, manual_name)
        metrics = validate_sections(sections, text_with_pics)
        parts.append({
            "sub_index": idx,
            "original_text": text_with_pics,
            "sections": sections,
            "metrics": metrics,
            "usage": usage,
        })
        total_cost += usage["cost_yuan"]

    return {
        "manual_name": manual_name,
        "n_parts": len(parts),
        "parts": parts,
        "total_sections": sum(len(p["sections"]) for p in parts),
        "total_pics": sum(p["metrics"]["n_pics_orig"] for p in parts),
        "total_cost_yuan": total_cost,
    }
