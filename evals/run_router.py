"""router 节点 eval：30+ 题人工标注，并发跑 router，看准确率"""
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from openai import OpenAI
from src.nodes.router import SYSTEM_PROMPT

client = OpenAI(api_key=config.QWEN_API_KEY, base_url=config.QWEN_BASE_URL)

# qwen-plus 单价（元 / 千 token）
PRICE_INPUT = 0.0008
PRICE_OUTPUT = 0.002


# 32 题：客服 8 + 中文产品 8 + 英文产品 6 + 边界（投诉/索赔/政策）6 + 复合 4
CASES = [
    # ── 明确客服题 ──
    {"id": "1",  "expected": "service", "q": "请问你们家的商品支持7天无理由退换货吗？需要自己承担运费吗？"},
    {"id": "6",  "expected": "service", "q": "我想了解一下你们的退款政策，退款多久能到账？信用卡会原路返回吗？"},
    {"id": "7",  "expected": "service", "q": "我收到的商品和图片不一样，颜色偏差很大，我要投诉！"},
    {"id": "12", "expected": "service", "q": "你们的快递员态度特别差，送货时还辱骂我，我要投诉！"},
    {"id": "18", "expected": "service", "q": "我想退货，但是已经超过7天无理由退换货期限了，还能退吗？"},
    {"id": "19", "expected": "service", "q": "请问你们的商品能提供纸质版说明书吗？电子版在哪里可以找到？"},
    {"id": "26", "expected": "service", "q": "请问你们的智能客服能解答哪些问题？智能客服解答不了的问题，怎么办？"},
    {"id": "41", "expected": "service", "q": "快递丢失了，怎么办？这种情况你们会怎么赔偿？多久能解决？"},

    # ── 明确产品题（中文）──
    {"id": "64",  "expected": "product", "q": "使用吹风机时，人员需要佩戴哪些防护装备？"},
    {"id": "69",  "expected": "product", "q": "该如何关闭吹风机？"},
    {"id": "74",  "expected": "product", "q": "如何用空调快速调节室内温度？"},
    {"id": "83",  "expected": "product", "q": "如何清洁空调的空气滤网？"},
    {"id": "86",  "expected": "product", "q": "如何快速组装蒸汽清洁机？"},
    {"id": "95",  "expected": "product", "q": "如何为洗碗机添加洗涤剂？"},
    {"id": "124", "expected": "product", "q": "我的DCB101型号电钻指示灯闪烁时，这些闪烁标识代表什么含义？"},
    {"id": "153", "expected": "product", "q": "考虑到燃油高度易燃且有毒，使用发电机时我需要注意什么？"},

    # ── 明确产品题（英文）──
    {"id": "241", "expected": "product", "q": "If this is the first time to use airfryer, What should I do before first use?"},
    {"id": "244", "expected": "product", "q": "How the ship steers?"},
    {"id": "250", "expected": "product", "q": "How do I use the jet wash function to clean the boat after using it?"},
    {"id": "255", "expected": "product", "q": "When I am sailing, how do I check the engine oil level to ensure continued sailing?"},
    {"id": "265", "expected": "product", "q": "How to use the energy saving mode of a coffee machine?"},
    {"id": "268", "expected": "product", "q": "How should I do if I want to empty the system before not in use, for frost protection or before maintenance?"},

    # ── 边界题：表面像产品但本质是售后/索赔/政策 ──
    {"id": "46", "expected": "service", "q": "我上个月购买的家电，使用不到一个月就出现故障，联系售后寄回维修，现在已经维修15天了，还没收到，而且我发现商品是翻新机，并非全新，要求退货退款并赔偿，请问该怎么处理？"},
    {"id": "47", "expected": "service", "q": "我在你们平台批量购买了100件商品用于企业采购，收到货后发现有20件存在质量问题，15件少发，而且发票开错了抬头，需要重新开具，同时想申请质量问题商品的换货和少发商品的补寄，请问流程是什么？"},
    {"id": "48", "expected": "service", "q": "我购买的食品类商品，收到后发现生产日期临近过期，而且包装破损，部分商品受潮无法食用，我要求退货退款并赔偿，同时担心食用了受潮商品影响健康，请问能提供相关保障吗？"},
    {"id": "50", "expected": "service", "q": "我购买的大家电需要上门安装，但是安装人员上门后说需要额外收取配件费，而且安装服务原本应该是免费的，同时安装人员操作不规范，导致家电出现轻微损坏，请问该怎么处理？"},
    {"id": "51", "expected": "service", "q": "我购买的商品在质保期内出现质量问题，寄回维修后，维修人员说需要更换配件，而且要收取配件费，但是质保期内应该免费维修，同时维修时间已经超过承诺的7天，请问该怎么处理？"},
    {"id": "53", "expected": "service", "q": "我想试用商品，但是试用期间商品出现故障，而且不是人为操作导致的，同时我想延长试用期限，请问可以吗？另外故障商品能更换吗？"},

    # ── 复合/容易混淆 ──
    {"id": "39",  "expected": "service", "q": "我购买的商品，使用一段时间后出现质量问题，还能售后吗？"},
    {"id": "44",  "expected": "service", "q": "请问你们的优惠券能用于所有商品吗？"},
    {"id": "82",  "expected": "product", "q": "不同型号空调的清洁频率是多少？"},
    {"id": "130", "expected": "product", "q": "电钻的三年有限保修包含哪些内容？"},
]


def call_with_usage(question: str) -> tuple[str, dict]:
    resp = client.chat.completions.create(
        model=config.QWEN_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Q: {question}\nA:"},
        ],
        temperature=0,
        max_tokens=4,
    )
    raw = (resp.choices[0].message.content or "").strip().lower()
    if "product" in raw:
        pred = "product"
    elif "service" in raw:
        pred = "service"
    else:
        pred = "product"  # fallback：产品题占多数

    u = resp.usage
    usage = {
        "prompt": u.prompt_tokens,
        "completion": u.completion_tokens,
        "total": u.total_tokens,
        "cost": (u.prompt_tokens * PRICE_INPUT + u.completion_tokens * PRICE_OUTPUT) / 1000,
        "raw": raw,
    }
    return pred, usage


def to_markdown(rows: list, elapsed: float) -> str:
    total = len(rows)
    correct = sum(1 for r in rows if r["pred"] == r["expected"])
    wrong = [r for r in rows if r["pred"] != r["expected"]]

    total_cost = sum(r["usage"]["cost"] for r in rows)
    total_tokens = sum(r["usage"]["total"] for r in rows)

    lines = [
        "# router 节点 — 32 题分类测试",
        "",
        f"**模型**: `{config.QWEN_MODEL}` | **耗时**: {elapsed:.1f}s（并发）",
        f"**准确率**: {correct}/{total} = **{correct/total*100:.1f}%**",
        f"**总用量**: {total_tokens} tokens | ¥{total_cost:.6f}",
        "",
        "---",
        "",
        "## 误判 case",
        "",
    ]
    if not wrong:
        lines.append("（无误判）")
    else:
        for r in wrong:
            lines += [
                f"### [id={r['id']}] 预期 `{r['expected']}` → 实际 `{r['pred']}`（原始: `{r['usage']['raw']}`）",
                f"> {r['q']}",
                "",
            ]

    lines += [
        "---",
        "",
        "## 全部结果",
        "",
        "| id | 预期 | 实际 | 对错 | 题目 |",
        "|---|---|---|---|---|",
    ]
    for r in rows:
        ok = "✅" if r["pred"] == r["expected"] else "❌"
        q_short = r["q"][:50].replace("|", "/").replace("\n", " ")
        if len(r["q"]) > 50:
            q_short += "..."
        lines.append(f"| {r['id']} | {r['expected']} | {r['pred']} | {ok} | {q_short} |")

    return "\n".join(lines)


if __name__ == "__main__":
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=len(CASES)) as pool:
        results = list(pool.map(lambda c: (c, *call_with_usage(c["q"])), CASES))
    elapsed = time.time() - t0

    rows = [
        {"id": c["id"], "q": c["q"], "expected": c["expected"], "pred": pred, "usage": usage}
        for c, pred, usage in results
    ]

    md = to_markdown(rows, elapsed)
    out_dir = os.path.join(os.path.dirname(__file__), "output")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "router_32cases.md")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(md)

    correct = sum(1 for r in rows if r["pred"] == r["expected"])
    total = len(rows)
    print(f"已写入: {out_path}")
    print(f"准确率: {correct}/{total} = {correct/total*100:.1f}% | "
          f"耗时: {elapsed:.1f}s | "
          f"RMB {sum(r['usage']['cost'] for r in rows):.6f}")
