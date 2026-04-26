"""跑 service_gen 节点：仅官方 3 个客服示例（看复读 + 风格），并发调用"""
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src.nodes.service_gen import client, SYSTEM_PROMPT


# qwen-plus 单价（元 / 千 token）
PRICE_INPUT = 0.0008
PRICE_OUTPUT = 0.002


OFFICIAL_CASES = [
    {
        "id": "官方_3",
        "q": "请问你们的商品能送到乡镇吗？需要额外加运费吗？多久能到？",
        "official": (
            "您好，我们的商品支持送到大部分乡镇哦，具体能否送达，取决于您的收货地址，"
            "您可以告诉我详细的收货地址，我帮您查询。"
            "送到乡镇一般不需要额外加运费，和市区运费一致；"
            "物流时效会比市区稍慢，正常情况下，下单后48小时发货，"
            "乡镇地区3-5天可收到，偏远乡镇可能需要5-7天哦。"
        ),
    },
    {
        "id": "官方_4",
        "q": "物流一直显示待揽收，是什么原因？",
        "official": (
            "您好，物流显示待揽收，大概率是商品已打包完成，等待快递员上门取件哦，"
            "一般24小时内会完成揽收；若超过24小时仍未揽收，您可以联系我们客服，"
            "我们会催促快递方尽快上门。"
        ),
    },
    {
        "id": "官方_5",
        "q": (
            "我购买的商品，售后维修后，使用不到10天又出现同样的故障，"
            "而且维修人员说这次故障是上次维修不彻底导致的，请问该怎么处理？"
        ),
        "official": (
            "您好，非常抱歉给您带来困扰！维修后短期内出现同样故障，"
            "且是上次维修不彻底导致的，属于我们的维修失误，"
            "支持免费重新维修，并延长维修质保期。"
            "请您提供维修单号、商品故障描述，我们立即安排专业维修人员处理。"
        ),
    },
]

# csv 客服题（无官方答案，看泛化）
CSV_CASES = [
    {"id": "csv_1", "q": "请问你们家的商品支持7天无理由退换货吗？需要自己承担运费吗？", "official": None},
    {"id": "csv_2", "q": "我想咨询一下，你们的售后维修服务范围是什么？如果是人为损坏的，能维修吗？维修费用怎么算？", "official": None},
    {"id": "csv_3", "q": "请问你们的商品能开发票吗？发票类型是什么？多久能收到呢？", "official": None},
    {"id": "csv_4", "q": "我收到商品后，发现包装破损了，怎么办？影响商品退换货吗？", "official": None},
    {"id": "csv_6", "q": "我想了解一下你们的退款政策，退款多久能到账？信用卡会原路返回吗？", "official": None},
    {"id": "csv_7", "q": "我收到的商品和图片不一样，颜色偏差很大，我要投诉！", "official": None},
    {"id": "csv_9", "q": "我收到的商品少了一件，联系客服说会补发，但是过了一周还没补发！", "official": None},
    {"id": "csv_12", "q": "你们的快递员态度特别差，送货时还辱骂我，我要投诉！", "official": None},
    {"id": "csv_18", "q": "我想退货，但是已经超过7天无理由退换货期限了，还能退吗？", "official": None},
]

ALL_CASES = OFFICIAL_CASES + CSV_CASES


def call_with_usage(question: str) -> tuple[str, dict]:
    resp = client.chat.completions.create(
        model=config.QWEN_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ],
        temperature=0.4,
        max_tokens=512,
    )
    answer = resp.choices[0].message.content
    u = resp.usage
    usage = {
        "prompt": u.prompt_tokens,
        "completion": u.completion_tokens,
        "total": u.total_tokens,
        "cost": (u.prompt_tokens * PRICE_INPUT + u.completion_tokens * PRICE_OUTPUT) / 1000,
    }
    return answer, usage


def to_markdown(results: list, elapsed: float) -> str:
    lines = [
        "# service_gen 节点 — 12 题客服题测试",
        "",
        f"**模型**: `{config.QWEN_MODEL}` | **耗时**: {elapsed:.1f}s（并发）",
        f"**官方 3 题（带参考答案）+ csv 9 题（无参考，看泛化）**",
        "",
        "---",
    ]
    total_cost = 0.0
    total_tokens = 0
    for c, answer, usage in results:
        total_cost += usage["cost"]
        total_tokens += usage["total"]
        lines += [
            "",
            f"## [{c['id']}]",
            "",
            f"### 问题",
            f"> {c['q']}",
            "",
        ]
        if c.get("official"):
            lines += [
                f"### 官方参考答案",
                "",
                c["official"],
                "",
            ]
        lines += [
            f"### service_gen 输出",
            "",
            answer,
            "",
            f"### 用量",
            f"- 输入: {usage['prompt']} | 输出: {usage['completion']} | 总: {usage['total']} tokens | 花费: ¥{usage['cost']:.6f}",
            "",
            "---",
        ]
    lines += [
        "",
        "## 汇总",
        f"- 总 tokens: {total_tokens}",
        f"- 总花费: ¥{total_cost:.6f}",
        f"- 并发耗时: {elapsed:.1f}s",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=len(ALL_CASES)) as pool:
        results = list(pool.map(lambda c: (c, *call_with_usage(c["q"])), ALL_CASES))
    elapsed = time.time() - t0

    md = to_markdown(results, elapsed)
    out_dir = os.path.join(os.path.dirname(__file__), "output")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "service_gen_12cases.md")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(md)

    print(f"已写入: {out_path}")
    print(f"耗时: {elapsed:.1f}s | 共 {sum(r[2]['total'] for r in results)} tokens "
          f"| ¥{sum(r[2]['cost'] for r in results):.6f}")
