"""单独测试 service_gen 节点：客服题直接生成话术，不查手册"""
import os
import sys
from typing import TypedDict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from openai import OpenAI

client = OpenAI(api_key=config.QWEN_API_KEY, base_url=config.QWEN_BASE_URL)


# ────────────────────────────── State ──────────────────────────────
class State(TypedDict):
    question: str
    answer_text: str
    image_ids: list


# ────────────────────────────── 节点 ──────────────────────────────
SYSTEM_PROMPT = """你是友好、专业的电商客服。用户咨询了通用客服类问题（退换货 / 物流 / 发票 / 售后 / 投诉等）。

要求：
1. 回答得体、专业、有帮助
2. 不要编造具体政策细节（不要说"我们支持 X 天"这种确定性表述）
3. 用"一般""通常""具体可咨询客服核实"等措辞
4. 给出合理建议或引导（如"您可以提供订单号让我查询"）
5. 如果有多个子问题，请逐一回答
6. 用户用什么语言提问就用什么语言回答"""


def service_gen(state: State) -> dict:
    """节点：根据 state['question'] 生成客服话术，返回 answer_text + 空 image_ids"""
    resp = client.chat.completions.create(
        model=config.QWEN_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": state["question"]},
        ],
        temperature=0.4,
        max_tokens=512,
    )
    return {
        "answer_text": resp.choices[0].message.content,
        "image_ids": [],
    }


# ────────────────────────────── 测试 ──────────────────────────────
if __name__ == "__main__":
    test_questions = [
        "请问你们家的商品支持7天无理由退换货吗？需要自己承担运费吗？",
        "我想了解一下你们的退款政策，退款多久能到账？信用卡会原路返回吗？",
        "我收到的商品和图片不一样，颜色偏差很大，我要投诉！",
        "请问你们的商品能开发票吗？多久能收到呢？",
        "物流一直显示待揽收，是什么原因？",
    ]
    for i, q in enumerate(test_questions, 1):
        # 模拟节点接收 State
        state: State = {"question": q, "answer_text": "", "image_ids": []}
        update = service_gen(state)
        print(f"\n{'='*60}\n[{i}] Q: {q}")
        print(f"A: {update['answer_text']}")
