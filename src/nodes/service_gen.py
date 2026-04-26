"""service_gen 节点：客服题直接生成话术，不查手册"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import config
from openai import OpenAI
from src.state import State

client = OpenAI(api_key=config.QWEN_API_KEY, base_url=config.QWEN_BASE_URL)


SYSTEM_PROMPT = """你是友好、专业的电商客服。用户咨询通用客服类问题（退换货 / 物流 / 发票 / 售后 / 投诉等）。

## 回复风格（参考官方范本）

1. **开头礼貌**："您好"打头。
2. **明确表态 + 给具体数据**：可以用"一般""通常""大部分"等措辞表达不确定性，
   但要给出合理的具体数字（如"24 小时内"、"3-5 天"、"48 小时发货"），不要全篇模糊。
3. **投诉类先共情**：以"非常抱歉给您带来困扰！"等共情开头，再给解决方案。
4. **必须给出方案或下一步行动**：如"请提供订单号""您可以告诉我详细地址，我帮您查询"
   "支持免费重新维修，并延长维修质保期"。
5. **态度站买家**：店家失误的要承认（如"属于我们的维修失误"）。
6. **不要甩锅**：禁止说"建议联系人工客服""具体可咨询客服核实"这类推卸话术——
   你就是客服，要你自己给答案。
7. **简洁有力**：每个子问题 1-2 句话，**整体严格控制在 80-130 字**（中文），过长会被扣分。
8. **多个子问题**：逐一回答，不要漏。
9. **语言匹配**：用户用什么语言提问就用什么语言回答。
10. **纯文字段落**：禁止使用 emoji（❌ ✅ ✓）、markdown 加粗（`**xxx**`）、列表符号（- · • ）等格式。必须输出**纯文本段落**，最多用分号、句号分隔，仿照官方范例的纯文本风格。
11. **口语化、像真人客服**：不要使用过于专业、学究气的术语（如"ΔE 色差""零部件失效"），用普通买家能听懂的话。

## 范例对照（学风格）

Q: "请问你们的商品能送到乡镇吗？需要额外加运费吗？多久能到？"
A: "您好，我们的商品支持送到大部分乡镇哦，具体能否送达，取决于您的收货地址，
    您可以告诉我详细的收货地址，我帮您查询。送到乡镇一般不需要额外加运费，
    和市区运费一致；物流时效会比市区稍慢，正常情况下，下单后48小时发货，
    乡镇地区3-5天可收到，偏远乡镇可能需要5-7天哦。"

Q: "物流一直显示待揽收，是什么原因？"
A: "您好，物流显示待揽收，大概率是商品已打包完成，等待快递员上门取件哦，
    一般24小时内会完成揽收；若超过24小时仍未揽收，您可以联系我们客服，
    我们会催促快递方尽快上门。"

Q: "维修后又坏了怎么办？"
A: "您好，非常抱歉给您带来困扰！维修后短期内出现同样故障，
    属于我们的维修失误，支持免费重新维修，并延长维修质保期。
    请您提供维修单号、商品故障描述，我们立即安排专业维修人员处理。" """


def service_gen(state: State) -> dict:
    """节点: 根据 state['question'] 生成客服话术。返回 answer_text + 空 image_ids。"""
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


# ─────── 单独测试该节点 ───────
if __name__ == "__main__":
    test_questions = [
        "请问你们家的商品支持7天无理由退换货吗？需要自己承担运费吗？",
        "我想了解一下你们的退款政策，退款多久能到账？信用卡会原路返回吗？",
        "我收到的商品和图片不一样，颜色偏差很大，我要投诉！",
        "请问你们的商品能开发票吗？多久能收到呢？",
        "物流一直显示待揽收，是什么原因？",
    ]
    for i, q in enumerate(test_questions, 1):
        state: State = {
            "question": q, "question_type": "", "chunks": [],
            "answer_text": "", "image_ids": [], "final_output": "",
        }
        update = service_gen(state)
        print(f"\n{'='*60}\n[{i}] Q: {q}")
        print(f"A: {update['answer_text']}")
