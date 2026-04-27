"""router 节点：把用户问题分到 product / service 两类，决定走哪条分支"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import config
from openai import OpenAI
from src.state import State

client = OpenAI(api_key=config.QWEN_API_KEY, base_url=config.QWEN_BASE_URL)


SYSTEM_PROMPT = """你是问题分类器。把用户问题分为以下两类，**只输出 product 或 service**（英文小写），不要输出其他任何文字、标点或解释。

# product —— 关于具体产品的操作类问题
涉及电钻、空调、洗碗机、空气净化器、健身设备、冰箱、吹风机、烤箱、咖啡机、船、相机、键盘、鼠标、蒸汽清洁机、发电机、空气炸锅等产品的：
- 使用方法、操作步骤
- 安装、组装、拆卸、连接
- 清洁、维护、保养、更换配件
- 功能介绍、规格说明、按钮含义
- 故障代码、指示灯含义、警告处理
- 配件、零部件相关

# service —— 通用电商客服问题
**不涉及具体产品物理操作**的售前/售中/售后咨询：
- 退换货、退款、订单
- 物流、运费、发货时效、配送范围
- 发票、票据
- 投诉（商品/服务/快递员等）
- 售后政策、保修条款、维修服务流程（政策层面）
- 商品质量问题反馈（消费者投诉/索赔诉求）
- 优惠券、试用装、以旧换新等政策

# 判断准则（按以下顺序匹配，命中即停）
- 问"如何使用 / 安装 / 清洁 / 操作 / 组装 / 拆卸 XX 产品" → product
- 问"能不能退 / 赔 / 补 / 开发票 / 投诉 / 索赔" → service
- 中文 / 英文产品题都是 product

# 关键：边界题判别（看问题的主语和重心）

**主语判别：**
- 主语是"你们 / 你们家 / 你们的商品（泛指平台货品）" + 问平台是否提供某项服务（说明书 / 发票 / 上门安装 / 优惠券 / 试用 / 终身维修等） → **service**
- 主语是具体产品名（电钻 / 空调 / 咖啡机 / 船 / 健身追踪器 等）+ 问该产品本身的属性（规格 / 保修内容 / 配件清单 / 三包条款 / 功能 / 故障含义） → **product**（手册里能查到）

**关于"质量问题 / 维修"：**
- "我买的商品坏了，要退 / 赔 / 重修 / 求处理" → **service**（投诉 / 索赔诉求）
- "XX 产品该如何维护 / 清洁 / 故障代码什么意思" → **product**（操作手册问题）

# 例子
Q: 请问你们的商品支持7天无理由退换货吗？
A: service

Q: 请问你们的商品能提供上门安装服务吗？
A: service

Q: 如何清洁空气净化器的滤网？
A: product

Q: 我的DCB101型号电钻指示灯闪烁时代表什么？
A: product

Q: 购买电钻后，应配备哪些附件？
A: product

Q: How do I turn on the boat's engine?
A: product

Q: How to use the energy saving mode of a coffee machine?
A: product

Q: 我收到的商品和图片不一样，颜色偏差很大，我要投诉！
A: service

Q: 我购买的商品在质保期内出现质量问题，维修人员要收配件费，该怎么处理？
A: service

Q: 使用吹风机时，人员需要佩戴哪些防护装备？
A: product"""


def _classify(question: str) -> str:
    """调一次 LLM 做二分类，返回 'product' 或 'service'"""
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
        return "product"
    if "service" in raw:
        return "service"
    # 兜底：400 题里产品占 ~88%，无法识别时偏向 product
    return "product"


def router(state: State) -> dict:
    """节点：根据 state['question'] 写入 question_type"""
    return {"question_type": _classify(state["question"])}


# ─────── 单独测试该节点 ───────
if __name__ == "__main__":
    samples = [
        "请问你们家的商品支持7天无理由退换货吗？",
        "如何清洁空气净化器的滤网？",
        "我的DCB101型号电钻指示灯闪烁时代表什么含义？",
        "How do I turn on the boat's engine to get the machine started?",
        "我收到的商品和图片不一样，颜色偏差很大，我要投诉！",
        "我购买的商品在质保期内出现质量问题，寄回维修后维修人员说要收配件费，该怎么处理？",
        "How to use the energy saving mode of a coffee machine?",
        "使用吹风机时，人员需要佩戴哪些防护装备？",
    ]
    for q in samples:
        state: State = {
            "question": q, "question_type": "", "chunks": [],
            "answer_text": "", "image_ids": [], "final_output": "",
        }
        result = router(state)
        print(f"[{result['question_type']:>7}] {q[:60]}")
