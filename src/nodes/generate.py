"""generate 节点：用 chunks 生成产品题答案 + 提取图片 ID

输入:  state["question"] + state["chunks"]
输出:  state["answer_text"](含 <PIC> 标记) + state["image_ids"]

关键流程:
  1. 把 chunks 格式化成 LLM 上下文(保留 [[PIC:xxx.png]] 占位符)
  2. LLM 基于上下文生成答案,在引用步骤时**保留 [[PIC:xxx.png]] 占位符**
  3. 后处理:regex 抠出所有 [[PIC:xxx]] → image_ids,把占位符替换为 <PIC>
"""
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import config
from openai import OpenAI
from src.state import State

client = OpenAI(api_key=config.QWEN_API_KEY, base_url=config.QWEN_BASE_URL)


SYSTEM_PROMPT = """你是友好、专业的产品客服。用户咨询具体产品的操作问题（使用 / 安装 / 清洁 / 故障等），你基于检索到的产品手册片段回答。

## 回答规则

1. **严格基于手册片段** —— 不要编造任何手册里没有的信息。如果片段里没有答案,回复"具体请查阅产品手册"。

2. **语言匹配** —— 用户用中文你用中文,用户用英文你用英文(English)。

3. **图片标记 [[PIC:xxx.png]]** —— 这是手册片段里的图片占位符。**如果你引用的步骤或部件涉及它,务必在你的回答中原样保留对应的 [[PIC:xxx.png]] 标记**。位置要准确(放在它对应的文字之后)。不要修改占位符内容,不要发明新的 PIC 标记。

4. **答案长度** —— 中文 80-200 字,英文 50-150 词。简洁但完整。

5. **风格** ——
   - 开头礼貌:"您好"(中文)/ "Hi"(英文)
   - 给具体步骤(可用编号 1. 2. 3.)
   - 涉及具体型号 / 数字 / 部件名时保留(如 "DCB101"、"Espresso button"、"40°C")
   - 不要输出 emoji、markdown 加粗、列表符号(-/•)
   - 纯文本段落,只用句号、分号、逗号

6. **多片段时** —— 多个片段都讲同一主题,综合它们的步骤,不要重复。如果片段相互冲突,以最具体的(含具体数字 / 型号 / 步骤)为准。

## 例子

[片段 1] 来自《空气净化器手册》—— 滤网清洁步骤
拔下电源插头。从机身后部取下滤网 [[PIC:M_5.png]] 用吸尘器或软刷清除灰尘 [[PIC:M_6.png]] 装回滤网,合上后盖。

用户问题:如何清洁空气净化器的滤网?

回答示例:
您好,清洁空气净化器滤网的步骤是:1. 拔下电源插头确保安全;2. 从机身后部取出滤网 [[PIC:M_5.png]] 3. 用吸尘器或软刷清除滤网上的灰尘 [[PIC:M_6.png]] 4. 装回滤网,合上后盖。建议每月清洁 1-2 次,灰尘较多的环境可增加频率。"""


def _format_chunks(chunks: list[dict]) -> str:
    """把 chunks 拼成给 LLM 的上下文(保留 [[PIC:xxx]] 占位符不动)"""
    if not chunks:
        return "(无检索片段)"
    parts = []
    for i, c in enumerate(chunks, 1):
        src = c.get("source", "").replace(".txt", "")
        title = c.get("title", "")
        prefix = f"[片段 {i}] 来自《{src}》"
        if title:
            prefix += f" —— {title}"
        text = c.get("text", "").strip()
        parts.append(f"{prefix}\n{text}")
    return "\n\n".join(parts)


def generate(state: State) -> dict:
    """节点:基于 chunks 生成答案,提取图片 ID"""
    chunks_text = _format_chunks(state["chunks"])
    user_msg = (
        f"## 检索到的手册片段\n\n{chunks_text}\n\n"
        f"## 用户问题\n\n{state['question']}\n\n"
        f"请基于上述片段回答用户。回答中需要的图片用 [[PIC:xxx.png]] 标记原样保留。"
    )

    resp = client.chat.completions.create(
        model=config.QWEN_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        temperature=0.4,
        max_tokens=600,
    )
    raw_answer = resp.choices[0].message.content or ""

    # 抠 [[PIC:xxx]] 文件名(去重保序)
    pic_filenames = re.findall(r'\[\[PIC:([^\]]+)\]\]', raw_answer)
    seen = set()
    image_ids = []
    for p in pic_filenames:
        if p not in seen:
            seen.add(p)
            image_ids.append(p)

    # 把 [[PIC:xxx]] 替换为 <PIC>(官方提交格式)
    answer_text = re.sub(r'\[\[PIC:[^\]]+\]\]', '<PIC>', raw_answer)

    return {
        "answer_text": answer_text,
        "image_ids": image_ids,
    }


# ─────── 单测:跑 3 题(中/英/含图)看实际输出 ───────
if __name__ == "__main__":
    from src.nodes.rag_search import rag_search

    test_questions = [
        "如何清洁空气净化器的滤网？",
        "我的DCB101型号电钻指示灯闪烁时代表什么含义？",
        "How to use the energy saving mode of a coffee machine?",
    ]
    for q in test_questions:
        state: State = {
            "question": q, "question_type": "product", "chunks": [],
            "answer_text": "", "image_ids": [], "final_output": "",
        }
        # 1. 检索
        state.update(rag_search(state))
        # 2. 生成
        result = generate(state)
        print(f"\n{'='*70}")
        print(f"Q: {q}")
        print(f"\nA: {result['answer_text']}")
        print(f"\nImage IDs: {result['image_ids']}")
