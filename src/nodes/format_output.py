"""format_output 节点：把答案拼成官方提交 csv 的 ret 字段"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.state import State


def format_output(state: State) -> dict:
    """节点：根据 question_type 拼出 state['final_output']

    格式（依据 README.md / 比赛说明）：
      - service：纯文本
      - product：JSON 数组 ["文本(含<PIC>)", ["图片id1", "图片id2"]]

    后续若官方实际格式不同，只改本函数。
    """
    answer = state["answer_text"]
    if state["question_type"] == "service":
        final = answer
    else:  # product
        final = json.dumps(
            [answer, state["image_ids"]],
            ensure_ascii=False,
        )
    return {"final_output": final}


# ─────── 单独测试该节点 ───────
if __name__ == "__main__":
    cases = [
        {
            "name": "客服题（纯文本）",
            "question_type": "service",
            "answer_text": "您好，我们支持7天无理由退换货，运费由我们承担。",
            "image_ids": [],
        },
        {
            "name": "产品题（含 <PIC> + 多图）",
            "question_type": "product",
            "answer_text": "请按下电源按钮 <PIC> 启动设备，然后选择模式 <PIC>。",
            "image_ids": ["bdcf3a28e3c7.png", "70c5e1f5acab.png"],
        },
        {
            "name": "产品题（无图，边缘情况）",
            "question_type": "product",
            "answer_text": "请参阅手册第三章。",
            "image_ids": [],
        },
    ]
    for i, c in enumerate(cases, 1):
        state: State = {
            "question": "", "question_type": c["question_type"],
            "chunks": [], "answer_text": c["answer_text"],
            "image_ids": c["image_ids"], "final_output": "",
        }
        result = format_output(state)
        print(f"[{i}] {c['name']}")
        print(f"    → {result['final_output']}")
        print()
