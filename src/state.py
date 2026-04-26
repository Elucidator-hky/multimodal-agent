"""所有节点共享的 State 定义"""
from typing import TypedDict


class State(TypedDict):
    question: str           # 输入: 用户问题
    question_type: str      # router 写: "product" / "service"
    chunks: list            # rag_search 写: [(Chunk, score), ...]
    answer_text: str        # generate / service_gen 写: 回答正文
    image_ids: list         # generate 写: 图片 ID 列表
    final_output: str       # format_output 写: 官方格式输出
