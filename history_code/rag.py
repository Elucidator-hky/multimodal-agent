"""RAG 引擎：检索增强生成"""
import re
import sys
import os

from openai import OpenAI

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src.knowledge_base import KnowledgeBase, Chunk


SYSTEM_PROMPT = """你是一个专业的产品客服助手。请基于检索到的产品知识库内容回答用户问题。

## 回答规则：

1. **基于知识库**：严格基于提供的知识库内容回答，不要编造信息。
2. **保留图片标记**：如果知识库内容中包含 <PIC> 标记，在回答中保留这些标记，并在回答末尾附上对应的图片引用列表。
3. **逐一回答**：如果用户的问题包含多个子问题，请逐一回答每个子问题。
4. **详细有深度**：回答要详细、专业、有深度，结构清晰。使用编号或分点回答。
5. **客服语气**：使用友好、专业的客服语气。
6. **无法回答时**：如果知识库内容不足以回答问题，请诚实告知，并给出合理的客服引导回复（如建议联系售后、查看官网等）。
7. **语言匹配**：如果用户用英文提问，请用英文回答；用中文提问则用中文回答。
8. **图片引用格式**：如果回答中包含 <PIC>，在回答最后用 [图片引用: xxx, yyy] 的格式列出所有图片引用名。

## 图片引用说明：
- 知识库中的 <PIC> 标记表示该位置有一张产品插图
- 每个 <PIC> 对应一个图片引用名（如 Camera_01）
- 在回答中适当引用这些图片来辅助说明"""


class RAGEngine:
    """RAG 引擎"""

    def __init__(self, kb: KnowledgeBase = None):
        self.kb = kb or KnowledgeBase()
        self.client = OpenAI(
            api_key=config.QWEN_API_KEY,
            base_url=config.QWEN_BASE_URL,
        )

    def load(self):
        """加载知识库索引"""
        self.kb.load()
        # 预加载 embedding 模型
        self.kb._get_model()
        print("RAG 引擎就绪")

    def build(self):
        """构建知识库索引"""
        self.kb.load_manuals()
        self.kb.build_index()
        self.kb.save()

    def retrieve(self, query: str, top_k: int = None) -> list[tuple[Chunk, float]]:
        """检索相关文档块"""
        if top_k is None:
            top_k = config.TOP_K
        return self.kb.search(query, top_k=top_k)

    def format_contexts(self, results: list[tuple[Chunk, float]]) -> str:
        """格式化检索结果为上下文文本"""
        contexts = []
        total_len = 0

        for i, (chunk, score) in enumerate(results):
            # 构建上下文块
            header = f"【知识库片段 {i+1}】来源: {chunk.source_file} | 标题: {chunk.title} | 相关度: {score:.3f}"
            text = chunk.text

            # 如果有图片引用，附加映射信息
            if chunk.image_refs:
                pic_info = " | ".join(
                    f"<PIC>→{ref}" for ref in chunk.image_refs
                )
                text += f"\n[图片映射: {pic_info}]"

            block = f"{header}\n{text}"

            if total_len + len(block) > config.MAX_CONTEXT_LENGTH:
                # 截断以控制上下文长度
                remaining = config.MAX_CONTEXT_LENGTH - total_len
                if remaining > 200:
                    contexts.append(block[:remaining] + "...(截断)")
                break

            contexts.append(block)
            total_len += len(block)

        return "\n\n---\n\n".join(contexts)

    def generate_answer(self, question: str, contexts: str) -> dict:
        """
        调用 Qwen API 生成回答。
        返回 {"answer": str, "image_refs": list}
        """
        user_prompt = f"""## 用户问题：
{question}

## 检索到的知识库内容：
{contexts}

请根据以上知识库内容回答用户问题。"""

        try:
            response = self.client.chat.completions.create(
                model=config.QWEN_MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.3,
                max_tokens=2048,
            )

            answer = response.choices[0].message.content

            # 提取图片引用
            image_refs = self._extract_image_refs(answer, contexts)

            return {
                "answer": answer,
                "image_refs": image_refs,
            }

        except Exception as e:
            print(f"API 调用失败: {e}")
            return {
                "answer": f"抱歉，服务暂时不可用，请稍后再试。错误信息: {str(e)}",
                "image_refs": [],
            }

    def _extract_image_refs(self, answer: str, contexts: str) -> list:
        """从回答和上下文中提取图片引用"""
        refs = []

        # 从回答末尾的 [图片引用: ...] 中提取
        ref_pattern = re.findall(r"\[图片引用[：:]\s*([^\]]+)\]", answer)
        for match in ref_pattern:
            for ref in re.split(r"[,，、\s]+", match):
                ref = ref.strip()
                if ref and ref not in refs:
                    refs.append(ref)

        # 如果回答中有 <PIC> 但没提取到引用，从上下文的图片映射中提取
        if "<PIC>" in answer and not refs:
            mapping_pattern = re.findall(r"<PIC>→(\w+)", contexts)
            for ref in mapping_pattern:
                if ref not in refs:
                    refs.append(ref)

        return refs

    def ask(self, question: str) -> dict:
        """完整的问答流程"""
        # 1. 检索
        results = self.retrieve(question)

        # 2. 格式化上下文
        contexts = self.format_contexts(results)

        # 3. 生成回答
        response = self.generate_answer(question, contexts)

        return response


if __name__ == "__main__":
    # 测试
    engine = RAGEngine()
    engine.load()

    test_questions = [
        "相机怎么充电？",
        "空调遥控器怎么用？",
        "How to clean the air purifier filter?",
    ]

    for q in test_questions:
        print(f"\n{'='*60}")
        print(f"问题: {q}")
        print(f"{'='*60}")
        result = engine.ask(q)
        print(f"回答: {result['answer'][:500]}")
        if result["image_refs"]:
            print(f"图片引用: {result['image_refs']}")
