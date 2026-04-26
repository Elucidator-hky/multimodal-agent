"""生成提交文件 submission.csv - 并发版本"""
import os
import sys
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src.knowledge_base import Chunk, KnowledgeBase
from src.rag import RAGEngine

# 并发数
WORKERS = 8


def process_one(engine: RAGEngine, qid: int, question: str) -> tuple:
    """处理单个问题，返回 (qid, answer)"""
    try:
        result = engine.ask(question)
        answer = result["answer"]
        if result["image_refs"] and "[图片引用" not in answer:
            refs_str = ", ".join(result["image_refs"])
            answer += f"\n[图片引用: {refs_str}]"
        return (qid, answer)
    except Exception as e:
        print(f"\n错误 Q{qid}: {e}")
        return (qid, "抱歉，暂时无法回答该问题，请联系人工客服。")


def _save(results: dict):
    # 将答案中的换行替换为字面 \n，避免 CSV 多行问题
    cleaned = {}
    for k, v in results.items():
        v = str(v).replace('\r\n', '\n').replace('\n', '\\n')
        cleaned[k] = v
    df_out = pd.DataFrame([
        {"id": k, "ret": v} for k, v in sorted(cleaned.items())
    ])
    df_out.to_csv(config.SUBMISSION_FILE, index=False, encoding="utf-8")


def generate():
    # 加载引擎
    engine = RAGEngine()
    try:
        engine.load()
    except FileNotFoundError:
        print("索引文件不存在，正在构建...")
        engine.build()

    # 读取问题
    df = pd.read_csv(config.QUESTION_FILE)
    print(f"总计 {len(df)} 个问题")

    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    # 断点续传
    results = {}
    if os.path.exists(config.SUBMISSION_FILE):
        existing = pd.read_csv(config.SUBMISSION_FILE)
        for _, row in existing.iterrows():
            results[row["id"]] = row["ret"]
        print(f"已有 {len(results)} 个结果，从断点继续")

    # 筛选未处理的
    todo = [(row["id"], row["question"]) for _, row in df.iterrows() if row["id"] not in results]
    total = len(df)
    print(f"待处理 {len(todo)} 个，并发数 {WORKERS}")

    start_time = time.time()
    lock = threading.Lock()

    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = {
            pool.submit(process_one, engine, qid, q): qid
            for qid, q in todo
        }
        for future in as_completed(futures):
            qid, answer = future.result()
            with lock:
                results[qid] = answer
                done = len(results)
                elapsed = time.time() - start_time
                speed = (done - (total - len(todo))) / max(elapsed, 1)
                remaining = (total - done) / max(speed, 0.01)
                print(f"\r[{done}/{total}] Q{qid} | "
                      f"{elapsed:.0f}s | ~{remaining:.0f}s剩余 | "
                      f"{speed:.1f}题/s", end="", flush=True)

                if done % 20 == 0:
                    _save(results)

    _save(results)
    print(f"\n\n完成！{config.SUBMISSION_FILE}")
    print(f"总耗时: {time.time() - start_time:.0f}s")


if __name__ == "__main__":
    generate()
