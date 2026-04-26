"""FastAPI 服务：POST /chat 接口"""
import time
import uuid
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI
from pydantic import BaseModel

from src.rag import RAGEngine

app = FastAPI(title="多模态客服智能体", version="1.0.0")
engine: RAGEngine = None


class ChatRequest(BaseModel):
    question: str
    images: list = []
    session_id: str = ""


class ChatData(BaseModel):
    answer: str
    session_id: str
    timestamp: int
    image_refs: list = []


class ChatResponse(BaseModel):
    code: int = 0
    msg: str = "success"
    data: ChatData


@app.on_event("startup")
async def startup():
    global engine
    engine = RAGEngine()
    try:
        engine.load()
    except FileNotFoundError:
        print("索引文件不存在，正在构建...")
        engine.build()
    print("服务启动完成")


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """问答接口"""
    session_id = request.session_id or str(uuid.uuid4())

    result = engine.ask(request.question)

    return ChatResponse(
        code=0,
        msg="success",
        data=ChatData(
            answer=result["answer"],
            session_id=session_id,
            timestamp=int(time.time()),
            image_refs=result.get("image_refs", []),
        ),
    )


@app.get("/health")
async def health():
    return {"status": "ok", "chunks": len(engine.kb.chunks) if engine else 0}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api:app", host="0.0.0.0", port=8000, reload=False)
