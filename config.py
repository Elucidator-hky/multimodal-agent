"""集中配置文件 — 敏感配置从 .env 读取（避免提交到 git）"""
import os
from pathlib import Path

# 加载 .env (无第三方依赖)
def _load_env():
    env_path = Path(__file__).parent / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))

_load_env()

# HuggingFace 镜像（国内加速）
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

# 项目根目录
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# 数据路径
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
KB_DIR = os.path.join(DATA_DIR, "KnowledgeBase", "手册")
IMAGE_DIR = os.path.join(KB_DIR, "插图")
QUESTION_FILE = os.path.join(DATA_DIR, "question-public.csv")

# 索引存储路径
INDEX_DIR = os.path.join(PROJECT_ROOT, "index")
FAISS_INDEX_PATH = os.path.join(INDEX_DIR, "faiss.index")
CHUNKS_PATH = os.path.join(INDEX_DIR, "chunks.pkl")

# 输出路径
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")
SUBMISSION_FILE = os.path.join(OUTPUT_DIR, "submission.csv")

# Embedding 模型(bge-m3 多语言,1024 维)
EMBEDDING_MODEL = "BAAI/bge-m3"
EMBED_DIM = 1024

# 模型缓存目录(避免反复下载)
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

# Qwen API（阿里云 DashScope OpenAI 兼容接口）
# Key 从环境变量读取，新机器需在 .env 文件中配置 QWEN_API_KEY
QWEN_API_KEY = os.getenv("QWEN_API_KEY", "")
QWEN_BASE_URL = os.getenv("QWEN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
QWEN_MODEL = os.getenv("QWEN_MODEL", "qwen-plus")

# RAG 参数
TOP_K = 5
MAX_CONTEXT_LENGTH = 12000  # 最大上下文字符数
