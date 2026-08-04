import os
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent

# 加载 .env 配置文件
load_dotenv(BASE_DIR / ".env")

def _resolve_path(path_str: str) -> str:
    """把 .env 中的相对路径转成绝对路径，确保任意目录下启动都能正确定位"""
    p = Path(path_str)
    if not p.is_absolute():
        p = BASE_DIR / p
    return str(p)

# DeepSeek LLM 配置
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "").strip()
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1").strip()
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat").strip()

# 将密钥写入环境变量，供 langchain-deepseek 自动读取
if DEEPSEEK_API_KEY:
    os.environ["DEEPSEEK_API_KEY"] = DEEPSEEK_API_KEY
    os.environ["DEEPSEEK_API_BASE"] = DEEPSEEK_BASE_URL

# LLM 温度：0.1 降低随机性，保证回答严谨
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.1"))

# Embedding 向量模型配置
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "BAAI/bge-small-zh-v1.5").strip()
MODEL_CACHE = _resolve_path(os.getenv("MODEL_CACHE", "./.model_cache"))

# HuggingFace 镜像源
# 必须在 import sentence-transformers / HuggingFaceEmbedding 之前设置生效
_hf_endpoint = os.getenv("HF_ENDPOINT", "https://hf-mirror.com").strip()
if _hf_endpoint:
    os.environ["HF_ENDPOINT"] = _hf_endpoint
    os.environ["HF_HOME"] = MODEL_CACHE

# 数据源配置
DATA_FOLDER = _resolve_path(os.getenv("DATA_FOLDER", "./data/clean"))

# 向量存储配置
VECTOR_STORE_PATH = _resolve_path(os.getenv("VECTOR_STORE_PATH", "./vector_store"))
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "campus_policy").strip()

# 检索配置
SIMILARITY_TOP_K = int(os.getenv("SIMILARITY_TOP_K", "5"))
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "512"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))

# 检索不到任何相关内容时的固定回复
NO_POLICY_REPLY = "暂无相关校园政策规定，请咨询学校行政部门"

# FastAPI 服务配置
API_HOST = os.getenv("API_HOST", "0.0.0.0").strip()
API_PORT = int(os.getenv("API_PORT", "8000"))
