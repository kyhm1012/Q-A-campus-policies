import uuid
import logging
import sys
from contextlib import asynccontextmanager
from typing import Optional, List

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from config import API_HOST, API_PORT, NO_POLICY_REPLY
from service.rag_index_service import init_index
from service.chat_chain_service import chat, clear_session

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# 全局标记：索引是否初始化成功
_index_ready = False


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期：启动时初始化向量索引"""
    global _index_ready
    try:
        logger.info("正在初始化向量索引...")
        init_index()
        _index_ready = True
        logger.info("向量索引初始化完成，服务就绪")
    except Exception as e:
        logger.error("向量索引初始化失败: %s", e)
        _index_ready = False
    yield

app = FastAPI(
    title="校园政策RAG智能问答系统",
    description="基于 LlamaIndex + LangChain + DeepSeek 的校园政策智能问答后端",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS 跨域支持，前端不跨域可以不用
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# 请求

class PolicyChatRequest(BaseModel):
    """政策问答请求"""
    question: str
    session_id: Optional[str] = None


class PolicyChatData(BaseModel):
    """政策问答返回数据"""
    answer: str
    source_files: List[str]
    session_id: str


class ClearChatRequest(BaseModel):
    """清空会话请求"""
    session_id: str

#响应

def _success(data=None, msg: str = "success"):
    return {"code": 200, "msg": msg, "data": data}


def _error(msg: str, code: int = 500):
    return {"code": code, "msg": msg, "data": None}


# 接口一：POST /api/v1/chat/policy — 发起政策问答
@app.post("/api/v1/chat/policy")
def policy_chat(request: PolicyChatRequest):
    """
    根据用户提问，RAG 检索校园 TXT 政策知识库，
    结合多轮对话上下文返回答案，并附带引用文档来源。

    - question: 用户问题（必填，不可为空字符串）
    - session_id: 会话 ID（可选，不传则后端自动生成）
    """
    # ---- 参数校验 ----
    if not request.question or not request.question.strip():
        return _error("参数不合法：question 不能为空", 400)

    question = request.question.strip()

    # 生成或使用已有 session_id
    session_id = request.session_id.strip() if request.session_id else f"sess_{uuid.uuid4().hex[:16]}"

    # ---- 索引就绪检查 ----
    if not _index_ready:
        return _error("服务正在初始化，请稍后重试", 500)

    # ---- 执行问答 ----
    try:
        answer, source_files = chat(question, session_id)
        return _success({
            "answer": answer,
            "source_files": source_files,
            "session_id": session_id,
        })
    except ValueError as e:
        # 配置错误（如 API Key 缺失）
        logger.error("配置错误: %s", e)
        return _error(str(e), 500)
    except RuntimeError as e:
        # DeepSeek API 调用失败
        logger.error("服务异常: %s", e)
        return _error(str(e), 500)
    except Exception as e:
        logger.exception("问答过程未知异常")
        return _error("服务异常，请稍后重试", 500)


# 接口二：POST /api/v1/chat/clear — 清空指定会话历史

@app.post("/api/v1/chat/clear")
def clear_chat(request: ClearChatRequest):
    """
    清空指定 session_id 对应的多轮对话记忆，开启全新一轮对话。

    - session_id: 需要清空的会话 ID（必填）
    """
    # ---- 参数校验 ----
    if not request.session_id or not request.session_id.strip():
        return _error("参数不合法：session_id 不能为空", 400)

    session_id = request.session_id.strip()

    # ---- 清空会话 ----
    try:
        clear_session(session_id)
        return {"code": 200, "msg": "当前会话历史已清空", "data": None}
    except Exception as e:
        logger.exception("清空会话异常")
        return _error("服务异常，请稍后重试", 500)


# 健康检查接口（辅助）

@app.get("/health")
def health_check():
    """健康检查端点"""
    return _success({"status": "ok", "index_ready": _index_ready})



if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api_server:app",
        host=API_HOST,
        port=API_PORT,
        reload=True,
    )
