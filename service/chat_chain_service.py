import logging
import warnings
from typing import Tuple, List

# LangChain 1.x 中 ConversationBufferMemory 已迁移到 langchain_classic 包
# 屏蔽 deprecation 提示（该类在 2.0 才移除，当前仍可正常使用）
warnings.filterwarnings("ignore", message=".*ConversationBufferMemory.*deprecated.*")
warnings.filterwarnings("ignore", message=".*langchain-community.*sunset.*")

from langchain_deepseek import ChatDeepSeek
from langchain_classic.memory import ConversationBufferMemory

from config import (
    DEEPSEEK_API_KEY,
    DEEPSEEK_BASE_URL,
    DEEPSEEK_MODEL,
    LLM_TEMPERATURE,
    NO_POLICY_REPLY,
)
from service.rag_index_service import retrieve

logger = logging.getLogger(__name__)

# 系统提示词模板

SYSTEM_PROMPT = """你是一个专业的校园政策智能问答助手。请严格根据下方【检索到的政策文档内容】回答用户的问题。

回答规则：
1. 只能基于【检索到的政策文档内容】进行回答，禁止编造、臆测任何政策条款或规定
2. 如果检索到的内容与用户问题无关、无法回答用户问题、或检索内容为空，请直接回复：{no_policy_reply}
3. 回答应当条理清晰，涉及流程、步骤、条件等内容时请使用编号分点列出
4. 如果用户追问上文内容（如"刚才说的流程可以线上办理吗"），请结合对话历史理解指代关系后作答
5. 不要透露"检索到以下内容"等内部结构信息，以自然流畅的语言直接回答

【检索到的政策文档内容】
{context}

【对话历史】
{history}

用户问题：{question}
请回答："""


# 会话管理（内存存储，按 session_id 隔离）
_sessions: dict = {}

#创建LLM实例
def _create_llm() -> ChatDeepSeek:
    if not DEEPSEEK_API_KEY:
        raise ValueError(
            "DeepSeek API Key 未配置！请在项目根目录 .env 文件中填写 DEEPSEEK_API_KEY=sk-xxxx"
        )

    return ChatDeepSeek(
        model=DEEPSEEK_MODEL,
        api_key=DEEPSEEK_API_KEY,
        base_url=DEEPSEEK_BASE_URL,
        temperature=LLM_TEMPERATURE,
    )

#获取或创建指定 session_id 的对话会话
def _get_or_create_session(session_id: str) -> dict:
    if session_id not in _sessions:
        logger.info("创建新会话: %s", session_id)
        _sessions[session_id] = {
            "llm": _create_llm(),
            "memory": ConversationBufferMemory(
                memory_key="history",
                human_prefix="用户",
                ai_prefix="AI",
            ),
        }
    return _sessions[session_id]

#RAG链路
def chat(question: str, session_id: str) -> Tuple[str, List[str]]:
    session = _get_or_create_session(session_id)
    memory = session["memory"]
    llm = session["llm"]

    # ---- 1. RAG 检索 ----
    context, source_files = retrieve(question)

    # ---- 2. 获取对话历史 ----
    history_dict = memory.load_memory_variables({})
    history = history_dict.get("history", "").strip()
    if not history:
        history = "（暂无对话历史）"

    # ---- 3. 组装提示词 ----
    prompt = SYSTEM_PROMPT.format(
        no_policy_reply=NO_POLICY_REPLY,
        context=context if context else "（未检索到任何相关内容）",
        history=history,
        question=question,
    )

    # ---- 4. 调用 LLM 生成回答 ----
    try:
        response = llm.invoke(prompt)
        answer = response.content.strip()
    except Exception as e:
        error_msg = str(e).lower()
        # 友好提示常见错误
        if "authentication" in error_msg or "401" in error_msg or "api key" in error_msg:
            raise RuntimeError(
                "DeepSeek API 密钥无效或已过期，请检查 .env 文件中的 DEEPSEEK_API_KEY"
            ) from e
        elif "connection" in error_msg or "timeout" in error_msg or "unreachable" in error_msg:
            raise RuntimeError(
                "网络请求失败，无法连接到 DeepSeek API，请检查网络连接后重试"
            ) from e
        elif "rate limit" in error_msg or "429" in error_msg:
            raise RuntimeError(
                "DeepSeek API 调用频率超限，请稍后重试"
            ) from e
        else:
            raise RuntimeError(f"DeepSeek API 调用失败: {e}") from e

    # ---- 5. 更新对话记忆 ----
    memory.save_context({"input": question}, {"output": answer})

    # ---- 6. 回答规则校验 ----
    # 如果 LLM 判断无相关政策，清空来源文件列表
    if NO_POLICY_REPLY in answer:
        source_files = []

    logger.info("会话 %s 问答完成，引用文件: %s", session_id, source_files)
    return answer, source_files

#清空指定 session_id 的对话历史。
def clear_session(session_id: str) -> bool:
    if session_id in _sessions:
        del _sessions[session_id]
        logger.info("会话 %s 历史已清空", session_id)
        return True
    logger.info("会话 %s 不存在，无需清空", session_id)
    return False
