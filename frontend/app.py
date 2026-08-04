import streamlit as st
import requests

# ========== 页面配置 ==========
st.set_page_config(
    page_title="校园政策智能问答",
    page_icon="🏫",
    layout="centered"
)

st.title("🏫 安徽工程大学校园政策智能问答助手")
st.caption("基于RAG技术的校园政策问答系统")

# ========== 初始化会话状态 ==========
if "messages" not in st.session_state:
    st.session_state.messages = []
if "session_id" not in st.session_state:
    st.session_state.session_id = None

# ========== 侧边栏 ==========
with st.sidebar:
    st.header("⚙️ 设置")
    if st.button("🗑️ 清空对话", use_container_width=True):
        st.session_state.messages = []
        st.session_state.session_id = None
        st.rerun()
    st.divider()
    st.caption("💡 本系统基于校园政策文档提供问答服务")

# ========== 显示历史消息 ==========
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "sources" in message and message["sources"]:
            with st.expander("📚 参考文献"):
                for src in message["sources"]:
                    st.write(f"- {src}")

# ========== 示例问题快捷按钮 ==========
if not st.session_state.messages:
    st.info("💡 试试问这些问题：")
    col1, col2, col3 = st.columns(3)
    examples = ["如何申请国家奖学金？", "转专业需要什么条件？", "选课流程是怎样的？"]
    for col, example in zip([col1, col2, col3], examples):
        with col:
            if st.button(example, use_container_width=True):
                st.session_state["input_value"] = example
                st.rerun()

# ========== 输入框 ==========
BACKEND_URL = "http://localhost:8000"

if prompt := st.chat_input("请输入您的政策问题..."):
    # 1. 显示用户消息
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # 2. 调用后端API
    with st.chat_message("assistant"):
        with st.spinner("🤔 正在查询政策库..."):
            try:
                response = requests.post(
                    f"{BACKEND_URL}/api/v1/chat/policy",
                    json={
                        "question": prompt,
                        "session_id": st.session_state.session_id
                    },
                    timeout=30
                )
                response.raise_for_status()
                data = response.json()

                if data.get("code") == 200:
                    result = data["data"]
                    answer = result.get("answer", "未找到相关答案")
                    sources = result.get("source_files", [])
                    session_id = result.get("session_id")
                    if session_id:
                        st.session_state.session_id = session_id
                else:
                    answer = f"❌ 服务错误：{data.get('msg', '未知错误')}"
                    sources = []

            except requests.exceptions.ConnectionError:
                answer = "❌ 无法连接到后端服务，请确认 `api_server.py` 已启动"
                sources = []
            except requests.exceptions.Timeout:
                answer = "❌ 请求超时，请稍后重试"
                sources = []
            except Exception as e:
                answer = f"❌ 请求失败：{str(e)}"
                sources = []

        st.markdown(answer)
        if sources:
            with st.expander("📚 参考文献"):
                for src in sources:
                    st.write(f"- {src}")

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "sources": sources
        })

# ========== 页脚 ==========
st.divider()
st.caption("© 2026 安徽工程大学 · 基于RAG的校园政策智能问答系统")