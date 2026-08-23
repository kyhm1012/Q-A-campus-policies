import os
import sys
from pathlib import Path

# =====================================================
# 0. 彻底接管 nltk 数据路径（必须在所有 import 之前）
# =====================================================
# 设置环境变量，让 nltk 和 huggingface 都使用本地目录
os.environ["NLTK_DATA"] = os.path.join(os.getcwd(), "nltk_data")
os.environ["HF_HOME"] = os.path.join(os.getcwd(), "hf_cache")
os.environ["TRANSFORMERS_CACHE"] = os.path.join(os.getcwd(), "hf_cache")

# 创建目录
for d in [os.environ["NLTK_DATA"], os.environ["HF_HOME"]]:
    os.makedirs(d, exist_ok=True)

# 提前导入 nltk 并设置路径
import nltk
nltk.data.path = [os.environ["NLTK_DATA"]] + nltk.data.path

# 下载必要的语料（静默）
nltk.download('stopwords', download_dir=os.environ["NLTK_DATA"], quiet=True)
nltk.download('punkt', download_dir=os.environ["NLTK_DATA"], quiet=True)

# =====================================================
# 1. 项目路径设置
# =====================================================
root_dir = Path(__file__).parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

# =====================================================
# 2. 其他 import
# =====================================================
import streamlit as st
import uuid
from service.chat_chain_service import chat
from service.rag_index_service import init_index

# ... 其余代码保持不变 ...

# ======================================
# 2. 页面配置
# ======================================
st.set_page_config(
    page_title="校园政策智能问答",
    page_icon="🏫",
    layout="centered"
)

st.title("🏫 安徽工程大学校园政策智能问答助手")
st.caption("基于RAG技术的校园政策问答系统")


# ======================================
# 3. 初始化索引（只执行一次）
# ======================================
@st.cache_resource
def load_rag_index():
    with st.spinner("📚 正在加载政策文档索引，请稍候..."):
        # 尝试多个可能的位置
        possible_paths = [
            Path(__file__).parent.parent / "data",
            Path.cwd() / "data",
            Path("/mount/src/q-a-campus-policies/data")
        ]

        data_dir = None
        for path in possible_paths:
            if path.exists():
                data_dir = path
                break

        if data_dir is None:
            st.error("❌ 找不到 data 文件夹，请确认部署文件结构")
            st.write("尝试过的路径：")
            for p in possible_paths:
                st.write(f"- {p}")
            return False

        try:
            init_index()
            return True
        except Exception as e:
            st.error(f"❌ 索引加载失败：{str(e)}")
            return False


if "index_ready" not in st.session_state:
    st.session_state.index_ready = load_rag_index()

if not st.session_state.index_ready:
    st.stop()

# ======================================
# 4. 初始化会话状态
# ======================================
if "messages" not in st.session_state:
    st.session_state.messages = []
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

# ======================================
# 5. 侧边栏
# ======================================
with st.sidebar:
    st.header("⚙️ 设置")
    if st.button("🗑️ 清空对话", use_container_width=True):
        st.session_state.messages = []
        st.session_state.session_id = str(uuid.uuid4())
        st.rerun()
    st.divider()
    st.caption("💡 本系统基于校园政策文档提供问答服务")

# ======================================
# 6. 显示历史消息
# ======================================
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "sources" in message and message["sources"]:
            with st.expander("📚 参考文献"):
                for src in message["sources"]:
                    st.write(f"- {src}")

# ======================================
# 7. 示例问题快捷按钮
# ======================================
if not st.session_state.messages:
    st.info("💡 试试问这些问题：")
    col1, col2, col3 = st.columns(3)
    examples = ["如何申请国家奖学金？", "转专业需要什么条件？", "选课流程是怎样的？"]
    for col, example in zip([col1, col2, col3], examples):
        with col:
            if st.button(example, use_container_width=True):
                st.session_state["input_value"] = example
                st.rerun()

# ======================================
# 8. 输入框 & 问答逻辑
# ======================================
input_value = st.session_state.pop("input_value", None)

if input_value:
    prompt = input_value
else:
    prompt = st.chat_input("请输入您的政策问题...")

if prompt:
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        with st.spinner("🤔 正在查询政策库..."):
            try:
                result = chat(prompt, st.session_state.session_id)

                # 调试信息
                st.write("📦 **调试信息**：数据类型：", type(result))
                st.write("📦 **内容**：", result)

                # 兼容不同的返回格式
                if isinstance(result, dict):
                    answer = result.get("answer", "未找到相关答案")
                    sources = result.get("source_files", [])
                    if result.get("session_id"):
                        st.session_state.session_id = result.get("session_id")
                elif isinstance(result, (list, tuple)):
                    if len(result) >= 1:
                        answer = result[0] if result[0] else "未找到相关答案"
                    else:
                        answer = "未找到相关答案"
                    sources = result[1] if len(result) > 1 else []
                    if len(result) > 2 and result[2]:
                        st.session_state.session_id = result[2]
                else:
                    answer = f"⚠️ 返回格式异常（{type(result)}）"
                    sources = []

            except Exception as e:
                answer = f"❌ 查询失败：{str(e)}"
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

# ======================================
# 9. 页脚
# ======================================
st.divider()
st.caption("© 2026 安徽工程大学 · 基于RAG的校园政策智能问答系统")