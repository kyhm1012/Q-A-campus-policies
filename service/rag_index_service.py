import os
import logging
from typing import Tuple, List

from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, StorageContext, Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.qdrant import QdrantVectorStore
import qdrant_client

from config import (
    EMBEDDING_MODEL_NAME,
    MODEL_CACHE,
    DATA_FOLDER,
    VECTOR_STORE_PATH,
    COLLECTION_NAME,
    SIMILARITY_TOP_K,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
)

logger = logging.getLogger(__name__)

# 全局单例：Embedding 模型、向量索引
_embed_model = None
_index = None

#Embedding 模型实例。第一次使用自动下载并缓存到 .model_cache 文件夹
def _get_embed_model() -> HuggingFaceEmbedding:

    global _embed_model
    if _embed_model is None:
        # 确保缓存目录存在
        os.makedirs(MODEL_CACHE, exist_ok=True)
        logger.info("正在加载 Embedding 模型: %s（首次运行需下载，请耐心等待）", EMBEDDING_MODEL_NAME)
        _embed_model = HuggingFaceEmbedding(
            model_name=EMBEDDING_MODEL_NAME,
            cache_folder=MODEL_CACHE,
        )
        logger.info("Embedding 模型加载完成")
    return _embed_model


def _check_data_folder() -> None:
    """检查数据文件夹是否存在且包含 TXT 文件"""
    if not os.path.exists(DATA_FOLDER):
        raise FileNotFoundError(f"数据文件夹不存在: {DATA_FOLDER}，请确认路径配置正确")

    txt_files = [f for f in os.listdir(DATA_FOLDER) if f.lower().endswith(".txt")]
    if not txt_files:
        raise FileNotFoundError(f"数据文件夹中没有 TXT 文件: {DATA_FOLDER}，请放入校园政策文档")


def _collection_exists(client: qdrant_client.QdrantClient) -> bool:
    """检查 Qdrant 中是否已存在指定 collection"""
    try:
        collections = client.get_collections()
        return any(c.name == COLLECTION_NAME for c in collections.collections)
    except Exception:
        return False

#构建全新的向量索引：加载文档 -> 文本分割 -> 生成向量 -> 存入 Qdrant
def _build_index_from_documents(
    vector_store: QdrantVectorStore,
    embed_model: HuggingFaceEmbedding,
) -> VectorStoreIndex:

    logger.info("开始从文档构建向量索引（首次运行，可能需要几分钟）...")

    # 1. 加载所有 TXT 文档（SimpleDirectoryReader 默认使用 UTF-8 编码读取）
    documents = SimpleDirectoryReader(
        input_dir=DATA_FOLDER,
        required_exts=[".txt"],
    ).load_data()
    logger.info("已加载 %d 个政策文档", len(documents))

    # 2. 文本节点分割
    splitter = SentenceSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    nodes = splitter.get_nodes_from_documents(documents)
    logger.info("文本分割完成，共生成 %d 个节点", len(nodes))

    # 3. 构建索引并存入 Qdrant
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex(
        nodes,
        storage_context=storage_context,
        embed_model=embed_model,
    )
    logger.info("向量索引构建完成，已持久化到: %s", VECTOR_STORE_PATH)
    return index

#初始化向量索引
def init_index() -> VectorStoreIndex:

    global _index
    if _index is not None:
        return _index

    # 前置检查
    _check_data_folder()

    # 确保向量存储目录存在
    os.makedirs(VECTOR_STORE_PATH, exist_ok=True)

    # 初始化 Embedding 模型
    embed_model = _get_embed_model()

    # 配置 LlamaIndex 全局设置（仅用 Embedding，不用 LLM —— LLM 由 LangChain 负责）
    Settings.embed_model = embed_model

    # 初始化 Qdrant 本地文件模式客户端
    client = qdrant_client.QdrantClient(path=VECTOR_STORE_PATH)
    vector_store = QdrantVectorStore(
        client=client,
        collection_name=COLLECTION_NAME,
    )

    if _collection_exists(client):
        # 向量库已存在，直接加载
        logger.info("检测到已有向量库，正在加载: %s", VECTOR_STORE_PATH)
        _index = VectorStoreIndex.from_vector_store(
            vector_store,
            embed_model=embed_model,
        )
        logger.info("向量库加载完成")
    else:
        # 首次运行，构建索引
        _index = _build_index_from_documents(vector_store, embed_model)

    return _index

#根据用户问题检索片段
def retrieve(question: str, top_k: int = SIMILARITY_TOP_K) -> Tuple[str, List[str]]:
    """
    Args:
        question: 用户问题
        top_k: 召回的节点数量
    """
    index = init_index()
    retriever = index.as_retriever(similarity_top_k=top_k)
    nodes = retriever.retrieve(question)

    context_parts = []
    source_files = []

    for node in nodes:
        text = node.get_content().strip()
        if not text:
            continue
        # 提取来源文件名
        file_name = node.metadata.get("file_name", "未知来源")
        context_parts.append(text)
        if file_name not in source_files:
            source_files.append(file_name)

    context = "\n\n".join(context_parts)
    return context, source_files
