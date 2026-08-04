# -*- coding: utf-8 -*-

import uuid
import logging
import sys

from config import NO_POLICY_REPLY
from service.rag_index_service import init_index
from service.chat_chain_service import chat, clear_session

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def main():
    print("=" * 60)
    print("  校园政策智能问答系统 — 终端调试模式")
    print("  输入问题开始问答，输入 exit 退出")
    print("=" * 60)
    print()

    # ---- 初始化向量索引 ----
    try:
        print("正在初始化向量索引...")
        init_index()
        print("向量索引初始化完成！\n")
    except FileNotFoundError as e:
        print(f"[错误] 初始化失败: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"[错误] 向量索引初始化失败: {e}")
        logger.exception("索引初始化异常")
        sys.exit(1)

    # ---- 生成会话 ID ----
    session_id = f"sess_{uuid.uuid4().hex[:16]}"
    print(f"当前会话 ID: {session_id}")
    print("-" * 60)

    # ---- 交互循环 ----
    while True:
        try:
            question = input("\n你的问题: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见！")
            break

        if question.lower() == "exit":
            print("再见！")
            break

        if not question:
            print("请输入有效问题。")
            continue

        try:
            answer, source_files = chat(question, session_id)

            # 显示回答
            print(f"\n{'=' * 60}")
            print(f"回答：\n{answer}")

            # 显示来源文件
            if source_files:
                print(f"\n参考文件：")
                for i, f in enumerate(source_files, 1):
                    print(f"  {i}. {f}")
            else:
                print("\n参考文件：无")

            print(f"{'=' * 60}")

        except ValueError as e:
            print(f"\n[配置错误] {e}")
        except RuntimeError as e:
            print(f"\n[服务异常] {e}")
        except Exception as e:
            print(f"\n[未知错误] {e}")
            logger.exception("问答过程异常")


if __name__ == "__main__":
    main()
