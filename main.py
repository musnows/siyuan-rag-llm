import os

# 设置tokenizers并行化以避免警告
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def main():
    print("Hello from siyuan-rag-llm!")


if __name__ == "__main__":
    main()
