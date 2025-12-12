"""
测试本地到远程 vLLM/OpenAI 兼容服务的连接。

用法示例（根据实际服务修改）：
    # Windows CMD / PowerShell 先设置代理（如需）
    set http_proxy=socks5://127.0.0.1:1080
    set https_proxy=socks5://127.0.0.1:1080
    set ALL_PROXY=socks5://127.0.0.1:1080

    python scripts/test_llm_connection.py ^
        --base-url http://10.13.12.164:7890/v1 ^
        --api-key abc123 ^
        --model Qwen/Qwen3-8B ^
        --message "你好，测试连接是否正常？"
"""

from __future__ import annotations

import argparse
from openai import OpenAI
import os


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="测试远程 vLLM/OpenAI 服务连接")
    p.add_argument("--base-url", required=True, help="服务 base_url，如 http://10.13.12.164:7890/v1")
    p.add_argument("--api-key", required=True, help="服务 api_key")
    p.add_argument("--model", required=True, help="模型名称，如 Qwen/Qwen3-8B")
    p.add_argument("--message", default="你好，测试连接是否正常？", help="测试消息")
    p.add_argument("--max-tokens", type=int, default=50, help="生成最大 token 数")
    p.add_argument("--proxy", help="代理地址，如 socks5://127.0.0.1:1080")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Proxy settings
    if args.proxy:
        os.environ["http_proxy"] = args.proxy
        os.environ["https_proxy"] = args.proxy
        os.environ["ALL_PROXY"] = args.proxy
        print(f"已配置代理: {args.proxy}")
    else:
        # Clear proxy if not specified
        os.environ.pop("http_proxy", None)
        os.environ.pop("https_proxy", None)
        os.environ.pop("ALL_PROXY", None)
        print("未配置代理")

    client = OpenAI(base_url=args.base_url, api_key=args.api_key)
    resp = client.chat.completions.create(
        model=args.model,
        messages=[{"role": "user", "content": args.message}],
        max_tokens=args.max_tokens,
    )
    print("Response:")
    print(resp.choices[0].message.content)


if __name__ == "__main__":
    main()
