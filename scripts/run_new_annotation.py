"""
对待标注的 CSV 逐条调用已启动的 vLLM/OpenAI 兼容服务，生成标注 JSONL。

用法示例（请根据实际服务修改 base_url/model/api_key）：
    python scripts/run_new_annotation.py \
        --input outputs/annotations/v3/to_annotate.csv \
        --output outputs/annotations/v3/new_official_ann.jsonl \
        --base-url http://10.13.12.164:7890/v1 \
        --api-key abc123 \
        --model Qwen/Qwen3-8B

依赖：pandas；服务需已按 docs/vllm_qwen_setup.md 启动。
"""

from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd
import sys
from pathlib import Path
import os

# 确保能找到 src 包
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.empirical import LLMAnnotator


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="调用 vLLM/OpenAI 服务对 CSV 逐条标注")
    p.add_argument("--input", required=True, help="待标注 CSV，需包含 content 列，可选 mid")
    p.add_argument("--output", required=True, help="输出 JSONL 路径")
    p.add_argument("--base-url", required=True, help="vLLM/OpenAI 服务 base_url，例如 http://10.13.12.164:7890/v1")
    p.add_argument("--api-key", required=True, help="服务 api_key")
    p.add_argument("--model", required=True, help="模型名称，例如 Qwen/Qwen3-8B")
    p.add_argument("--max-tokens", type=int, default=512, help="生成最大 token 数")
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
        
    df = pd.read_csv(args.input)
    if "content" not in df.columns:
        raise ValueError("输入 CSV 需要包含 content 列")
    mids = df["mid"] if "mid" in df.columns else ["" for _ in range(len(df))]

    ann = LLMAnnotator(
        provider="openai",
        api_key=args.api_key,
        model=args.model,
        base_url=args.base_url,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Resume logic: Read existing MIDs
    existing_mids = set()
    if output_path.exists():
        import json
        with output_path.open("r", encoding="utf-8") as f:
            for line in f:
                try:
                    data = json.loads(line)
                    if "mid" in data:
                        existing_mids.add(str(data["mid"]))
                except:
                    pass
        print(f"Found {len(existing_mids)} existing annotations. Resuming...")

    mode = "a" if output_path.exists() else "w"
    with output_path.open(mode, encoding="utf-8") as f_out:
        for idx, (mid, text) in enumerate(zip(mids, df["content"].fillna("")), start=1):
            if str(mid) in existing_mids:
                continue
                
            res = ann.annotate(str(text), max_tokens=args.max_tokens)
            rec = res.to_dict()
            rec["mid"] = mid
            rec["original_text"] = text
            f_out.write(__import__("json").dumps(rec, ensure_ascii=False) + "\n")
            f_out.flush()  # Ensure write to disk
            if idx % 50 == 0:
                print(f"[Progress] {idx}/{len(df)}")

    print(f"完成标注，输出: {output_path}")


if __name__ == "__main__":
    main()
