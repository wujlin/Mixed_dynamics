#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class SampleRow:
    mid: str
    text: str
    llm_emotion: str
    llm_risk: str


def _iter_jsonl(paths: list[Path]) -> Iterable[dict]:
    for path in paths:
        with path.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    print(
                        f"[warn] JSON decode failed: {path}:{line_no}",
                        file=sys.stderr,
                    )


def _extract_row(obj: dict) -> SampleRow | None:
    mid = obj.get("mid")
    if mid is None:
        return None
    text = obj.get("original_text") or obj.get("text") or ""
    llm_emotion = obj.get("emotion_class")
    llm_risk = obj.get("risk_class")
    if not text or llm_emotion is None or llm_risk is None:
        return None
    return SampleRow(
        mid=str(mid),
        text=str(text).replace("\r\n", "\n").replace("\r", "\n"),
        llm_emotion=str(llm_emotion),
        llm_risk=str(llm_risk),
    )


def _reservoir_sample(rows: Iterable[SampleRow], n: int, seed: int) -> list[SampleRow]:
    rng = random.Random(seed)
    sample: list[SampleRow] = []
    seen = 0
    for row in rows:
        seen += 1
        if len(sample) < n:
            sample.append(row)
            continue
        j = rng.randrange(seen)
        if j < n:
            sample[j] = row
    if len(sample) < n:
        raise ValueError(f"样本不足：需要 n={n}，但只有 {len(sample)} 条可用记录")
    return sample


def _write_human_csv(path: Path, rows: list[SampleRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "mid",
                "text",
                "human_emotion",
                "human_risk",
                "notes",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "mid": row.mid,
                    "text": row.text,
                    "human_emotion": "",
                    "human_risk": "",
                    "notes": "",
                }
            )


def _write_llm_key_csv(path: Path, rows: list[SampleRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "mid",
                "llm_emotion",
                "llm_risk",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "mid": row.mid,
                    "llm_emotion": row.llm_emotion,
                    "llm_risk": row.llm_risk,
                }
            )


def _write_meta_json(path: Path, *, inputs: list[Path], n: int, seed: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "inputs": [str(p) for p in inputs],
        "n": n,
        "seed": seed,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="从LLM标注JSONL中抽取盲评人工验证样本（输出 human.csv + llm_key.csv）。"
    )
    parser.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help="一个或多个JSONL路径（需包含 mid, original_text/text, emotion_class, risk_class）",
    )
    parser.add_argument("--n", type=int, default=300, help="抽样数量（默认：300）")
    parser.add_argument("--seed", type=int, default=202501, help="随机种子（默认：202501）")
    parser.add_argument(
        "--out-dir",
        default="outputs/llm_validation",
        help="输出目录（默认：outputs/llm_validation）",
    )
    parser.add_argument(
        "--prefix",
        default="qwen3_validation",
        help="输出文件名前缀（默认：qwen3_validation）",
    )
    args = parser.parse_args()

    inputs = [Path(p) for p in args.inputs]
    for p in inputs:
        if not p.exists():
            print(f"[error] 输入文件不存在：{p}", file=sys.stderr)
            return 2

    out_dir = Path(args.out_dir)
    prefix = args.prefix
    out_human = out_dir / f"{prefix}_human.csv"
    out_llm = out_dir / f"{prefix}_llm_key.csv"
    out_meta = out_dir / f"{prefix}_meta.json"

    seen_mid: set[str] = set()

    def rows_stream() -> Iterable[SampleRow]:
        for obj in _iter_jsonl(inputs):
            row = _extract_row(obj)
            if row is None:
                continue
            if row.mid in seen_mid:
                continue
            seen_mid.add(row.mid)
            yield row

    rows = _reservoir_sample(rows_stream(), n=args.n, seed=args.seed)
    rows_sorted = sorted(rows, key=lambda r: r.mid)

    _write_human_csv(out_human, rows_sorted)
    _write_llm_key_csv(out_llm, rows_sorted)
    _write_meta_json(out_meta, inputs=inputs, n=args.n, seed=args.seed)

    print(f"[ok] human 模板：{out_human}")
    print(f"[ok] llm key：{out_llm}")
    print(f"[ok] meta：{out_meta}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

