#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Confusion:
    labels: list[str]
    matrix: list[list[int]]  # rows=true, cols=pred

    def total(self) -> int:
        return sum(sum(r) for r in self.matrix)

    def diag(self) -> int:
        return sum(self.matrix[i][i] for i in range(len(self.labels)))

    def row_sums(self) -> list[int]:
        return [sum(r) for r in self.matrix]

    def col_sums(self) -> list[int]:
        n = len(self.labels)
        return [sum(self.matrix[i][j] for i in range(n)) for j in range(n)]


def _normalize_emotion(label: str) -> str:
    v = label.strip().upper()
    alias = {"HIGH": "H", "MEDIUM": "M", "LOW": "L"}
    return alias.get(v, v)


def _normalize_risk(label: str) -> str:
    v = label.strip().lower()
    alias = {"no_risk": "norisk", "no-risk": "norisk", "no risk": "norisk"}
    return alias.get(v, v)


def _read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)


def _build_lookup(rows: list[dict], key: str) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for row in rows:
        mid = (row.get(key) or "").strip()
        if not mid:
            continue
        if mid in out:
            raise ValueError(f"重复 mid：{mid}（文件：{key}）")
        out[mid] = row
    return out


def _confusion(true_labels: list[str], pred_labels: list[str], label_order: list[str]) -> Confusion:
    idx = {lab: i for i, lab in enumerate(label_order)}
    n = len(label_order)
    mat = [[0 for _ in range(n)] for _ in range(n)]
    for t, p in zip(true_labels, pred_labels, strict=True):
        if t not in idx or p not in idx:
            raise ValueError(f"标签超出预期：true={t}, pred={p}, labels={label_order}")
        mat[idx[t]][idx[p]] += 1
    return Confusion(labels=label_order, matrix=mat)


def _precision_recall_f1(conf: Confusion) -> dict[str, dict[str, float]]:
    n = len(conf.labels)
    rows = conf.row_sums()
    cols = conf.col_sums()
    out: dict[str, dict[str, float]] = {}
    for i, lab in enumerate(conf.labels):
        tp = conf.matrix[i][i]
        fp = cols[i] - tp
        fn = rows[i] - tp
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        out[lab] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": float(rows[i]),
        }
    return out


def _accuracy(conf: Confusion) -> float:
    total = conf.total()
    return conf.diag() / total if total else 0.0


def _macro_f1(prf: dict[str, dict[str, float]]) -> float:
    if not prf:
        return 0.0
    return sum(m["f1"] for m in prf.values()) / len(prf)


def _weighted_f1(prf: dict[str, dict[str, float]]) -> float:
    total = sum(m["support"] for m in prf.values())
    if total == 0:
        return 0.0
    return sum(m["f1"] * m["support"] for m in prf.values()) / total


def _cohen_kappa(conf: Confusion) -> float:
    n = conf.total()
    if n == 0:
        return 0.0
    p_o = _accuracy(conf)
    rows = conf.row_sums()
    cols = conf.col_sums()
    p_e = sum((r / n) * (c / n) for r, c in zip(rows, cols, strict=True))
    if math.isclose(1.0 - p_e, 0.0):
        return 0.0
    return (p_o - p_e) / (1.0 - p_e)


def _bootstrap_ci(
    true_labels: list[str],
    pred_labels: list[str],
    label_order: list[str],
    *,
    seed: int,
    n_boot: int,
) -> dict[str, list[float]]:
    if n_boot <= 0:
        return {}
    rng = random.Random(seed)
    n = len(true_labels)
    metrics = {"accuracy": [], "kappa": [], "macro_f1": [], "weighted_f1": []}
    for _ in range(n_boot):
        idxs = [rng.randrange(n) for _ in range(n)]
        t = [true_labels[i] for i in idxs]
        p = [pred_labels[i] for i in idxs]
        conf = _confusion(t, p, label_order)
        prf = _precision_recall_f1(conf)
        metrics["accuracy"].append(_accuracy(conf))
        metrics["kappa"].append(_cohen_kappa(conf))
        metrics["macro_f1"].append(_macro_f1(prf))
        metrics["weighted_f1"].append(_weighted_f1(prf))

    def pct(xs: list[float], q: float) -> float:
        xs_sorted = sorted(xs)
        if not xs_sorted:
            return 0.0
        k = (len(xs_sorted) - 1) * q
        f = math.floor(k)
        c = math.ceil(k)
        if f == c:
            return xs_sorted[int(k)]
        return xs_sorted[f] * (c - k) + xs_sorted[c] * (k - f)

    out: dict[str, list[float]] = {}
    for name, xs in metrics.items():
        out[name] = [pct(xs, 0.025), pct(xs, 0.975)]
    return out


def _format_float(x: float) -> str:
    return f"{x:.3f}"


def _write_confusion_csv(path: Path, conf: Confusion) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["true\\pred", *conf.labels])
        for lab, row in zip(conf.labels, conf.matrix, strict=True):
            writer.writerow([lab, *row])


def _make_report(
    *,
    title: str,
    conf: Confusion,
    prf: dict[str, dict[str, float]],
    ci: dict[str, list[float]] | None,
) -> dict:
    acc = _accuracy(conf)
    kappa = _cohen_kappa(conf)
    macro_f1 = _macro_f1(prf)
    weighted_f1 = _weighted_f1(prf)
    return {
        "title": title,
        "n": conf.total(),
        "accuracy": acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "cohen_kappa": kappa,
        "per_class": prf,
        "confusion": {"labels": conf.labels, "matrix": conf.matrix},
        "bootstrap_ci_95": ci or {},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="计算LLM标注与人工标注的一致性指标（accuracy/F1/kappa）。")
    parser.add_argument("--human-csv", required=True, help="人工标注CSV（mid,text,human_emotion,human_risk,...)")
    parser.add_argument("--llm-csv", required=True, help="LLM key CSV（mid,llm_emotion,llm_risk）")
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
    parser.add_argument("--seed", type=int, default=202501, help="随机种子（用于bootstrap，默认：202501）")
    parser.add_argument("--bootstrap", type=int, default=2000, help="bootstrap次数（0关闭；默认：2000）")
    args = parser.parse_args()

    human_path = Path(args.human_csv)
    llm_path = Path(args.llm_csv)
    if not human_path.exists():
        print(f"[error] human CSV 不存在：{human_path}", file=sys.stderr)
        return 2
    if not llm_path.exists():
        print(f"[error] llm CSV 不存在：{llm_path}", file=sys.stderr)
        return 2

    human_rows = _read_csv(human_path)
    llm_rows = _read_csv(llm_path)
    human_by_mid = _build_lookup(human_rows, "mid")
    llm_by_mid = _build_lookup(llm_rows, "mid")

    mids = sorted(set(human_by_mid.keys()) & set(llm_by_mid.keys()))
    if not mids:
        raise ValueError("human 与 llm 文件 mid 无交集")

    emotion_true: list[str] = []
    emotion_pred: list[str] = []
    risk_true: list[str] = []
    risk_pred: list[str] = []
    joint_correct = 0
    skipped = 0

    for mid in mids:
        h = human_by_mid[mid]
        p = llm_by_mid[mid]
        he = (h.get("human_emotion") or "").strip()
        hr = (h.get("human_risk") or "").strip()
        pe = (p.get("llm_emotion") or "").strip()
        pr = (p.get("llm_risk") or "").strip()
        if not he or not hr:
            skipped += 1
            continue
        he_n = _normalize_emotion(he)
        pe_n = _normalize_emotion(pe)
        hr_n = _normalize_risk(hr)
        pr_n = _normalize_risk(pr)
        emotion_true.append(he_n)
        emotion_pred.append(pe_n)
        risk_true.append(hr_n)
        risk_pred.append(pr_n)
        if he_n == pe_n and hr_n == pr_n:
            joint_correct += 1

    n_used = len(emotion_true)
    if n_used == 0:
        raise ValueError("没有可用样本：请先在 human CSV 中填写 human_emotion/human_risk")

    emotion_labels = ["H", "M", "L"]
    risk_labels = ["risk", "norisk"]

    conf_e = _confusion(emotion_true, emotion_pred, emotion_labels)
    prf_e = _precision_recall_f1(conf_e)
    ci_e = _bootstrap_ci(
        emotion_true,
        emotion_pred,
        emotion_labels,
        seed=args.seed,
        n_boot=args.bootstrap,
    )
    rep_e = _make_report(title="emotion_arousal", conf=conf_e, prf=prf_e, ci=ci_e)

    conf_r = _confusion(risk_true, risk_pred, risk_labels)
    prf_r = _precision_recall_f1(conf_r)
    ci_r = _bootstrap_ci(
        risk_true,
        risk_pred,
        risk_labels,
        seed=args.seed + 1,
        n_boot=args.bootstrap,
    )
    rep_r = _make_report(title="risk_content", conf=conf_r, prf=prf_r, ci=ci_r)

    joint_acc = joint_correct / n_used
    out = {
        "n_total_rows_intersection": len(mids),
        "n_used_with_human_labels": n_used,
        "n_skipped_missing_human_labels": skipped,
        "emotion": rep_e,
        "risk": rep_r,
        "joint_exact_match_accuracy": joint_acc,
        "label_distributions": {
            "human_emotion": dict(Counter(emotion_true)),
            "llm_emotion": dict(Counter(emotion_pred)),
            "human_risk": dict(Counter(risk_true)),
            "llm_risk": dict(Counter(risk_pred)),
        },
    }

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.prefix
    out_json = out_dir / f"{prefix}_metrics.json"
    out_md = out_dir / f"{prefix}_metrics.md"
    out_conf_e = out_dir / f"{prefix}_confusion_emotion.csv"
    out_conf_r = out_dir / f"{prefix}_confusion_risk.csv"

    out_json.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _write_confusion_csv(out_conf_e, conf_e)
    _write_confusion_csv(out_conf_r, conf_r)

    md_lines = []
    md_lines.append(f"# LLM vs Human Validation Metrics ({prefix})")
    md_lines.append("")
    md_lines.append(f"- n used: {n_used} (skipped missing human labels: {skipped})")
    md_lines.append(f"- joint exact-match accuracy (emotion+risk): {_format_float(joint_acc)}")
    md_lines.append("")

    def section(rep: dict, label_order: list[str]) -> None:
        md_lines.append(f"## {rep['title']}")
        md_lines.append(f"- accuracy: {_format_float(rep['accuracy'])}")
        md_lines.append(f"- macro-F1: {_format_float(rep['macro_f1'])}")
        md_lines.append(f"- weighted-F1: {_format_float(rep['weighted_f1'])}")
        md_lines.append(f"- Cohen's kappa: {_format_float(rep['cohen_kappa'])}")
        ci = rep.get("bootstrap_ci_95") or {}
        if ci:
            md_lines.append(
                f"- 95% bootstrap CI (accuracy): [{_format_float(ci['accuracy'][0])}, {_format_float(ci['accuracy'][1])}]"
            )
            md_lines.append(
                f"- 95% bootstrap CI (kappa): [{_format_float(ci['kappa'][0])}, {_format_float(ci['kappa'][1])}]"
            )
        md_lines.append("")
        md_lines.append("| class | precision | recall | F1 | support |")
        md_lines.append("|---|---:|---:|---:|---:|")
        for lab in label_order:
            m = rep["per_class"][lab]
            md_lines.append(
                f"| {lab} | {_format_float(m['precision'])} | {_format_float(m['recall'])} | {_format_float(m['f1'])} | {int(m['support'])} |"
            )
        md_lines.append("")

    section(rep_e, emotion_labels)
    section(rep_r, risk_labels)

    out_md.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    print(f"[ok] metrics: {out_json}")
    print(f"[ok] report: {out_md}")
    print(f"[ok] confusion (emotion): {out_conf_e}")
    print(f"[ok] confusion (risk): {out_conf_r}")
    print(f"[info] n_used={n_used}, joint_accuracy={_format_float(joint_acc)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

