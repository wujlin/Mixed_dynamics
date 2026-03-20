#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
META_COLS = ["mid", "user_name", "verify_typ", "publish_time", "content"]
MIN_VALID_DATE = pd.Timestamp("2020-03-01")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="构建统一 Long-COVID 语料并清理明显误命中")
    p.add_argument(
        "--out-dir",
        default="outputs/annotations/derived/unified_long_covid_corpus",
        help="输出目录",
    )
    return p.parse_args()


def _read_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path, dtype=str, low_memory=False, lineterminator="\n", on_bad_lines="skip")
    except Exception:
        return pd.read_csv(path, dtype=str, low_memory=False, on_bad_lines="skip")


def _clean_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns={c: c.lstrip("\ufeff") for c in df.columns})
    if "id" in df.columns and "mid" not in df.columns:
        df = df.rename(columns={"id": "mid"})
    if "ser_name" in df.columns and "user_name" not in df.columns:
        df = df.rename(columns={"ser_name": "user_name"})
    if "text" in df.columns and "content" not in df.columns:
        df = df.rename(columns={"text": "content"})
    if "original_text" in df.columns and "content" not in df.columns:
        df = df.rename(columns={"original_text": "content"})
    for col in META_COLS:
        if col not in df.columns:
            df[col] = ""
    out = df[META_COLS].copy()
    for col in ["mid", "user_name", "verify_typ", "content"]:
        out[col] = out[col].fillna("").astype(str).str.strip()
    out["mid"] = out["mid"].str.replace(r"\.0$", "", regex=True)
    out["publish_time"] = pd.to_datetime(out["publish_time"], errors="coerce")
    return out[out["mid"] != ""].copy()


def _load_jsonl(path: Path) -> pd.DataFrame:
    rows: List[Dict[str, str]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
            except json.JSONDecodeError:
                continue
            mid = str(obj.get("mid", "")).strip()
            if not mid:
                continue
            rows.append(
                {
                    "mid": mid,
                    "emotion_class": str(obj.get("emotion_class", "")).strip(),
                    "risk_class": str(obj.get("risk_class", "")).strip(),
                }
            )
    return pd.DataFrame(rows).drop_duplicates(subset=["mid"], keep="first")


def _best_record(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["quality"] = (
        out["user_name"].ne("").astype(int)
        + out["verify_typ"].ne("").astype(int)
        + out["content"].ne("").astype(int)
        + out["publish_time"].notna().astype(int)
    )
    out = out.sort_values(
        ["mid", "quality", "publish_time"],
        ascending=[True, False, True],
        na_position="last",
    )
    return out.drop_duplicates(subset=["mid"], keep="first").drop(columns=["quality"])


def _recover_topic_raw_records(target_mids: set[str]) -> pd.DataFrame:
    if not target_mids:
        return pd.DataFrame(columns=META_COLS + ["raw_source_files"])

    rows = []
    for path in sorted((ROOT / "dataset/Topic_data").glob("*.csv")):
        df = _clean_cols(_read_csv(path))
        sub = df[df["mid"].isin(target_mids)].copy()
        if len(sub) == 0:
            continue
        sub["raw_source_file"] = path.name
        rows.append(sub)

    if not rows:
        return pd.DataFrame(columns=META_COLS + ["raw_source_files"])

    full = pd.concat(rows, ignore_index=True)
    src = (
        full.groupby("mid")["raw_source_file"]
        .apply(lambda s: "|".join(sorted(set(s.tolist()))))
        .reset_index()
        .rename(columns={"raw_source_file": "raw_source_files"})
    )
    best = _best_record(full.drop(columns=["raw_source_file"]))
    return best.merge(src, on="mid", how="left")


def _nonmedical_exclude_reason(text: str) -> str:
    s = str(text or "")
    rules = [
        (
            "pasc_standards",
            r"太平洋地区标准大会|PASC执行委员会主席|PASC成立于197|一图读懂 \| 太平洋地区标准大会",
        ),
        (
            "pasc_group_inc",
            r"PASC Group Inc|急性后医疗保健公司PASC Group|PACS\)美国IP|PASC Group美国IP",
        ),
        (
            "pasc_music_catalog",
            r"\bPASC\s+\d{2,4}\b",
        ),
        (
            "pasc_fandom",
            r"现pasc|汉服paSC|民国paSC|萨菲罗斯x克劳德超话",
        ),
        (
            "pasc_arts_name",
            r"Pedro Pascal|佩德罗[·・ ]帕斯卡|Mariano Pascual|Progressive Arts Studio Collective|Signal-Return",
        ),
        (
            "metaphor_nonhealth",
            r"我们观众是新冠后遗症|导演跟那什么新冠后遗症一样",
        ),
    ]
    for reason, pat in rules:
        if re.search(pat, s, flags=re.IGNORECASE):
            return reason
    return ""


def _strip_urls(text: str) -> str:
    s = str(text or "")
    s = re.sub(r"https?://\S+", "", s, flags=re.IGNORECASE)
    s = re.sub(r"O网页链接", "", s)
    return s.strip()


def _row_exclude_reason(row: pd.Series) -> str:
    text = str(row.get("content", "") or "")
    reason = _nonmedical_exclude_reason(text)
    if reason:
        return reason

    publish_time = row.get("publish_time")
    if pd.isna(publish_time):
        return "missing_publish_time"
    if pd.Timestamp(publish_time) < MIN_VALID_DATE:
        return "pre_covid_time"

    raw_sources = str(row.get("raw_source_files", "") or "")
    text_no_url = _strip_urls(text)
    long_covid_marker = re.search(
        r"新冠|长新冠|后新冠|long covid|covid|sars-cov-2|pasc|post-acute",
        text_no_url,
        flags=re.IGNORECASE,
    )
    if raw_sources == "PASC.csv" and not long_covid_marker:
        return "pasc_context_free"
    return ""


def main() -> None:
    args = parse_args()
    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    base_meta = _clean_cols(_read_csv(ROOT / "dataset/Topic_data/#新冠后遗症#_filtered.csv"))
    base_meta["corpus_source"] = "master_base"

    official_meta = _clean_cols(_read_csv(ROOT / "dataset/Topic_data/官媒补充_flat.csv"))
    official_meta["corpus_source"] = "master_official"

    batch3_meta = _clean_cols(_read_csv(ROOT / "outputs/annotations/intermediate/to_annotate_batch3_clean.csv"))
    batch3_meta["corpus_source"] = "batch3"

    v2_media_meta = _clean_cols(_read_csv(ROOT / "outputs/annotations/intermediate/to_annotate_v2_media_exposed.csv"))
    v2_media_meta["corpus_source"] = "v2_media_exposed"

    ann_master = _load_jsonl(ROOT / "outputs/annotations/master/long_covid_annotations_master.jsonl")
    ann_batch3 = _load_jsonl(ROOT / "outputs/annotations/batches/batch_03_expanded/new_batch3.jsonl")
    ann_v2_media = _load_jsonl(ROOT / "outputs/annotations/batches/batch_v2_media_exposed/new_v2_media_exposed.jsonl")
    ann_union = pd.concat([ann_master, ann_batch3, ann_v2_media], ignore_index=True).drop_duplicates(subset=["mid"], keep="first")

    meta_union = pd.concat([base_meta, official_meta, batch3_meta, v2_media_meta], ignore_index=True)
    missing_mids = set(ann_union["mid"]) - set(meta_union["mid"])
    recovered = _recover_topic_raw_records(missing_mids)
    if len(recovered):
        recovered["corpus_source"] = "recovered_from_topic_raw"
    else:
        recovered = pd.DataFrame(columns=META_COLS + ["raw_source_files", "corpus_source"])

    meta_all = pd.concat([meta_union, recovered], ignore_index=True, sort=False)
    meta_all = _best_record(meta_all)

    missing_time_mids = set(meta_all.loc[meta_all["publish_time"].isna(), "mid"].tolist())
    recovered_time = _recover_topic_raw_records(missing_time_mids)
    if len(recovered_time):
        recovered_time = recovered_time.set_index("mid")
        meta_all = meta_all.set_index("mid")
        for col in ["user_name", "verify_typ", "publish_time", "content"]:
            meta_all[col] = meta_all[col].combine_first(recovered_time[col])
        if "raw_source_files" not in meta_all.columns:
            meta_all["raw_source_files"] = ""
        meta_all["raw_source_files"] = meta_all["raw_source_files"].combine_first(recovered_time["raw_source_files"])
        meta_all = meta_all.reset_index()

    d = meta_all.merge(ann_union, on="mid", how="inner", validate="one_to_one")
    d["exclude_reason"] = d.apply(_row_exclude_reason, axis=1)
    d["keep_in_unified_corpus"] = d["exclude_reason"].eq("")

    all_path = out_dir / "unified_long_covid_corpus_all.csv"
    kept_path = out_dir / "unified_long_covid_corpus_kept.csv"
    excl_path = out_dir / "unified_long_covid_corpus_excluded.csv"
    ann_all_path = out_dir / "unified_long_covid_annotations_all.jsonl"
    ann_kept_path = out_dir / "unified_long_covid_annotations_kept.jsonl"
    d.to_csv(all_path, index=False, encoding="utf-8-sig")
    d[d["keep_in_unified_corpus"]].to_csv(kept_path, index=False, encoding="utf-8-sig")
    d[~d["keep_in_unified_corpus"]].to_csv(excl_path, index=False, encoding="utf-8-sig")

    ann_cols = ["mid", "emotion_class", "risk_class"]
    with ann_all_path.open("w", encoding="utf-8") as f:
        for row in d[ann_cols].to_dict(orient="records"):
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    with ann_kept_path.open("w", encoding="utf-8") as f:
        for row in d.loc[d["keep_in_unified_corpus"], ann_cols].to_dict(orient="records"):
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "master_base_rows": int(len(base_meta)),
        "master_official_rows": int(len(official_meta)),
        "batch3_meta_rows": int(len(batch3_meta)),
        "v2_media_meta_rows": int(len(v2_media_meta)),
        "annotation_union_rows": int(len(ann_union)),
        "missing_meta_recovered": int(len(recovered)),
        "union_before_cleaning": int(len(d)),
        "excluded_rows": int((~d["keep_in_unified_corpus"]).sum()),
        "excluded_by_reason": {
            k: int(v)
            for k, v in d.loc[~d["keep_in_unified_corpus"], "exclude_reason"].value_counts().to_dict().items()
        },
        "final_unified_corpus_rows": int(d["keep_in_unified_corpus"].sum()),
        "date_min": str(pd.to_datetime(d.loc[d["keep_in_unified_corpus"], "publish_time"], errors="coerce").min()),
        "date_max": str(pd.to_datetime(d.loc[d["keep_in_unified_corpus"], "publish_time"], errors="coerce").max()),
        "output_all_csv": str(all_path),
        "output_kept_csv": str(kept_path),
        "output_excluded_csv": str(excl_path),
        "output_all_annotations_jsonl": str(ann_all_path),
        "output_kept_annotations_jsonl": str(ann_kept_path),
    }
    (out_dir / "unified_long_covid_corpus_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
