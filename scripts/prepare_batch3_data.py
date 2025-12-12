"""
准备 Batch 3 待标注数据
功能：
1. 扫描 dataset/Topic_data 下所有 CSV 文件。
2. 排除已知的已处理文件 (merged_topic_official.csv, #新冠后遗症#_filtered.csv 等)。
3. 合并剩余所有 CSV。
4. 读取 outputs/annotations/master/long_covid_annotations_master.jsonl 获取已标注 MID。
5. 过滤掉已标注的样本。
6. 输出 outputs/annotations/intermediate/to_annotate_batch3.csv。
"""

import pandas as pd
import glob
import os
import json
import argparse
from pathlib import Path

def load_master_mids(jsonl_path):
    mids = set()
    if not os.path.exists(jsonl_path):
        print(f"Warning: Master file {jsonl_path} not found. Assuming no existing annotations.")
        return mids
    
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line)
                if 'mid' in data and data['mid']:
                    mids.add(str(data['mid']))
            except:
                pass
    return mids

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="dataset/Topic_data")
    parser.add_argument("--master_ann", default="outputs/annotations/master/long_covid_annotations_master.jsonl")
    parser.add_argument("--output", default="outputs/annotations/intermediate/to_annotate_batch3.csv")
    args = parser.parse_args()

    # 1. Load Master MIDs
    master_mids = load_master_mids(args.master_ann)
    print(f"Loaded {len(master_mids)} existing MIDs from master.")

    # 2. Scan CSVs
    all_files = glob.glob(os.path.join(args.data_dir, "*.csv"))
    
    # Exclude list
    excludes = [
        "merged_topic_official.csv",
        "官媒补充_flat.csv",
        "#新冠后遗症#_filtered.csv"
    ]
    
    dfs = []
    for f in all_files:
        fname = os.path.basename(f)
        if fname in excludes:
            continue
            
        try:
            # 尝试读取，处理可能的编码或坏行
            # 微博数据通常含有换行符，需特别注意
            df = pd.read_csv(f, lineterminator='\n', on_bad_lines='skip')
            
            # 标准化列名 (有些文件可能叫 id 或 mid)
            if 'id' in df.columns and 'mid' not in df.columns:
                df.rename(columns={'id': 'mid'}, inplace=True)
            
            if 'mid' in df.columns:
                # 确保 mid 是 string
                df['mid'] = df['mid'].astype(str)
                dfs.append(df)
                print(f"Loaded {fname}: {len(df)} rows")
            else:
                print(f"Skipping {fname}: No 'mid' column")
        except Exception as e:
            print(f"Error reading {fname}: {e}")

    if not dfs:
        print("No new data found.")
        return

    # 3. Merge
    full_df = pd.concat(dfs, ignore_index=True)
    print(f"Total raw rows: {len(full_df)}")

    # 4. Dedup internally
    full_df.drop_duplicates(subset=['mid'], inplace=True)
    print(f"Unique new rows: {len(full_df)}")

    # 5. Filter existing
    new_df = full_df[~full_df['mid'].isin(master_mids)].copy()
    print(f"New samples to annotate: {len(new_df)}")

    # 6. Ensure required columns
    required_cols = ["mid", "user_name", "verify_typ", "publish_time", "content"]
    # 填充缺失列
    for col in required_cols:
        if col not in new_df.columns:
            # 尝试映射 text -> content
            if col == 'content' and 'text' in new_df.columns:
                new_df['content'] = new_df['text']
            elif col == 'content' and '微博正文' in new_df.columns:
                new_df['content'] = new_df['微博正文']
            # 尝试映射 微博id -> mid
            elif col == 'mid' and '微博id' in new_df.columns:
                new_df['mid'] = new_df['微博id']
            # 尝试映射 发布时间 -> publish_time
            elif col == 'publish_time' and '发布时间' in new_df.columns:
                new_df['publish_time'] = new_df['发布时间']
            # 尝试映射 用户名称 -> user_name
            elif col == 'user_name' and '用户名称' in new_df.columns:
                new_df['user_name'] = new_df['用户名称']
            else:
                new_df[col] = ""

    # 保存
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    new_df[required_cols].to_csv(args.output, index=False)
    print(f"Saved to {args.output}")

if __name__ == "__main__":
    main()
