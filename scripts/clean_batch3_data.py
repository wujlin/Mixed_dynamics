"""
Batch 3 数据清洗脚本
功能：
1. 读取 outputs/annotations/intermediate/to_annotate_batch3.csv
2. 过滤掉长度小于 10 的短文本 (通常是无效信息或纯表情)。
3. 基于 content 去重 (保留第一条)。
   解释：虽然 MID 不同，但内容完全一致通常是转发或营销号刷屏，标注一条即可代表一类。
4. 输出 outputs/annotations/intermediate/to_annotate_batch3_clean.csv
"""

import pandas as pd
import argparse
from pathlib import Path
import sys

# 确保能找到 src 包
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.empirical.text_preprocessor import preprocess_weibo_text, is_valid_for_annotation

def main():
    input_path = "outputs/annotations/intermediate/to_annotate_batch3.csv"
    output_path = "outputs/annotations/intermediate/to_annotate_batch3_clean.csv"
    
    print(f"Loading {input_path}...")
    df = pd.read_csv(input_path)
    original_len = len(df)
    
    print("Applying text preprocessing (removing @, URLs, location, etc.)...")
    # 清洗文本 (保留话题，但去重标签)
    df['content'] = df['content'].fillna("").astype(str).apply(
        lambda x: preprocess_weibo_text(x, max_length=1000, keep_hashtags=True)
    )
    
    # 1. Length & Validity Filter
    # 使用 is_valid_for_annotation 进行更智能的过滤
    df = df[df['content'].apply(lambda x: is_valid_for_annotation(x, min_length=8))]
    after_len_filter = len(df)
    print(f"Filtered invalid/short texts: {original_len - after_len_filter} removed.")
    
    # 2. Content Deduplication (Cleaned content)
    df = df.drop_duplicates(subset=['content'])
    final_len = len(df)
    print(f"Removed cleaning-based duplicates: {after_len_filter - final_len} removed.")
    
    print(f"Final count: {final_len} (Original: {original_len})")
    print(f"Reduction: {100 * (original_len - final_len) / original_len:.2f}%")
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Saved cleaned data to {output_path}")

if __name__ == "__main__":
    main()
