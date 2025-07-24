#!/usr/bin/env python3

import json
import random
import sys
import os
from pathlib import Path


JSONL_FILE = "/work/s245302/multiclassification/data/toxicity_dataset_ver2.jsonl"
OUTPUT_DIR_1 = "/work/s245302/multiclassification/analysis_results/sample_texts_ver1"
OUTPUT_DIR_2 = "/work/s245302/multiclassification/analysis_results/sample_texts_ver2"


def save_samples(samples, output_dir):
	"""サンプルのテキストをファイルに保存する関数"""
	os.makedirs(output_dir, exist_ok=True)
	saved_count = 0
	
	for i, sample in enumerate(samples):
		if 'text' in sample:
			sample_id = sample.get('id', f'unknown_{i}')
			output_file = Path(output_dir) / f"sample_{sample_id}.txt"
			
			# テキストファイルに保存
			with open(output_file, 'w', encoding='utf-8') as f:
				f.write(sample['text'])
			
			# 保存したことを通知
			print(f"[サンプル {i+1}] を {output_file} に保存しました")
			saved_count += 1
		else:
			print(f"[サンプル {i+1}] テキストフィールドがありません")
	
	return saved_count


def main():
	# ファイルが存在するか確認
	jsonl_path = Path(JSONL_FILE)
	if not jsonl_path.exists():
		print(f"Error: ファイル {JSONL_FILE} が見つかりません。", file=sys.stderr)
		sys.exit(1)
	
	# データを読み込む
	with open(jsonl_path, 'r', encoding='utf-8') as f:
		data = [json.loads(line) for line in f]
	
	# 行数をカウント
	total_lines = len(data)
	print(f"データセット内の総サンプル数: {total_lines}")
	
	# データを範囲1（1847件目まで）と範囲2（1848件目以降）に分割
	data_ver1 = data[:1847]
	data_ver2 = data[1847:]
	
	print(f"範囲1（1847件目まで）のサンプル数: {len(data_ver1)}")
	print(f"範囲2（1848件目以降）のサンプル数: {len(data_ver2)}")
	
	# 範囲1からランダムに10件選択
	if len(data_ver1) < 10:
		samples_ver1 = data_ver1
		print("注意: 範囲1のサンプル数が10未満のため、すべてのサンプルを使用します。")
	else:
		samples_ver1 = random.sample(data_ver1, 10)
	
	# 範囲2からランダムに10件選択
	if len(data_ver2) < 10:
		samples_ver2 = data_ver2
		print("注意: 範囲2のサンプル数が10未満のため、すべてのサンプルを使用します。")
	else:
		samples_ver2 = random.sample(data_ver2, 10)
	
	# 範囲1のサンプルを保存
	print("\n範囲1（1847件目まで）からのサンプルを保存します:")
	print("----------------------------------------")
	saved_count_1 = save_samples(samples_ver1, OUTPUT_DIR_1)
	print("----------------------------------------")
	print(f"範囲1の処理が完了しました。{saved_count_1}件のファイルを {OUTPUT_DIR_1} に保存しました。")
	
	# 範囲2のサンプルを保存
	print("\n範囲2（1848件目以降）からのサンプルを保存します:")
	print("----------------------------------------")
	saved_count_2 = save_samples(samples_ver2, OUTPUT_DIR_2)
	print("----------------------------------------")
	print(f"範囲2の処理が完了しました。{saved_count_2}件のファイルを {OUTPUT_DIR_2} に保存しました。")
	
	print(f"\n合計: {saved_count_1 + saved_count_2}件のファイルを保存しました。")


if __name__ == "__main__":
	main()