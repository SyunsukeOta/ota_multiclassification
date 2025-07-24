import json

def compare_files(file_path1, file_path2, compare_fields=None, exclude_fields=None):
	"""2つのファイルを行単位で比較し、指定したフィールドまたは除外したフィールド以外の異なる箇所を検出する
	
	Args:
		file_path1 (str): 比較対象の1つ目のファイルパス
		file_path2 (str): 比較対象の2つ目のファイルパス
		compare_fields (list, optional): 比較するフィールド名のリスト。指定した場合はこれらのフィールドのみ比較
		exclude_fields (list, optional): 比較から除外するフィールド名のリスト。compare_fieldsが指定されていない場合のみ使用
		
	Returns:
		tuple: (異なる行の番号リスト, 詳細な差分情報)
	"""
	# ファイルを読み込む
	with open(file_path1, 'r', encoding='utf-8') as f1:
		lines1 = f1.readlines()
	
	with open(file_path2, 'r', encoding='utf-8') as f2:
		lines2 = f2.readlines()
	
	# 各ファイルの行数
	len1 = len(lines1)
	len2 = len(lines2)
	
	print(f"{file_path1} の行数: {len1}")
	print(f"{file_path2} の行数: {len2}")
	
	# 比較対象の行数を決定（より少ない方の行数を採用）
	compare_length = min(len1, len2)
	print(f"比較する行数: {compare_length}")
	
	# デフォルト値の設定
	if exclude_fields is None:
		exclude_fields = ['id']  # デフォルトでidを除外
	
	if compare_fields is None:
		mode = "exclude"  # 除外モード
	else:
		mode = "include"  # 包含モード
	
	# 異なる行を保存
	diff_lines = []
	
	# 行単位で比較
	for i in range(compare_length):
		try:
			item1 = json.loads(lines1[i])
			item2 = json.loads(lines2[i])
			
			# 比較対象のフィールドを決定
			if mode == "include":
				# 指定されたフィールドのみを比較
				item1_filtered = {k: v for k, v in item1.items() if k in compare_fields}
				item2_filtered = {k: v for k, v in item2.items() if k in compare_fields}
				field_desc = ', '.join(compare_fields)
			else:
				# 除外フィールド以外を比較
				item1_filtered = {k: v for k, v in item1.items() if k not in exclude_fields}
				item2_filtered = {k: v for k, v in item2.items() if k not in exclude_fields}
				field_desc = f"{'id' if len(exclude_fields) == 1 else ', '.join(exclude_fields)}以外のフィールド"
			
			# フィルタリングしたアイテムを比較
			if item1_filtered != item2_filtered:
				diff_lines.append(i+1)  # 行番号は1から開始
		except json.JSONDecodeError:
			# JSONデコードエラーの場合はスキップ
			continue
	
	# 異なる行の件数を表示
	print(f"\n{field_desc}が異なる行数: {len(diff_lines)} 件")
	
	# 異なる行の詳細情報を取得
	difference_details = []
	for line_num in diff_lines:
		# インデックスは0始まりなので調整
		idx = line_num - 1
		try:
			item1 = json.loads(lines1[idx])
			item2 = json.loads(lines2[idx])
			
			# IDを取得
			id1 = item1.get('id', 'ID not found')
			id2 = item2.get('id', 'ID not found')
			
			# 異なるフィールドを特定
			diff_fields_dict = {}
			
			if mode == "include":
				all_fields = set(compare_fields)
			else:
				all_fields = set(item1.keys()) | set(item2.keys())
				all_fields -= set(exclude_fields)  # 除外フィールドを削除
			
			for field in all_fields:
				val1 = item1.get(field, 'フィールドなし')
				val2 = item2.get(field, 'フィールドなし')
				if val1 != val2:
					# テキストなどの長い値は省略表示
					if isinstance(val1, str) and len(val1) > 50:
						val1 = val1[:50] + "..."
					if isinstance(val2, str) and len(val2) > 50:
						val2 = val2[:50] + "..."
					diff_fields_dict[field] = (val1, val2)
			
			difference_details.append((line_num, id1, id2, diff_fields_dict))
		except json.JSONDecodeError:
			difference_details.append((line_num, 'JSONデコードエラー', 'JSONデコードエラー', {}))
	
	return diff_lines, difference_details

def display_differences(difference_details, limit=10):
	"""差分を表示する
	
	Args:
		difference_details (list): compare_filesが返す詳細情報
		limit (int, optional): 表示する最大件数。デフォルトは10
	"""
	print(f"\n===== フィールドが異なる行の詳細（最初の{limit}件）=====")
	for i, (line_num, id1, id2, diff_fields) in enumerate(difference_details[:limit]):
		print(f"行 {line_num}: dataset1のID={id1}, dataset2のID={id2}")
		print(f"  異なるフィールド:")
		for field, (val1, val2) in diff_fields.items():
			print(f"    {field}: dataset1='{val1}', dataset2='{val2}'")
		print("-" * 80)  # 見やすさのための区切り線

def display_line_numbers(diff_lines, limit=20):
	"""異なる行番号のリストを表示する
	
	Args:
		diff_lines (list): 異なる行番号のリスト
		limit (int, optional): 表示する最大件数。デフォルトは20
	"""
	print("\n===== 異なる行の一覧 =====")
	print(diff_lines[:limit])
	print(f"...他 {len(diff_lines) - limit if len(diff_lines) > limit else 0}件")

if __name__ == "__main__":
	# 比較対象のファイルパス
	file1 = '/work/s245302/multiclassification/data/toxicity_dataset.jsonl'
	file2 = '/work/s245302/multiclassification/data/toxicity_dataset_ver2.jsonl'
	
	# id以外のフィールドの比較
	print("\n=== id以外のフィールドの比較 ===")
	diff_lines, difference_details = compare_files(file1, file2)  # デフォルトでid以外を比較
	display_differences(difference_details)
	display_line_numbers(diff_lines)
