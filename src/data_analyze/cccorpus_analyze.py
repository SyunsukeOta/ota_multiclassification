#!/usr/bin/env python3
import os
import sys
import argparse
import re
from tqdm import tqdm

def get_wet_dict(file_path, max_records=1, filter='jpn'):
	records = []
	current_record = ""
	record_count = 0

	try:
		file_size = os.path.getsize(file_path)
		print(f"ファイル {file_path} のサイズ: {file_size} バイト")
		print(f"ファイル {file_path} から WARC レコードを抽出中...")

		with open(file_path, 'r', encoding='utf-8') as f:
			for line in tqdm(f, total=file_size, unit='B', unit_scale=True):
				if line.startswith("WARC/1.0") and current_record:
					current_dict = {}
					current_dict['record'] = current_record
					set_content_info2dict(current_dict)
					del current_dict['record']
					if 'identified_content_language' in current_dict and 'jpn' in current_dict['identified_content_language']:
						records.append(current_dict)
					record_count += 1
					if max_records and record_count >= max_records:
						break
					current_record = line
				else:
					current_record += line
		return records

	except Exception as e:
		print(f"エラー: {e}")

def set_content_info2dict(record_dict):
	if not record_dict.get('record'):
		return None
	
	set_info2dict(record_dict, content_type='WARC', text_type='Str', info_name='Type')
	set_info2dict(record_dict, content_type='WARC', text_type='Str', info_name='Target-URI')
	set_info2dict(record_dict, content_type='WARC', text_type='Str', info_name='Date')
	set_info2dict(record_dict, content_type='WARC', text_type='Str', info_name='Record-ID')
	set_info2dict(record_dict, content_type='WARC', text_type='Str', info_name='Refers-To')
	set_info2dict(record_dict, content_type='WARC', text_type='Str', info_name='Block-Digest')
	set_info2dict(record_dict, content_type='WARC', text_type='Str', info_name='Identified-Content-Language')
	set_info2dict(record_dict, content_type='Content', text_type='Str', info_name='Type')
	set_info2dict(record_dict, content_type='Content', text_type='Num', info_name='Length')
	set_info2dict(record_dict, content_type='Content', text_type='Content', info_name='Content')
	return record_dict

def set_info2dict(record_dict, content_type='WARC', text_type='Str', info_name=''):
	if not info_name or not record_dict.get('record'):
		return None

	if text_type == 'Str':
		pattern = rf'{content_type}-{info_name}: (.+)'
	elif text_type == 'Num':
		pattern = rf'{content_type}-{info_name}: (\d+)'
	elif text_type == 'Content':
		pattern = r'\r?\n\r?\n(.+)'
	
	if text_type == 'Content':
		match = re.search(pattern, record_dict['record'], re.DOTALL)
	else:	
		match = re.search(pattern, record_dict['record'])
	
	if match:
		dict_key = info_name.lower().replace('-', '_')
		if text_type == 'Str':
			record_dict[dict_key] = match.group(1).strip()
		elif text_type == 'Num':
			record_dict[dict_key] = int(match.group(1))
		elif text_type == 'Content':
			record_dict[dict_key] = match.group(1)

def devide_content(content):
	if content is '':
		return None

	ends_with_period = content.endswith('。')
	arr = content.split('。')
	if arr[-1] == '':
		arr.pop()
	result = [item + '。' for item in arr]
	return result

def main():
	parser = argparse.ArgumentParser(description='WARCファイルからレコードを抽出する')
	parser.add_argument('warc_file', help='処理するWARCファイルのパス')
	parser.add_argument('-o', '--output', help='抽出したレコードを保存するファイル')
	parser.add_argument('-m', '--max-records', type=int, help='処理する最大レコード数')
	
	args = parser.parse_args()
	
	# ファイルの存在確認
	if not os.path.exists(args.warc_file):
		print(f"エラー: ファイル {args.warc_file} が見つかりません")
		return 1

	records = get_wet_dict(args.warc_file, args.max_records)
	for record in records:
		if 'content' in record:
			record['content'] = devide_content(record['content'])
			print(record['content'])
	# print(records)
	print("finished!!")	

	return 0

if __name__ == "__main__":
	sys.exit(main())