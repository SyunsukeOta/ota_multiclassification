import json
from collections import defaultdict
import matplotlib.pyplot as plt
import os
from pathlib import Path

def count_characters_in_jsonl(file_path, search_char='。', start=0, end=None):
    """
    JSONLファイルの指定範囲から特定の文字の数をカウントする関数
    
    Args:
        file_path (str): JSONLファイルのパス
        search_char (str): カウントする文字（デフォルトは句点「。」）
        start (int): 開始行番号（0から始まる）
        end (int): 終了行番号（この行は含まない）
    
    Returns:
        dict: 文字数ごとのカウント結果
    """
    # 文字数をカウントするための辞書を初期化
    char_counts = defaultdict(int)
    
    # 行カウンタ
    line_count = 0
    
    # JSONLファイルを1行ずつ読み込む
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            # 指定範囲外の行はスキップ
            if line_count < start:
                line_count += 1
                continue
            
            if end is not None and line_count >= end:
                break
                
            try:
                # JSONとしてパース
                data = json.loads(line)
                
                # textフィールドがある場合のみ処理
                if 'text' in data:
                    # 指定された文字の数をカウント
                    char_count = data['text'].count(search_char)
                    
                    # 0-99は20単位、100以上は100単位でバケット作成
                    if char_count < 100:
                        bucket = (char_count // 20) * 20
                        bucket_label = f"{bucket}~{bucket+19}"
                    else:
                        bucket = (char_count // 100) * 100
                        bucket_label = f"{bucket}~{bucket+99}"
                        
                    # 辞書にカウントを追加
                    char_counts[bucket_label] += 1
            except json.JSONDecodeError:
                # JSONデコードエラーの場合はスキップ
                continue
                
            line_count += 1
    
    # 通常の辞書に変換して返す
    return dict(char_counts)

def plot_character_count_distribution(count_dict, output_path, title_suffix="", char_name="Periods"):
    """
    文字数の分布をグラフ化して保存する関数
    
    Args:
        count_dict (dict): カウント結果の辞書
        output_path (str): 出力ファイルパス
        title_suffix (str): グラフタイトルの接尾辞
        char_name (str): 文字の名前（グラフタイトルと軸ラベルに使用）
    """
    # 出力ディレクトリが存在しない場合は作成
    output_dir = os.path.dirname(output_path)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # グラフのフィギュアとaxesを作成
    plt.figure(figsize=(12, 6))
    
    # バケットを適切にソート
    def get_bucket_start(bucket):
        return int(bucket.split('~')[0])
    
    buckets = sorted(count_dict.keys(), key=get_bucket_start)
    counts = [count_dict[bucket] for bucket in buckets]
    
    # X軸のラベルを作成
    bucket_positions = range(len(buckets))
    
    # 棒グラフを作成
    plt.bar(bucket_positions, counts, color='skyblue', width=0.6)
    
    # X軸のラベルを設定
    plt.xticks(bucket_positions, buckets, rotation=45, ha='right')
    
    # グラフのタイトルと軸ラベルを設定
    plt.title(f'Distribution of {char_name} per Text (0-99: 20 units, 100+: 100 units) {title_suffix}', fontsize=15)
    plt.xlabel(f'Number of {char_name}', fontsize=12)
    plt.ylabel('Number of Texts', fontsize=12)
    
    # グリッド線の表示
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # レイアウトの調整
    plt.tight_layout()

    # グラフを保存
    plt.savefig(output_path, dpi=300)
    print(f"グラフを保存しました: {output_path}")
    
    # リソースの解放
    plt.close()

if __name__ == "__main__":
    import argparse
    
    # コマンドライン引数の設定
    parser = argparse.ArgumentParser(description='JSONLファイル内のテキストに含まれる特定の文字をカウントしてグラフ化します')
    parser.add_argument('--file_path', type=str, default='data/toxicity_dataset_ver2.jsonl',
                        help='分析するJSONLファイルのパス')
    parser.add_argument('--search_char', type=str, default='。',
                        help='カウントする文字（デフォルトは句点「。」）')
    parser.add_argument('--char_name', type=str, default='Periods',
                        help='グラフに表示する文字の名前（デフォルトはPeriods）')
    parser.add_argument('--output_dir', type=str, default='imgs',
                        help='グラフを保存するディレクトリ（デフォルトはimgs）')
    
    args = parser.parse_args()
    
    # ファイルパスを設定
    file_path = args.file_path
    
    # 出力パスを指定
    char_name_slug = args.char_name.lower().replace(' ', '_')
    ver1_output_path = f'{args.output_dir}/{char_name_slug}_count_distribution_ver1.png'
    ver2_output_path = f'{args.output_dir}/{char_name_slug}_count_distribution_ver2.png'
    
    # Ver1データ（最初の1847件）の文字数をカウント
    ver1_result = count_characters_in_jsonl(file_path, args.search_char, start=0, end=1847)
    
    # Ver2データ（後の2000件）の文字数をカウント
    ver2_result = count_characters_in_jsonl(file_path, args.search_char, start=1847, end=3847)
    
    # Ver1データの結果を表示
    print(f"Ver1データの{args.char_name}ごとのテキストフィールド数（0-99: 20単位, 100+: 100単位）:")
    for bucket in sorted(ver1_result.keys(), key=lambda x: int(x.split('~')[0])):
        print(f'"{bucket}": {ver1_result[bucket]}')
    
    # Ver1データの結果をグラフ化して保存
    plot_character_count_distribution(ver1_result, ver1_output_path, 
                                    title_suffix="(Ver1 Data)", 
                                    char_name=args.char_name)
    
    # Ver2データの結果を表示
    print(f"\nVer2データの{args.char_name}ごとのテキストフィールド数（0-99: 20単位, 100+: 100単位）:")
    for bucket in sorted(ver2_result.keys(), key=lambda x: int(x.split('~')[0])):
        print(f'"{bucket}": {ver2_result[bucket]}')
    
    # Ver2データの結果をグラフ化して保存
    plot_character_count_distribution(ver2_result, ver2_output_path, 
                                    title_suffix="(Ver2 Data)", 
                                    char_name=args.char_name)
    
    print("\n両方のグラフを保存しました。")