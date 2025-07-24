import json
from collections import defaultdict
import matplotlib.pyplot as plt
import os
from pathlib import Path

def count_text_lengths(file_path, start=0, end=None):
    # 1000文字単位でカウントするための辞書を初期化
    char_count_dict = defaultdict(int)
    
    # JSONLファイルを1行ずつ読み込む
    count = 0
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            # 指定範囲のみ処理
            if count < start:
                count += 1
                continue
            if end is not None and count >= end:
                break
                
            try:
                # JSONとしてパース
                data = json.loads(line)
                # textフィールドの文字数を取得
                if 'text' in data:
                    text_length = len(data['text'])
                    # 1000文字ごとの区間に分類
                    bucket = (text_length // 1000) * 1000
                    # 辞書にカウントを追加
                    char_count_dict[bucket] += 1
            except json.JSONDecodeError:
                # JSONデコードエラーの場合はスキップ
                continue
            count += 1
    
    # 通常の辞書に変換して返す
    return dict(char_count_dict)

def plot_text_length_distribution(count_dict, output_path, title=None):
    # 出力ディレクトリが存在しない場合は作成
    output_dir = os.path.dirname(output_path)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # グラフのフィギュアとaxesを作成
    plt.figure(figsize=(12, 6))
    
    # データの準備
    lengths = sorted(count_dict.keys())
    counts = [count_dict[length] for length in lengths]
    
    # X軸のラベルを作成
    labels = [f"{length}~{length+999}" for length in lengths]
    
    # 棒グラフを作成
    plt.bar(range(len(lengths)), counts, color='skyblue', width=0.6)
    
    # X軸のラベルを設定
    plt.xticks(range(len(lengths)), labels, rotation=45, ha='right')
    
    # グラフのタイトルと軸ラベルを設定
    if title:
        plt.title(title, fontsize=15)
    else:
        plt.title('Text Length Distribution (1000 character units)', fontsize=15)
    plt.xlabel('Text Length (characters)', fontsize=12)
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

# ファイルパスを指定
file_path = '/work/s245302/multiclassification/data/toxicity_dataset_ver2.jsonl'
output_dir = '/work/s245302/multiclassification/imgs'

# Ver.1のデータ（最初の1847件）をカウントして分析
ver1_result = count_text_lengths(file_path, start=0, end=1847)
print("Ver.1（1847件）の1000文字ごとのtextフィールド数:")
for length, count in sorted(ver1_result.items()):
    print(f'"{length}～{length+999}文字": {count}')

# Ver.1のグラフを作成
ver1_output_path = f'{output_dir}/text_length_distribution_ver1.png'
plot_text_length_distribution(ver1_result, ver1_output_path, 
                            title='Text Length Distribution - Ver.1 Data (1000 character units)')

# Ver.2のデータ（後の2000件）のみをカウントして分析
ver2_result = count_text_lengths(file_path, start=1847, end=3847)
print("Ver.2（2000件）の1000文字ごとのtextフィールド数:")
for length, count in sorted(ver2_result.items()):
    print(f'"{length}～{length+999}文字": {count}')

# Ver.2のグラフを作成
ver2_output_path = f'{output_dir}/text_length_distribution_ver2.png'
plot_text_length_distribution(ver2_result, ver2_output_path, 
                            title='Text Length Distribution - Ver.2 Data (1000 character units)')