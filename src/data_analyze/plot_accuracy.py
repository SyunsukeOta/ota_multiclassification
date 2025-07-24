import pandas as pd
import matplotlib.pyplot as plt
import os
import glob
from pathlib import Path

def plot_metric_per_epoch(csv_file, metric_name, output_dir='analysis_results'):
    """指定したメトリクスについてepochごとのグラフを作成する"""
    # CSVファイルを読み込む
    df = pd.read_csv(csv_file)
    
    # グラフのフィギュアとaxesを作成
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # グラフの背景と格子設定
    ax.set_facecolor('#f8f8f8')
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # epochとメトリクスのプロット
    ax.plot(df['epoch'], df[metric_name], marker='o', linestyle='-', 
            color='#1f77b4', linewidth=2, markersize=8)
    
    # グラフのタイトルと軸ラベルを設定
    model_name = os.path.basename(os.path.dirname(os.path.dirname(csv_file)))
    category_name = os.path.basename(csv_file).replace('_log.csv', '')
    metric_title = metric_name.capitalize()
    ax.set_title(f'{metric_title} per Epoch for {category_name} - {model_name}', fontsize=15)
    ax.set_xlabel('Epoch', fontsize=16)
    ax.set_ylabel(metric_title, fontsize=16)
    
    # Y軸の範囲を0.0から1.0に設定
    ax.set_ylim([0.0, 1.0])
    
    # 出力ディレクトリが存在しない場合は作成
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # ファイル名の作成
    base_name = os.path.basename(csv_file).split('.')[0]
    output_file = os.path.join(output_dir, f'{base_name}_{metric_name}_plot.png')
    
    # グラフを保存
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    print(f"{metric_title}グラフを保存しました: {output_file}")
    
    plt.close()

def plot_multiple_metrics(directory, metric_name, output_dir='analysis_results'):
    """ディレクトリ内の全ての_log.csvファイルから指定メトリクスデータを読み込み1つのグラフに表示する"""
    # ディレクトリ内の全ての_log.csvファイルを検索
    csv_files = list(Path(directory).glob('*_log.csv'))
    
    if not csv_files:
        print(f"エラー: {directory} に_log.csvファイルが見つかりませんでした")
        return
    
    # グラフのフィギュアとaxesを作成
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # グラフの背景と格子設定
    ax.set_facecolor('#f8f8f8')
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # カラーマップの設定（各カテゴリを異なる色で表示）
    colors = plt.cm.tab10(range(len(csv_files)))
    
    # 各ファイルのデータをプロット
    for i, csv_file in enumerate(csv_files):
        df = pd.read_csv(csv_file)
        
        # カテゴリ名を抽出（ファイル名から_log.csvを削除）
        category = os.path.basename(csv_file).replace('_log.csv', '')
        
        # epochとメトリクスのプロット
        ax.plot(df['epoch'], df[metric_name], marker='o', linestyle='-', 
                color=colors[i], linewidth=2, markersize=6, label=category)
    
    # グラフのタイトルと軸ラベルを設定
    model_name = os.path.basename(os.path.dirname(os.path.dirname(directory)))
    metric_title = metric_name.capitalize()
    ax.set_title(f'{metric_title} per Epoch for All Categories - {model_name}', fontsize=16)
    ax.set_xlabel('Epoch', fontsize=16)
    ax.set_ylabel(metric_title, fontsize=16)
    
    # Y軸の範囲を0.0から1.0に設定
    ax.set_ylim([0.0, 1.0])
    
    # 凡例を追加
    ax.legend(loc='lower right', frameon=True, fontsize=16)
    
    # 出力ディレクトリが存在しない場合は作成
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # ファイル名の作成
    output_file = os.path.join(output_dir, f'combined_{metric_name}_plot.png')
    
    # グラフを保存
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    print(f"複合{metric_title}グラフを保存しました: {output_file}")
    
    plt.close()

def plot_all_metrics_for_file(csv_file, output_dir='analysis_results'):
    """単一のCSVファイルから全てのメトリクス（accuracy、precision、recall、f1）のグラフを作成する"""
    for metric in ['accuracy', 'precision', 'recall', 'f1']:
        plot_metric_per_epoch(csv_file, metric, output_dir)

def plot_all_metrics_for_directory(directory, output_dir='analysis_results'):
    """ディレクトリ内の全てのCSVファイルについて、全てのメトリクスの複合グラフを作成する"""
    for metric in ['accuracy', 'precision', 'recall', 'f1']:
        plot_multiple_metrics(directory, metric, output_dir)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='CSVファイルからepochごとのメトリクスグラフを作成します')
    parser.add_argument('--csv_file', type=str, help='解析する単一のCSVファイルのパス')
    parser.add_argument('--directory', type=str, help='複数のCSVファイルを含むディレクトリのパス')
    parser.add_argument('--output_dir', type=str, default='imgs', 
                        help='グラフを保存するディレクトリ (デフォルト: imgs)')
    parser.add_argument('--metric', type=str, choices=['accuracy', 'precision', 'recall', 'f1', 'all'],
                        default='all', help='プロットするメトリクス (デフォルト: all)')
    
    args = parser.parse_args()
    
    if args.directory:
        if args.metric == 'all':
            plot_all_metrics_for_directory(args.directory, args.output_dir)
        else:
            plot_multiple_metrics(args.directory, args.metric, args.output_dir)
    elif args.csv_file:
        if args.metric == 'all':
            plot_all_metrics_for_file(args.csv_file, args.output_dir)
        else:
            plot_metric_per_epoch(args.csv_file, args.metric, args.output_dir)
    else:
        print("エラー: --csv_file または --directory オプションを指定してください")
        parser.print_help()