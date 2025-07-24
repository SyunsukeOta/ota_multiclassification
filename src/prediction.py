from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    BitsAndBytesConfig,
)
import torch
import json
from tqdm import tqdm
from tap import Tap
from pathlib import Path

# Model関連のインポートを調整
from src.models import Model  # オリジナルのプログラムと同じModelクラスを使用


class Args(Tap):
    model_dir: str  # モデルが保存されているディレクトリ
    input_path: str  # 入力ファイルパス
    output_path: str  # 出力ファイルパス
    
    batch_size: int = 8
    max_seq_len: int = 512
    template_type: int = 2  # オリジナルのプログラムと同じテンプレートタイプ
    
    use_bf16: bool = torch.cuda.is_bf16_supported()
    
    @property
    def torch_dtype(self):
        return torch.bfloat16 if self.use_bf16 else torch.float16


def predict(args, texts):
    # トークナイザーを読み込む
    use_fast = True  # modelによって調整が必要
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_dir,
        model_max_length=args.max_seq_len,
        use_fast=use_fast,
    )
    
    # モデルを読み込む（オリジナルのModelクラスを使用）
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=args.torch_dtype,
    )
    
    # モデルのラベル数や設定を学習時と同じにする
    # モデルの設定ファイルがある場合は読み込む
    config_path = Path(args.model_dir) / "config.json"
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = json.load(f)
        num_labels = config.get("num_labels", 9)  # デフォルト値は適宜調整
        lora_r = config.get("lora_r", 32)
    else:
        num_labels = 9  # デフォルト値は適宜調整
        lora_r = 32
    
    model = Model(
        model_name=args.model_dir,
        num_labels=num_labels,
        lora_r=lora_r,
        gradient_checkpointing=False,  # 予測時は不要
    )
    
    # モデルをGPUに移動（利用可能な場合）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()  # 評価モードに設定
    
    # データセットからテキストを抽出
    text_samples = []
    for item in texts:
        title = item.get("title", "")
        body = item.get("body", item.get("text", ""))  # bodyかtextのどちらかを使用
        
        # テンプレートタイプに応じて入力テキストを構築
        if args.template_type == 0:
            text = f"タイトル: {title}\n本文: {body}\nラベル: "
        elif args.template_type == 1:
            text = f"タイトル: {title}\n本文: {body}"
        elif args.template_type == 2:
            text = f"{title}\n{body}"
        else:
            text = body  # デフォルト
        
        text_samples.append(text)
    
    # バッチ処理で予測を実行
    predictions = []
    for i in tqdm(range(0, len(text_samples), args.batch_size)):
        batch_texts = text_samples[i:i+args.batch_size]
        
        # テキストのトークン化
        inputs = tokenizer(
            batch_texts, 
            truncation=True, 
            padding=True, 
            return_tensors="pt",
            max_length=args.max_seq_len
        )
        
        # GPUに送る（利用可能な場合）
        # token_type_idsを削除（モデルがこのパラメータを受け付けない）
        if 'token_type_ids' in inputs:
            del inputs['token_type_ids']
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # 予測実行（勾配計算なし）
        with torch.no_grad():
            outputs = model(**inputs)
            batch_preds = outputs.logits.argmax(dim=-1)
            predictions.extend(batch_preds.cpu().numpy().tolist())
    
    return predictions


def main(args):
    print('Loading dataset...')
    datasets = load_dataset("json", data_files=args.input_path)
    datasets = datasets["train"]
    
    # データセットの情報を表示
    print(f"Dataset size: {len(datasets)}")
    print(f"Dataset columns: {datasets.column_names}")
    print(f"First example: {datasets[0]}")
    
    # 予測を行う
    print('Predicting...')
    predictions = predict(args, datasets)
    
    # 結果を保存
    print('Saving results...')
    results = []
    for i, pred in enumerate(predictions):
        # データセットにtweet_idなどの識別子がある場合はそれを使用
        sample_id = datasets[i].get('tweet_id', datasets[i].get('id', i))
        results.append({
            'id': sample_id,
            'prediction': pred
        })
    
    # 結果をファイルに保存
    with open(args.output_path, 'w') as f:
        for result in results:
            f.write(json.dumps(result) + '\n')
    
    print(f"Results saved to {args.output_path}")


if __name__ == "__main__":
    args = Args().parse_args()
    main(args)
