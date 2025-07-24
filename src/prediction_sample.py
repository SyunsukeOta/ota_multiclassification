
from datasets import load_dataset
from transformers import (
	AutoTokenizer,
	AutoModel,
	AutoModelForSequenceClassification,
	BitsAndBytesConfig,
	DataCollatorWithPadding,
	Trainer,
	TrainingArguments,
	TrainerCallback,
)
from transformers.modeling_outputs import (
	SequenceClassifierOutput,
)
from peft import (
	LoraConfig,
	get_peft_model,
	prepare_model_for_kbit_training,
	TaskType
)
import evaluate
import numpy as np
import torch
import json

import src.utils as utils
from src.multi_classification_sample import Classifier

from tqdm import tqdm
from datetime import datetime
from pathlib import Path
from tap import Tap
import copy

class Args(Tap):
	model_dir: str = "./outputs/llm-jp-3-1.8b/2025-02-20/10-00-29.700866/checkpoint-16"
	input_path: str
	output_path: str
	output_dir: str = ""
	num_categories: int = 7
	num_labels: int = 2

	num_epochs: int = 1
	batch_size: int = 8
	per_device_batch_size: int = 2

	test_rate: float = 0.3
	learning_rate: float = 2e-5
	weight_decay: float = 0.01
	warmup_ratio: float = 0.05
	lora_rank: int = 16
	lora_dropout: float = 0.05

	use_bf16: bool = torch.cuda.is_bf16_supported()

	@property
	def torch_dtype(self):
		return torch.bfloat16 if self.use_bf16 else torch.float16

	def process_args(self):
		if not self.output_dir:
			basename = self.model_name.split("/")[-1]
			date, time = datetime.now().strftime("%Y-%m-%d/%H-%M-%S.%f").split("/")
			output_dir = Path("outputs", basename, date, time)
			output_dir.mkdir(parents=True)
			self.output_dir = output_dir


	def training_args(self):
		x = TrainingArguments(
			output_dir=args.output_dir,
			per_device_train_batch_size=self.per_device_batch_size,
			per_device_eval_batch_size=self.per_device_batch_size,
			gradient_accumulation_steps=self.batch_size // self.per_device_batch_size,
			bf16=self.use_bf16,
			fp16=not self.use_bf16,
		)
		print('training_args: ', x.to_dict())
		return x


	def peft_config(self):
		x = LoraConfig(
			r=self.lora_rank,
			lora_alpha=self.lora_rank * 2,
			lora_dropout=self.lora_dropout,
			inference_mode=False,
			target_modules="all-linear",
		)
		print('peft_config: ', x.to_dict())
		return x

class CustomDataCollator(DataCollatorWithPadding):
	def __call__(self, features):
		batch = super().__call__(features)
		# 予測モードでは空のラベルを返す
		if "labels" not in batch or batch["labels"] is None:
			batch["labels"] = torch.zeros((len(features), args.num_categories), dtype=torch.long)
		return batch

def predict(args, texts):
	# トークナイザーを読み込む
	tokenizer = AutoTokenizer.from_pretrained(args.model_dir)

	# モデルを読み込む
	quantization_config = BitsAndBytesConfig(
		load_in_4bit=True,
		bnb_4bit_quant_type="nf4",
		bnb_4bit_use_double_quant=True,
		bnb_4bit_compute_dtype=args.torch_dtype,
	)
	model = AutoModel.from_pretrained(
		args.model_dir,
		device_map="auto",
		torch_dtype=args.torch_dtype,
		trust_remote_code=True,
		quantization_config=quantization_config,
	)
	model = prepare_model_for_kbit_training(model)
	model = get_peft_model(model, args.peft_config())
	model = Classifier(model, args.num_labels, dtype=args.torch_dtype, num_categories=args.num_categories)
	print("model loading is successful")

	# data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
	data_collator = CustomDataCollator(tokenizer=tokenizer)

	# トレーナーを作成
	trainer = Trainer(
		model=model,
		args=args.training_args(),
		tokenizer=tokenizer,
		# processing_class=tokenizer,
		data_collator=data_collator,
	)

	# テキストの前処理
	def preprocess_function(examples):
		return tokenizer(examples["text"], truncation=True)
	tokenized_dataset = texts.map(preprocess_function, batched=True)
	# tokenized_dataset = tokenized_dataset.map(lambda x: {**x, "labels": [0]*args.num_categories})
	# tokenized_dataset = tokenized_dataset.map(lambda x: {**x, "labels": None})
	tokenized_dataset = tokenized_dataset.map(lambda x: {
    **x, 
    "labels": torch.zeros(args.num_categories, dtype=torch.long)  # または適切な形式
	})
	# 試しに10件だけ
	tokenized_dataset = tokenized_dataset.select(range(10))

	# データをbfloat16に変換
	# tokenized_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])
	# tokenized_dataset = tokenized_dataset.map(lambda x: {k: v.to(args.torch_dtype) for k, v in x.items()})

	# 入力データをトークナイズ
	# inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)

	# データローダー
	# data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
	# dataset = torch.utils.data.TensorDataset(inputs['input_ids'], inputs['attention_mask'])
	# dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, collate_fn=data_collator)

	# print(f"Dataset lenght: {len(dataset)}")
	# print(f"First item: {dataset[0]}")
	# for i, tensor in enumerate(dataset[0]):
	# 	print(f"Tensor {i} shape: {tensor.shape}")
	# print(f"Inputs keys: {inputs.keys()}")
	# print(f"input_ids shape: {inputs['input_ids'].shape}")
	# print(f"token_type_ids shape: {inputs['token_type_ids'].shape}")
	# print(f"attention_mask shape: {inputs['attention_mask'].shape}")

	print('before predict tokenized_dataset: ', tokenized_dataset)
	print(f"Columns: {tokenized_dataset.column_names}")
	print(f"First item: {tokenized_dataset[0]}")

	# モデルを使って予測
	predictions_output = trainer.predict(tokenized_dataset)
	# predictions = trainer.predict(inputs)
	# predictions = trainer.predict(dataloader)
	print('予測結果の形状: ',predictions_output.predictions.shape)

	# 予測結果の処理
	# 形状が (サンプル数, カテゴリ数, ラベル数) の場合
	predictions_array = predictions_output.predictions
	category_results = []

	# カテゴリごとに予測結果を処理
	# for category_idx in range(args.num_categories):
	# 	# 各カテゴリの予測結果を取得し、最も確率の高いクラスのインデックスを取得
	# 	category_preds = torch.argmax(torch.tensor(predictions_array[:, category_idx, :]), dim=1)
	# 	category_results.append(category_preds)

	# 結果を整形（例: カテゴリごとの予測をまとめる）
	# final_predictions = torch.stack(category_results, dim=1)

	final_predictions = torch.argmax(torch.tensor(predictions_output.predictions), dim=1)
	return final_predictions

def main(args):
	print('Hello!!')
	datasets = load_dataset("json", data_files=args.input_path)
	print(args.input_path)
	# dataset に labels のカラムを追加
	datasets = datasets["train"]
	datasets = datasets.add_column("labels", [[]]*len(datasets))

	# print(f"Labels column type: {type(datasets['labels'])}")
	# print(f"First item in labels: {datasets['labels'][0]}")
	# print(f"Length of labels: {len(datasets['labels'])}")
	# if len(datasets['labels']) > 0:
	# 		print(f"Shape of first item in labels: {len(datasets['labels'][0])}")


	print('@@datasets@@', datasets)
	# datasets = datasets.remove_columns(["tweet_id"])
	# print('@@deleted datasets@@', datasets)

	# print(f"dataset_shape: {datasets['train'].shape}")
	# with open(args.input_path, 'r') as f:
	# 	data = [json.loads(line) for line in f]

	# texts = [item['text'] for item in data]
	# tweet_ids = [item['tweet_id'] for item in data]

	# 予測を行う
	print('Predicting...')
	predictions = predict(args, texts=datasets)

	# 結果を保存する
	print('Saving results...')
	print(f"Predictions shape: {predictions.shape}")
	# results = [{'tweet_id': tweet_id, 'prediction': prediction.item()} for tweet_id, prediction in zip(tweet_ids, predictions)]
	# with open(args.output_path, 'w') as f:
	# 	for result in results:
	# 		f.write(json.dumps(result) + '\n')
	# 例：結果をJSONとして保存
	results = []
	for i, sample_pred in enumerate(predictions):
		# データセットから必要な情報を取得（例：IDなど）
		sample_info = datasets[i]
		result = {
			'id': i,  # または sample_info['id'] などデータセットに含まれる識別子
			'predictions': sample_pred.tolist()  # テンソルをリストに変換
		}
		results.append(result)

	# 結果を保存
	with open(args.output_path, 'w') as f:
		json.dump(results, f, indent=2)



if __name__ == "__main__":
	args = Args().parse_args()
	main(args)