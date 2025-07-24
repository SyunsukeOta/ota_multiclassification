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

from tqdm import tqdm
from datetime import datetime
from pathlib import Path
from tap import Tap
import copy

class Args(Tap):
	# 6 + 1 categories: 'obscene', 'discriminatory', 'violent', 'illegal', 'personal', 'corporate', 'others'
	dataset_path: str = "data/toxicity_dataset.jsonl"
	# dataset_path: str = "data/toxicity_dataset_mini.jsonl"
	dataset_name: str = "imdb"
	model_name: str = "sbintuitions/modernbert-ja-30m"
	# model_name: str = "llm-jp/llm-jp-3-1.8b"
	output_dir: str = ""
	num_categories: int = 7
	num_labels: int = 2
	
	num_epochs: int = 10
	batch_size: int = 8
	per_device_batch_size: int = 2

	test_rate: float = 0.3
	learning_rate: float = 2e-5
	weight_decay: float = 0.01
	warmup_ratio: float = 0.05
	lora_rank: int = 16
	lora_dropout: float = 0.05
	datasplit_seed: str = 42
	
	use_bf16: bool = torch.cuda.is_bf16_supported()

	categories: str = []
		
	@property
	def torch_dtype(self):
		print('!!!use_bf16: ', self.use_bf16)
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
			num_train_epochs=self.num_epochs,
			eval_strategy="epoch",
			logging_strategy="epoch",
			save_strategy="epoch",
			optim="adamw_torch",
			learning_rate=self.learning_rate,
			weight_decay=self.weight_decay,
			lr_scheduler_type="inverse_sqrt",
			warmup_ratio=args.warmup_ratio,
			bf16=self.use_bf16,
			fp16=not self.use_bf16,
			load_best_model_at_end=True,
			dataloader_drop_last=False,
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
			# task_type=TaskType.SEQ_CLS # AutoModelの場合は指定する必要はなし
		)
		print('peft_config: ', x.to_dict())
		return x

	def log(self, metrics: dict) -> None:
		for category in self.categories:
			category_metrics = {
				f"accuracy": metrics.get(f"eval_{category}_accuracy", -1),
				f"precision": metrics.get(f"eval_{category}_precision", -1),
				f"recall": metrics.get(f"eval_{category}_recall", -1),
				f"f1": metrics.get(f"eval_{category}_f1", -1),
				"epoch": metrics.get("epoch", -1),
				"step": metrics.get("step", -1)
			}
			log_file = self.output_dir / f"{category}_log.csv"
			utils.log(category_metrics, log_file)
			tqdm.write(
				f"epoch: {category_metrics['epoch']} \t"
				f"{category}_accuracy: {category_metrics[f'accuracy']:.4f} \t"
				f"{category}_precision: {category_metrics[f'precision']:.4f} \t"
				f"{category}_recall: {category_metrics[f'recall']:.4f} \t"
				f"{category}_f1: {category_metrics[f'f1']:.4f}"
			)
			print(category, "ok!!")


class LoggingCallback(TrainerCallback):
	def __init__(self, args):
		self.args = args

	def on_epoch_end(self, args, state, control, **kwargs):
		if state.log_history:
			metrics = state.log_history[-1]
			print('state_log_history: ', state.log_history)
			print('metrics: ', metrics)
			self.args.log(metrics)
		else:
			print("No log history")

class Classifier(torch.nn.Module):
	def __init__(self, encoder, num_labels, bias=False, dtype=None, num_categories=1):
		super().__init__()
		self.add_module("encoder", encoder)
		self.num_categories = num_categories
		for category_index in range(self.num_categories):
			self.add_module(f"classifier_{category_index}",
			torch.nn.Linear(encoder.config.hidden_size,
			num_labels,
			bias=bias,
			dtype=dtype))
		self.loss_fn = torch.nn.CrossEntropyLoss()
	
	def forward(self, input_ids, attention_mask, labels):
		if labels is not None:
			outputs = self.encoder(
				input_ids=input_ids,
				attention_mask=attention_mask,
			)
			seq_length = attention_mask.sum(dim=1)
			eos_hidden_states = outputs.last_hidden_state[
				torch.arange(
					seq_length.size(0),
					device=outputs.last_hidden_state.device,
				),
				seq_length - 1,
			]
			logits = []
			for i in range(self.num_categories):
				curr_classifier = self.__getattr__(f"classifier_{i}")
				logits.append(curr_classifier(eos_hidden_states))
			logits = torch.stack(logits, dim=1)
			# print('@@a@@',logits, labels)

			flatten_logits = torch.reshape(logits, (logits.shape[0]*logits.shape[1], logits.shape[2]))
			flatten_labels = torch.reshape(labels, (labels.shape[0]*labels.shape[1],))
			# print('@@b@@',logits, labels)

			loss = self.loss_fn(flatten_logits, flatten_labels)
			# print('@@loss@@', loss)
		else:
			loss = None

		return SequenceClassifierOutput(
			loss=loss,
			logits=logits,
		)
		

def main(args):
	tokenizer = AutoTokenizer.from_pretrained(args.model_name)
	
	# 1. get dataset
	dataset = load_dataset("json", data_files=args.dataset_path, split="train")
	
	# 2. shape data
	# print(type(dataset.column_names["train"]))
	dataset = dataset.remove_columns(["id", "label"])
	
	# 2.1 set categories to label list
	categories = dataset.column_names
	categories.remove("text")
	args.categories = categories
	dataset = dataset.add_column("label", [[]]*len(dataset))
	# print(type(dataset), dataset)
	def update_label(dataset, category):
		dataset["label"].append(1 if dataset[category] == "yes" else 0)
		return dataset
	for category in categories:
		dataset = dataset.map(lambda example: update_label(example, category))

	# # 2.2 make tokenzied train/test dataset and data collator
	split_dataset = dataset.train_test_split(test_size=args.test_rate, seed=args.datasplit_seed)
	
	def preprocess_function(examples):
		return tokenizer(examples["text"], truncation=True, max_length=512)
	tokenized_dataset = split_dataset.map(preprocess_function, batched=True)
	data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
	# print(tokenized_dataset, data_collator)
	print('@@tokenized_dataset@@', tokenized_dataset)

	accuracy = evaluate.load("accuracy")
	precision_metric = evaluate.load("precision")
	recall_metric = evaluate.load("recall")
	f1_metric = evaluate.load("f1")

	def compute_metrics(eval_pred):
		predictions, labels = eval_pred

		# shape を確認
		print(f"Predictions shape before argmax: {predictions.shape}")
		print(f"Labels shape: {labels.shape}")
  
		# カテゴリーごとの予測値を取得
		category_predictions = [predictions[:, i, :] for i in range(args.num_categories)]
		argmax_predictions = np.argmax(category_predictions, axis=2)
		print(f"Argmax predictions shape: {argmax_predictions.shape}")

		# 転置
		labels = np.transpose(labels)

		# shape の最終確認
		print(f"Final Predictions shape: {argmax_predictions.shape}")
		print(f"Final Labels shape: {labels.shape}")

		# 各カテゴリごとの評価
		category_metrics = {}
		for i in range(args.num_categories):
			category_accuracy = accuracy.compute(
				predictions=argmax_predictions[i].tolist(),
				references=labels[i].tolist(),
			)
			category_precision = precision_metric.compute(
				predictions=argmax_predictions[i].tolist(),
				references=labels[i].tolist(),
				average="binary"
      )
			category_recall = recall_metric.compute(
				predictions=argmax_predictions[i].tolist(),
				references=labels[i].tolist(),
				average="binary"
			)
			category_f1 = f1_metric.compute(
				predictions=argmax_predictions[i].tolist(),
				references=labels[i].tolist(),
				average="binary"
			)
			# print(f"**before calc accuracy, args.categories type: {type(args.categories)}")
			category_metrics[f"{args.categories[i]}_accuracy"] = category_accuracy["accuracy"]
			category_metrics[f"{args.categories[i]}_precision"] = category_precision["precision"]
			category_metrics[f"{args.categories[i]}_recall"] = category_recall["recall"]
			category_metrics[f"{args.categories[i]}_f1"] = category_f1["f1"]

		print(f"Category metrics: {category_metrics}")
		# args.log(category_metrics)
		return category_metrics

	quantization_config = BitsAndBytesConfig(
		load_in_4bit=True,
		bnb_4bit_quant_type="nf4",
		bnb_4bit_use_double_quant=True,
		bnb_4bit_compute_dtype=args.torch_dtype,
	)
	# model = AutoModelForSequenceClassification.from_pretrained(
	model = AutoModel.from_pretrained(
		args.model_name,
		device_map="auto",
		torch_dtype=args.torch_dtype,
		trust_remote_code=True,
		quantization_config=quantization_config,
	)

	# 3 model
	print(f"Loaded model uses {model.get_memory_footprint()} bytes")
	model = prepare_model_for_kbit_training(model)
	model = get_peft_model(model, args.peft_config())
	model.print_trainable_parameters()
	model = Classifier(model, args.num_labels, dtype=args.torch_dtype, num_categories=args.num_categories)
	
	# 4. training
	trainer = Trainer(
		model=model,
		args=args.training_args(),
		train_dataset=tokenized_dataset["train"],
		eval_dataset=tokenized_dataset["test"],
		processing_class=tokenizer,
		data_collator=data_collator,
		compute_metrics=compute_metrics,
		callbacks=[LoggingCallback(args)],
	)

	train_result = trainer.train()
	trainer.save_metrics("train", train_result.metrics)
	trainer.save_metrics("eval", trainer.evaluate())
	
	# trainer.save_state()
	trainer.save_model(args.output_dir)
	tokenizer.save_pretrained(args.output_dir)

	# config.jsonを保存
	model.encoder.config.save_pretrained(args.output_dir)
	# args.log(train_result.metrics, 'train')
	# args.log(trainer.evaluate().metrics, 'eval')


	# 5. prediction
	# with open(args.input_path, 'r') as f:
	# 	data = [json.loads(line) for line in f]
	# texts = [item['text']for item in data]

	# inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
	# predictions = trainer.predict(inputs)
	# predictions = torch.argmax(torch.tensor(predictions), dim=1)

	# # 6. save results
	# results = [{'tweet_id': tweet_id, 'prediction': prediction.item()} for tweet_id, prediction in zip(tweet_ids, predictions)]
	# with open(args.output_path, 'w') as f:
	# 	for result in results:
	# 		f.write(json.dumps(result) + '\n')

if __name__ == '__main__':
	args = Args().parse_args()
	main(args)