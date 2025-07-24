from datasets import load_dataset, Dataset, concatenate_datasets
from setfit import (
	SetFitModel,
	Trainer,
	TrainingArguments,
	sample_dataset
)
from transformers import TrainerCallback
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np

import utils
import random
import json
from datetime import datetime
from pathlib import Path
from tap import Tap
from typing import Literal
from collections import Counter
from tqdm import tqdm

class Args(Tap):
	yes_dataset_path: str = "data/toxicity_ver2_allyesdata.jsonl"
	allno_data_path: str = "data/toxicity_ver2_allnodata.jsonl"
	corpus_path: str = "data/CC-MAIN-2023-23/CC-MAIN-20230527223515-20230528013515/00000-ja-sentence.txt"
	num_labels: int = 2
	max_length: int = 256

	model_name: str = "cl-nagoya/ruri-v3-310m"
	output_dir: str = ""

	datasplit_rate: float = 0.3
	allno_datasplit_rate: float = 0.1
	datasplit_seed: int = 42
	seed: int = 42

	hypothesis: int = 1
	train_no_rate: float = 1.0

	num_samples: int = 8
	num_epochs: int = 5
	batch_size: int = 8
	sampling_strategy: Literal["oversampling", "undersampling", "unique"] = "oversampling"
	target_strategy: Literal["one-vs-rest", "multi-output", "classifier-chain"] = "multi-output"

	device: Literal["cuda", "cpu"] = "cuda" if torch.cuda.is_available() else "cpu"
	suppress_dynamo_errors: bool = True
	debug: bool = True

	def process_args(self):
		if self.output_dir:
			self.output_dir = Path( self.output_dir )
		else:
			basename = self.model_name.split("/")[-1]
			date, time = datetime.now().strftime("%Y-%m-%d/%H-%M-%S.%f").split("/")
			output_dir = Path("outputs", basename, date, time)
			self.output_dir = output_dir
		self.output_dir.mkdir(parents=True)
		log_path = Path( self.output_dir, "parameters.txt" )
		self.log_file = log_path.open( mode='w', buffering=1 )
		print( json.dumps({ "yes_dataset_path": self.yes_dataset_path,
					"model_name": self.model_name }),
					file=self.log_file )

	def training_args(self):
		x = TrainingArguments(
			output_dir=self.output_dir,
			sampling_strategy=self.sampling_strategy,
			batch_size=self.batch_size,
			num_epochs=self.num_epochs,
			logging_dir=self.output_dir,
		)
		print(json.dumps(x.to_dict(), default=str), file=self.log_file)
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
				f"accuracy: {category_metrics[f'accuracy']:.4f} \t"
				f"precision: {category_metrics[f'precision']:.4f} \t"
				f"recall: {category_metrics[f'recall']:.4f} \t"
				f"f1: {category_metrics[f'f1']:.4f}"
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

def main(args):
	# 1. Process arguments
	def truncate_text(example):
		example["text"] = example["text"][:args.max_length]
		return example
	
	# 1.1 Loading least toxic yes_dataset
	yes_dataset = load_dataset("json", data_files=args.yes_dataset_path, split="train")
	yes_dataset = yes_dataset.remove_columns(["label"])
	yes_dataset = yes_dataset.map(truncate_text)
	tasknames = [task for task in yes_dataset.column_names if task not in ["id", "text"]]
	
	# 1.2 Loading non-toxic yes_dataset
	no_dataset = load_dataset("json", data_files=args.allno_data_path, split="train")
	no_dataset = no_dataset.remove_columns(["label"])
	no_dataset = no_dataset.map(truncate_text)
	# if args.debug:
	# 	print("tasknames: ", tasknames)
	# 	print("yes_dataset", yes_dataset)
	# 	print("no_dataset", no_dataset)

	# 1.3 Create category rank list
	label_yes_counts = {taskname: Counter(yes_dataset[taskname])["yes"] for taskname in tasknames}
	category_rank = [label for (label, _) in sorted(label_yes_counts.items(), key=lambda x: x[1])]
	if args.debug:
		print("Category: ", tasknames)
		print("label_yes_counts:", label_yes_counts)
		print("Category rank:", category_rank)
	# Category rank: ['others', 'personal', 'illegal', 'violent', 'discriminatory', 'corporate', 'obscene']

	# 2. Create train, test dataset from yes_dataset and no_dataset
	# 2.1 split yes_dataset
	yes_dataset = yes_dataset.add_column(name="flag", column=[0]*len(yes_dataset))
	yes_df = yes_dataset.to_pandas()

	for category in category_rank:
		print(category)
		# sort df
		yes_df = yes_df.sort_values(by=[category, 'flag'], ascending=False).reset_index(drop=True)

		# count yes num
		yes_count = label_yes_counts[category]
		exist_flag_count = yes_df['flag'].iloc[0:yes_count].sum()
		# print(f"- before flag: {exist_flag_count}, {yes_df['flag'].iloc[0:yes_count].tolist()}")
		# print(f"- {category}: {yes_df[category].iloc[0:yes_count].tolist()}")

		# make random ids list
		set_flag_count = min(args.num_samples, yes_count) - exist_flag_count
		if set_flag_count > 0:
			selected_yes_ids = set(random.sample(
				range(exist_flag_count, yes_count),
				set_flag_count
			))

		# set flag by ids list
		for idx in selected_yes_ids:
			yes_df.at[idx, "flag"] = 1
		# print(f"- selected_yes_ids: {len(selected_yes_ids)}, {selected_yes_ids}")
		# print(f"- after flag: {yes_count}, {yes_df['flag'].iloc[0:yes_count].tolist()}")
		# print("---------------")

	# split by flag
	yes_train_df = yes_df[yes_df["flag"] == 1].reset_index(drop=True)
	yes_test_df = yes_df[yes_df["flag"] == 0].reset_index(drop=True)
	y_train_dataset = Dataset.from_pandas(yes_train_df, features=yes_dataset.features).remove_columns("flag")
	y_test_dataset = Dataset.from_pandas(yes_test_df, features=yes_dataset.features).remove_columns("flag")

	# 2.2 split no_dataset
	# len of each dataset
	yes_len = yes_dataset.num_rows
	no_len = no_dataset.num_rows
	y_train_len = y_train_dataset.num_rows
	y_test_len = y_test_dataset.num_rows
	n_train_len = y_train_len
	n_test_len = min(no_len - n_train_len, int(y_test_len * no_len / yes_len))
	if args.debug:
		print("yes_dataset:", yes_len)
		print("- y_train_dataset:", y_train_len)
		print("- y_test_dataset:", y_test_len)
		print("no_dataset:", no_len)
		print("- n_train_len:", n_train_len)
		print("- n_test_len:", n_test_len)

	no_df = no_dataset.shuffle(seed=args.seed).to_pandas()
	no_train_df = no_df.iloc[-n_train_len:].reset_index(drop=True)
	no_test_df = no_df.iloc[:n_test_len].reset_index(drop=True)
	n_train_dataset = Dataset.from_pandas(no_train_df, features=no_dataset.features)
	n_test_dataset = Dataset.from_pandas(no_test_df, features=no_dataset.features)

	# 2.3 Concatenate yes and no datasets
	train_dataset = concatenate_datasets([y_train_dataset, n_train_dataset])
	test_dataset = concatenate_datasets([y_test_dataset, n_test_dataset])
	if args.debug:
		print("train_dataset:", train_dataset)
		print(f"- y_train_dataset:", y_train_dataset)
		print(f"- n_train_dataset:", n_train_dataset)
		print("test_dataset:", test_dataset)
		print(f"- y_test_dataset:", y_test_dataset)
		print(f"- n_test_dataset:", n_test_dataset)

	return

	# The following copy of string is necesssary to avoid the Pickle warnings.
	def encode_labels(example):
		values = [1 if example[k] == "yes" else 0 for k in category_rank]
		category = category_rank[next((i for i, v in enumerate(values) if v), 0)]
		return {
			"category": category,
			"labels": values
		}
	yes_dataset = yes_dataset.map(encode_labels)
	no_dataset = no_dataset.map(encode_labels)
	# if args.debug:
	# 	print(yes_dataset[:10])
	# 	print(no_dataset[:10])

	yes_dataset = yes_dataset.train_test_split(test_size=args.datasplit_rate, seed=args.datasplit_seed)
	no_dataset = no_dataset.train_test_split(test_size=args.allno_datasplit_rate, seed=args.datasplit_seed)

	if args.debug:
		print("hypothesis:", args.hypothesis)
		print("train_no_rate:", args.train_no_rate)
	allno_select_len = 0
	if args.hypothesis == 1:
		# Hypothesis 1
		print("label_yes_counts: ", label_yes_counts)
		max_key = max(label_yes_counts, key=label_yes_counts.get)
		max_value = label_yes_counts[max_key]
		print("max_key: ", max_key, "max_value: ", max_value)
		allno_select_len = int(max_value * args.train_no_rate)
		# print("obscene x0.5", no_dataset["train"][:int(max_value*0.5)])
	elif args.hypothesis == 2:
		# Hypothesis 2
		allyes_train_len = yes_dataset["train"].num_rows
		allno_select_len = int(allyes_train_len * args.train_no_rate)
		print("yes_dataset['train']:", allyes_train_len)
		# print("allyes x0.5", no_dataset["train"].select(range(int(allyes_train_len*0.5))))

	train = concatenate_datasets([yes_dataset["train"], no_dataset["train"].select(range(int(allno_select_len)))])
	print("train:", train)

	# TODO: It is necessary to sample instances balancing emotion labels.
	train_yes_dataset = sample_dataset(
    yes_dataset["train"],
    label_column="category",
    num_samples=args.num_samples
  )
	test_yes_dataset = yes_dataset["test"]




	if args.suppress_dynamo_errors:
		import torch._dynamo
		torch._dynamo.config.suppress_errors = True




if __name__ == "__main__":
	args = Args().parse_args()
	main(args)
