#!/usr/bin/env python
# coding=utf-8

import logging
import os
import random
import sys
from dataclasses import dataclass, field
from typing import Optional, List
import torch
import datasets
import numpy as np
from datasets import load_dataset, load_metric

# https://github.com/kittchy/transformers_multitask/blob/main/src/run_glue_edit.py

# torch.set_default_tensor_type("torch.cuda.FloatTensor")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

import transformers
from transformers import (
	AutoConfig,
	AutoModel,
	AutoModelForSequenceClassification,
	AutoTokenizer,
	DataCollatorWithPadding,
	EvalPrediction,
	HfArgumentParser,
	PretrainedConfig,
	Trainer,
	TrainingArguments,
	default_data_collator,
	set_seed,
	EarlyStoppingCallback,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version
from transformers.utils.versions import require_version


# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.17.0.dev0")

require_version(
	"datasets>=1.8.0",
	"To fix: pip install -r examples/pytorch/text-classification/requirements.txt",
)


def list_field(default=None, metadata=None):
	return field(default_factory=lambda: default, metadata=metadata)


logger = logging.getLogger(__name__)


@dataclass
class DataTrainingArguments:
	"""
	Arguments pertaining to what data we are going to input our model for training and eval.

	Using `HfArgumentParser` we can turn this class
	into argparse arguments to be able to specify them on
	the command line.
	"""

	early_stopping_step: Optional[int] = field(
		default=5, metadata={"help": "Eearly stopping steps (default is 3)"}
	)
	dataset_name: Optional[str] = field(
		default=None,
		metadata={"help": "The name of the dataset to use (via the datasets library)."},
	)

	dataset_config_name: Optional[str] = field(
		default=None,
		metadata={
			"help": "The configuration name of the dataset to use (via the datasets library)."
		},
	)
	max_seq_length: int = field(
		default=128,
		metadata={
			"help": "The maximum total input sequence length after tokenization. Sequences longer "
			"than this will be truncated, sequences shorter will be padded."
		},
	)
	overwrite_cache: bool = field(
		default=False,
		metadata={"help": "Overwrite the cached preprocessed datasets or not."},
	)
	pad_to_max_length: bool = field(
		default=True,
		metadata={
			"help": "Whether to pad all samples to `max_seq_length`. "
			"If False, will pad the samples dynamically when batching to the maximum length in the batch."
		},
	)
	max_train_samples: Optional[int] = field(
		default=None,
		metadata={
			"help": "For debugging purposes or quicker training, truncate the number of training examples to this "
			"value if set."
		},
	)
	max_eval_samples: Optional[int] = field(
		default=None,
		metadata={
			"help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
			"value if set."
		},
	)
	max_predict_samples: Optional[int] = field(
		default=None,
		metadata={
			"help": "For debugging purposes or quicker training, truncate the number of prediction examples to this "
			"value if set."
		},
	)
	# task1
	train_file_task1: Optional[str] = field(
		default=None,
		metadata={"help": "A csv or a json file containing the training data."},
	)
	validation_file_task1: Optional[str] = field(
		default=None,
		metadata={"help": "A csv or a json file containing the validation data."},
	)
	test_file_task1: Optional[str] = field(
		default=None,
		metadata={"help": "A csv or a json file containing the test data."},
	)
	# task2
	train_file_task2: Optional[str] = field(
		default=None,
		metadata={"help": "A csv or a json file containing the training data."},
	)
	validation_file_task2: Optional[str] = field(
		default=None,
		metadata={"help": "A csv or a json file containing the validation data."},
	)
	test_file_task2: Optional[str] = field(
		default=None,
		metadata={"help": "A csv or a json file containing the test data."},
	)


@dataclass
class ModelArguments:
	"""
	Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
	"""

	model_name_or_path: str = field(
		metadata={
			"help": "Path to pretrained model or model identifier from huggingface.co/models",
		},
	)
	config_name: Optional[str] = field(
		default=None,
		metadata={
			"help": "Pretrained config name or path if not the same as model_name"
		},
	)
	tokenizer_name: Optional[str] = field(
		default=None,
		metadata={
			"help": "Pretrained tokenizer name or path if not the same as model_name"
		},
	)
	cache_dir: Optional[str] = field(
		default=None,
		metadata={
			"help": "Where do you want to store the pretrained models downloaded from huggingface.co"
		},
	)
	use_fast_tokenizer: bool = field(
		default=True,
		metadata={
			"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."
		},
	)
	model_revision: str = field(
		default="main",
		metadata={
			"help": "The specific model version to use (can be a branch name, tag name or commit id)."
		},
	)
	use_auth_token: bool = field(
		default=False,
		metadata={
			"help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
			"with private models)."
		},
	)


@dataclass
class Task:
	id: int
	name: str
	type: str
	num_labels: int


def tokenize_seq_classification_dataset(
	tokenizer, raw_datasets, task_id, task_name, data_args, training_args
):

	# Padding strategy
	if data_args.pad_to_max_length:
		padding = "max_length"
	else:
		# We will pad later, dynamically at batch creation, to the max sequence length in each batch
		padding = False

	# padding strategy
	if data_args.max_seq_length > tokenizer.model_max_length:
		logger.warning(
			f"The max_seq_length passed ({data_args.max_seq_length}) is larger than the maximum length for the"
			f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
		)

	max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)

	def tokenize_text(examples):
		result = tokenizer(
			examples["sentence"],
			padding=padding,
			max_length=max_seq_length,
			truncation=True,
		)
		examples["labels"] = examples.pop("label")
		result["task_ids"] = [task_id] * len(examples["labels"])

		return result

	# トークナイズとパディング
	def tokenize_and_pad_text(examples):

		result = tokenize_text(examples)
		examples["labels"] = [
			[l] + [-100] * (max_seq_length - 1) for l in examples["labels"]
		]
		return result

	with training_args.main_process_first(desc="dataset map pre-processing"):
		# col_to_remove = ["idx", "sentence"]

		train_dataset = raw_datasets["train"].map(
			tokenize_and_pad_text,
			batched=True,
			load_from_cache_file=not data_args.overwrite_cache,
			# remove_columns=col_to_remove,
			desc="Running tokenizer on dataset",
		)

		validation_dataset = raw_datasets["validation"].map(
			tokenize_text,
			batched=True,
			load_from_cache_file=not data_args.overwrite_cache,
			# remove_columns=col_to_remove,
			desc="Running tokenizer on dataset",
		)
	return train_dataset, validation_dataset


def load_seq_classification_dataset(
	task_id,
	task_name,
	tokenizer,
	data_args,
	training_args,
	model_args,
	train_path,
	eval_path,
	num_labels,
):

	data_files = {
		"train": train_path,
		"validation": eval_path,
	}
	raw_datasets = load_dataset(
		"csv", data_files=data_files, cache_dir=model_args.cache_dir
	)

	train_dataset, validation_dataset = tokenize_seq_classification_dataset(
		tokenizer,
		raw_datasets,
		task_id,
		task_name,
		data_args,
		training_args,
	)

	task_info = Task(
		id=task_id,
		name=task_name,
		num_labels=num_labels,
		type="seq_classification",
	)

	return (train_dataset, validation_dataset, task_info)


def load_datasets(tokenizer, data_args, training_args, model_args):

	# task1 negaposi
	(
		train_dataset_task1,
		validation_dataset_task1,
		task1_info,
	) = load_seq_classification_dataset(
		0,
		"negaposi",
		tokenizer,
		data_args,
		training_args,
		model_args,
		data_args.train_file_task1,
		data_args.validation_file_task1,
		3,
	)
	# task2 category
	(
		train_dataset_task2,
		validation_dataset_task2,
		task2_info,
	) = load_seq_classification_dataset(
		1,
		"category",
		tokenizer,
		data_args,
		training_args,
		model_args,
		data_args.train_file_task2,
		data_args.validation_file_task2,
		5,
	)

	# merge train datasets

	train_dataset_df = train_dataset_task1.to_pandas().append(
		train_dataset_task2.to_pandas()
	)
	train_dataset = datasets.Dataset.from_pandas(train_dataset_df)
	train_dataset.shuffle(seed=123)

	# Append validation datasets
	validation_dataset = [validation_dataset_task1, validation_dataset_task2]
	dataset = datasets.DatasetDict(
		{"train": train_dataset, "validation": validation_dataset}
	)
	tasks = [task1_info, task2_info]
	return tasks, dataset


# text Classification
class ClassificationHead(torch.nn.Module):
	def __init__(self, hidden_size: int, num_labels: int, dropout_p=0.1):
		super().__init__()
		self.num_labels = num_labels
		self.dropout = torch.nn.Dropout(dropout_p)
		self.classifier = torch.nn.Linear(hidden_size, num_labels)

		self._init_weights()

	def forward(self, pooled_output, labels=None, **kwargs):
		pooled_output = self.dropout(pooled_output)
		logits = self.classifier(pooled_output)

		# calculate loss
		loss = None
		if labels is not None:
			if labels.dim() != 1:
				# remove padding
				labels = labels[:, 0]
			loss_fct = torch.nn.CrossEntropyLoss(label_smoothing=0.05)
			# print(f"logits:{logits.view(-1, self.num_labels)}")
			# print(f"logits:{labels.long().view(-1)}")
			loss = loss_fct(logits.view(-1, self.num_labels), labels.long().view(-1))
		return (logits, loss)

	def _init_weights(self):
		self.classifier.weight.data.normal_(mean=0.0, std=0.02)
		if self.classifier.bias is not None:
			self.classifier.bias.data.zero_()


# multi tasking classification
class MultiTaskModel(torch.nn.Module):
	def __init__(self, encoder_name: str, tasks: List[Task]):
		super().__init__()

		self.encoder = AutoModel.from_pretrained(encoder_name)

		self.output_heads = torch.nn.ModuleDict()
		self.output_heads[str(tasks[0].id)] = ClassificationHead(
			hidden_size=self.encoder.config.hidden_size, num_labels=tasks[0].num_labels
		)
		self.output_heads[str(tasks[1].id)] = ClassificationHead(
			hidden_size=self.encoder.config.hidden_size, num_labels=tasks[1].num_labels
		)

	"""
	https://huggingface.co/transformers/v3.0.2/model_doc/bert.html#transformers.BertModel
	-------parameter-----
	input_id : Indices of input sequence tokens in the vocabulary.
	attention_mask : Mask to avoid performing attention on padding token indices. Mask values selected in [0, 1]: 1 for tokens that are NOT MASKED, 0 for MASKED tokens.
	token_type : Segment token indices to indicate first and second portions of the inputs. Indices are selected in [0, 1]: 0 corresponds to a sentence A token, 1 corresponds to a sentence B token   
	position_id : Indices of positions of each input sequence tokens in the position embeddings. Selected in the range
	head_mask : Mask to nullify selected heads of the self-attention modules. Mask values selected in [0, 1]: 1 indicates the head is not masked, 0 indicates the head is masked.
	inuts_embeds : Optionally, instead of passing input_ids you can choose to directly pass an embedded representation. This is useful if you want more control over how to convert input_ids indices into associated vectors than the model's internal embedding lookup matrix.
	encoder_hidden : Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention if the model is configured as a decoder.
	output_attentions :  Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in the cross-attention if the model is configured as a decoder. Mask values selected in [0, 1]: 1 for tokens that are NOT MASKED, 0 for MASKED tokens.
	labels :If set to True, the attentions tensors of all attention layers are returned. See attentions under returned tensors for more detail.

	---------return-------- 

	"""

	def forward(
		self,
		input_ids: torch.LongTensor = None,
		attention_mask: torch.FloatTensor = None,
		token_type_ids: torch.LongTensor = None,
		position_id: torch.LongTensor = None,
		head_mask: torch.FloatTensor = None,
		inputs_embeds: torch.FloatTensor = None,
		output_attentions: torch.FloatTensor = None,
		labels=None,
		task_ids=None,
		**kwargs,
	):
		outputs = self.encoder(
			input_ids=input_ids,
			attention_mask=attention_mask,
			token_type_ids=token_type_ids,
			position_ids=position_id,
			head_mask=head_mask,
			inputs_embeds=inputs_embeds,
			# encoder_hidden_state=encoder_hidden_state,
			output_attentions=output_attentions,
		)

		pooler_output = outputs["pooler_output"]

		# unique_task_ids_list = torch.unique(task_ids).tolist()
		unique_task_ids_list = [0, 1]

		loss_list = []
		logits = []
		for unique_task_id in unique_task_ids_list:
			task_id_filter = task_ids == unique_task_id

			(task_logits, task_loss) = self.output_heads[str(unique_task_id)].forward(
				pooler_output[task_id_filter],
				labels=None if labels is None else labels[task_id_filter],
			)
			# logit
			if task_logits == []:
				logits.append(torch.tensor(0).to(device))
			else:
				logits.append(task_logits)
			# loss
			if torch.any(torch.isnan(task_loss)):
				loss_list.append(torch.tensor(0).to(device))
			else:
				loss_list.append(task_loss)

		task_logits = []
		count1 = 0
		count2 = 0
		pad = torch.tensor([-100.0, -100.0]).to(device)
		for id in task_ids:
			if id == 0:
				temp = logits[0][count1]
				l = torch.cat((temp, pad), 0)
				task_logits.append(l)
				count1 += 1
			elif id == 1:
				task_logits.append(logits[1][count2])
				count2 += 1
			else:
				logger.error("例外発生")
				sys.exit(1)

		loss = torch.stack(loss_list)
		loss_ave = loss.sum() / len(unique_task_ids_list)
		task_logits = torch.stack(task_logits)
		outputs = (loss_ave, task_logits, task_ids)

		return outputs


def main():
	# See all possible arguments in src/transformers/training_args.py
	# or by passing the --help flag to this script.
	# We now keep distinct sets of args, for a cleaner separation of concerns.

	parser = HfArgumentParser(
		(ModelArguments, DataTrainingArguments, TrainingArguments)
	)

	if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
		# If we pass only one argument to the script and it's the path to a json file,
		# let's parse it to get our arguments.
		model_args, data_args, training_args = parser.parse_json_file(
			json_file=os.path.abspath(sys.argv[1])
		)
	else:
		(
			model_args,
			data_args,
			training_args,
		) = parser.parse_args_into_dataclasses()

	# Setup logging
	logging.basicConfig(
		format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
		datefmt="%m/%d/%Y %H:%M:%S",
		handlers=[logging.StreamHandler(sys.stdout)],
	)

	log_level = logging.DEBUG
	logger.setLevel(log_level)
	datasets.utils.logging.set_verbosity(log_level)
	transformers.utils.logging.set_verbosity(log_level)
	transformers.utils.logging.enable_default_handler()
	transformers.utils.logging.enable_explicit_format()

	# Log on each process the small summary:
	logger.warning(
		f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
		+ f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
	)
	logger.info(f"Training/evaluation parameters {training_args}")

	# Detecting last checkpoint.
	last_checkpoint = None
	if (
		os.path.isdir(training_args.output_dir)
		and training_args.do_train
		and not training_args.overwrite_output_dir
	):
		last_checkpoint = get_last_checkpoint(training_args.output_dir)
		if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
			raise ValueError(
				f"Output directory ({training_args.output_dir}) already exists and is not empty. "
				"Use --overwrite_output_dir to overcome."
			)
		elif (
			last_checkpoint is not None and training_args.resume_from_checkpoint is None
		):
			logger.info(
				f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
				"the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
			)

	# Set seed before initializing model.
	set_seed(training_args.seed)

	tokenizer = AutoTokenizer.from_pretrained(
		model_args.model_name_or_path,
		cache_dir=model_args.cache_dir,
		use_fast=model_args.use_fast_tokenizer,
		revision=model_args.model_revision,
		use_auth_token=True if model_args.use_auth_token else None,
	)

	# datasetsの作成
	tasks, raw_datasets = load_datasets(tokenizer, data_args, training_args, model_args)

	# モデルの定義
	model = MultiTaskModel(model_args.model_name_or_path, tasks)

	# train dataの取得
	if training_args.do_train:
		if "train" not in raw_datasets:
			raise ValueError("--do_train requires a train dataset")
		train_dataset = raw_datasets["train"]
		if data_args.max_train_samples is not None:
			train_dataset = train_dataset.select(range(data_args.max_train_samples))

	# validation dataの取得
	if training_args.do_eval:
		if (
			"validation" not in raw_datasets
			and "validation_matched" not in raw_datasets
		):
			raise ValueError("--do_eval requires a validation dataset")
		eval_dataset = raw_datasets["validation"]
		if data_args.max_eval_samples is not None:
			new_ds = []
			for ds in eval_dataset:
				new_ds.append(ds.select(range(data_args.max_eval_samples)))

			eval_dataset = new_ds

	# Log a few random samples from the training set:
	if training_args.do_train:
		for index in random.sample(range(len(train_dataset)), 2):

			logger.info(
				f"Sample {index}/{len(train_dataset)} of the training set: {train_dataset[index]}."
			)

	eval_datasets_df = eval_dataset[0].to_pandas().append(eval_dataset[1].to_pandas())
	eval_datasets = datasets.Dataset.from_pandas(eval_datasets_df)
	eval_datasets.shuffle(seed=123)

	# metric = load_metric("accuracy")
	# 評価時にメトリクスを計算するために使用される関数です。
	# EvalPredictionを受け取り、メトリクスの値のディクショナリ文字列を返す必要があります。
	def compute_metrics(p: EvalPrediction):
		# output
		preds = p.predictions[0]
		task_ids = p.predictions[1]
		unique_task_ids_list = [1, 0]
		# true label
		label = p.label_ids.astype(int)
		# metric

		# precision, recall, f1, accuracy = [], [], [], []
		accuracy = []

		for unique_task_id in unique_task_ids_list:
			task_id_filter = task_ids == unique_task_id
			p = preds[task_id_filter]
			l = label[task_id_filter]
			if len(p) != len(l):
				sys.exit(1)
			p = np.argmax(p, axis=1)
			if len(p) != 0:
				# Remove ignored index (special tokens)
				# results = metric.compute(predictions=p, references=l)
				# precision.append(results["overall_precision"])
				# recall.append(results["overall_recall"])
				# f1.append(results["overall_f1"])
				# accuracy.append(results["accuracy"])
				result = (p == l).astype(np.float32).mean().item()
				accuracy.append(result)
			else:
				accuracy.append(0)

		# result
		return {
			# "precision_task1": precision[0],
			# "precision_task2": precision[1],
			# "recall_task1": recall[0],
			# "recall_task2": recall[1],
			# "f1_task1": f1[0],
			# "f1_task2": f1[1],
			"accuracy_task1": accuracy[0],
			"accuracy_task2": accuracy[1],
		}

	# Initialize our Trainer
	trainer = Trainer(
		model=model,
		args=training_args,
		train_dataset=train_dataset if training_args.do_train else None,
		eval_dataset=eval_datasets if training_args.do_eval else None,
		compute_metrics=compute_metrics,
		tokenizer=tokenizer,
		# callbacks=[
		#    EarlyStoppingCallback(early_stopping_patience=data_args.early_stopping_step)
		# ],
	)

	# Training
	if training_args.do_train:
		checkpoint = None
		if training_args.resume_from_checkpoint is not None:
			checkpoint = training_args.resume_from_checkpoint
		elif last_checkpoint is not None:
			checkpoint = last_checkpoint

		train_result = trainer.train(resume_from_checkpoint=checkpoint)
		metrics = train_result.metrics
		max_train_samples = (
			data_args.max_train_samples
			if data_args.max_train_samples is not None
			else len(train_dataset)
		)
		metrics["train_samples"] = min(max_train_samples, len(train_dataset))

		trainer.save_model()  # Saves the tokenizer too for easy upload

		trainer.log_metrics("train", metrics)
		trainer.save_metrics("train", metrics)
		trainer.save_state()

	# Evaluation
	if training_args.do_eval:

		# タスクごとに評価
		for eval_d, task in zip(eval_dataset, tasks):
			logger.info(f"*** Evaluate of {task.name} ***")

			metrics = trainer.evaluate(eval_dataset=eval_d)

			max_eval_samples = (
				data_args.max_eval_samples
				if data_args.max_eval_samples is not None
				else len(eval_d)
			)
			metrics["eval_samples"] = min(max_eval_samples, len(eval_d))

			trainer.log_metrics(f"eval_{task.name}", metrics)
			trainer.save_metrics(f"eval_{task.name}", metrics)

	# if training_args.do_predict:

	#     logger.info("*** Predict ***")

	#     # Loop to handle MNLI double evaluation (matched, mis-matched)
	#     tasks = [data_args.task_name]
	#     predict_datasets = [predict_dataset]
	#     if data_args.task_name == "mnli":
	#         tasks.append("mnli-mm")
	#         predict_datasets.append(raw_datasets["test_mismatched"])

	#     for predict_dataset, task in zip(predict_datasets, tasks):
	#         # Removing the `label` columns because it contains -1 and Trainer won't like that.
	#         predict_dataset = predict_dataset.remove_columns("label")
	#         predictions = trainer.predict(
	#             predict_dataset, metric_key_prefix="predict"
	#         ).predictions
	#         predictions = (
	#             np.squeeze(predictions)
	#             if is_regression
	#             else np.argmax(predictions, axis=1)
	#         )

	#         output_predict_file = os.path.join(
	#             training_args.output_dir, f"predict_results_{task}.txt"
	#         )
	#         if trainer.is_world_process_zero():
	#             with open(output_predict_file, "w") as writer:
	#                 logger.info(f"***** Predict results {task} *****")
	#                 writer.write("index\tprediction\n")
	#                 for index, item in enumerate(predictions):
	#                     if is_regression:
	#                         writer.write(f"{index}\t{item:3.3f}\n")
	#                     else:
	#                         item = label_list[item]
	#                         writer.write(f"{index}\t{item}\n")

	kwargs = {
		"finetuned_from": model_args.model_name_or_path,
		"tasks": "text-classification",
	}

	if training_args.push_to_hub:
		trainer.push_to_hub(**kwargs)
	else:
		trainer.create_model_card(**kwargs)


def _mp_fn(index):
	# For xla_spawn (TPUs)
	main()


if __name__ == "__main__":
	main()