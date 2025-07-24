from datasets import load_dataset
from transformers import (
	AutoTokenizer,
	AutoModel,
	AutoModelForSequenceClassification,
	BitsAndBytesConfig,
	DataCollatorWithPadding,
	Trainer,
	TrainingArguments,
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

from datetime import datetime
from pathlib import Path
from tap import Tap


class Args(Tap):
	num_labels: int = 2
	dataset_name: str = "imdb"
	model_name: str = "llm-jp/llm-jp-3-1.8b"
	output_dir: str = ""

	num_epochs: int = 1
	batch_size: int = 64
	per_device_batch_size: int = 2

	learning_rate: float = 2e-5
	weight_decay: float = 0.01
	warmup_ratio: float = 0.05
	lora_rank: int = 16
	lora_dropout: float = 0.05

	debug: bool = False
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
			num_train_epochs=self.num_epochs,
			eval_strategy="epoch",
			save_strategy="epoch",
			optim="adamw_torch",
			learning_rate=self.learning_rate,
			weight_decay=self.weight_decay,
			lr_scheduler_type="inverse_sqrt",
			warmup_ratio=args.warmup_ratio,
			bf16=self.use_bf16,
			fp16=not self.use_bf16,
			load_best_model_at_end=True,
		)
		print(x.to_dict())
		return x

	def peft_config(self):
		x = LoraConfig(
			r=self.lora_rank,
			lora_alpha=self.lora_rank * 2,
			lora_dropout=self.lora_dropout,
			inference_mode=False,
			target_modules="all-linear",
		)
		print(x.to_dict())
		return x


class Classifier(torch.nn.Module):
	def __init__(self, encoder, num_labels, bias=False, dtype=None):
		super().__init__()
		self.add_module("encoder", encoder)
		self.add_module("classifier",
			torch.nn.Linear(encoder.config.hidden_size,
				num_labels,
				bias=bias,
				dtype=dtype))
		self.loss_fn = torch.nn.CrossEntropyLoss()
		# print('--Classifier init--')
		# print(f'num_labels:{num_labels}, type: {type(num_labels)}')
		# print(f'encoder:{self.encoder}, type: {type(self.encoder)}')
		# print(f'classifier:{self.classifier}, type: {type(self.classifier)}')
		self.forwardcount = 1


	def forward(self, input_ids, attention_mask, labels):
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
		logits = self.classifier(eos_hidden_states)
		loss = self.loss_fn(logits, labels)

		print(f'--Classifier {self.forwardcount}_forward--')
		print('@@a@@',logits, labels)
		# self.forwardcount += 1
		# print(f'input_ids:{input_ids}, type: {type(input_ids)}, shape: {input_ids.shape}')
		# print(f'attention_mask:{attention_mask}, type: {type(attention_mask)}, shape: {attention_mask.shape}')
		# print(f'labels:{labels}, type: {type(labels)}, shape: {labels.shape}')
		# print(f'outputs:{outputs}, type: {type(outputs)}')
		# print(f'logits:{logits}, type: {type(logits)}, shape: {logits.shape}')
		# print(f'loss:{loss}, type: {type(loss)}, shape: {loss.shape}')
		# print(f'labels:{labels}, type: {labels}')
		return SequenceClassifierOutput(
			loss=loss,
			logits=logits,
		)


def main(args):
	tokenizer = AutoTokenizer.from_pretrained(args.model_name)

	dataset = load_dataset(args.dataset_name)
	if args.debug:
		dataset["train"] = dataset["train"].select(range(128))
		dataset["test"] = dataset["test"].select(range(16))
	def preprocess_function(examples):
		return tokenizer(examples["text"], truncation=True)
	tokenized_dataset = dataset.map(preprocess_function, batched=True)
	#print(dataset["train"][:2])
	#print(tokenized_dataset["train"][:2])
	data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

	accuracy = evaluate.load("accuracy")
	def compute_metrics(eval_pred):
		predictions, labels = eval_pred
		predictions = np.argmax(predictions, axis=1)
		print('--compute metrics--')
		print(f'labels:{labels}, type: {type(labels)}')
		return accuracy.compute(predictions=predictions, references=labels)

	# https://huggingface.co/docs/peft/developer_guides/quantization
	quantization_config = BitsAndBytesConfig(
		load_in_4bit=True,
		bnb_4bit_quant_type="nf4",
		bnb_4bit_use_double_quant=True,
		bnb_4bit_compute_dtype=args.torch_dtype,
	)
	model = AutoModel.from_pretrained(
		args.model_name,
		device_map="auto",
		torch_dtype=args.torch_dtype,
		trust_remote_code=True,
		quantization_config=quantization_config,
	)
	print(f"Loaded model uses {model.get_memory_footprint()} bytes")
	model = prepare_model_for_kbit_training(model)
	model = get_peft_model(model, args.peft_config())
	model.print_trainable_parameters()
	model = Classifier(model, args.num_labels, dtype=args.torch_dtype)

	# Original code was borrowed from https://huggingface.co/docs/transformers/tasks/sequence_classification
	# id2label = {0: "NEGATIVE", 1: "POSITIVE"}
	# label2id = {"NEGATIVE": 0, "POSITIVE": 1}
	# model = AutoModelForSequenceClassification.from_pretrained(
	#     model_name,
	#     num_labels=2,
	#     id2label=id2label,
	#     label2id=label2id
	# )

	trainer = Trainer(
		model=model,
		args=args.training_args(),
		train_dataset=tokenized_dataset["train"],
		eval_dataset=tokenized_dataset["test"],
		processing_class=tokenizer,
		data_collator=data_collator,
		compute_metrics=compute_metrics,
	)

	train_result = trainer.train()
	# trainer.save_metrics("train", train_result.metrics)
	# trainer.save_metrics("eval", trainer.evaluate())
	# trainer.save_model(args.output_dir)
	# tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
	args = Args().parse_args()
	main(args)
