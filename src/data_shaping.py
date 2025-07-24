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
import copy

class Args(Tap):
  dataset_path: str = "data/toxicity_dataset.jsonl"
  dataset_name: str = "imdb"
  model_name: str = "llm-jp/llm-jp-3-1.8b"
  datasplit_seed: str = 42
  
  use_bf16: bool = torch.cuda.is_bf16_supported()
  
  @property
  def torch_dtype(self):
    return torch.bfloat16 if self.use_bf16 else torch.float16



def main(args):
  tokenizer = AutoTokenizer.from_pretrained(args.model_name)
  
  # 1. get dataset
  dataset = load_dataset("json", data_files=args.dataset_path, split="train") # Datasetとして取るため
  imdb_dataset = load_dataset(args.dataset_name)
  
  # 2. shape data
  # print(type(dataset.column_names["train"]))
  dataset = dataset.remove_columns(["id", "label"])
  
  # 2.1 make categories
  categories = dataset.column_names
  categories.remove("text")
  # categories = [item for item in categories if item not in ["id", "label", "text"]]
  # print(categories)
  
  # 2.2 devide obscene
  curr_categories = categories.copy()
  curr_category = curr_categories.pop(0)
  # print(curr_category, curr_categories)
  curr_dataset = copy.deepcopy(dataset)
  curr_dataset = curr_dataset.remove_columns(curr_categories)
  # print(curr_dataset)
  curr_dataset = curr_dataset.rename_column(curr_category, "label")
  print(curr_dataset)
  curr_dataset = curr_dataset.map(lambda example: {"label": 1 if example["label"] == "yes" else 0})
  print('rename yes/no -> 1/0', curr_dataset.unique("label"))

  # 2.3 make tokenzied train/test dataset and data collator
  split_dataset = curr_dataset.train_test_split(test_size=0.3, seed=args.datasplit_seed)
  def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True)
  tokenized_dataset = split_dataset.map(preprocess_function, batched=True)
  data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
  print(tokenized_dataset, data_collator)

  # 3.1 quantization
  quantization_config = BitsAndBytesConfig(
		load_in_4bit=True,
		bnb_4bit_quant_type="nf4",
		bnb_4bit_use_double_quant=True,
		bnb_4bit_compute_dtype=args.torch_dtype,
	)


  # 3.2 model init
  model = AutoModelForSequenceClassification.from_pretrained(
		args.model_name,
		device_map="auto",
		torch_dtype=args.torch_dtype,
		trust_remote_code=True,
		quantization_config=quantization_config,
	)
  # 

if __name__ == '__main__':
  args = Args().parse_args()
  main(args)