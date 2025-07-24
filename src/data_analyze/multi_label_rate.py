from datasets import load_dataset
import json
from tap import Tap
from collections import Counter

class Args(Tap):
  dataset_path: str = "data/toxicity_ver2_allyesdata.jsonl"
  # dataset_path: str = "data/toxicity_dataset_ver2.jsonl"
  max_length: int = 32

def main(args):
  # def truncate_text(example):
  #   example["text"] = example["text"][:args.max_length]
  #   return example

  dataset = load_dataset("json", data_files=args.dataset_path, split="train")
  dataset = dataset.remove_columns(["label", "id", "text"])
  # dataset = dataset.map(truncate_text)
  tasknames = dataset.column_names
  tasknames = [task for task in tasknames if task not in ["id", "text"]]
  print("Category: ", tasknames)
  # Category:  ['obscene', 'discriminatory', 'violent', 'illegal', 'personal', 'corporate', 'others']

  # make category rank list
  label_yes_counts = {taskname: Counter(dataset[taskname])["yes"] for taskname in tasknames}
  category_rank = [label for (label, _) in 
                  sorted(label_yes_counts.items(), key=lambda x: x[1])]
  print("Category rank:", category_rank)
  # Category rank: ['others', 'personal', 'illegal', 'violent', 'discriminatory', 'corporate', 'obscene']

  # set category name
  def encode_labels(example):
    values = [1 if example[k] == "yes" else 0 for k in category_rank]
    category = category_rank[next((i for i, v in enumerate(values) if v), 0)]
    return {
      "category": category,
      "labels": values
    }
  dataset = dataset.map(encode_labels)
  print(dataset[:10])

  
if __name__ == '__main__':
  args = Args().parse_args()
  main(args)
