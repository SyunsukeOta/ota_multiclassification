from tap import Tap
import random

class Args(Tap):
  corpus_release: str = "CC-MAIN-2023-23"
  group: str = "CC-MAIN-20230527223515-20230528013515"
  file_count: int = 800
  category_len: int = 7
  categories: str = ['others', 'personal', 'illegal', 'violent', 'discriminatory', 'corporate', 'obscene']
  out_text_path: str = "analysis_results/setfit_random_toxic_texts.txt"

def main(args):
  print("corpus_time: ", args.corpus_release)
  print("group: ", args.group)
  print("file_count: ", args.file_count)
  i = 0
  ja_text_path = f"data/{args.corpus_release}/{args.group}/{i:05d}-ja-sentence.txt"
  ja_out_label_path = f"prediction_data/{args.corpus_release}/{args.group}/{i:05d}-ja-label.txt"
  print(i, ja_text_path, ja_out_label_path)
  label_ids = {i: [] for i in range(args.category_len)}
  # print("label_ids: ", label_ids)
  
  with open(ja_out_label_path, "r", encoding="utf-8") as f:
    for i, line in enumerate(f.readlines()):
      labels = [int(label) for label in line.strip().split(",")]
      for category_i in range(args.category_len):
        if labels[category_i] == 1:
          label_ids[category_i].append(i)
      

  with open(ja_text_path, "r", encoding="utf-8") as f:
    texts = f.readlines()

  # for category_i in range(args.category_len):
  # 'others', 'personal', 'illegal', 'violent', 'discriminatory', 'corporate', 'obscene'
  category_i = 2
  print(f"category {category_i} has {len(label_ids[category_i])} labels")
  random_text_ids = random.sample(label_ids[category_i], 10)
  for i, random_text_id in enumerate(random_text_ids):
    print(f"id {i}: {texts[random_text_id]}")

  with open(args.out_text_path, "w", encoding="utf-8") as f:
    for category_i in range(args.category_len):
      f.write(f"Category {args.categories[category_i]}:\n")
      random_text_ids = random.sample(label_ids[category_i], 10)
      for i, random_text_id in enumerate(random_text_ids):
        f.write(f"{texts[random_text_id]}")



if __name__ == "__main__":
  args = Args().parse_args()
  main(args)
