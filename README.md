# LoRA を用いた他クラス分類モデル

## フォルダ構成
analysis_results: 分析結果の表データ

log: 実行ログファイルの保存先

src: 実行するソースコードの保存先
  - data_analysis
    - label_rate_analysis.py
  - multi_classification_sample.py: 他クラス分類モデルの学習
  - prediction_a.py: データの有害ラベルの予測

imgs: 学習済みモデルの分析結果の画像データ

outputs: 学習済みモデルの保存先
prediction_data: 予測データの保存先

multi-classification.sh: multi_classification_sample.py の実行

prediction_anime.sh: アニメに関するツイートデータを取得し、prediction_a.py を実行

## 他クラス分類モデルの学習 (multi_classification_sample.py)

## 使用するデータについて
使用データ: [LLM-jp Toxicity Dataset v2](https://gitlab.llm-jp.nii.ac.jp/datasets/llm-jp-toxicity-dataset-v2)

各データにおける yes の個数(合計 3847 件、カテゴリは 7 個)
{'0': 1965, '1': 1468, '2': 382, '3': 28, '4': 3, '5': 0, '6': 1, '7': 0}

- ダウンサンプリングのために no のデータを削ろうと考えたが、半数のデータがどれか一つはラベルがついていた。
- しかし、2 つ以上の yes がついているデータは少なかった (10.7%) 
** 各カテゴリの yes/no の割合が 50% に近くなるようにデータを削る必要があるが、yes のデータは削りたくない**
- とりあえず、全てが no のデータを削ってみて割合を調べてみる。
  1. まず、全てが no のデータを削る
  2. 次に、そのデータセットから各カラムに対して yes/no の割合・数を調べる

実行結果 (1882 件)
  ```
  'obscene': yes..63.39%(1193), no..36.61%(689)
  'discriminatory': yes..22.21%(418), no..77.79%(1464)
  'violent': yes..8.98%(169), no..91.02%(1713)
  'illegal': yes..3.24%(61), no..96.76%(1821)
  'personal': yes..2.18%(41), no..97.82%(1841)
  'corporate': yes..22.48%(423), no..77.52%(1459)
  'others': yes..1.54%(29), no..98.46%(1853)
  ```
- `obscene` の `yes` のデータの個数が他のカテゴリよりも圧倒的に多いため、 `obscene` だけ `yes` の割合が大きくなりすぎた

- そもそも、`other`, `illegal`, `personal` のデータが 100 件も無いためそのままでは学習には使えないのでは？
  - 最終の評価指標 (`accuracy`, `f1`, `recall`, `precision`) の計算は各カテゴリ毎で計算している
  - 学習における `forward` での `logits`, `labels` を用いた `loss` の計算には各カテゴリのデータを同時に使用している。
  - このままでは、`loss` の計算に不均衡データの影響が出てしまうため一旦取り外して計算してみる
    - 使用するカテゴリ: `obscene`, `discriminatory`, `violent`, `corporate`

### `label` について
このデータセットには、label カテゴリがあり、`toxic`, `nontoxic`, `has_toxic_expression` という値のいずれかが割り当てられる。


### 学習について
各カテゴリで yes/no の割合が違うため様々な場合で学習を行う
- 使用するカテゴリ: `obscene`, `discriminatory`, `violent`, `corporate`
1. 元のデータセット(`data/toxicity_dataset_ver2.jsonl`)
2. いずれかが `yes` であるデータセット(`data/toxicity_ver2_allyesdata.jsonl`)
  ```
  1839
  columns:  ['obscene', 'discriminatory', 'violent', 'corporate']
  'obscene': yes..64.87%(1193), no..35.13%(646)
  'discriminatory': yes..22.73%(418), no..77.27%(1421)
  'violent': yes..9.19%(169), no..90.81%(1670)
  'corporate': yes..23.00%(423), no..77.00%(1416)
  ```
3. さらに `violent` を取り除いて、`corporate`, `discriminatory` の `yes` の割合を増やしつつ、`obscene` のデータの割合を減らす(すべて、できるだけ 50% に近づける)(`data/toxicity_ver2_downsampling.jsonl`)
- `violent` を除去
```
1802
columns:  ['obscene', 'discriminatory', 'corporate']
'obscene': yes..66.20%(1193), no..33.80%(609)
'discriminatory': yes..23.20%(418), no..76.80%(1384)
'corporate': yes..23.47%(423), no..76.53%(1379)
```
- `corporate`, `discriminatory` が `no` のデータを削除する
```
828
columns:  ['obscene', 'discriminatory', 'corporate']
'obscene': yes..26.45%(219), no..73.55%(609)
'discriminatory': yes..50.48%(418), no..49.52%(410)
'corporate': yes..51.09%(423), no..48.91%(405)
```
  - `obscene` が `no` のデータを減らしすぎた。原因としては、`corporate`, `discriminatory` が `no` のデータを全て削除したことが原因
  - 次は、`obscene` の`yes/no` の割合が 50% になるまで行う。
```
1193 584
['id', 'text', 'label', 'obscene', 'discriminatory', 'corporate']
delete_len:  584
columns:  ['obscene', 'discriminatory', 'corporate']
'obscene': yes..50.00%(609), no..50.00%(609)
'discriminatory': yes..34.32%(418), no..65.68%(800)
'corporate': yes..34.73%(423), no..65.27%(795)
```

学習結果
```json
{
  "epoch": 0.9953051643192489,
  "eval_corporate_accuracy": 0.6092896174863388,
  "eval_corporate_f1": 0.42570281124497994,
  "eval_corporate_precision": 0.424,
  "eval_corporate_recall": 0.4274193548387097,
  "eval_discriminatory_accuracy": 0.5710382513661202,
  "eval_discriminatory_f1": 0.29596412556053814,
  "eval_discriminatory_precision": 0.358695652173913,
  "eval_discriminatory_recall": 0.25190839694656486,
  "eval_loss": 1.608913779258728,
  "eval_obscene_accuracy": 0.5792349726775956,
  "eval_obscene_f1": 0.5549132947976878,
  "eval_obscene_precision": 0.5614035087719298,
  "eval_obscene_recall": 0.5485714285714286,
  "eval_runtime": 95.8014,
  "eval_samples_per_second": 3.82,
  "eval_steps_per_second": 1.91,
  "total_flos": 0.0,
  "train_loss": 1.7642282449974205,
  "train_runtime": 824.0257,
  "train_samples_per_second": 1.034,
  "train_steps_per_second": 0.129
}
```

## todo
- ファイル名の整理: 実行する python ファイルにおいて、似たような内容のファイルが複数存在するためこれらを整理する
  - モデルの学習: imdb-sentiment-analysis.py, multi_classification_sample.py, sentiment-analysis.py, undersampling_multi_classification.py
  - 有害ラベルの予測: prediction_a.py, prediction_sample.py, prediction.py
- モデルの学習の中身を README に記載
- 不均衡データ (LLM-jp Toxicity Dataset v2) をダウンサンプリングさせて 1 epoch で学習
  - ダウンサンプリングしたデータを Trainer に投げて複数 epoch で学習させる方法が思いつかないので、一旦 1 epoch で実行
  - データが複数カテゴリあるので、どのカテゴリにも属していないデータとそれ以外のデータの割合を調べてみる