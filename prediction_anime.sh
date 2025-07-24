#PBS -q gLiotq
#PBS -l select=1:ncpus=4:mem=128G:ngpus=1
#PBS -v DOCKER_IMAGE=imc.tut.ac.jp/transformers-pytorch-cuda118:4.37.2,DOCKER_OPTIONS="--volume=/lwork:/lwork"
#PBS -k doe -j oe -o ./log

cd ${PBS_O_WORKDIR}

TORCH_HOME=/lwork/${LOGNAME}/.cache/torch
TRANSFORMERS_CACHE=/lwork/${LOGNAME}/.cache/transformers
HF_HOME=/lwork/${LOGNAME}/.cache/huggingface
TRITON_CACHE_DIR=/lwork/${LOGNAME}/.cache/triton
export TORCH_HOME TRANSFORMERS_CACHE HF_HOME TRITON_CACHE_DIR

echo "hello"

INPUT_DIR=/lwork/n213304/anime_tweet_analyze/data/raw/target_anime_tweet_text_period

OUTPUT_DIR=/work/s245302/multiclassification/prediction_data/anime

# MODEL_DIR=/work/s245302/multiclassification/outputs/llm-jp-3-1.8b/2025-02-20/10-00-29.700866/
MODEL_DIR=/work/s245302/multiclassification/outputs/llm-jp-3-1.8b/2025-02-25/10-03-58.042401/

# mkdir -p $OUTPUT_DIR

echo "start prediction"
for input_path in $INPUT_DIR/*.jsonl
do
	filename=$(basename "$input_path" .jsonl)
	output_path="${OUTPUT_DIR}/${filename}.json"
	model_path="${MODEL_DIR}"
	if [ ! -f $output_path ]; then
		echo "input_path: $input_path"
		# poetry run  src/multi_classification_sample.py --input_path ${input_path} --output_path ${output_path}
		# poetry run python src/prediction.py --model_dir ${model_path} --input_path ${input_path} --output_path ${output_path}
		poetry run python src/prediction_a.py --model_dir ${model_path} --input_path ${input_path} --output_path ${output_path}
		# poetry run python src/prediction_sample.py --model_dir ${model_path} --input_path ${input_path} --output_path ${output_path}
	else
		echo "skip: $output_path"
		continue
	fi
	break
done

mv "./log/${PBS_JOBID}.OU" "./log/${PBS_JOBNAME}.o${PBS_JOBID%.xregistry*}"


# poetry run accelerate launch --mixed_precision=bf16 src/prediction_emo_polarity_fine_to01.py --model_name rinna/japanese-gpt-neox-3.6b --batch_size 32 --epochs 2 --lr 5e-05 --input_path ${input_path} --output_path ${output_path}

# qsub prediction_anime.sh
