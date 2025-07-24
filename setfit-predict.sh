#PBS -q gLrchq
#PBS -l select=1:ncpus=4:mem=64G:ngpus=1
#PBS -v SINGULARITY_IMAGE=imc.tut.ac.jp/transformers-pytorch-cuda118:4.46.3
#PBS -k doe -j oe -o ./log

cd ${PBS_O_WORKDIR}

TORCH_HOME=/work/${LOGNAME}/.cache/torch
TRANSFORMERS_CACHE=/work/${LOGNAME}/.cache/transformers
HF_HOME=/work/${LOGNAME}/.cache/huggingface
TRITON_CACHE_DIR=/work/${LOGNAME}/.cache/triton
export TORCH_HOME TRANSFORMERS_CACHE HF_HOME TRITON_CACHE_DIR

#ADHOC FIX to avoid "undefined symbol: cuModuleGetFunction" error.
export TRITON_LIBCUDA_PATH=/usr/local/cuda/lib64/stubs

# export TORCH_LOGS=""
export TORCHDYNAMO_DISABLE=1
export TORCH_COMPILE_DISABLE=1
export PYTHONOPTIMIZE=1
export PYTORCH_NO_CUDA_MEMORY_CACHING=1

# only run group 0
CORPUS_RELEASE="CC-MAIN-2023-23"
for group in $(ls data/${CORPUS_RELEASE}); do
  file_count=$(find data/${CORPUS_RELEASE}/${group} -type f | wc -l)
  mkdir -p "prediction_data/${CORPUS_RELEASE}/${group}"
  # echo "file_count: ${file_count} for group: ${group}"
  poetry run python src/setfit-predict.py \
    --corpus_release ${CORPUS_RELEASE} \
    --group ${group} \
    --file_count ${file_count}
  break
done


# poetry run python src/setfit-predict.py

mv "./log/${PBS_JOBID}.OU" "./log/${PBS_JOBNAME}.o${PBS_JOBID%.xregistry*}"
