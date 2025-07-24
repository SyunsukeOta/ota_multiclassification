#PBS -q gLrchq
#PBS -l select=1:ncpus=4:mem=128G:ngpus=1
#PBS -v DOCKER_IMAGE=imc.tut.ac.jp/transformers-pytorch-cuda118:4.37.2,DOCKER_OPTIONS="--volume=/lwork:/lwork"
#PBS -k doe -j oe -o ./log

cd ${PBS_O_WORKDIR}

TORCH_HOME=/lwork/${LOGNAME}/.cache/torch
TRANSFORMERS_CACHE=/lwork/${LOGNAME}/.cache/transformers
HF_HOME=/lwork/${LOGNAME}/.cache/huggingface
TRITON_CACHE_DIR=/lwork/${LOGNAME}/.cache/triton
export TORCH_HOME TRANSFORMERS_CACHE HF_HOME TRITON_CACHE_DIR

# pip show datasets

poetry run python src/multi_classification_sample.py

mv "./log/${PBS_JOBID}.OU" "./log/${PBS_JOBNAME}.o${PBS_JOBID%.xregistry*}"
