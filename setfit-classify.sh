START_TIME=$(date +%s)
echo "step1 start: $(date)"

cd ${PBS_O_WORKDIR}
TORCH_HOME=/work/${LOGNAME}/.cache/torch
TRANSFORMERS_CACHE=/work/${LOGNAME}/.cache/transformers
HF_HOME=/work/${LOGNAME}/.cache/huggingface
UV_CACHE_DIR=/work/${LOGNAME}/.cache/uv
PIP_CACHE_DIR=/work/${LOGNAME}/.cache/pip
TRITON_CACHE_DIR=/work/${LOGNAME}/.cache/triton
export TORCH_HOME TRANSFORMERS_CACHE HF_HOME UV_CACHE_DIR PIP_CACHE_DIR TRITON_CACHE_DIR

#ADHOC FIX to avoid "undefined symbol: cuModuleGetFunction" error.
export TRITON_LIBCUDA_PATH=/usr/local/cuda/lib64/stubs

#ADHOC FIX to avoid torch._inductor.exc.InductorError
#For more detail, see https://github.com/pytorch/pytorch/issues/119054
export TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas

export TORCH_USE_CUDA_DSA=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "hypothesis: ${hypothesis}, train_no_rate: ${train_no_rate}"
source .venv/bin/activate
time uv run python src/setfit-classify.py --hypothesis=${hypothesis} --train_no_rate=${train_no_rate}

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
echo "step1 is finished: $(date)"
echo "step1's time: $((DURATION / 3600))hours, $((DURATION % 3600 / 60))minutes, $((DURATION % 60))seconds"

mv "./log/${PBS_JOBID}.OU" "./log/${PBS_JOBNAME}.o${PBS_JOBID%.xregistry*}"
