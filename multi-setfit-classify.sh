GPUQOPTS="-q gLrchq -l select=1:ncpus=4:mem=192G:ngpus=2"
DOCKER_IMAGE="imc.tut.ac.jp/transformers-pytorch-cuda118:4.37.2"

echo "Start setfit classify"
for hypothesis in 1 2; do
  for train_no_rate in 0.5 1.0 2.0; do
    qsub ${GPUQOPTS} -N setfit_classify_hypothesis${hypothesis}_train_no_rate${train_no_rate} \
      -k doe -j oe -o ./log \
      -v DOCKER_IMAGE=${DOCKER_IMAGE},hypothesis=${hypothesis},train_no_rate=${train_no_rate} \
      setfit-classify.sh
  done
done
