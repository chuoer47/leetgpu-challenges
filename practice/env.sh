# Source this file on the 4090 server: source practice/env.sh
# Reuses the existing environment without installing or replacing packages.
export CUDA_HOME="${LEETGPU_CUDA_HOME:-/home/amax01/miniconda3/envs/leetcuda}"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export NVCC_CCBIN="${NVCC_CCBIN:-$CUDA_HOME/bin/x86_64-conda-linux-gnu-g++}"
