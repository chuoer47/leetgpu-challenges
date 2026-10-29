#include <cuda_runtime.h>
#include <limits.h>

#define WARP_SIZE 32
__device__ __forceinline__ int warp_reduce(int val) {
#pragma unroll
  for (int mask = WARP_SIZE >> 1; mask >= 1; mask >>= 1) {
    val = max(val,__shfl_xor_sync(0xffffffff, val, mask));
  }
  return val;
}

__global__ void perfix_sum_kernel(
    const int* __restrict__ input,
    int* __restrict__ prefix,
    int N
){
    __shared__ int block_sums[256];
    int tid = threadIdx.x;
    int blockSize = blockDim.x;
    if (tid == 0) prefix[0] = 0;

    int elems_per_thread = (N + blockSize - 1) / blockSize;
    int start = tid * elems_per_thread;
    int end = min(N, start + elems_per_thread);

    int local_sum = 0;
    for (int i = start;i < end; ++i){
        local_sum += input[i];
    }
    block_sums[tid] = local_sum;
    __syncthreads();

    if (tid == 0){
        int acc = 0;
        for (int i = 0;i < blockSize;++i){
            int tmp = block_sums[i];
            block_sums[i] = acc;
            acc += tmp;
        }
    }
    __syncthreads();

    int offset = block_sums[tid];
    int running = offset;
    for (int i = start;i < end;++i){
        running += input[i];
        prefix[i + 1] = running;
    }
}

__global__ void max_window_sum_kernel(
    int* __restrict__ prefix,
    int* output,
    int N, int M, int window_size
){
    __shared__ int sdata[256 / WARP_SIZE];
    int tid = threadIdx.x;
    int blockSize = blockDim.x;
    int lane = tid & 31;
    int warp_id = tid >> 5;
    int local_max = INT_MIN;

    for (int i = tid; i < M; i += blockSize) {
        int sum = prefix[i + window_size] - prefix[i];
        local_max = max(local_max, sum);
    }
    local_max = warp_reduce(local_max);
    if (lane == 0) sdata[warp_id] = local_max;
    __syncthreads();

    if (warp_id == 0) {
        local_max = (lane < 256 / WARP_SIZE)?sdata[lane]:INT_MIN;
        local_max = warp_reduce(local_max);
        *output = local_max;
    }
}

// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const int* input, int* output, int N, int window_size) {
    int M = N - window_size + 1;
    int* prefix = nullptr;
    cudaMalloc(&prefix,(N + 1) * sizeof(int));

    int threads = 256;
    perfix_sum_kernel<<<1,threads>>>(input,prefix,N);
    max_window_sum_kernel<<<1,threads>>>(prefix,output,N,M,window_size);

    cudaFree(prefix);
}
