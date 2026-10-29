#include <cuda_runtime.h>
#define BN 256
#define WARP_SIZE 32

template <const int kWarpSize = WARP_SIZE>
__device__ __forceinline__ int warp_reduce_sum(int val) {
#pragma unroll
  for (int mask = kWarpSize >> 1; mask >= 1; mask >>= 1) {
    val += __shfl_xor_sync(0xffffffff, val, mask);
  }
  return val;
}

__global__ void kernel(
    const int* __restrict__ input,
    int* output, int N, int S, int E
){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    int sum = 0;
    for (int i=idx;i + S <= E;i += stride){
        sum += input[i + S];
    }

    __shared__ int smem[BN / WARP_SIZE];
    int lane = threadIdx.x & 31;
    int warpId= threadIdx.x >> 5;
    
    sum = warp_reduce_sum<WARP_SIZE>(sum);
    if (lane == 0) smem[warpId] = sum;

    __syncthreads();

    if (warpId == 0){
        sum = (lane < BN / WARP_SIZE)?smem[lane]:0;
        sum = warp_reduce_sum<WARP_SIZE>(sum);
        if (lane == 0) atomicAdd(output,sum);
    }

}

// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const int* input, int* output, int N, int S, int E) {
    cudaMemset(output, 0, sizeof(int));

    int numSMs;
    cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, 0);

    int M = E - S + 1;
    int blocksNeeded = (M + BN - 1) / BN;
    int blocksPerGrid =  blocksNeeded < numSMs ? blocksNeeded : numSMs;

    kernel<<<blocksPerGrid, BN>>>(input,output,N,S,E);

    cudaDeviceSynchronize();
}
