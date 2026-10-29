#include <cuda_runtime.h>
#include <cuda_runtime.h>
#define THREADS 256
#define WARP_SIZE 32

template <const int kWarpSize = WARP_SIZE>
__device__ __forceinline__ int warp_reduce_sum(int val) {
#pragma unroll
  for (int mask = kWarpSize >> 1; mask >= 1; mask >>= 1) {
    val += __shfl_xor_sync(0xffffffff, val, mask);
  }
  return val;
}


__global__ void count_kernel(
    const int* __restrict__ input,
    int* output,
    int N, int K
){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    int N4 = N / 4;
    int sum = 0;
    const int4* in4 = reinterpret_cast<const int4*>(input);
    for (int i = idx;i < N4;i += stride){
        int4 p = in4[i];
        sum += (p.x == K)?1:0;
        sum += (p.y == K)?1:0;
        sum += (p.z == K)?1:0;
        sum += (p.w == K)?1:0;
    }

    int i = N4 * 4 + idx;
    if (i < N){
        sum += (input[i] == K)?1:0;
    }

    __shared__ int smem[THREADS / WARP_SIZE];
    int lane = threadIdx.x & 31;
    int warpId = threadIdx.x >> 5;

    sum = warp_reduce_sum<WARP_SIZE>(sum);
    if (lane == 0) smem[warpId] = sum;
    __syncthreads();

    if (warpId == 0){
      sum = (lane < THREADS / WARP_SIZE) ? smem[lane] : 0;
      sum = warp_reduce_sum<WARP_SIZE>(sum);
      if (lane == 0)  atomicAdd(output,sum);
    }

}

// input, output are device pointers
extern "C" void solve(const int* input, int* output, int N, int M, int K, int P) {
    N = N * M * K;
    K = P;
    int numSMs;
    cudaDeviceGetAttribute(&numSMs,cudaDevAttrMultiProcessorCount,0);
    int grids = min((N + THREADS - 1)/THREADS, numSMs);
    int blocks = THREADS;

    count_kernel<<<grids,blocks>>>(input,output,N,K);

    cudaDeviceSynchronize();
}