#include <cuda_runtime.h>
#define BN 256

#define WARP_SIZE 32

template <const int kWarpSize = WARP_SIZE>
__device__ __forceinline__ float warp_reduce_sum_f32(float val) {
#pragma unroll
  for (int mask = kWarpSize >> 1; mask >= 1; mask >>= 1) {
    val += __shfl_xor_sync(0xffffffff, val, mask);
  }
  return val;
}

__global__ void kernel(
    const float* __restrict__ predictions,
    const float* __restrict__ targets, 
    float* mse, 
    int N
){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    float sum = 0.0f;

    int N4 = N / 4;
    const float4* p4 = reinterpret_cast<const float4*>(predictions);
    const float4* t4 = reinterpret_cast<const float4*>(targets);
    for (int i = idx;i < N4;i += stride){
      float4 a = p4[i];
      float4 b = t4[i];
      sum += (a.x - b.x) * (a.x - b.x) + \
        (a.y - b.y) * (a.y - b.y) + \
        (a.z - b.z) * (a.z - b.z) + \
        (a.w - b.w) * (a.w - b.w);
    }
    int i = N4 * 4 + idx;
    if (i < N){
      float d = predictions[i] - targets[i];
      sum += d*d;
    }

    __shared__ float warpsSums[BN / WARP_SIZE];
    int lane = threadIdx.x & 31;
    int warpId = threadIdx.x >> 5;

    sum = warp_reduce_sum_f32<WARP_SIZE>(sum);
    if (lane == 0) warpsSums[warpId] = sum;
    __syncthreads();

    if (warpId == 0){
      sum = (lane < BN / WARP_SIZE) ? warpsSums[lane] : 0.0f;
      sum = warp_reduce_sum_f32<WARP_SIZE>(sum);
      if (lane == 0)  atomicAdd(mse,sum / N);
    }


}

// predictions, targets, mse are device pointers
extern "C" void solve(const float* predictions, const float* targets, float* mse, int N) {
    cudaMemset(mse,0,sizeof(float));

    int numSMs;
    cudaDeviceGetAttribute(&numSMs,cudaDevAttrMultiProcessorCount,0);
    
    int blocksNeeded = (N + BN - 1) / BN;
    int blocksPerGrid = min(blocksNeeded,numSMs * 8);
    kernel<<<blocksPerGrid,BN>>>(predictions,targets,mse,N);

    cudaDeviceSynchronize();
}





