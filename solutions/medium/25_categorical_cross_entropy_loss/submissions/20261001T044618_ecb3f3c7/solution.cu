#include <cuda_runtime.h>

#define BN 32
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
    const float* __restrict__ logits,
    const int* __restrict__ true_labels,
    float* loss,
    int N,
    int C
){
    int row_idx = blockIdx.x * blockDim.y + threadIdx.y;
    if (row_idx >= N) return;
    float s = 0.0f;
    const float* p = logits + row_idx * C;
    for (int i = threadIdx.x;i < C;i += WARP_SIZE){
        s += __expf(p[i]);
    }
    
    s = warp_reduce_sum_f32<WARP_SIZE>(s);

    if (threadIdx.x == 0){
        float tmp = (__logf(s) - p[true_labels[row_idx]]) / N;
        atomicAdd(loss,tmp);
    }

}

// logits, true_labels, loss are device pointers
extern "C" void solve(const float* logits, const int* true_labels, float* loss, int N, int C) {
    cudaMemset(loss, 0, sizeof(float));

    // 每个block负责BN行
    // Block内部，一个WARP负责一行
    dim3 threadsPerBlock(WARP_SIZE, BN);          // x = 32, y = BN
    dim3 blocksPerGrid((N + BN - 1) / BN);        // 注意分母是 BN，不是 N
    kernel<<<blocksPerGrid,threadsPerBlock>>>(logits,true_labels,loss,N,C);

    cudaDeviceSynchronize();
}
