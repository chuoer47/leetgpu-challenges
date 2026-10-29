#include <cuda_runtime.h>
#define BN 32
#define BM 8
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
    int* output, 
    int N, int M, int S_ROW, int E_ROW, int S_COL,int E_COL
){
    int sum = 0;
    int row_stride = gridDim.y * blockDim.y;
    int col_stride = gridDim.x * blockDim.x;

    for (int row = S_ROW + blockIdx.y * blockDim.y + threadIdx.y;row <= E_ROW;row += row_stride){
        const int* row_ptr = input + row * M;
        for (int col = S_COL + blockIdx.x * blockDim.x + threadIdx.x;col <= E_COL;col += col_stride){
            sum += row_ptr[col];
        }
    }

    sum = warp_reduce_sum<WARP_SIZE>(sum);

    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    int lane = tid & 31;
    int warp_id = tid >> 5;
    __shared__ int smem[BN * BM / 32];
    if (lane == 0) smem[warp_id] = sum;
    __syncthreads();

    if (warp_id == 0){
        sum = (lane < BN * BM / 32)?smem[lane]:0;
        sum = warp_reduce_sum<WARP_SIZE>(sum);
        if (lane == 0){
            atomicAdd(output,sum);
        } 
    }

}

// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(
    const int* input, 
    int* output, 
    int N, int M, int S_ROW, int E_ROW, int S_COL,int E_COL) {
        cudaMemset(output, 0, sizeof(int));
        int CN = E_ROW - S_ROW + 1;
        int CM = E_COL - S_COL + 1;
        dim3 grid(
            min((CM + BM - 1)/BM,16),
            min((CN + BN - 1)/BN,16));
        dim3 block(BM,BN);
        kernel<<<grid,block>>>(input,output,N,M,S_ROW,E_ROW,S_COL,E_COL);

        cudaDeviceSynchronize();
}
