#include <cuda_runtime.h>
#define threads 256
#define WARP_SIZE 32

template <const int kWarpSize = WARP_SIZE>
__device__ __forceinline__ float warp_reduce_sum(float val) {
#pragma unroll
  for (int mask = kWarpSize >> 1; mask >= 1; mask >>= 1) {
    val += __shfl_xor_sync(0xffffffff, val, mask);
  }
  return val;
}

__global__ void get_u_kernel(
    const float* __restrict__ input, 
    float *u,
    int N,int C
){
    __shared__ float smem[threads / WARP_SIZE];
    int idx = threadIdx.x;
    int col = blockIdx.x;
    int stride = blockDim.x;
    float sum = 0.0f;
    for (int row = idx;row < N;row += stride){
        sum += input[row * C + col];
    }

    sum = warp_reduce_sum<WARP_SIZE>(sum);
    int lane = idx & 31;
    int warp_id = idx >> 5;
    if (lane == 0) smem[warp_id] = sum;

    __syncthreads();

    if (warp_id == 0){
        sum = (lane < threads / WARP_SIZE)?smem[lane]:0.0f;
        sum = warp_reduce_sum<WARP_SIZE>(sum);
        if (lane == 0){
            u[col] = sum / N;
        }
    }
}

__global__ void get_detla_kernel(
    const float* __restrict__ input, 
    float *u,
    float *detla,
    int N,int C
){
    __shared__ float smem[threads / WARP_SIZE];
    int idx = threadIdx.x;
    int col = blockIdx.x;
    int stride = blockDim.x;
    float sum = 0.0f;
    for (int row = idx;row < N;row += stride){
        float d = (input[row * C + col] - u[col]);
        sum += d*d;
    }

    sum = warp_reduce_sum<WARP_SIZE>(sum);
    int lane = idx & 31;
    int warp_id = idx >> 5;
    if (lane == 0) smem[warp_id] = sum;

    __syncthreads();

    if (warp_id == 0){
        sum = (lane < threads / WARP_SIZE)?smem[lane]:0.0f;
        sum = warp_reduce_sum<WARP_SIZE>(sum);
        if (lane == 0){
            detla[col] = sum / N;
        }
    }
}


__global__ void batch_norm_kernel(
    const float* __restrict__ input, 
    const float* __restrict__ gamma, 
    const float* __restrict__ beta, 
    float* __restrict__ u,
    float* __restrict__ detla,
    float* output,
    int N, int C, float eps
){
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    int row = idx / C;
    int col = idx % C;
    if (row >= N || col >= C) return;
    float x = input[row*C + col];
    float uj = u[col];
    float dj = detla[col];
    float y = gamma[col] * (x - uj) / sqrt(dj + eps) + beta[col];
    output[row*C + col] = y;
}

// input, gamma, beta, output are device pointers
extern "C" void solve(
    const float* input, 
    const float* gamma, 
    const float* beta, 
    float* output,
    int N, int C, float eps) {
        float *u;
        float *detla;
        size_t bytes = (size_t)C * sizeof(float);
        cudaMalloc(&u,bytes);
        cudaMalloc(&detla,bytes);

        get_u_kernel<<<C,threads>>>(input,u,N,C);
        get_detla_kernel<<<C,threads>>>(input,u,detla,N,C);

        int blocks = (N*C + threads - 1) / threads;
        batch_norm_kernel<<<blocks, threads>>>(
            input, gamma, beta, u, detla, output, N, C, eps
        );

        cudaFree(u);
        cudaFree(detla);

        cudaDeviceSynchronize();
}

