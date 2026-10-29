#include <cuda_runtime.h>

constexpr int BLOCK_PER_ROWS = 4;
constexpr int WARP_SIZE = 32;
constexpr int BLOCK_SIZE = 256;
constexpr int WARP_NUM = BLOCK_SIZE / WARP_SIZE;

__device__ __forceinline__ float warp_reduce(float sum){
    #pragma unroll
    for (int offset=WARP_SIZE >> 1;offset > 0;offset>>=1){
        sum += __shfl_down_sync(0xffffffff,sum,offset);
    }
    return sum;
}

__device__ __forceinline__ float block_reduce(float sum){
    __shared__ float warp_sums[WARP_NUM];
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    sum = warp_reduce(sum);
    if (lane == 0) warp_sums[warp] = sum;
    __syncthreads();
    if (warp == 0){
        sum = (lane < WARP_NUM)?warp_sums[lane]:0.0f;
        sum = warp_reduce(sum);
    }
    return sum;
}

__global__ void layer_norm_kernel(
    const float* __restrict__ input, 
    const float* __restrict__ weight, 
    const float* __restrict__ bias, 
    float* output,
    int N, int C, float eps
){
    __shared__ float shared_mean;
    __shared__ float shared_inv_std;
    
    int stride = blockDim.x;
    const float* row = input + blockIdx.x * C;
    // 先求u
    float u = 0.0f;
    for (int i=threadIdx.x;i < C;i += stride){
        u += row[i];
    }
    u = block_reduce(u) / C;
    if (threadIdx.x == 0) shared_mean = u;
    __syncthreads();
    u = shared_mean;

    // 求 detla
    float detla = 0.0f;
    for (int i=threadIdx.x;i < C;i += stride){
        float tmp = (row[i] - u);
        detla += tmp * tmp;
    }
    detla = rsqrtf(block_reduce(detla) / C + eps);
    if (threadIdx.x == 0) shared_inv_std = detla;
    __syncthreads();
    detla = shared_inv_std;
    
    // 计算output
    for (int i=threadIdx.x;i < C;i += stride){
        output[blockIdx.x * C + i] = weight[i] * (row[i] - u) * detla + bias[i];
    } 
}


// input, weight, bias, output are device pointers
extern "C" void solve(
    const float* input, 
    const float* weight, 
    const float* bias, 
    float* output,
    int N, int C, float eps) {
        layer_norm_kernel<<<N, BLOCK_SIZE>>>(input,weight,bias,output,N,C,eps);
}
