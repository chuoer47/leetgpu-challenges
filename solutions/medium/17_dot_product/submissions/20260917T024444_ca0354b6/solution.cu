#include <cuda_runtime.h>

// 1.创建 N / BLOCK_SIZE 全局内存 smem，每个BLOCK 完成 BLOCK_SIZE 的 dot product，然后存到 smem
// 2.smem 在进行一个 reduce
constexpr int BLOCK_SIZE = 256;
constexpr int WARP_SIZE = 32;
constexpr int WARP_NUM = BLOCK_SIZE / WARP_SIZE;

template<int WARP_SIZE=32>
__device__ __forceinline__ float warp_scan(float x){
#pragma unroll
    for (int offset=1;offset < WARP_SIZE; offset <<= 1){
        float y = __shfl_up_sync(0xffffffff,x,offset);
        if ((threadIdx.x & 31) >= offset){
            x += y;
        }
    }
    return x;
}

template<int WARP_SIZE=32>
__device__ __forceinline__ float warp_reduce(float sum){
#pragma unroll
    for (int offset=1;offset < WARP_SIZE; offset <<= 1){
        sum += __shfl_down_sync(0xffffffff,sum,offset);
    }
    return sum;
}

__global__ void dot_product_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* block_reduce,
    int N
){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;

    __shared__ float warp_sums[WARP_NUM];

    // WARP 归约
    float sum = (idx < N)?A[idx] * B[idx]:0.f;
    sum = warp_reduce<WARP_SIZE>(sum);

    if (lane == 0){
        warp_sums[warp] = sum;
    }
    __syncthreads();

    // 跨 WARP 归约
    if (warp == 0){
        sum = (lane < WARP_NUM)?warp_sums[lane]:0.f;
        sum = warp_reduce<WARP_SIZE>(sum);
        if (lane == 0){
            block_reduce[blockIdx.x] = sum;
        }
    }
}

__global__ void reduce_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int N
){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0.0f;
    if (idx < N) sum = input[idx];

    const int WARP_NUM = BLOCK_SIZE / WARP_SIZE;
    int warpId = threadIdx.x / WARP_SIZE;
    int laneId = threadIdx.x % WARP_SIZE;
    __shared__ float sum_s[WARP_NUM];

#pragma unroll
    for (int s=WARP_SIZE >> 1;s > 0;s >>= 1)
        sum += __shfl_down_sync(0xffffffff,sum,s);
    
    if (laneId == 0)
        sum_s[warpId] = sum;
    
    __syncthreads();

    if (threadIdx.x < WARP_SIZE){
        sum = (threadIdx.x < WARP_NUM)?sum_s[threadIdx.x]:0.0f;
#pragma unroll
        for (int s=WARP_NUM >> 1;s>0;s>>=1)
            sum += __shfl_down_sync(0xffffffff,sum,s);
        if (threadIdx.x == 0)
            atomicAdd(output,sum);
    }

}

// A, B, result are device pointers
extern "C" void solve(const float* A, const float* B, float* result, int N) {
    if (N <= 0) return;
    cudaMemset(result, 0, sizeof(float));

    int num_blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    float* block_reduce = nullptr;
    cudaMalloc(&block_reduce,num_blocks * sizeof(float));

    dot_product_kernel<<<num_blocks,BLOCK_SIZE>>>(A,B,block_reduce,N);

    int nn = (num_blocks + BLOCK_SIZE - 1) / BLOCK_SIZE;
    reduce_kernel<<<nn,BLOCK_SIZE>>>(block_reduce,result,num_blocks);
    cudaFree(block_reduce);

}
