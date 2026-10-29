#include <cuda_runtime.h>

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

__global__ void scan_blocks(
    const float* __restrict__ input,
    float* __restrict__ output,
    float* __restrict__ block_sums,
    int N
){
    __shared__ float warp_sums[WARP_NUM];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    float x = (idx < N)?input[idx]:0.f;
    float scanned = warp_scan<WARP_SIZE>(x);
    
    int lane = tid & 31;
    int warp = tid >> 5;
    if (lane == WARP_SIZE - 1) warp_sums[warp] = scanned;
    __syncthreads();

    if (warp == 0){
        float v = (lane < WARP_NUM) ? warp_sums[lane] :0.f;
        v = warp_scan<WARP_SIZE>(v);
        if (lane < WARP_NUM)
            warp_sums[lane] = v;
    }
    __syncthreads();

    float warp_offset = 0.f;
    if (warp > 0)
        warp_offset = warp_sums[warp - 1];
    scanned += warp_offset;

    if (idx < N) output[idx] = scanned;

    if (tid == BLOCK_SIZE - 1) block_sums[blockIdx.x] = scanned;
}

__global__ void add_block_offsets(
    float* output,
    const float* block_sums,
    int N
){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    if (blockIdx.x == 0) return;
    output[idx] += block_sums[blockIdx.x - 1];
}

void scan_recursive(float* data,int N){
    if (N <= 1) return;
    int num_blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    if (num_blocks == 1){
        float* dummy_sums = nullptr;
        cudaMalloc(&dummy_sums,sizeof(float));
        scan_blocks<<<1,BLOCK_SIZE>>>(data,data,dummy_sums,N);
        cudaFree(dummy_sums);
        return;
    }

    float* next_sums = nullptr;
    cudaMalloc(&next_sums,num_blocks * sizeof(float));
    scan_blocks<<<num_blocks,BLOCK_SIZE>>>(data,data,next_sums,N);
    scan_recursive(next_sums,num_blocks);
    add_block_offsets<<<num_blocks, BLOCK_SIZE>>>(data,next_sums,N);

    cudaFree(next_sums);
}



// input, output are device pointers
extern "C" void solve(const float* input, float* output, int N) {
    if (N <= 0) return;
    int num_blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    float* block_sums = nullptr;
    cudaMalloc(&block_sums,num_blocks * sizeof(float));
    scan_blocks<<<num_blocks,BLOCK_SIZE>>>(input,output,block_sums,N);
    if (num_blocks > 1) scan_recursive(block_sums,num_blocks);
    if (num_blocks > 1) add_block_offsets<<<num_blocks,BLOCK_SIZE>>>(output,block_sums,N);
    cudaFree(block_sums);
}
