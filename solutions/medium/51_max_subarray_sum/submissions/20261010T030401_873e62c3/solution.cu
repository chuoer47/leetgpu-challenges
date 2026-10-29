#include <cuda_runtime.h>
#include <limits.h>
#define THREADS 256
#define WARP_SIZE 32

__device__ __forceinline__ int warp_reduce(int val) {
#pragma unroll
  for (int mask = WARP_SIZE >> 1; mask >= 1; mask >>= 1) {
    val = max(val,__shfl_xor_sync(0xffffffff, val, mask));
  }
  return val;
}


__global__ void get_max_nums_kernel(
    const int* __restrict__ input,
    int* max_nums,
    int N,int window_size,int M
){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int sum = 0;
    if (idx + window_size - 1 >= N) return;
    for (int i = 0;i < window_size;i += 1){
        sum += input[idx + i];
    }
    max_nums[idx] = sum;
}

__global__ void get_output_kernel(
    int* max_nums,
    int* output,
    int M
){
    int idx = threadIdx.x;
    int warp_id = idx >> 5;
    int lane = idx & 31;
    __shared__ int smem[THREADS / WARP_SIZE];

    int mx = INT_MIN;
    for (int i = idx;i < M;i += blockDim.x){
        mx = max(mx,max_nums[i]);
    }
    mx = warp_reduce(mx);
    if (lane == 0) smem[warp_id] = mx;
    __syncthreads();
    if (warp_id == 0){
        mx = (lane < THREADS / WARP_SIZE)?smem[lane]:INT_MIN;
        mx =  warp_reduce(mx);
        if (lane == 0) *output = mx;
    }
}


// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const int* input, int* output, int N, int window_size) {
    int M = N - window_size + 1;
    int* max_nums;
    size_t bytes = (size_t) M * sizeof(int);
    cudaMalloc(&max_nums,bytes);

    int blocks = (M + THREADS - 1) / THREADS;
    get_max_nums_kernel<<<blocks,THREADS>>>(input,max_nums,N,window_size,M);
    
    get_output_kernel<<<1,THREADS>>>(max_nums,output,M);

    cudaFree(max_nums);
}
