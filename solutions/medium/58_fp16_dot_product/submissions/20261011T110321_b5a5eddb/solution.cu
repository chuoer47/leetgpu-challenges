#include <cuda_fp16.h>
#include <cuda_runtime.h>

constexpr int BLOCK_SIZE = 256;
constexpr int WARP_SIZE = 32;
constexpr int WARP_NUM = BLOCK_SIZE / WARP_SIZE;
constexpr int VEC_SIZE = 8;
constexpr int MAX_BLOCKS = 1024;

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

__device__ __forceinline__ float2 unpack_half2(unsigned int bits){
    half lo = __ushort_as_half(static_cast<unsigned short>(bits));
    half hi = __ushort_as_half(static_cast<unsigned short>(bits >> 16));
    half2 h2 = __halves2half2(lo,hi);
    return __half22float2(h2);
}

__device__ __forceinline__ float dot2(unsigned int a,unsigned int b, float acc){
    float2 fa = unpack_half2(a);
    float2 fb = unpack_half2(b);

    acc = fmaf(fa.x,fb.x,acc);
    acc = fmaf(fa.y,fb.y,acc);
    return acc;
}

__global__ void dot_product_vec8(
    const half* __restrict__ A,
    const half* __restrict__ B,
    float* partial,
    int N
){
    int tid = threadIdx.x;
    int idx = tid + blockDim.x * blockIdx.x;
    int stride = gridDim.x * blockDim.x;
    int num_vec = N / VEC_SIZE;
    float sum0 = 0.0;
    float sum1 = 0.0;
    float sum2 = 0.0;
    float sum3 = 0.0;
    const uint4* A4 = reinterpret_cast<const uint4*>(A);
    const uint4* B4 = reinterpret_cast<const uint4*>(B);
    for (int i = idx;i < num_vec;i += stride){
        uint4 a = A4[i];
        uint4 b = B4[i];
        sum0 = dot2(a.x,b.x,sum0);
        sum1 = dot2(a.y,b.y,sum1);
        sum2 = dot2(a.w,b.w,sum2);
        sum3 = dot2(a.z,b.z,sum3);
    }
    
    float sum = (sum0 + sum1) + (sum2 + sum3);
    if (blockIdx.x == 0 && tid < N - num_vec * VEC_SIZE){
        int tail_idx = num_vec * VEC_SIZE + tid;
        sum = fmaf(__half2float(A[tail_idx]),__half2float(B[tail_idx]),sum);
    }

    sum = block_reduce(sum);
    if (tid == 0){
        partial[blockIdx.x] = sum;
    }
}

__global__ void final_reduce(
    const float* __restrict__ partial,
    half* __restrict__ result,
    int num_blocks
) {
    int tid = threadIdx.x;
    float sum = 0.0f;
    for (int i = tid; i < num_blocks; i += BLOCK_SIZE) {
        sum += partial[i];
    }
    sum = block_reduce(sum);
    if (tid == 0) {
        result[0] = __float2half_rn(sum);
    }
}

// A, B, result are device pointers
extern "C" void solve(const half* A, const half* B, half* result, int N) {
    if (N <= 0){
        cudaMemset(result, 0, sizeof(half));
        return;
    }
    int num_vec = N / VEC_SIZE;
    int num_blocks = (num_vec + BLOCK_SIZE - 1) / BLOCK_SIZE;
    num_blocks = (num_blocks < 1)?1:num_blocks;
    num_blocks = (num_blocks > MAX_BLOCKS)?MAX_BLOCKS:num_blocks;

    float* partial = nullptr;
    cudaMalloc(&partial,num_blocks * sizeof(float));

    dot_product_vec8<<<num_blocks,BLOCK_SIZE>>>(A,B,partial,N);
    final_reduce<<<1,BLOCK_SIZE>>>(partial,result,num_blocks);
    cudaFree(partial);
}
