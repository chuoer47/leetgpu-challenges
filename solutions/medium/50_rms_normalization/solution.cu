#include <cuda_runtime.h>
#define THREADS 256
#define WARP_SIZE 32

template <const int kWarpSize = WARP_SIZE>
__device__ __forceinline__ float warp_reduce_sum(float val) {
#pragma unroll
  for (int mask = kWarpSize >> 1; mask >= 1; mask >>= 1) {
    val += __shfl_xor_sync(0xffffffff, val, mask);
  }
  return val;
}

__global__ void get_rms_kernel(
    const float* __restrict__ input, 
    float* rms, 
    int N
){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    float sum = 0.0f;

    int N4 = N / 4;
    const float4* in4 = reinterpret_cast<const float4*>(input);
    for (int i = idx;i < N4;i += stride){
        float4 a = in4[i];
        sum += a.x*a.x + a.y * a.y + a.z * a.z + a.w * a.w;
    }
    int i = N4 * 4 + idx;
    if (i < N){
        sum += input[i] * input[i];
    }

    __shared__ float warpsSums[THREADS / WARP_SIZE];
    int lane = threadIdx.x & 31;
    int warpId = threadIdx.x >> 5;

    sum = warp_reduce_sum<WARP_SIZE>(sum);
    if (lane == 0) warpsSums[warpId] = sum;
    __syncthreads();

    if (warpId == 0){
      sum = (lane < THREADS / WARP_SIZE) ? warpsSums[lane] : 0.0f;
      sum = warp_reduce_sum<WARP_SIZE>(sum);
      if (lane == 0)  rms[blockIdx.x] = sum / N;
    }
}

__global__ void get_y_kernel(
    const float* input, 
    float gamma, 
    float beta, 
    float* rms,
    float* output, 
    int N, float eps, int grids
){
    float rms_s = 0.0f;
    int lane = threadIdx.x & 31;
    for (int i = lane;i < grids;i += WARP_SIZE){
        rms_s += rms[i];
    }
    rms_s = warp_reduce_sum<WARP_SIZE>(rms_s);

    float r_rms = rsqrtf(rms_s + eps); 
    float scale = gamma * r_rms; 

    int N4 = N / 4;
    const float4* in4  = reinterpret_cast<const float4*>(input);
    float4*       out4 = reinterpret_cast<float4*>(output);

    int idx    = blockDim.x * blockIdx.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = idx; i < N4; i += stride) {
        float4 a = in4[i];
        float4 r;
        r.x = fmaf(a.x, scale, beta);
        r.y = fmaf(a.y, scale, beta);
        r.z = fmaf(a.z, scale, beta);
        r.w = fmaf(a.w, scale, beta);
        out4[i] = r;
    }

    int i = N4 * 4 + idx;
    if (i < N) {
        output[i] = fmaf(input[i], scale, beta);
    }
}

// input, output are device pointers
extern "C" void solve(
    const float* input, 
    float gamma, 
    float beta, 
    float* output, 
    int N, float eps) {
        // 总思路：2个kernel完成；第一个kernel求局部的rms；第二个kernel计算结果
        int numSMs;
        cudaDeviceGetAttribute(&numSMs,cudaDevAttrMultiProcessorCount,0);
        int grids = min((N + THREADS - 1)/THREADS, numSMs);
        int blocks = THREADS;

        float *rms;
        size_t bytes = (size_t)grids * sizeof(float);
        cudaMalloc(&rms,bytes);

        get_rms_kernel<<<grids,blocks>>>(input,rms,N);
        get_y_kernel<<<grids,blocks>>>(input,gamma,beta,rms,output,N,eps,grids);

        cudaFree(rms);
        cudaDeviceSynchronize();
}
