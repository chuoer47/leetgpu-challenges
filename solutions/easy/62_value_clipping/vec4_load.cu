/*
PASS: 8 cases (example + functional + performance)
GPU: NVIDIA GeForce RTX 4090 D; median solve + sync: 0.0402 ms; CUDA event: 0.0206 ms
*/

#include <algorithm>
#include <cuda_runtime.h>

#define TILE 256

__device__ float clip(float x, float lo, float hi) {
    return fmaxf(lo, fminf(x, hi));
}

__global__ void clip_kernel(const float* __restrict__ input, float* __restrict__ output, float lo,
                            float hi, int N) {
    int vecN = N / 4;
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    const float4* input4 = reinterpret_cast<const float4*>(input);
    float4* output4 = reinterpret_cast<float4*>(output);
    if (idx < vecN) {
        float4 v = input4[idx];
        v.x = clip(v.x, lo, hi);
        v.y = clip(v.y, lo, hi);
        v.z = clip(v.z, lo, hi);
        v.w = clip(v.w, lo, hi);
        output4[idx] = v;
    } else {
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            int scalarIdx = vecN * 4 + i;
            if (scalarIdx < N) {
                output[scalarIdx] = clip(input[scalarIdx], lo, hi);
            }
        }
    }
}

// input, output are device pointers
extern "C" void solve(const float* input, float* output, float lo, float hi, int N) {
    int vecN = N / 4;
    int totalThreads = vecN + (N % 4 > 0 ? 1 : 0);

    int blocksPerGrid = (totalThreads + TILE - 1) / TILE;
    clip_kernel<<<blocksPerGrid, TILE>>>(input, output, lo, hi, N);
}
