#include <cuda_runtime.h>
#include <math_constants.h>

// 定义常量
#define WARP_SIZE 32
#define WARPS_PER_BLOCK 8
#define BM WARPS_PER_BLOCK
#define BN 32
#define MAX_D 128
#define MAX_D_PER_LANE (MAX_D / WARP_SIZE)

// WARP内归约sum
__device__ __forceinline__ float warp_reduce_sum(float v){
#pragma unroll
    for (int offset = WARP_SIZE >> 1; offset >= 1;offset >>= 1){
        v += __shfl_down_sync(0xffffffff,v,offset);
    }
    return v;
}

__global__ void fused_attention_kernel(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    float* __restrict__ O,
    int M, int N, int d,
    float inv_sqrt_d
){
    __shared__ float Ks[BN][MAX_D];
    __shared__ float Vs[BN][MAX_D];

    const int warp = threadIdx.x / WARP_SIZE;
    const int lane = threadIdx.x % WARP_SIZE;

    const int q_raw = blockIdx.x * BM + warp;
    const bool valid = q_raw < M;
    const int q = valid ? q_raw : (M - 1); // 加载最后一行，但是不参与后续计算

    float q_reg[MAX_D_PER_LANE];
    float acc[MAX_D_PER_LANE];

#pragma unroll
    for (int k = 0;k < MAX_D_PER_LANE; ++k){
        q_reg[k] = 0.f;
        acc[k] = 0.f;
    }

    // 每个线程只加载至多 MAX_D_PER_LANE 元素
    int kk = 0;
    for (int j = lane;j < d;j+=WARP_SIZE,++kk){
        q_reg[kk] = Q[(size_t)q*d + j];
    }

    float m = -CUDART_INF_F; // 当前见过的最大 score
    float l = 0.f; // 当前 softmax 分母

    for (int n0 = 0;n0 < N;n0 += BN){ // 切割 Ktile，Vtile，按照BN进行

        // 所有线程都去搬运，搬运Ks[BN][MAX_D]、Vs[BN][MAX_D]
        for (int idx = threadIdx.x;idx < BN * d;idx += blockDim.x){
            int r = idx / d;
            int c = idx % d;
            int n = n0 + r;

            float kv = (n < N)?K[(size_t)n*d + c]:0.0f;
            float vv = (n < N)?V[(size_t)n*d + c]:0.0f;

            Ks[r][c] = kv;
            Vs[r][c] = vv;
        }
        __syncthreads();

        int tile_n = min(BN,N - n0);
        for (int r = 0;r < tile_n;++r){ // 处理 Ktile/Vtile 一行，行为 r
            float partial = 0.f;
            int k = 0;
            // 一个线程只处理点积的一部分，然后再归约
            for (int j = lane;j < d;j += WARP_SIZE,++k){
                partial += q_reg[k] * Ks[r][j];
            }
            partial = warp_reduce_sum(partial);
            float score = __shfl_sync(0xffffffff,partial,0) * inv_sqrt_d; // 广播 partial

            // online 的精华代码，边遍历边更新
            // WARP 里面的 0线程 负责更新
            float alpha = 0.f;
            float beta = 0.f;
            if (lane == 0){
                float m_new = fmaxf(m,score);
                alpha = __expf(m - m_new);
                beta = __expf(score - m_new);
                l = l * alpha + beta;
                m = m_new;
            }
            alpha = __shfl_sync(0xffffffff,alpha,0);
            beta = __shfl_sync(0xffffffff,beta,0);

            // acc 只负责 MAX_D_PER_LANE 个
            k = 0;
            for (int j = lane;j < d;j += WARP_SIZE,++k){
                acc[k] = acc[k] * alpha + beta * Vs[r][j];
            }
        }
        __syncthreads();
    }
    if (!valid) return;
    float inv_l = 0.f;
    if (lane == 0) inv_l = 1.0f/l;
    inv_l = __shfl_sync(0xffffffff,inv_l,0);

    float* o_ptr = O + (size_t)q * d;
    int k = 0;
    for (int j = lane;j < d;j+=WARP_SIZE,++k){
        o_ptr[j] = acc[k] * inv_l;
    }
}



// Q, K, V, output are device pointers
extern "C" void solve(
    const float* Q,
    const float* K,
    const float* V,
    float* output,
    int M,
    int N,
    int d) {
        float inv_sqrt_d = 1.0f / sqrtf((float)d);
        // 一个 block 负责 BM 行
        // block WARP_SIZE * WARPS_PER_BLOCK 个线程，一共有 WARPS_PER_BLOCK 个 WARP
        // 每个 WARP 负责一行，一行有 WARP_SIZE 个线程
        dim3 block(WARP_SIZE * WARPS_PER_BLOCK);
        dim3 grid((M + BM - 1) / BM);
        fused_attention_kernel<<<grid,block>>>(Q,K,V,output,M,N,d,inv_sqrt_d);
}
