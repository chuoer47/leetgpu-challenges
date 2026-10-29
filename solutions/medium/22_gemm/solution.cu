#include <cuda_fp16.h>
#include <cuda_runtime.h>

#define BM 64
#define BN 64
#define BK 16
#define TM 4
#define TN 4

__global__ void gemm_kernel(
    const half* __restrict__ A,
    const half* __restrict__ B,
    half* __restrict__ C,
    int M, int N, int K,
    float alpha, float beta
){
    int m0 = blockIdx.y * BM;
    int n0 = blockIdx.x * BN;

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tid = ty * blockDim.x + tx;
    int nthreads = blockDim.x * blockDim.y;

    __shared__ half As[BM][BK];
    __shared__ half Bs[BK][BN];

    float acc[TM][TN] = {0.0f};
    for (int k0 = 0;k0 < K;k0 += BK){
        for (int idx = tid;idx < BM * BK;idx += nthreads){
            int i = idx / BK;
            int j = idx % BK;
            int row = m0 + i;
            int col = k0 + j;
            As[i][j] = (row < M && col < K)?A[row * K + col]:__float2half(0.0f);
        }
        for (int idx = tid;idx < BK * BN;idx += nthreads){
            int i = idx / BN;
            int j = idx % BN;
            int row = k0 + i;
            int col = n0 + j;
            Bs[i][j] = (row < K && col < N)?B[row*N + col]:__float2half(0.0f);
        }

        __syncthreads();

        for (int k = 0;k < BK; ++k){
            float a_frag[TM];
            float b_frag[TN];
            #pragma unroll
            for (int i = 0;i < TM;++i){
                a_frag[i] = __half2float(As[ty * TM + i][k]); 
            }
            #pragma unroll
            for (int j = 0;j < TN; ++j){
                b_frag[j] = __half2float(Bs[k][tx * TN + j]);
            }
            #pragma unroll
            for (int i = 0;i < TM;++i){
                #pragma unroll
                for (int j = 0;j< TN;++j){
                    acc[i][j] += a_frag[i] * b_frag[j];
                }
            }

        }

        __syncthreads();
    }

    #pragma unroll
    for (int i = 0;i < TM; ++i){
        int row = m0 + ty * TM + i;
        if (row < M){
            #pragma unroll
            for (int j = 0;j < TN;++j){
                int col = n0 + tx * TN + j;
                if (col < N){
                    float old = __half2float(C[row * N + col]);
                    float val = alpha * acc[i][j] + beta * old;
                    C[row * N + col] = __float2half(val);
                }
            }
        }
    }
}

// A, B, and C are device pointers
extern "C" void solve(const half* A, const half* B, half* C, int M, int N, int K, float alpha,
                      float beta) {
    dim3 block(16,16);
    dim3 grid((N + BN - 1)/BN,(M + BM - 1)/BM);
    gemm_kernel<<<grid,block>>>(A,B,C,M,N,K,alpha,beta);
}
