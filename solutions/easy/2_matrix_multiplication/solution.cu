
#include <cuda_runtime.h>

// A[M,N] @ B[N,K] = C[M,K]
// Each block computes BM x BK outputs.
// Each thread computes TM x TK outputs.
// BN is the reduction tile size.

template <int BM, int BN, int BK, int TM, int TK, bool VECTOR_LOAD>
__global__ void register_gemm(const float* __restrict__ A, const float* __restrict__ B,
                              float* __restrict__ C, int M, int N, int K) {
    constexpr int TX = BK / TK;
    constexpr int TY = BM / TM;
    constexpr int NT = TX * TY;

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tid = ty * TX + tx;

    const int m0 = blockIdx.y * BM;
    const int k0 = blockIdx.x * BK;

    __shared__ float As[BN][BM + 1];
    __shared__ float Bs[BN][BK];

    float accum[TM][TK] = {};

    // Reduction along N
    for (int n0 = 0; n0 < N; n0 += BN) {

        if constexpr (VECTOR_LOAD) {
            // Load A: [BM, BN] -> As[BN, BM+1]
            for (int v = tid; v < BM * BN / 4; v += NT) {
                int m = v / (BN / 4);
                int n = (v % (BN / 4)) * 4;
                float4 value = make_float4(0, 0, 0, 0);
                if (m0 + m < M && n0 + n + 3 < N)
                    value = *reinterpret_cast<const float4*>(A + (m0 + m) * N + n0 + n);
                As[n][m] = value.x;
                As[n + 1][m] = value.y;
                As[n + 2][m] = value.z;
                As[n + 3][m] = value.w;
            }

            // Load B: [BN, BK]
            for (int v = tid; v < BN * BK / 4; v += NT) {
                int n = v / (BK / 4);
                int k = (v % (BK / 4)) * 4;
                float4 value = make_float4(0, 0, 0, 0);
                if (n0 + n < N && k0 + k + 3 < K)
                    value = *reinterpret_cast<const float4*>(B + (n0 + n) * K + k0 + k);
                Bs[n][k] = value.x;
                Bs[n][k + 1] = value.y;
                Bs[n][k + 2] = value.z;
                Bs[n][k + 3] = value.w;
            }

        } else {

            // Scalar load A
            for (int v = tid; v < BM * BN; v += NT) {
                int m = v / BN;
                int n = v % BN;
                As[n][m] = m0 + m < M && n0 + n < N ? A[(m0 + m) * N + n0 + n] : 0.0f;
            }

            // Scalar load B
            for (int v = tid; v < BN * BK; v += NT) {
                int n = v / BK;
                int k = v % BK;
                Bs[n][k] = n0 + n < N && k0 + k < K ? B[(n0 + n) * K + k0 + k] : 0.0f;
            }
        }

        __syncthreads();

// Register tiling
#pragma unroll
        for (int n = 0; n < BN; ++n) {
            float a[TM], b[TK];

#pragma unroll
            for (int i = 0; i < TM; ++i)
                a[i] = As[n][ty + i * TY];

#pragma unroll
            for (int j = 0; j < TK; ++j)
                b[j] = Bs[n][tx + j * TX];

#pragma unroll
            for (int i = 0; i < TM; ++i) {
#pragma unroll
                for (int j = 0; j < TK; ++j)
                    accum[i][j] = fmaf(a[i], b[j], accum[i][j]);
            }
        }

        __syncthreads();
    }

// Store C
#pragma unroll
    for (int i = 0; i < TM; ++i) {
        int m = m0 + ty + i * TY;

#pragma unroll
        for (int j = 0; j < TK; ++j) {
            int k = k0 + tx + j * TX;

            if (m < M && k < K)
                C[m * K + k] = accum[i][j];
        }
    }
}

extern "C" void solve(const float* A, const float* B, float* C, int M, int N, int K) {
    constexpr int BM = 64;
    constexpr int BN = 16;
    constexpr int BK = 64;
    constexpr int TM = 4;
    constexpr int TK = 4;

    dim3 block(BK / TK, BM / TM);
    dim3 grid((K + BK - 1) / BK, (M + BM - 1) / BM);

    if (/*0 &&*/ N % 4 == 0 && K % 4 == 0) {
        register_gemm<BM, BN, BK, TM, TK, true><<<grid, block>>>(A, B, C, M, N, K);
    } else {
        register_gemm<BM, BN, BK, TM, TK, false><<<grid, block>>>(A, B, C, M, N, K);
    }

    cudaDeviceSynchronize();
}
