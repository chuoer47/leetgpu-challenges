#include <cuda_runtime.h>
#define BM 64
#define BN 64
#define BK 16
#define TM 4
#define TN 4

__global__ void gemm_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K
){
    int m0 = blockIdx.y * BM;
    int n0 = blockIdx.x * BN;

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tid = ty * blockDim.x + tx;
    int nthreads = blockDim.x * blockDim.y;

    __shared__ float As[BM][BK];
    __shared__ float Bs[BK][BN];

    float acc[TM][TN] = {0.0f};
    for (int k0 = 0;k0 < K;k0 += BK){
        for (int idx = tid;idx < BM * BK;idx += nthreads){
            int i = idx / BK;
            int j = idx % BK;
            int row = m0 + i;
            int col = k0 + j;
            As[i][j] = (row < M && col < K)?A[row * K + col]:0.0f;
        }
        for (int idx = tid;idx < BK * BN;idx += nthreads){
            int i = idx / BN;
            int j = idx % BN;
            int row = k0 + i;
            int col = n0 + j;
            Bs[i][j] = (row < K && col < N)?B[row*N + col]:0.0f;
        }

        __syncthreads();

        for (int k = 0;k < BK; ++k){
            float a_frag[TM];
            float b_frag[TN];
            #pragma unroll
            for (int i = 0;i < TM;++i){
                a_frag[i] = As[ty * TM + i][k]; 
            }
            #pragma unroll
            for (int j = 0;j < TN; ++j){
                b_frag[j] = Bs[k][tx * TN + j];
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
                    C[row * N + col] = acc[i][j];
                }
            }
        }
    }
}

__global__ void set_identity_kernel(
    float* mat,
    int N
){
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int total = N*N;
    for (int i = idx;i < total;i += blockDim.x * gridDim.x){
        int row = i / N;
        int col = i % N;
        mat[i] = (row == col) ? 1.0f :0.0f;
    }
}

// input, output are device pointers
extern "C" void solve(const float* input, float* output, int N, int P) {
    if (P == 1){
        size_t bytes = (size_t)N*N * sizeof(float);
        cudaMemcpy(output, input, bytes, cudaMemcpyDeviceToDevice);
        return;
    }
    float *d_result, *d_base, *d_temp;
    size_t bytes = (size_t)N*N * sizeof(float);
    cudaMalloc(&d_result,bytes);
    cudaMalloc(&d_base,bytes);
    cudaMalloc(&d_temp,bytes);

    int threads = 256;
    int blocks = (N*N + threads - 1) / threads;
    set_identity_kernel<<<blocks,threads>>>(d_result,N);

    cudaMemcpy(d_base, input, bytes, cudaMemcpyDeviceToDevice);

    dim3 block(16,16);
    dim3 grid((N + BN - 1)/BN,(N + BM - 1)/BM);

    while (P > 0){
        if (P & 1){
            gemm_kernel<<<grid,block>>>(d_result,d_base,d_temp,N,N,N);
            float *tmp = d_result;
            d_result = d_temp;
            d_temp = tmp;
        }
        gemm_kernel<<<grid,block>>>(d_base,d_base,d_temp,N,N,N);
        float *tmp = d_base;
        d_base = d_temp;
        d_temp = tmp;

        P >>= 1;
    }

    cudaMemcpy(output, d_result, bytes, cudaMemcpyDeviceToDevice);
    cudaFree(d_result);
    cudaFree(d_base);
    cudaFree(d_temp);

    cudaDeviceSynchronize();
}
