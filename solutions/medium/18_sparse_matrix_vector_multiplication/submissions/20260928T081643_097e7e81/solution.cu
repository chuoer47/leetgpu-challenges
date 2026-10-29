#include <cuda_runtime.h>
#include <cstdint>
#include <cstddef>

template <bool USE_VEC4>
__global__ void count_nnz_kernel(
    const float* __restrict__ A,
    int* __restrict row_nnz,
    int M,int N
){
    const int row = blockIdx.x;
    if (row >= M) return;
    const float* rp = A + (size_t)row * (size_t)N;

    int cnt = 0;
    if (USE_VEC4){
        const float4* rp4 = reinterpret_cast<const float4*>(rp);
        const int N4 = N >> 2;
        for (int i = threadIdx.x; i < N4; i += blockDim.x){
            const float4 v = rp4[i];
            cnt += (v.x != 0.0f) + (v.y != 0.0f)
                    + (v.z != 0.0f) + (v.w != 0.0f);
        }
        for (int i = (N4 << 2) + threadIdx.x; i < N; i += blockDim.x) {
            cnt += (rp[i] != 0.0f);
        }
    } else{
        for (int i = threadIdx.x; i < N; i += blockDim.x) {
            cnt += (rp[i] != 0.0f);
        }
    }

    #pragma unroll
    for (int off = 16;off > 0;off>>=1){
        cnt += __shfl_down_sync(0xffffffffu, cnt, off);
    }

    __shared__ int warp_sums[32];
    const int warp_id   = threadIdx.x >> 5;
    const int lane      = threadIdx.x & 31;
    const int num_warps = blockDim.x >> 5;

    if (lane == 0) warp_sums[warp_id] = cnt;
    __syncthreads();   // 全程只此一次

    if (warp_id == 0) {
        int v = (lane < num_warps) ? warp_sums[lane] : 0;
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            v += __shfl_down_sync(0xffffffffu, v, off);
        }
        if (lane == 0) row_nnz[row] = v;
    }
}

__global__ void scan_row_nnz_kernel(
    const int* __restrict__ row_nnz,
    int* __restrict row_ptr,
    int M
){
    extern __shared__ int smem[];
    int* cur = smem;
    int* nxt = smem + blockDim.x;

    const int tid = threadIdx.x;
    const int nthreads = blockDim.x;
    const int chunk = (M + nthreads - 1) / nthreads;
    const int start = tid * chunk;
    const int end = (start + chunk < M) ? (start + chunk) : M;

    int local = 0;
    for (int i = start;i < end;i++){
        local += row_nnz[i];
    }

    cur[tid] = local;
    __syncthreads();
    
    for (int off = 1;off < nthreads; off <<= 1){
        const int v = (tid >= off)?cur[tid - off]:0;
        nxt[tid] = cur[tid] + v;
        __syncthreads();
        int *tmp = cur;cur = nxt;nxt = tmp;
    }

    const int offset = (tid > 0)?cur[tid - 1]:0;
    int acc = offset;
    for (int i = start;i < end;++i){
        row_ptr[i] = acc;
        acc += row_nnz[i];
    }

    if (tid == nthreads - 1){
        row_ptr[M] = cur[tid];
    }
}

__global__ void fill_csr_kernel(
    const float* __restrict__ A,
    const int* __restrict__ row_ptr,
    int* __restrict__ col_idx,
    float* __restrict__ values,
    int M, int N
){
    const int row = blockIdx.x;
    if (row >= M) return;
    const float* rp = A + (size_t)row *(size_t)N;

    __shared__ int s_pos;
    if (threadIdx.x == 0) s_pos = row_ptr[row];
    __syncthreads();

    const int lane = threadIdx.x & 31;
    const unsigned int FULL = 0Xffffffffu;
    const int total_iter = (N + blockDim.x - 1) / blockDim.x;

    for (int k = 0;k < total_iter;++k){
        const int i = threadIdx.x + k * blockDim.x;
        const float v = (i < N)?rp[i]:0.0f;
        const bool nz = (v!=0.0f);

        const unsigned mask = __ballot_sync(FULL, nz);
        const int warp_nz = __popc(mask);

        int base = 0;
        if (lane == 0 && warp_nz > 0){
            base = atomicAdd(&s_pos,warp_nz);
        }
        base = __shfl_sync(FULL,base,0);

        if (nz){
            const unsigned before = mask &((1u << lane) - 1);
            const int p = base + __popc(before);
            values[p] = v;
            col_idx[p] = i;
        }

    }
}

__global__ void csr_spmv_kernel(
    const int* __restrict__ row_ptr,
    const int* __restrict__ col_idx,
    const float* __restrict__ values,
    const float* __restrict__ x,
    float* __restrict__ y,
    int M
){
    const int warps_per_block = blockDim.x >> 5;
    const int warp_id = blockIdx.x * warps_per_block + (threadIdx.x >> 5);
    const int lane = threadIdx.x & 31;
    if (warp_id >= M) return;

    const int start = row_ptr[warp_id];
    const int end = row_ptr[warp_id + 1];
    float sum = 0.0f;
    for (int i = start + lane;i < end;i += 32){
        sum += values[i] * x[col_idx[i]];
    }

    #pragma unroll
    for (int off = 16;off > 0;off >>= 1){
        sum += __shfl_down_sync(0xffffffffu,sum,off);
    }
    if (lane == 0) y[warp_id] = sum;
}

extern "C" void solve(
    const float* A,
    const float* x,
    float* y,
    int M,
    int N,
    int nnz
) {
    if (M <= 0 || N <= 0) return;

    const size_t alloc_nnz = (nnz > 0)?(size_t)nnz:(size_t)M * (size_t)N;
    int* d_row_nnz = nullptr;
    int* d_row_ptr = nullptr;
    int* d_col_idx = nullptr;
    float* d_values = nullptr;

    cudaMalloc(&d_row_nnz,sizeof(int) * (size_t)M);
    cudaMalloc(&d_row_ptr,sizeof(int) * (size_t)(M + 1));
    cudaMalloc(&d_col_idx,sizeof(int) * alloc_nnz);
    cudaMalloc(&d_values,sizeof(float) * alloc_nnz);

    const int block = 256;
    const int scan_block = 256;
    const int scan_shmem = 2 * scan_block * (int)sizeof(int);
    const bool use_vec4 = 
        ((N & 3) == 0) &&
        (reinterpret_cast<uintptr_t>(A) % 16 == 0);
    
    if (use_vec4){
        count_nnz_kernel<true><<<M,block>>>(A,d_row_nnz,M,N);
    }else{
        count_nnz_kernel<false><<<M,block>>>(A,d_row_nnz,M,N);
    }

    scan_row_nnz_kernel<<<1,scan_block,scan_shmem>>>(
        d_row_nnz, d_row_ptr, M
    );

    fill_csr_kernel<<<M,block>>>(A,d_row_ptr,d_col_idx,d_values,M,N);

    const int warps_per_block = block / 32;
    const int grid = (M + warps_per_block - 1) / warps_per_block;
    csr_spmv_kernel<<<grid,block>>>(
        d_row_ptr,d_col_idx,d_values,x,y,M
    );

    cudaFree(d_row_nnz);
    cudaFree(d_row_ptr);
    cudaFree(d_col_idx);
    cudaFree(d_values);

    cudaDeviceSynchronize();
}