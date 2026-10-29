#include <cuda_runtime.h>

constexpr int TILE = 256;

__device__ __forceinline__ unsigned int float_to_key(float f){
    unsigned int u = __float_as_uint(f);
    return (u & 0x80000000u) ? ~u:(u ^ 0x80000000u);
}

__global__ void count_bit_kernel(const float* input,int N,
                                unsigned int mask, unsigned int prefix,
                                unsigned int bit, unsigned int* count){
    unsigned int local = 0;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x;i < N;i += blockDim.x * gridDim.x){
        unsigned int key = float_to_key(input[i]);
        if ((key&mask) == prefix && (key&bit)){
            local++;
        }
    }

    __shared__ unsigned int sdata[TILE];
    int tid = threadIdx.x;
    sdata[tid] = local;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0;s >>= 1){
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    if (tid == 0 && sdata[0] > 0){
        atomicAdd(count,sdata[0]);
    }
}

// 统计 key > threshold 的元素个数
__global__ void count_gt_kernel(const float* input, int N,
                                unsigned int threshold, unsigned int* count) {
    unsigned int local = 0;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x) {
        unsigned int key = float_to_key(input[i]);
        if (key > threshold) {
            local++;
        }
    }

    __shared__ unsigned int sdata[256];
    int tid = threadIdx.x;
    sdata[tid] = local;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    if (tid == 0 && sdata[0] > 0) {
        atomicAdd(count, sdata[0]);
    }
}

__global__ void collect_kernel(const float* input,int N,
                                unsigned int threshold,int k,int count_gt,
                                float* output,unsigned int* pos_gt,
                                unsigned int* pos_eq){
    for (int i = blockIdx.x * blockDim.x + threadIdx.x;i < N;i += blockDim.x * gridDim.x){
        float val = input[i];
        unsigned int key = float_to_key(val);

        if (key > threshold){
            unsigned int pos = atomicAdd(pos_gt,1);
            output[pos] = val;
        }else if (key == threshold){
            unsigned int pos = atomicAdd(pos_eq, 1);
            unsigned int need_eq = (unsigned int)(k - count_gt);
            if (pos < need_eq){
                output[count_gt + pos] = val;
            }
        }
    }
}


// 用 +inf 填充排序数组的尾部
__global__ void fill_inf_kernel(float* data, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        data[i] = __int_as_float(0x7f800000); // +inf
    }
}

// Bitonic Sort 单步
__global__ void bitonic_sort_step(float* data, int m, int j, int size) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < m; i += blockDim.x * gridDim.x) {
        int ij = i ^ j;
        if (i < ij) {
            bool ascending = ((i & size) == 0);
            unsigned int key_i = float_to_key(data[i]);
            unsigned int key_ij = float_to_key(data[ij]);

            if (ascending) {
                if (key_i > key_ij) {
                    float tmp = data[i];
                    data[i] = data[ij];
                    data[ij] = tmp;
                }
            } else {
                if (key_i < key_ij) {
                    float tmp = data[i];
                    data[i] = data[ij];
                    data[ij] = tmp;
                }
            }
        }
    }
}

// 将升序排序后的前 k 个元素反转，得到降序输出
__global__ void reverse_copy_kernel(const float* src, float* dst, int k) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < k; i += blockDim.x * gridDim.x) {
        dst[i] = src[k - 1 - i];
    }
}

// input, output are device pointers
extern "C" void solve(const float* input, float* output, int N, int k) {
    if (k <= 0) return;
    if (k > N) k = N;

    int threads = TILE;
    int blocks = (N + threads - 1) / threads;
    if (blocks > 65535) blocks = 65535;

    unsigned int *d_count,*d_count_gt,*d_pos_gt,*d_pos_eq;
    cudaMalloc(&d_count,sizeof(unsigned int));
    cudaMalloc(&d_count_gt,sizeof(unsigned int));
    cudaMalloc(&d_pos_gt,sizeof(unsigned int));
    cudaMalloc(&d_pos_eq,sizeof(unsigned int));

    unsigned int mask = 0;
    unsigned int prefix = 0;
    int kk = k;

    for (int b = 31;b >= 0; --b){
        unsigned int bit = 1u << b;
        cudaMemset(d_count, 0, sizeof(unsigned int));

        count_bit_kernel<<<blocks,threads>>>(input,N,mask,prefix,bit,d_count);

        unsigned int cnt1;
        cudaMemcpy(&cnt1,d_count,sizeof(unsigned int),cudaMemcpyDeviceToHost);

        if (cnt1 >= (unsigned int)kk){
            prefix |= bit;
        }else{
            kk -= cnt1;
        }
        mask |= bit;
    }

    unsigned int threshold = prefix;

    // 统计严格大于 threshold 的数量
    cudaMemset(d_count_gt, 0, sizeof(unsigned int));
    count_gt_kernel<<<blocks,threads>>>(input,N,threshold,d_count_gt);

    unsigned int count_gt;
    cudaMemcpy(&count_gt, d_count_gt,sizeof(unsigned int),cudaMemcpyDeviceToHost);
    int count_gt_int = (int)count_gt;

    cudaMemset(d_pos_gt,0,sizeof(unsigned int));
    cudaMemset(d_pos_eq,0,sizeof(unsigned int));
    
    collect_kernel<<<blocks,threads>>>(input,N,threshold,k,count_gt_int,
    output,d_pos_gt,d_pos_eq);

    // 对output中的k个元素进行排序
    int m = 1;
    while (m < k) m <<= 1;
    float* d_sort;
    cudaMalloc(&d_sort,m * sizeof(float));
    cudaMemcpy(d_sort,output,k * sizeof(float),cudaMemcpyDeviceToDevice);

    if (m > k){
        int fill_blocks = (m - k + threads - 1) / threads;
        if (fill_blocks > 65535) fill_blocks=65535;
        fill_inf_kernel<<<fill_blocks, threads>>>(d_sort + k,m - k);
    }

    for (int size = 2;size <= m; size <<= 1){
        for (int j = size >> 1;j > 0;j >>= 1){
            int sort_blocks = (m + threads - 1) / threads;
            if (sort_blocks > 65535) sort_blocks=65535;
            bitonic_sort_step<<<sort_blocks,threads>>>(d_sort,m,j,size); 
        }
    }

    int rev_blocks = (k + threads - 1) / threads;
    if (rev_blocks > 65535) rev_blocks = 65535;
    reverse_copy_kernel<<<rev_blocks,threads>>>(d_sort, output, k);

    cudaFree(d_count);
    cudaFree(d_count_gt);
    cudaFree(d_pos_gt);
    cudaFree(d_pos_eq);
    cudaFree(d_sort);
}
