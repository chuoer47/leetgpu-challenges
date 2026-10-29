#include <cuda_runtime.h>
constexpr int TILE = 1024;

__global__ void histogram_kernel(
    const int* input,
    int* histogram,
    int N,
    int num_bins
){  
    extern __shared__ unsigned int histo_s[];
    for (unsigned int bin=threadIdx.x;bin < num_bins;bin += blockDim.x){
        histo_s[bin] = 0;
    }
    __syncthreads();

    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = gridDim.x * blockDim.x;
    for (int i = idx;i < N;i += stride){
        atomicAdd(&(histo_s[input[i]]),1);
    }

    __syncthreads();

    for (unsigned int bin=threadIdx.x;bin < num_bins;bin += blockDim.x){
        int bin_value = histo_s[bin];
        if (bin_value > 0){
            atomicAdd(&(histogram[bin]),bin_value);
        }
    }


}


// input, histogram are device pointers
extern "C" void solve(const int* input, int* histogram, int N, int num_bins) {
    cudaMemset(histogram,0,num_bins * sizeof(int));

    size_t shared_memory_size = num_bins * sizeof(unsigned int);

    int threads = TILE;
    int block = (N + threads - 1) / threads;
    histogram_kernel<<<block,threads,shared_memory_size>>>(input,histogram,N,num_bins);

    cudaDeviceSynchronize();
}   
