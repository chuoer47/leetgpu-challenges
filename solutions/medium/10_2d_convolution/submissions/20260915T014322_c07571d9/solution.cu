#include <cuda_runtime.h>
#define TILE 256
#define MAX_KERNEL_SIZE 32

__global__ void convolution2d_kernel(
    const float* __restrict__ input,
    const float* __restrict__ kernel,
    float* output,
    int input_rows,
    int input_cols,
    int kernel_rows,
    int kernel_cols
){
    int output_rows = input_rows - kernel_rows + 1;
    int output_cols = input_cols - kernel_cols + 1;
    int N = output_rows * output_cols;
    int tix = threadIdx.x;
    // 搬运 kernel -> share mem
    __shared__ float K_[MAX_KERNEL_SIZE][MAX_KERNEL_SIZE];
    for (int i = tix;i < kernel_rows * kernel_cols;i+=blockDim.x){
        int r = i / kernel_cols;
        int c = i % kernel_cols;
        K_[r][c] = kernel[i]; 
    }
    __syncthreads();
    // 确定当前 thread 负责的 output
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    int r = idx / output_cols;
    int c = idx % output_cols;
    // 很蠢的方法，来回读....
    float sum = 0.f;
    for (int i=0;i<kernel_rows;i++){
        for (int j=0;j<kernel_cols;j++){
            sum += K_[i][j] * input[(r + i)*input_cols + (c + j)];
        }
    }
    output[idx] = sum;
}

// input, kernel, output are device pointers
extern "C" void solve(
    const float* input,
    const float* kernel,
    float* output,
    int input_rows,
    int input_cols,
    int kernel_rows,
    int kernel_cols) {        
        int output_rows = input_rows - kernel_rows + 1;
        int output_cols = input_cols - kernel_cols + 1;
        // kernel很小，可直接存
        // 每个线程负责一个 output 的位置
        int N = output_rows * output_cols;
        int threadsPerBlock = TILE;
        int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
        convolution2d_kernel<<<blocksPerGrid, threadsPerBlock>>>(input,kernel, output,
        input_rows,input_cols,kernel_rows,kernel_cols);
        cudaDeviceSynchronize();
}
