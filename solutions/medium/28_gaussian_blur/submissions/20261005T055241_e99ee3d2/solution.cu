#include <cuda_runtime.h>
#define TILE_H 16
#define TILE_W 16
#define THREADS (TILE_H * TILE_W)   // 256
#define MAX_KERNEL_SIZE 21

__global__ void gaussian_blur_kernel(
    const float* __restrict__ input, 
    const float* __restrict__ kernel, 
    float* output, 
    int input_rows,
    int input_cols, 
    int kernel_rows, 
    int kernel_cols
){
    int kh2 = kernel_rows / 2;
    int kw2 = kernel_cols / 2;
    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    int r0 = blockIdx.y * TILE_H;
    int c0 = blockIdx.x * TILE_W; 
    int sr0 = r0 - kh2;
    int sw0 = c0 - kw2;
    int tile_w = TILE_W + 2 * kw2;
    int tile_h = TILE_H + 2 * kh2;
    int I_total = tile_w * tile_h;
    
    __shared__ float K_[MAX_KERNEL_SIZE][MAX_KERNEL_SIZE];
    extern __shared__ float I[];

    for (int i = tid; i < kernel_rows * kernel_cols; i += THREADS) {
        K_[i / kernel_cols][i % kernel_cols] = kernel[i];
    }

    for (int i = tid; i < I_total; i += THREADS){
        int rr = i / tile_w;
        int cc = i % tile_w;
        int r = rr + sr0;
        int c = cc + sw0;
        I[i] = (r >= 0 && r <input_rows && c>=0 && c<input_cols)?input[r*input_cols + c] :0.0f;
    } 

    __syncthreads();

    int r = r0 + threadIdx.y;
    int c = c0 + threadIdx.x;
    if (r >= input_rows || c >= input_cols) return;

    float sum = 0.0f;
    for (int i = 0;i < kernel_rows; i++){
        for (int j = 0;j < kernel_cols;j++){
            sum += K_[i][j] * I[(threadIdx.y + i)*tile_w +(threadIdx.x + j)];
        }
    }
    output[r * input_cols + c] = sum;
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
        dim3 block(TILE_W,TILE_H);
        dim3 grid(
            (input_cols + TILE_W - 1) / TILE_W,
            (input_rows + TILE_H - 1) / TILE_H
        );
        int kh2 = kernel_rows / 2;
        int kw2 = kernel_cols / 2;
        size_t shared_memory_size = (TILE_H + 2*kh2) *(TILE_W + 2 * kw2)* sizeof(float);
        gaussian_blur_kernel<<<grid,block,shared_memory_size>>>(
            input,kernel,output,input_rows,input_cols,kernel_rows,kernel_cols
        );

        cudaDeviceSynchronize();
}
