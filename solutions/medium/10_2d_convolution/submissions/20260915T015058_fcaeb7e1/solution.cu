#include <cuda_runtime.h>

#define TILE_H 16
#define TILE_W 16
#define THREADS (TILE_H * TILE_W)   // 256
#define MAX_KERNEL_SIZE 32

__global__ void convolution2d_kernel(
    const float* __restrict__ input,
    const float* __restrict__ kernel,
    float* output,
    int input_rows, int input_cols,
    int kernel_rows, int kernel_cols)
{
    int output_rows = input_rows  - kernel_rows + 1;
    int output_cols = input_cols  - kernel_cols + 1;

    int tile_in_rows = TILE_H + kernel_rows - 1;
    int tile_in_cols = TILE_W + kernel_cols - 1;

    __shared__ float K_[MAX_KERNEL_SIZE][MAX_KERNEL_SIZE];
    __shared__ float I_[TILE_H + MAX_KERNEL_SIZE - 1][TILE_W + MAX_KERNEL_SIZE - 1];

    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    int out_r0 = blockIdx.y * TILE_H;
    int out_c0 = blockIdx.x * TILE_W;

    // 1) kernel 搬到 shared
    for (int i = tid; i < kernel_rows * kernel_cols; i += THREADS) {
        K_[i / kernel_cols][i % kernel_cols] = kernel[i];
    }

    // 2) input tile 搬到 shared
    for (int i = tid; i < tile_in_rows * tile_in_cols; i += THREADS) {
        int lr = i / tile_in_cols;
        int lc = i % tile_in_cols;
        int gr = out_r0 + lr;
        int gc = out_c0 + lc;
        I_[lr][lc] = (gr < input_rows && gc < input_cols)
                     ? input[gr * input_cols + gc]
                     : 0.f;
    }

    __syncthreads();

    int r = out_r0 + threadIdx.y;
    int c = out_c0 + threadIdx.x;
    if (r >= output_rows || c >= output_cols) return;

    // 3) 卷积：全部读 shared memory
    float sum = 0.f;
    #pragma unroll
    for (int i = 0; i < kernel_rows; ++i) {
        #pragma unroll
        for (int j = 0; j < kernel_cols; ++j) {
            sum += K_[i][j] * I_[threadIdx.y + i][threadIdx.x + j];
        }
    }

    output[r * output_cols + c] = sum;
}

extern "C" void solve(
    const float* input, const float* kernel, float* output,
    int input_rows, int input_cols, int kernel_rows, int kernel_cols)
{
    int output_rows = input_rows - kernel_rows + 1;
    int output_cols = input_cols - kernel_cols + 1;

    dim3 block(TILE_W, TILE_H);
    dim3 grid((output_cols + TILE_W - 1) / TILE_W,
              (output_rows + TILE_H - 1) / TILE_H);

    convolution2d_kernel<<<grid, block>>>(
        input, kernel, output,
        input_rows, input_cols, kernel_rows, kernel_cols);

    cudaDeviceSynchronize();
}