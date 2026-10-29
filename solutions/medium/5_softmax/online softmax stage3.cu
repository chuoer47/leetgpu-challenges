/*
PASS: 12 cases (example + functional + performance)
GPU: NVIDIA GeForce RTX 4090 D; median solve + sync: 0.1594 ms; CUDA event: 0.1442 ms
*/

#include <cuda_runtime.h>
#include <float.h>

#define WARP_SIZE 32

struct __align__(8) MD {
  float m;
  float d;
};

template <const int kWarpSize = WARP_SIZE>
__device__ __forceinline__ MD warp_reduce_md_op(MD value) {
  unsigned int mask = 0xffffffff;
#pragma unroll
  for (int stride = kWarpSize >> 1; stride >= 1; stride >>= 1) {
    MD other;
    other.m = __shfl_xor_sync(mask, value.m, stride, kWarpSize);
    other.d = __shfl_xor_sync(mask, value.d, stride, kWarpSize);

    bool value_bigger = (value.m > other.m);
    MD bigger_m = value_bigger ? value : other;
    MD smaller_m = value_bigger ? other : value;

    value.d = bigger_m.d + smaller_m.d * __expf(smaller_m.m - bigger_m.m);
    value.m = bigger_m.m;
  }
  return value;
}

// ================= Stage 1: per block reduce, output block-wise MD =================
template<const int NUM_THREADS=256>
__global__ void online_softmax_stage1_kernel(const float* x, MD* block_md_out, int N) {
  int local_tid = threadIdx.x;
  int global_tid = blockIdx.x * NUM_THREADS + threadIdx.x;
  const int WARP_NUM = NUM_THREADS / WARP_SIZE;
  int warp_id = local_tid / WARP_SIZE;
  int lane_id = local_tid % WARP_SIZE;

  MD val;
  val.m = (global_tid < N) ? x[global_tid]:-FLT_MAX;
  val.d = (global_tid < N) ? 1.0f:0.0f;

  __shared__ MD shared[WARP_NUM];
  MD res = warp_reduce_md_op<WARP_SIZE>(val);
  if(lane_id == 0){
    shared[warp_id] = res;
  }
  __syncthreads();

  if(local_tid < WARP_SIZE){
    MD block_res = (local_tid < WARP_NUM) ? shared[local_tid] : MD{-FLT_MAX,0.0f};
    block_res = warp_reduce_md_op<WARP_NUM>(block_res);
    if(local_tid == 0){
      block_md_out[blockIdx.x] = block_res;
    }
  }
}

// ================= Stage2: single block merge all block MD into global MD =================
template<const int NUM_THREADS=256>
__global__ void online_softmax_stage2_kernel(const MD* block_md_in, MD* global_md_out, int num_blocks) {
  int local_tid = threadIdx.x;
  MD val = { -FLT_MAX, 0.0f };

  // grid stride loop to consume all block MD
  for(int i = local_tid; i < num_blocks; i += NUM_THREADS){
    MD b = block_md_in[i];
    // merge val and b
    MD bigger, smaller;
    if(val.m > b.m){
      bigger = val;
      smaller = b;
    }else{
      bigger = b;
      smaller = val;
    }

    val.d = bigger.d + smaller.d *  __expf(smaller.m - bigger.m);
    val.m = bigger.m;
  }

  // block reduce to get final global MD
  const int WARP_NUM = NUM_THREADS / WARP_SIZE;
  int warp_id = local_tid / WARP_SIZE;
  int lane_id = local_tid % WARP_SIZE;
  __shared__ MD shared[WARP_NUM];

  MD res = warp_reduce_md_op<WARP_SIZE>(val);
  if(lane_id == 0){
    shared[warp_id] = res;
  }
  __syncthreads();

  if(local_tid < WARP_SIZE){
    MD block_res = (local_tid < WARP_NUM) ? shared[local_tid] : MD{-FLT_MAX,0.0f};
    block_res = warp_reduce_md_op<WARP_NUM>(block_res);
    if(local_tid == 0){
      *global_md_out = block_res;
    }
  }
}

// ================= Stage3: compute softmax value using global m,d =================
template<const int NUM_THREADS=256>
__global__ void online_softmax_stage3_kernel(const float* x, float* y, const MD* global_md, int N) {
  int global_tid = blockIdx.x * NUM_THREADS + threadIdx.x;
  MD g = *global_md;
  float inv_d = __fdividef(1.0f, g.d);
  if(global_tid < N){
    y[global_tid] = __expf(x[global_tid] - g.m) * inv_d;
  }
}

extern "C" void solve(const float* input, float* output, int N) {
  if (N <= 0)
    return;
  const int threadsPerBlock = 256;
  int blocksPerGrid = (N + threadsPerBlock -1)/threadsPerBlock;

  MD* d_block_md = nullptr;
  MD* d_global_md = nullptr;

  cudaMalloc(&d_block_md, blocksPerGrid * sizeof(MD));
  cudaMalloc(&d_global_md, sizeof(MD));

  // Stage 1: every block compute local MD
  online_softmax_stage1_kernel<256><<<blocksPerGrid, threadsPerBlock>>>(input, d_block_md, N);

  // Stage2: single block merge all block MD to get global m,d
  online_softmax_stage2_kernel<256><<<1, threadsPerBlock>>>(d_block_md, d_global_md, blocksPerGrid);

  // Stage3: calculate softmax
  online_softmax_stage3_kernel<256><<<blocksPerGrid, threadsPerBlock>>>(input, output, d_global_md, N);

  cudaDeviceSynchronize();

  cudaFree(d_block_md);
  cudaFree(d_global_md);
}
