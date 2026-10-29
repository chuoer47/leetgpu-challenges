#include <cstddef>
#include <cuda_runtime.h>

// Environment smoke test, separate from the user's exercise solutions.
__global__ void add(const float* a, const float* b, float* c, size_t n) {
    size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n)
        c[i] = a[i] + b[i];
}

extern "C" void solve(const float* a, const float* b, float* c, size_t n) {
    add<<<(n + 255) / 256, 256>>>(a, b, c, n);
}
