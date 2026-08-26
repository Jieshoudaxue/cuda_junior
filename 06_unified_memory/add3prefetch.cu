#include <cmath>
#include <cstdio>

#include "cuda_error.cuh"

typedef double real;
const real EPSILON = 1e-15;
// typedef float real;
// const real EPSILON = 1e-6;

const real a = 1.23;
const real b = 2.34;
const real c = 3.57;

__global__ void add(const double *px, const double *py, double *pz) {
    const int tid = blockDim.x * blockIdx.x + threadIdx.x;
    pz[tid] = px[tid] + py[tid];
}

void check(const double *pz, const int N) {
    bool has_error = false;
    for (int i = 0; i < N; ++i) {
        if (fabs(pz[i] < c) > EPSILON) {
            has_error = true;
        }
    }
    printf("%s\n", has_error ? "Has errors" : "No errors");
}

int main(void) {
    const int N = 1e8;
    const int M = sizeof(double) * N;
    double *px, *py, *pz;

    CHECK_CUDA_CALL(cudaMallocManaged((void **)&px, M));
    CHECK_CUDA_CALL(cudaMallocManaged((void **)&py, M));
    CHECK_CUDA_CALL(cudaMallocManaged((void **)&pz, M));

    for (int i = 0; i < N; i ++) {
        px[i] = a;
        py[i] = b;
    }

    const int block_size = 128;
    const int grid_size = (N - 1) / block_size + 1;

    int device_id = 0;
    // 获取当前机器的 GPU 设备 ID，使用 nvidia-smi 也能看得到
    CHECK_CUDA_CALL(cudaGetDevice(&device_id));
    // 使用统一内存需要尽量保持数据的局部性（让数据靠近对应的处理器），避免缺页异常，以提高性能
    // cudaMemPrefetchAsync 函数原型：
    // cudaError_t cudaMemPrefetchAsync(const void *devPtr, size_t count, int dstDevice, cudaStream_t stream);
    // cudaMemPrefetchAsync 函数的作用就是将一块统一内存数据迁移到主机或设备内存中，提高数据局部性。
    // 以下面的调用为例，作用是将 M 大小的统一内存数据，迁移到 GPU 设备显存中。
    CHECK_CUDA_CALL(cudaMemPrefetchAsync(px, M, device_id, NULL));
    CHECK_CUDA_CALL(cudaMemPrefetchAsync(py, M, device_id, NULL));
    CHECK_CUDA_CALL(cudaMemPrefetchAsync(pz, M, device_id, NULL));

    add<<<grid_size, block_size>>>(px, py, pz);

    // 在使用统一内存时，要尽可能多用 cudaMemPrefetchAsync 函数，提高数据局部性，规避缺页。
    // 即使这样，使用统一内存也比不使用要简洁，而且由于可以申请超量的内存，因此很多时候必须使用统一内存。

    // 这里的调用，作用是将 M 大小的统一内存数据，迁移到主机内存中（cudaCpuDeviceId 代表主机设备号）。
    CHECK_CUDA_CALL(cudaMemPrefetchAsync(pz, M, cudaCpuDeviceId, NULL));

    check(pz, N);

    CHECK_CUDA_CALL(cudaFree(px));
    CHECK_CUDA_CALL(cudaFree(py));
    CHECK_CUDA_CALL(cudaFree(pz));
    
    return 0;
}