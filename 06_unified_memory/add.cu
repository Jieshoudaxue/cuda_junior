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

// 使用统一内存，核函数并没有什么区别
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

    // 统一内存在设备上是当作全局内存使用的（重中之重），而且必须在主机端定义和分配内存。
    // 即动态分配统一内存，只能在主机端使用 cudaMallocManaged 分配，而不能在核函数中调用。
    // cudaError_t cudaMallocManaged(void **devPtr, size_t size, unsigned flags = cudaMemAttachGlobal);
    // cudaMemAttachGlobal: 表示分配的全局内存可由 GPU 设备访问
    // cudaMemAttachHost: 暂时不讨论
    CHECK_CUDA_CALL(cudaMallocManaged((void **)&px, M));
    CHECK_CUDA_CALL(cudaMallocManaged((void **)&py, M));
    CHECK_CUDA_CALL(cudaMallocManaged((void **)&pz, M));

    // 分配的统一内存变量，既可以被设备访问，也可以被主机访问。省去了主机到设备，设备到主机的拷贝步骤。
    // 使用统一内存使得 CUDA 编程更简单，而且可能提供比手动移动数据更好的性能（底层有优化）。
    // 当然，使用统一内存最大的优势是可以申请超过 GPU 显存容量的内存，虽然会降低一些性能（CPU 内存更多，但速度较低），但可以让较大的程序跑起来。
    // 另外，同一个程序，可以同时使用统一内存和非统一内存。
    for (int i = 0; i < N; i ++) {
        px[i] = a;
        py[i] = b;
    }

    const int block_size = 128;
    const int grid_size = (N - 1) / block_size + 1;
    add<<<grid_size, block_size>>>(px, py, pz);

    // 由于核函数的调用是异步的，因此在主机端访问统一内存前，需要调用 cudaDeviceSynchronize，确保核函数对统一内存的访问已经结束。
    // 对于 CC 6（帕斯卡）及以后的 GPU，使用了第二代统一内存，就不需要这里的同步操作了。
    // 我的设备是 MX450, CC7.5，因此可以移除下面的同步函数，不影响结果。
    CHECK_CUDA_CALL(cudaDeviceSynchronize());
    check(pz, N);

    CHECK_CUDA_CALL(cudaFree(px));
    CHECK_CUDA_CALL(cudaFree(py));
    CHECK_CUDA_CALL(cudaFree(pz));
    
    return 0;
}