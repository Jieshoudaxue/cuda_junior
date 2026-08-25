#include <cmath>
#include <cstdio>

#include "cuda_error.cuh"

// GPU 的全局内存可以动态分配，也可以静态分配，即静态全局内存变量，见 05_cuda_memory/memory2static.cu，
// 统一内存可以动态分配，也可以静态分配，即静态统一内存变量，需要在 __device__ 的后面再加一个 __managed__。
// 静态统一内存变量要在所有函数（主机+设备）之外定义，可见范围是所在翻译单元的所有函数（主机+设备）。
// 通用的两种定义方式：
// 定义单个变量：__device__ __managed__ T x;
// 定义固定长度的数组：__device__ __managed__ T y[N];
__device__ __managed__ int ret[1000];

__global__ void addAB(int a, int b) {
    ret[threadIdx.x] = a + b + threadIdx.x;
}

int main(void) {
    addAB<<<1, 1000>>>(10, 100);
    cudaDeviceSynchronize();

    for (int i = 0; i < 1000; i ++) {
        printf("%d: A + B = %d\n", i, ret[i]);
    }
}