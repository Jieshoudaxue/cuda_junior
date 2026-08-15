#include <cstdio>

#include "cuda_error.cuh"

// 静态全局内存变量使用 __device__ 修饰，在所有主机和设备函数外定义，
// 静态全局内存变量的作用域是整个网格，生命周期是整个应用程序（由编译器确定），位置在 GPU 设备，但位于片外，可读可写。
// 类比： CUDA 的静态全局内存变量类似 C++ 的全局变量所在的全局数据区
// 通用的两种定义方式：
// 定义单个变量：__device__ T x;
// 定义固定长度的数组：__device__ T y[N];
__device__ int d_x = 1;
__device__ int d_y[2];

__global__ void my_kernel(void) {
    // 在核函数中，可直接对静态全局内存变量进行访问，可读可写，就像 C++ 中的全家变量
    d_y[0] += d_x;
    d_y[1] += d_x;
    printf("d_x = %d, d_y[0] = %d, d_y[1] = %d\n", d_x, d_y[0], d_y[1]);
}

int main(void) {
    int h_y[2] = {100, 200};

    // 主机代码不能直接访问静态全局内存变量，因为他是属于设备的，
    // 但可以使用 cudaMemcpyToSymbol 和 cudaMemcpyFromSymbol 在静态全家内存与主机内存间传输数据。
    // cudaMemcpyToSymbol 用于将主机数据拷贝到静态全家内存中，接口定义：
    // cudaError_t cudaMemcpyToSymbol(const void* symbol, const void* src, size_t count, size_t offset = 0, cudaMemcpyKind kind = cudaMemcpyHostToDevice);
    // const void* symbol: 静态全局内存变量名
    // const void* src: 主机内存缓冲区指针
    // size_t count: 复制的字节数
    // size_t offset = 0: 从 symbol 对应设备地址开始偏移的字节数
    // cudaMemcpyKind kind = cudaMemcpyHostToDevice: 可选参数
    CHECK_CUDA_CALL(cudaMemcpyToSymbol(d_y, h_y, sizeof(int) * 2));

    my_kernel<<<1, 1>>>();
    CHECK_CUDA_CALL(cudaDeviceSynchronize());

    // cudaMemcpyFromSymbol 用于将静态全家内存数据拷贝到主机内存，接口定义：
    // cudaError_t cudaMemcpyFromSymbol(void* dst, const void* symbol, size_t count, size offset = 0, cudaMemcpyKind kind = cudaMemcpyDeviceToHost);
    // void* dst: 主机内存缓冲区指针
    // const void* symbol: 静态全局内存变量名
    // size_t count: 复制的字节数
    // size_t offset = 0: 从 symbol 对应设备地址开始偏移的字节数
    // cudaMemcpyKind kind = cudaMemcpyHostToDevice: 可选参数
    CHECK_CUDA_CALL(cudaMemcpyFromSymbol(h_y, d_y, sizeof(int) * 2));
    printf("h_y[0] = %d, h_y[1] = %d\n", h_y[0], h_y[1]);

    return 0;
}