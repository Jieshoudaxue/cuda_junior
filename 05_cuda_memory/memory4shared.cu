#include <cstdio>

#include "cuda_error.cuh"

__global__ void my_kernel() {
    // 共享内存（shared memory）作用域是单个线程块，生命周期是所属线程块的生命周期，位置在 SM 上，且位于 GPU 片上，可读可写，
    // 共享内存在硬件上与寄存器类似，位于 GPU 片上，速度仅次于寄存器，数量也有限，
    // 通常情况下，单个线程块共享内存上限是 48～96KB，一个 SM 的共享内存上限也是 48～96KB，
    // 在编程方面，共享内存介于寄存器-局部内存与全局内存之间，生命周期等于所属线程快，他的主要作用是减少对全局内存的访问。
    // 特别注意：每个线程块的线程只能访问自己所属线程块的共享内存，两个线程块不能通过共享内存进行通信。
    // 类比：共享内存在 C++ 中找不到类似用法，属于 CUDA 独有的，因为 CPU 没有线程块的概念。
    // 共享内存变量使用 __shared__ 修饰，在核函数内定义，可以是数组，也可以是单个变量
    // 定义单个变量： __shared__ T x;
    // 定义固定长度的数组：_ __shared__ T y[N];
    __shared__ int shared_data[8];

    int bid = blockIdx.x;
    int tid = threadIdx.x;
    shared_data[tid] = bid * 100 + tid;

    // 同步同一个Block的线程
    __syncthreads();

    printf("block %d, thread %d: %d\n", bid, tid, shared_data[tid]);
}

int main(void) {
    my_kernel<<<2, 8>>>();
    CHECK_CUDA_CALL(cudaDeviceSynchronize());

    return 0;
}