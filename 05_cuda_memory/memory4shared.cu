#include <cstdio>

#include "cuda_error.cuh"

__global__ void my_kernel_1() {
    // 共享内存（shared memory）可见范围是单个线程块，生命周期是所属线程块的生命周期，位置在 SM 上，且位于 GPU 片上，可读可写，
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

    // __syncthreads 用于同步单个线程块里的线程，只能用于核函数（也包括被核函数调用的设备函数），
    // __syncthreads 提供的是一个屏障（barrier）：
    // 1. 当线程块中的所有线程都执行到 __syncthreads() 时，才会继续往下执行。先到的线程会等待后到的线程。
    // 2. 在屏障之前，每个线程执行的语句（尤其是对共享/全局内存的写）必须完成，才能让任何线程越过屏障。
    // 即屏障之前的工作全部完成 → 所有线程汇合 → 屏障之后的工作才开始。 这就是"排队"的精确含义。
    // 在这里，__syncthreads 用来确保赋值过程不被干扰，一般来说，在利用共享内存中的数据之前，都要进行线程块内的同步操作，以确保共享内存数组中的所有元素都已经更新完毕。
    __syncthreads();

    printf("my_kernel_1: block %d, thread %d: %d\n", bid, tid, shared_data[tid]);
}

__global__ void my_kernel_2() {
    // 上面的 reduce_shared 使用的是静态的共享内存，即编译时指定共享内存的大小
    // 这里的核函数使用动态共享内存，当调用核函数时指定共享内存大小。
    // 注意：动态共享内存和静态共享内存没有性能的区别
    extern __shared__ int shared_data[];

    int bid = blockIdx.x;
    int tid = threadIdx.x;
    shared_data[tid] = bid * 100 + tid;

    __syncthreads();

    printf("my_kernel_2: block %d, thread %d: %d\n", bid, tid, shared_data[tid]);
}

int main(void) {
    my_kernel_1<<<2, 8>>>();
    CHECK_CUDA_CALL(cudaDeviceSynchronize());

    // 第三个参数，用来指定动态共享内存的大小
    my_kernel_2<<<2, 8, 8>>>();
    CHECK_CUDA_CALL(cudaDeviceSynchronize());

    return 0;
}