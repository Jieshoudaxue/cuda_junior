#include <cstdio>

__global__ void hello_from_gpu() {
    printf("hello world from the gpu!\n");
}



int main() {
    // 这里的 <<<2, 4>>> 表示启动两个线程块，每个线程块中有四个线程，共有 2 * 4 = 8 个线程。
    // 这 8 个线程执行同样的 CUDA 核函数，即 “单指令-多线程”。
    hello_from_gpu<<<2, 4>>>();
    cudaDeviceSynchronize();


    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    printf("最大线程数/线程块: %d\n", prop.maxThreadsPerBlock);
    printf("线程块各维度上限 (x, y, z): (%d, %d, %d)\n",
           prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
    printf("网格各维度上限 (x, y, z): (%d, %d, %d)\n",
           prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
    printf("每个线程块最大共享内存: %zu 字节\n", prop.sharedMemPerBlock);

    return 0;
}