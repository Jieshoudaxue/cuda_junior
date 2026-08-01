#include <cstdio>

// 这个核函数由 <<<2, 4>>> 启动，
// 这里的 2 是 grid_size,这个值存储在内建变量 gridDim.x 中，表示网格中线程块的数量。
// 这里的 4 是 block_size,这个值存储在内建变量 blockDim.x 中，表示线程块中线程的数量。
// blockIdx.x 也是一个内建变量，表示线程块的索引号，范围是 [0, gridDim.x - 1]。
// threadIdx.x 也是一个内建变量，表示线程的索引号，范围是 [0, blockDim.x - 1]。
// 因此，内建变量 blockIdx.x 和 threadIdx.x 可以唯一标识一个线程。
__global__ void hello_from_gpu() {
    const int bid = blockIdx.x;
    const int tid = threadIdx.x;
    printf("hello world from block %d(gridDim.x: %d), thread %d(bolckDim.x: %d)!\n", bid, gridDim.x, tid, blockDim.x);
}

// 这个例子仅讨论一维网格和一维线程块的情况，
int main() {
    // 这里的 <<<2, 4>>> 表示启动两个线程块，每个线程块中有四个线程，共有 2 * 4 = 8 个线程。
    // 这 8 个线程并发执行同样的 CUDA 核函数，即 “单指令-多线程”，（single instruction, multiple thread，SIMT）
    hello_from_gpu<<<2, 4>>>();
    cudaDeviceSynchronize();
    return 0;
}