#include <cstdio>

// 这个核函数由 <<<1, block_size>>> 启动，
// grid_size 是整数 1, 则 gridDim.x = 1, gridDim.y = 1, gridDim.z = 1
// block_size 是 dim3 结构体变量，blockDim.x = 2, blockDim.y = 4, blockDim.z = 1
// 内建变量 blockIdx 和 threadIdx 是 uint3 结构体类型的变量，也具有 x, y, z 三个成员，
// blockIdx 是线程块的索引，三个成员的范围是 [0, gridDim.x - 1], [0, gridDim.y - 1], [0, gridDim.z - 1]
// threadIdx 是线程的索引，三个成员的范围是 [0, blockDim.x - 1], [0, blockDim.y - 1], [0, blockDim.z - 1]
__global__ void hello_from_gpu() {
    const int bx = blockIdx.x;
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    printf("hello world from block: %d and thread: (%d, %d)!\n", bx, tx, ty);
}


// 这个样例讨论多维网格和多维线程块的情况
int main() {
    // <<<>>> 的参数永远是 <<<grid_size, block_size>>>，
    // 这两个参数支持整数传參，但实际上他们都是 dim3（dimension，维度）结构体类型的变量。
    // dim3 有三个成员 x, y, z，分别表示三维空间的三个维度。
    // 如果 <<<>>> 中的参数是整数，那么这个整数就是 dim3 的 x 成员，y 和 z 成员默认为 1。这种就是一维网格和一维线程块的情况。
    // 如果 <<<>>> 中的参数是 dim3 结构体变量，那么这个 dim3 的 x, y, z 成员就分别表示三维空间的三个维度。这种就是多维网格和多维线程块的情况。
    // grid_size 的值存储在内建变量 gridDim 中，block_size 的值存储在内建变量 blockDim 中。
    const dim3 block_size(2, 4);
    hello_from_gpu<<<1, block_size>>>();
    cudaDeviceSynchronize();


    // cudaDeviceProp prop;
    // cudaGetDeviceProperties(&prop, 0);

    // printf("最大线程数/线程块: %d\n", prop.maxThreadsPerBlock);
    // printf("线程块各维度上限 (x, y, z): (%d, %d, %d)\n",
    //        prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
    // printf("网格各维度上限 (x, y, z): (%d, %d, %d)\n",
    //        prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
    // printf("每个线程块最大共享内存: %zu 字节\n", prop.sharedMemPerBlock);

    return 0;
}