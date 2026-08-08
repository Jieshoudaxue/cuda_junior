#include <cstdio>

// 这个核函数由 <<<1, block_size>>> 启动，
// grid_size 是整数 1, 则 gridDim.x = 1, gridDim.y = 1, gridDim.z = 1
// block_size 是 dim3 结构体变量，blockDim.x = 2, blockDim.y = 4, blockDim.z = 1
// 因此，线程总数是 grid_size * block_size = 1 * (2 * 4 * 1) = 8 个线程
// 内建变量 blockIdx 和 threadIdx 是 uint3 结构体类型的变量，也具有 x, y, z 三个成员，
// blockIdx 是线程块的索引，三个成员的范围是 [0, gridDim.x - 1], [0, gridDim.y - 1], [0, gridDim.z - 1]
// threadIdx 是线程的索引，三个成员的范围是 [0, blockDim.x - 1], [0, blockDim.y - 1], [0, blockDim.z - 1]
__global__ void hello_from_gpu() {
    // block 的索引
    const int bx = blockIdx.x;
    // block 内的线程局部坐标
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    printf("hello world from block: %d and thread: (%d, %d)!\n", bx, tx, ty);

    // 类似 C 语言的二维数组，CUDA 的线程在硬件上也是一维的连续排列，多维只是一种逻辑上的便利，
    // 因此，CUDA 线程在 block 内的局部索引也可以用线性化的一维方式表示，计算公式如下：
    const int linear_tid = threadIdx.z * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;
    printf("linear_tid: %d\n", linear_tid);

    // 同上，block 在 grid 内的索引也可以用线性化的一维方式表示，计算公式如下：
    const int linear_bid = blockIdx.z * gridDim.x * gridDim.y + blockIdx.y * gridDim.x + blockIdx.x;

    // grid 内的线程全局坐标
    const int global_tx = blockDim.x * blockIdx.x + threadIdx.x;
    const int global_ty = blockDim.y * blockIdx.y + threadIdx.y;
    const int global_tz = blockDim.z * blockIdx.z + threadIdx.z;
    printf("global thread coordinate: (%d, %d, %d)\n", global_tx, global_ty, global_tz);

    //计算线程在整个 Grid 中的一维全局索引
    const int threadsPerBlock = blockDim.x * blockDim.y * blockDim.z;
    const int global_linear_tid = linear_bid * threadsPerBlock + linear_tid;
    printf("global_linear_tid: %d\n", global_linear_tid);

    // 一个 block 内的线程可以进一步细分为多个 thread warp，即线程束，
    // cpu 的最小调度单位是线程，而 GPU 的最小调度单位是线程束，
    // 线程束中的线程按照 SIMT (single instruction, multiple thread) 的方式同时执行相同的指令，
    // 特别需要说明的是，线程束内的所有线程执行相同的内核代码，但每个线程可能遵循代码中的不同分支。
    // 也就是说，尽管程序的所有线程执行相同的代码，但线程不需要遵循相同的执行路径。
    // 每个线程束包含固定 32 个线程，warpSize 是一个内建变量，表示线程束的大小，永远为 32 。
    // 对于这个样例，总线程为 8 个线程，所以只有一个线程束，其中 8 个活跃，24 个非活跃，只是占位。
    printf("warpSize: %d\n", warpSize);
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
    return 0;
}