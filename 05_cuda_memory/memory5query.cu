#include <cstdio>

#include "cuda_error.cuh"

int main(void) {
    int device_id = 0;
    CHECK_CUDA_CALL(cudaSetDevice(device_id));

    cudaDeviceProp prop;
    CHECK_CUDA_CALL(cudaGetDeviceProperties(&prop, device_id));

    printf("Basic info of Device:\n");
    printf("    Device id: %d\n", device_id);
    printf("    Device name: %s\n", prop.name);
    printf("    Compute capability: %d.%d\n", prop.major, prop.minor);
    // 一个 GPU 设备由若干个 SM 组成，每个 SM 有一定数量的寄存器，共享内存+L1缓存，若干整数和不同浮点数的处理核，若干张量核,
    // CUDA 程序在调度时，一个 block 的线程会被分配到一个 SM 上。具体调度的细节，程序本身无法控制，由调度器自己决定。
    printf("    Number of SM: %d\n", prop.multiProcessorCount);

    // SM 的占有率（theoretical occupancy）：一个 SM 上被分配的线程数量与 SM 支持的最大线程数量之比。占有率越高，一般程序性能越高，但占有率受多个因素限制。
    // 第一种情况：设备代码的寄存器和共享内存使用非常少，可以忽略不计的情况下，占有率由调用核函数时指定的 block_size 决定。
    // 举例：我的设备下面三个值分别是 14, 16, 1024。
    // 如果 <<<2, 768>>> ，由于单个 SM 最多只能有 1024 个，因此一个 SM 只能被分配 1 个 block，占有率为 768/1024 = 75%。
    // 如果 <<<16, 64>>>，由于单个 SM 最多支持 16 个 block 和 1024 个线程，而 16 × 64 = 1024， 不大于 1024，因此一个 SM 被分配全部 16 个 block，占有率为 1024/1024 = 100%。
    // 如果 <<<16, 48>>>，由于单个 SM 最多支持 16 个 block 和 1024 个线程，而 16 × 48 = 768，不大于 1024，因此一个 SM 被分配全部 16 个 block，占有率为 768/1024 = 75%。
    // 针对上面这个例子，由于线程束固定为 32 个。因此每个 block 48 个线程，会被分配 2 个线程束（其中一个有 16 个空闲线程），总共被分配 16 × 2 = 32 个线程束，而不是 768 / 32 = 24 个。
    // 因此通常情况下，block size 应为 32 的整数倍，否则就会有空闲线程，损失占有率。
    printf("    Maximum number of blocks per SM: %d\n", prop.maxBlocksPerMultiProcessor);
    printf("    Maximum number of threads per SM: %d\n", prop.maxThreadsPerMultiProcessor);


    printf("Memory info of Device:\n");
    printf("    Amount of global memory: %g GB\n", prop.totalGlobalMem/(1024.0*1024*1024));
    printf("    Amount of constant memory: %g KB\n", prop.totalConstMem/1024.0);
    // 第二种情况：设备代码的寄存器和共享内存使用比较多，此时每个 SM 有限的寄存器和共享内存会影响占有率
    // 提示：每个线程块的线程数和每个线程块的共享内存使用量由程序员明确控制，并且可以进行调整以实现所需的占用率。但程序员对寄存器使用的控制有限，因为编译器和运行时系统会尝试优化寄存器使用。
    // 举例：我的设备下面两个值分别是 64 * 1024，64 KB
    // 假设一个线程块使用共享内存为 8KB，那么这个 SM 最多只能容纳 64 / 8 = 8 个线程块，而不是 16 个。如果 <<<16, 64>>>，那么 SM 占有率为 8 × 64 / 1024 = 50%。
    printf("    Maximum amount of registers per SM: %d * 1024\n", prop.regsPerMultiprocessor/1024);
    printf("    Maximum amount of shared memory Per SM: %g KB\n", prop.sharedMemPerMultiprocessor/1024.0);
    printf("    Maximum amount of registers per block: %d * 1024\n", prop.regsPerBlock/1024);
    printf("    Maximum amount of shared memory per block: %g KB\n", prop.sharedMemPerBlock/1024.0);
}