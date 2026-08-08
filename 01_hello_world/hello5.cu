#include <cstdio>
#include <stdint.h>

// 这个样例讨论网格和线程块的最大尺寸
int main() {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    // 网格各维度最大值分别为：2^31 −1 = 2147483647、65535 和 65535
    printf("网格各维度上限: (%d, %d, %d)\n",
           prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
    // 线程块各维度最大值分别为：1024、1024 和 64
    printf("线程块各维度上限: (%d, %d, %d)\n",
           prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
    // 但是，一个线程块最多只能有 1024 个线程，而不是 1024 * 1024 * 64 = 67108864 个线程
    printf("单个线程块的最大线程数: %d\n", prop.maxThreadsPerBlock);

    // 如果网格和线程块的尺寸都按最大值设置，那么总线程数为：18,446,741,874,820,512,768
    // 即使只使用一维网格，如果按上限设置，最多能使用 (2^31 −1) * 1024 = 2,199,023,255,552 个线程，即两万亿左右，
    // 这个数量通常是远大于一般编程问题所需要的线程数的，这表明 cuda 并行计算能力的理论上限是非常高的。
    // 通常情况下，GPU 实际的处理核个数在几千到几万个之间，只要线程数比处理核个数多几倍时，就能充分利用 GPU 的并行计算能力了。
    const uint64_t max_grid_size = prop.maxGridSize[0] * prop.maxGridSize[1] * prop.maxGridSize[2];
    const uint64_t max_block_size = prop.maxThreadsPerBlock;
    const uint64_t max_thread_count = max_grid_size * max_block_size;
    printf("网格和线程块的最大尺寸下的总线程数: %llu\n", (unsigned long long)max_thread_count);
    return 0;
}