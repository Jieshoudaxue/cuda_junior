#include <cstdio>

#include "cuda_error.cuh"

const unsigned int WIDTH = 8;
const unsigned int BLOCK_SIZE = 16;
const unsigned int FULL_MASK = 0xffffffff;

__global__ void test_warp_primitives(void) {
    // grid_size 是 1 ，block_size 是 16,
    // 因此，这里的 threadIdx.x 的范围是从 0 到 15
    int tid = threadIdx.x;
    // 关注这里的打印技巧，虽然核函数是多线程并发执行的，但是printf 的打印输出内容是整齐排列的，很像串行打印的效果：
    // threadIdx.x:     0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 
    // 核心原因是同一个线程束内的线程执行到 printf 时，硬件会对这些线程按线程号从小到大进行串行化输出
    if (tid == 0) {
        printf("threadIdx.x:    ");
    }
    printf("%2d ", tid);
    if (tid == 0) {
        printf("\n");
    }

    // __ballot_sync 和 __all_sync 和 __any_sync 是线程束表决函数（warp vote functions），函数原型：
    // unsigned int __ballot_sync(unsigned mask, int predicate);
    // int __all_sync(unsigned mask, int predicate);
    // int __any_sync(unsigned mask, int predicate);
    
    // unsigned int __ballot_sync(unsigned mask, int predicate);
    // __ballot_sync 用于在指定的线程掩码内，收集所有活跃线程对某个条件的判断结果，并以掩码的形式返回，
    // 参数 mask 用于指定哪些线程参与这次投票，掩码中为 1 的线程才会参与，
    // 参数 predicate 是每个参与线程的条件判断，如果为零/false，则对应的线程掩码为 0。如果非零/true，则对应的线程掩码为 1。

    // 当前核函数共 16 个线程，所以参与表决的线程是 16 个，
    // 以这里为例，针对 tid 为 0 的线程，不满足 tid > 0, 则 predicate 为 0 ; 其他线程满足 tid > 0, 则 predicate 为 1，因此 mask1 的值为 0xfffe
    unsigned int mask1 = __ballot_sync(FULL_MASK, tid > 0);
    // 以这里为例，针对 tid 为 0 的线程，满足 tid == 0, 则 predicate 为 1 ; 其他线程不满足 tid == 0, 则 predicate 为 0，因此 mask1 的值为 0x1
    unsigned int mask2 = __ballot_sync(FULL_MASK, tid == 0);
    if (tid == 0) {
        printf("FULL_MASK   = %x\n", FULL_MASK);
    }
    if (tid == 1) {
        printf("mask1       = %x\n", mask1);
    }
    if (tid == 0) {
        printf("mask2       = %x\n", mask2);
    }

    // int __all_sync(unsigned mask, int predicate);
    // __all_sync 用于在指定的线程掩码内，收集所有活跃线程对某个条件的判断结果，只有所有的线程均为 非零/true，最终返回结果才会为 1, 只要有一个为 零/false, 则最终返回结果就会为 0。
    
    // 当前核函数共 16 个线程，所以参与表决的线程是 16 个，
    // 以这里为例，针对 tid 为 0 的线程，predicate 为 0 ，则最终返回结果为 0
    int result = __all_sync(FULL_MASK, tid);
    if (tid == 0) {
        printf("__all_sync(FULL_MASK): %d\n", result);
    }
    // 以这里为例，mask1 为 0xfffe, 线程 0 不参与表决，则所有不参与表决的线程，predicate 为非零 ，则返回结果为 1
    result = __all_sync(mask1, tid);
    if (tid == 1) {
        printf("__all_sync    (mask1): %d\n", result);
    }

    // int __any_sync(unsigned mask, int predicate);
    // __any_sync 用于在指定的线程掩码内，收集所有活跃线程对某个条件的判断结果, 只要有一个为 非零/true, 则最终返回结果就会为 1。

    // 当前核函数共 16 个线程，所以参与表决的线程是 16 个，
    // 以这里为例，针对 tid 为 0 的线程，predicate 为 0 ，其他所有线程，predicate 为非零，则最终返回结果为 1
    result = __any_sync(FULL_MASK, tid);
    if (tid == 0) {
        printf("__any_sync(FULL_MASK): %d\n", result);
    }
    // 以这里为例，mask2 为 0x1, 只有线程 0 参与表决，且 predicate 为 0，则返回结果为 0
    result = __any_sync(mask2, tid);
    if (tid == 0) {
        printf("__any_sync    (mask2): %d\n", result);
    }


    // 由于 block_size 是 16, WIDTH 是 8，
    // 因此，这里的 lane_id 的范围是 0 到 7，通过这种方式可以将线程束的 32 个线程分段，变成多个子线程束，lane id 的取值范围是 [0, WIDTH-1]
    // 打印结果是：
    // lane_id:         0  1  2  3  4  5  6  7  0  1  2  3  4  5  6  7 
    int lane_id = tid % WIDTH;
    if (tid == 0) {
        printf("lane_id:        ");
    }
    printf("%2d ", lane_id);
    if (tid == 0) {
        printf("\n");
    }

    // __shfl_sync 和 __shfl_up_sync 和 __shfl_down_sync 和 __shfl_xor_sync 是线程束洗牌函数（warp shuffle functions），
    // 洗牌范围由 width 确定，默认是 32，即线程束的线程个数，如果 width 小于 32, 则将线程束分段，划分为多个子线程束，每个子线程束分别洗牌
    // T __shfl_sync(unsigned mask, T var, int srcLane, int width = warpSize);
    // T __shfl_up_sync(unsigned mask, T var, unsigned delta, int width = warpSize);
    // T __shfl_down_sync(unsigned mask, T var, unsigned delta, int width = warpSize);
    // T __shfl_xor_sync(unsigned mask, T var, int laneMask, int width = warpSize);

    // T __shfl_sync(unsigned mask, T var, int srcLane, int width = warpSize);
    // __shfl_sync 的作用是从指定范围 [0, width-1] 的某一个线程中读取 var , 并返回给所有参与线程，类似广播
    // mask 是线程掩码，表示哪些线程参与洗牌
    // var 是参与线程提供的值
    // srcLane 代表 lane id 范围为 [0, width-1] 内的某一个线程
    // width 是可选参数，用于控制 srcLane 的范围

    // 以这里为例，当前核函数共 16 个线程，所以参与表决的线程是 16 个。WIDTH 为 8 ，则将 16 个线程分为两段，分别洗牌。
    // 针对 [0, 7] 线程，srcLane 为 2, 即从线程 2 取 var 为 2(tid) ，返回给所有 [0, 7] 线程。
    // 针对 [8, 15] 线程，srcLane 为 2, 即从线程 10 取 var 为 10(tid) ，返回给所有 [8, 15] 线程。
    // 最终打印结果为： 2  2  2  2  2  2  2  2 10 10 10 10 10 10 10 10
    int value = __shfl_sync(FULL_MASK, tid, 2, WIDTH);
    if (tid == 0) {
        printf("__shfl_sync:        ");
    }
    printf("%2d ", value);
    if (tid == 0) {
        printf("\n");
    }

    // T __shfl_up_sync(unsigned mask, T var, unsigned delta, int width = warpSize);
    // __shfl_up_sync 的作用是让指定范围 [0, width-1] 的所有线程，都从比自己小 delta 的线程中获取 var 并返回，即小 ID 的线程将 var 上移
    // delta 用于计算 srcLane， srcLane = curLane - delta, curLane 代表 [0, width-1] 内的某一个线程
    // 如果 srcLane < 0, 即越过了所在子段的左边界，则返回值就是调用者自己的 var。
    
    // 以这里为例，当前核函数共 16 个线程，所以参与表决的线程是 16 个。WIDTH 为 8 ，则将 16 个线程分为两段，分别洗牌。
    // 针对 [0, 7] 线程，则依次从 [0, 0, ..., 6] 线程取 var (tid) ，返回给自己。
    // 针对 [8, 15] 线程，则依次从 [8, 8, ..., 14] 线程取 var (tid) ，返回给自己。
    // 最终打印结果是： 0  0  1  2  3  4  5  6  8  8  9 10 11 12 13 14 
    value = __shfl_up_sync(FULL_MASK, tid, 1, WIDTH);
    if (tid == 0) {
        printf("__shfl_up_sync:     ");
    }
    printf("%2d ", value);
    if (tid == 0) {
        printf("\n");
    }

    // T __shfl_down_sync(unsigned mask, T var, unsigned delta, int width = warpSize);
    // __shfl_down_sync 的作用是让指定范围 [0, width-1] 的所有线程，都从比自己大 delta 的线程中获取 var 并返回，即大 ID 的线程将 var 下移
    // delta 用于计算 srcLane， srcLane = curLane + delta, curLane 代表 [0, width-1] 内的某一个线程
    // 如果 srcLane > width-1, 即越过了所在子段的右边界，则返回值就是调用者自己的 var。

    // 以这里为例，当前核函数共 16 个线程，所以参与表决的线程是 16 个。WIDTH 为 8 ，则将 16 个线程分为两段，分别洗牌。
    // 针对 [0, 7] 线程，则依次从 [1, ..., 7, 7] 线程取 var (tid) ，返回给自己。
    // 针对 [8, 15] 线程，则依次从 [9, ..., 15, 15] 线程取 var (tid) ，返回给自己。
    // 最终打印结果是： 1  2  3  4  5  6  7  7  9 10 11 12 13 14 15 15
    value = __shfl_down_sync(FULL_MASK, tid, 1, WIDTH);
    if (tid == 0) {
        printf("__shfl_down_sync:   ");
    }
    printf("%2d ", value);
    if (tid == 0) {
        printf("\n");
    }

    // T __shfl_xor_sync(unsigned mask, T var, int laneMask, int width = warpSize);
    // __shfl_xor_sync 的作用是让指定范围 [0, width-1] 的所有线程，从 srcLane = curLane ^ laneMask 中获取 var 并返回，实际效果是临近的线程两两交互 var。

    // 以这里为例，当前核函数共 16 个线程，所以参与表决的线程是 16 个。WIDTH 为 8 ，则将 16 个线程分为两段，分别洗牌。
    // 针对 [0, 7] 线程，则依次从 [1  0  3  2  5  4  7  6] 线程取 var (tid) ，返回给自己。
    // 针对 [8, 15] 线程，则依次从 [9  8 11 10 13 12 15 14] 线程取 var (tid) ，返回给自己。
    // 最终打印结果是： 1  0  3  2  5  4  7  6  9  8 11 10 13 12 15 14
    value = __shfl_xor_sync(FULL_MASK, tid, 1, WIDTH);
    if (tid == 0) {
        printf("__shfl_xor_sync:    ");
    }
    printf("%2d ", value);
    if (tid == 0) {
        printf("\n");
    }
}

// 这里举例讲解 线程束表决函数（warp vote functions） 和 线程束洗牌函数（warp shuffle functions），
// 他们都以 _sync 结尾，而且都具备隐式的线程束同步功能，能自动处理数据竞争问题
int main(void) {
    test_warp_primitives<<<1, BLOCK_SIZE>>>();
    CHECK_CUDA_CALL(cudaDeviceSynchronize());

    return 0;
}