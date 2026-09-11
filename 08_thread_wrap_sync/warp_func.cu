#include <cstdio>

#include "cuda_error.cuh"

const unsigned int WIDTH = 8;
const unsigned int BLOCK_SIZE = 16;
const unsigned int FULL_MASK = 0xffffffff;

__global__ void test_warp_primitives(void) {
    // grid_size 是 1 ，block_size 是 16,
    // 因此，这里的 threadIdx.x 的范围是从 0 到 15
    int tid = threadIdx.x;
    // 由于 block_size 是 16, WIDTH 是 8，
    // 因此，这里的 lane_id 的范围是 0 到 7
    int lane_id = tid % WIDTH;

    // 关注这里的打印技巧，虽然核函数是多线程并发执行的，但是printf 的打印输出内容是整齐排列的，很像串行打印的效果：
    // threadIdx.x:     0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 
    // TODO
    if (tid == 0) {
        printf("threadIdx.x:    ");
    }
    printf("%2d ", tid);
    if (tid == 0) {
        printf("\n");
    }

    if (tid == 0) {
        printf("lane_id:        ");
    }
    printf("%2d ", lane_id);
    if (tid == 0) {
        printf("\n");
    }

    unsigned int mask1 = __ballot_sync(FULL_MASK, tid > 0);
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

    int result = __all_sync(FULL_MASK, tid);
    if (tid == 0) {
        printf("__all_sync(FULL_MASK): %d\n", result);
    }
    result = __all_sync(mask1, tid);
    if (tid == 1) {
        printf("__all_sync    (mask1): %d\n", result);
    }

    result = __any_sync(FULL_MASK, tid);
    if (tid == 0) {
        printf("__any_sync(FULL_MASK): %d\n", result);
    }
    result = __any_sync(mask2, tid);
    if (tid == 0) {
        printf("__any_sync    (mask2): %d\n", result);
    }

    int value = __shfl_sync(FULL_MASK, tid, 2, WIDTH);
    if (tid == 0) {
        printf("__shfl_sync:        ");
    }
    printf("%2d ", value);
    if (tid == 0) {
        printf("\n");
    }

    value = __shfl_up_sync(FULL_MASK, tid, 1, WIDTH);
    if (tid == 0) {
        printf("__shfl_up_sync:     ");
    }
    printf("%2d ", value);
    if (tid == 0) {
        printf("\n");
    }

    value = __shfl_down_sync(FULL_MASK, tid, 1, WIDTH);
    if (tid == 0) {
        printf("__shfl_down_sync:   ");
    }
    printf("%2d ", value);
    if (tid == 0) {
        printf("\n");
    }

    value = __shfl_xor_sync(FULL_MASK, tid, 1, WIDTH);
    if (tid == 0) {
        printf("__shfl_xor_sync:    ");
    }
    printf("%2d ", value);
    if (tid == 0) {
        printf("\n");
    }
}

int main(void) {
    test_warp_primitives<<<1, BLOCK_SIZE>>>();
    CHECK_CUDA_CALL(cudaDeviceSynchronize());

    return 0;
}