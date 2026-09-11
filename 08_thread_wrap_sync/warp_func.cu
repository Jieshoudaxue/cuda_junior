#include <cstdio>

#include "cuda_error.cuh"

const unsigned int WIDTH = 8;
const unsigned int BLOCK_SIZE = 16;
const unsigned int FULL_MASK = 0xffffffff;

__global__ void test_warp_primitives(void) {
    int tid = threadIdx.x;
    int lane_id = tid % WIDTH;

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