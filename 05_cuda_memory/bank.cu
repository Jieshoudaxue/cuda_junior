#include "cstdio"

#include "cuda_error.cuh"

typedef float real;

const int NUM_REPEATS = 10;
const int TILE_DIM = 32;

// 利用共享内存实现对全局内存的读写都是合并的，在我的机器上，平均运行时间是 0.0150368 ms。
// 优于 memory6gobal.cu 的 transpose1 的 0.0213312 ms，但比 memory6gobal.cu 的 transpose2 的 0.010992 ms 要慢，说明还有优化空间
__global__ void transpose1(const real *A, real *B, const int N) {
    // 一片（tile）是 32×32，每个线程块处理一片，线程块的大小也是 32×32，
    // 这里为每个线程块申请同样 32×32 大小的共享内存
    __shared__ real S[TILE_DIM][TILE_DIM];

    int ix = blockIdx.x * TILE_DIM + threadIdx.x;
    int iy = blockIdx.y * TILE_DIM + threadIdx.y;

    // 这里是将每个线程块负责的一片子矩阵数据，从全局数据总拷贝到当前线程块的共享内存中，
    // 此时对全局内存 A 的读是合并的。
    if (ix < N && iy < N) {
        S[threadIdx.y][threadIdx.x] = A[iy * N + ix];
    }
    // 调用 __syncthreads ，确保拷贝过程不被干扰
    // 一般来说，在利用共享内存中的数据之前，都要进行线程块内的同步操作，以确保共享内存数组中的所有元素都已经更新完毕。
    __syncthreads();

    // 借助共享内存，这里对全局内存 B 的写，也可以使用合并的方式
    if ( ix < N && iy < N) {
        B[iy * N + ix] = S[threadIdx.x][threadIdx.y];
    }
}

// 共享内存不是铁板一块，而是分为 32 个同等宽度（4/8 字节）的能被同时访问的内存 bank，
// bank 的标号从 0 ～ 31, 而每个 bank 有很多层，每一层有 4/8 个字节，绝大多数是 4 字节，
// 一个 32×32的共享内存矩阵，x 方向就是 bank 的标号，y 方向就是 bank 的层级。
// 一个线程束的 32 个线程，如果同时访问不同的 32 个 bank，则只触发一次内存事务（memory transaction），即数据传输（data transfer），效率最高。
// 如果一个线程束的 32 个线程，同时访问同一个 bank 的不同层的数据，访问多少层，就会出发多少次内存事务，即出现 bank 冲突，效率较低，应当避免。

// 这个例子利用共享内存实现对全局内存的读写都是合并的，且避免了 bank 冲突，在我的机器上，平均运行时间是 0.0088352 ms，性能最好。
__global__ void transpose2(const real *A, real *B, const int N) {
    // transpose1 中，共享内存矩阵 S 为 32×32，
    // 写的时候，线程束的每个线程访问不同的 bank，没有 bank 冲突，
    // 但读的时候，线程束的每个线程访问同一个 bank 的不同层数据，导致严重的 bank 冲突，使得加速效果不明显
    // 这里将共享内存矩阵 S 改为 32×33，于是矩阵的每一行的第一个元素在 bank 中的位置是错开的，从而避免了共享内存的读取时的冲突
    __shared__ real S[TILE_DIM][TILE_DIM+1];

    int ix = blockIdx.x * TILE_DIM + threadIdx.x;
    int iy = blockIdx.y * TILE_DIM + threadIdx.y;
    if (ix < N && iy < N) {
        S[threadIdx.y][threadIdx.x] = A[iy * N + ix];
    }
    __syncthreads();

    if (ix < N && iy < N) {
        B[iy * N + ix] = S[threadIdx.x][threadIdx.y];
    }
}

void timing(const real *d_A, real *d_B, const int N, const int task) {
    const int grid_size_x = (N + TILE_DIM - 1) / TILE_DIM;
    const int grid_size_y = grid_size_x;
    const dim3 block_size(TILE_DIM, TILE_DIM);
    const dim3 grid_size(grid_size_x, grid_size_y);

    float t_sum = 0;
    float t2_sum = 0;
    for (int repeat = 0; repeat <= NUM_REPEATS; ++repeat) {
        cudaEvent_t start, stop;
        CHECK_CUDA_CALL(cudaEventCreate(&start));
        CHECK_CUDA_CALL(cudaEventCreate(&stop));
        CHECK_CUDA_CALL(cudaEventRecord(start));
        cudaEventQuery(start);
        
        switch(task) {
            case 0:
                transpose1<<<grid_size, block_size>>>(d_A, d_B, N);
                break;
            case 1:
                transpose2<<<grid_size, block_size>>>(d_A, d_B, N);
                break;
            default:
                printf("Error: wrong task\n");
                exit(1);
                break;
        }

        CHECK_CUDA_CALL(cudaEventRecord(stop));
        CHECK_CUDA_CALL(cudaEventSynchronize(stop));
        float elapsed_time;
        CHECK_CUDA_CALL(cudaEventElapsedTime(&elapsed_time, start, stop));
        printf("Time = %g ms\n", elapsed_time);

        if (repeat > 0) {
            t_sum += elapsed_time;
            t2_sum += elapsed_time * elapsed_time;
        }
        
        CHECK_CUDA_CALL(cudaEventDestroy(start));
        CHECK_CUDA_CALL(cudaEventDestroy(stop));
    }

    const float t_ave = t_sum / NUM_REPEATS;
    const float t_err = sqrt(t2_sum/NUM_REPEATS - t_ave * t_ave);
    printf("Time = %g +- %g ms\n", t_ave, t_err);
}

void print_matrix(const int N, const real *A) {
    for (int iy = 0; iy < N; iy ++) {
        for (int ix = 0; ix < N; ix ++) {
            printf("%g\t", A[iy * N + ix]);
        }
        printf("\n");
    }
}

// 利用共享内存改善全局内存的访问模式，使得对全局内存的读和写都是合并的
int main(void) {
    const int N = 128;
    const int N2 = N * N;
    const int M = sizeof(real) * N2;

    real *h_A = (real *)malloc(M);
    real *h_B = (real *)malloc(M);
    for (int i = 0; i < N2; ++i) {
        h_A[i] = i;
    }

    real *d_A, *d_B;
    CHECK_CUDA_CALL(cudaMalloc(&d_A, M));
    CHECK_CUDA_CALL(cudaMalloc(&d_B, M));
    CHECK_CUDA_CALL(cudaMemcpy(d_A, h_A, M, cudaMemcpyHostToDevice));

    printf("\ntranspose with shared memory bank conflict: \n");
    timing(d_A, d_B, N, 0);

    printf("\ntranspose without shared memory bank conflict: \n");
    timing(d_A, d_B, N, 1);

    CHECK_CUDA_CALL(cudaMemcpy(h_B, d_B, M, cudaMemcpyDeviceToHost));
    if (N <= 8) {
        printf("\nA = \n");
        print_matrix(N, h_A);
        printf("\nB = \n");
        print_matrix(N, h_B);
    }

    free(h_A);
    free(h_B);
    CHECK_CUDA_CALL(cudaFree(d_A));
    CHECK_CUDA_CALL(cudaFree(d_B));
    
    return 0;
}