#include "cstdio"

#include "cuda_error.cuh"

typedef float real;

const int NUM_REPEATS = 10;
const int TILE_DIM = 32;

__global__ void copy(const real *A, real *B, const int N) {
    // 核函数是可以直接访问普通全局变量的
    const int ix = blockIdx.x * TILE_DIM + threadIdx.x;
    const int iy = blockIdx.y * TILE_DIM + threadIdx.y;
    if (ix < N && iy < N) {
        B[iy * N + ix] = A[iy * N + ix];
    }
}

__global__ void transpose1(const real *A, real *B, const int N) {
    const int ix = blockIdx.x * blockDim.x + threadIdx.x;
    const int iy = blockIdx.y * blockDim.y + threadIdx.y;
    if (ix < N && iy < N) {
        B[ix * N + iy] = A[iy * N + ix];
    }
}

__global__ void transpose2(const real *A, real *B, const int N) {
    const int ix = blockIdx.x * blockDim.x + threadIdx.x;
    const int iy = blockIdx.y * blockDim.y + threadIdx.y;
    if (ix < N && iy < N) {
        B[iy * N + ix] = A[ix * N + iy];
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
                copy<<<grid_size, block_size>>>(d_A, d_B, N);
                break;
            case 1:
                transpose1<<<grid_size, block_size>>>(d_A, d_B, N);
                break;
            case 2:
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

// 对全局内存的访问将触发内存事务（memory transaction），也就是数据传输（data transfer），一次数据传输处理的数据量在默认情况下是 32 字节。
// 当 CUDA 线程从全局内存请求数据时，相关的线程束会根据每个线程访问的数据大小以及内存地址在线程间的分布情况，将该线程束中所有线程的内存请求合并成若干次数据传输，即若干次 32 字节传输。
// 合并度（degree of coalescing），或者叫内存利用率，它等于线程束请求的字节数除以由该请求导致的所有数据传输处理的字节数。内存利用率越高，核函数中与全局内存访问有关的部分的性能就更好；利用率低则意味着对显存带宽的浪费。
// 好的情况：如果线程束中的连续线程请求内存中连续的 4 字节，那么该线程束将总共请求 128 字节的内存，并且这所需的 128 字节将通过四次 32 字节的内存事务获取。这实现了 100% 的内存利用率。也就是说，100% 的内存流量都被该线程束利用了。
// 坏的情况：连续的线程访问内存中彼此相距 32 字节或更远的数据元素。在这种情况下，线程束将被迫为每个线程发起一次 32 字节的内存事务，内存传输的总字节数将是 32 字节 * 32 线程/线程束 = 1024 字节。然而，实际使用的内存量仅为 128 字节（线程束中每个线程 4 字节），因此内存利用率仅为 128 / 1024 = 12.5%。这是对内存系统非常低效的使用。
// 数据传输对数据地址的要求：在一次数据传输中，从全局内存转移到 L2 缓存的一片内存的首地址一定是一个最小粒度（这里是 32 字节）的整数倍。
// 而使用 CUDA 运行时 API 函数（如我们常用的 cudaMalloc）分配的内存的首地址，至少是 256 字节的整数倍，因此满足数据传输对地址的要求。`
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
    
    // 这里的矩阵拷贝，读和写都实现了 100% 的合并度，速度最快，时间最短。在我的机器上，单次拷贝平均耗时：0.0068736 ms
    printf("copy: \n");
    timing(d_A, d_B, N, 0);

    // 这里的矩阵转置，读的时候实现了 100% 的合并度，而写的时候没有实现合并。在我的机器上，单次拷贝平均耗时：0.0213312 ms
    printf("\ntranspose with coalesced read: \n");
    timing(d_A, d_B, N, 1);
    
    // 这里的矩阵转置，读的时候没有实现合并，而写的时候实现了 100% 的合并度，在我的机器上，单次拷贝平均耗时：0.010992 ms
    // 这里的耗时比上面的转置更短，是因为读虽然没有合并，但全局内存的读默认情况下是有缓存机制的，而写是没有的，请读者注意。
    // 因此，在不能同时满足读取和写入都是合并的情况下，一般来说应当尽量做到合并的写入。
    printf("\ntranspose with coalesced write: \n");
    timing(d_A, d_B, N, 2);

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