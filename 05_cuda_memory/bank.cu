#include "cstdio"

#include "cuda_error.cuh"

typedef float real;

const int NUM_REPEATS = 10;
const int TILE_DIM = 32;

__global__ void transpose1(const real *A, real *B, const int N) {
    __shared__ real S[TILE_DIM][TILE_DIM];

    int bx = blockIdx.x * TILE_DIM;
    int by = blockIdx.y * TILE_DIM;

    int nx1 = bx + threadIdx.x;
    int ny1 = by + threadIdx.y;

    if (nx1 < N && ny1 < N) {
        S[threadIdx.y][threadIdx.x] = A[ny1 * N + nx1];
    }
    __syncthreads();

    int nx2 = bx + threadIdx.y;
    int ny2 = by + threadIdx.x;
    if ( nx2 < N && ny2 < N) {
        B[nx2 * N + ny2] = S[threadIdx.x][threadIdx.y];
    }
}

__global__ void transpose2(const real *A, real *B, const int N) {
    __shared__ real S[TILE_DIM][TILE_DIM];
    int bx = blockIdx.x * TILE_DIM;
    int by = blockIdx.y * TILE_DIM;

    int nx1 = bx + threadIdx.x;
    int ny1 = by + threadIdx.y;
    if (nx1 < N && ny1 < N) {
        S[threadIdx.y][threadIdx.x] = A[ny1 * N + nx1];
    }
    __syncthreads();

    int nx2 = bx + threadIdx.y;
    int ny2 = by + threadIdx.x;
    if (nx2 < N && ny2 < N) {
        B[nx2 * N + ny2] = S[threadIdx.x][threadIdx.y];
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