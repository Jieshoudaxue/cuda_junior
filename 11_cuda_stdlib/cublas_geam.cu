#include <cublas_v2.h>
#include <cstdio>

#include "cuda_error.cuh"

// R 即 rows, 行， C 即 columns, 列
// 以矩阵 A 为例，R = 2, C = 3
void print_matrix(int R, int C, double *A, const char *name) {
    printf("%s = \n", name);
    for (int r = 0; r < R; ++r) {
        for (int c = 0; c < C; ++c) {
            // 这里依次打印的是 A[0], A[2], A[4], A[1], A[3], A[5]，即按列主序打印矩阵 A 的元素
            printf("%10.0f", A[c * R + r]);
        }
        printf("\n");
    }
}

// 05_cuda_memory/memory6global.cu 里，我们实现了矩阵（方阵）转置的核函数，
// 本样例我们使用 cuBLAS 库的 cublasDgeam 函数来进行矩阵转置操作，cublasDgeam 是通用矩阵加法和乘法运算函数，
int main(void) {
    int M = 8, N = 6;
    int MN = M * N;
    double *h_A = (double *)malloc(sizeof(double) * MN);
    double *h_B = (double *)malloc(sizeof(double) * MN);
    for (int i = 0; i < MN; i++) {
        h_A[i] = i;
    }
    print_matrix(M, N, h_A, "A");

    double *g_A, *g_B;
    CHECK_CUDA_CALL(cudaMalloc((void **)&g_A, sizeof(double) * MN));
    CHECK_CUDA_CALL(cudaMalloc((void **)&g_B, sizeof(double) * MN));

    cublasSetVector(MN, sizeof(double), h_A, 1, g_A, 1);

    cublasHandle_t handle;
    cublasCreate(&handle);
    double alpha = 1.0;
    double beta = 0.0;

    // cublasDgeam 的 geam 是 General Matrix Addition and Multiplication 的缩写，表示通用矩阵加法，D 指 double 类型
    // cublasDgeam 的计算公式为 C = alpha * transa(A) + beta * transb(B)
    // 这里令 beta = 0，因此只保留 alpha * transa(A)，其中 transa(A) = A^T
    // 进行矩阵转置操作：B = A^T    
    // 输出矩阵 B 的尺寸是 N × M，内存中仍然是 MN 个元素
    cublasDgeam(handle, CUBLAS_OP_T, CUBLAS_OP_N, N, M, &alpha, g_A, M, &beta, g_A, M, g_B, N);

    cublasDestroy(handle);

    cublasGetVector(MN, sizeof(double), g_B, 1, h_B, 1);
    print_matrix(N, M, h_B, "B = A^T");

    free(h_A);
    free(h_B);
    CHECK_CUDA_CALL(cudaFree(g_A));
    CHECK_CUDA_CALL(cudaFree(g_B));
    return 0;
}