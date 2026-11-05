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
            printf("%10.6f", A[c * R + r]);
        }
        printf("\n");
    }
}

// 本样例演示了如何使用 cuBLAS 库的 cublasDgemm 函数来进行矩阵乘法运算
/*
    A = 
    0.000000  2.000000  4.000000
    1.000000  3.000000  5.000000
    B = 
    0.000000  3.000000
    1.000000  4.000000
    2.000000  5.000000
    C = A * B = 
    10.000000 28.000000
    13.000000 40.000000
*/
int main(void) {
    // 矩阵 A 是 m(2) × k(3) 的矩阵，按数学和编程中的通用约定，m(2) 是行，k(3) 是列，
    // 矩阵 B 是 k(3) × n(2) 的矩阵，按数学和编程中的通用约定，k(3) 是行，n(2) 是列，
    // 矩阵 C 是 A * B 的结果，两个矩阵相乘的前提是 A 的列必须等于 B 的行，乘积的结果是，行取 A 的行，列取 B 的列，因此 C 是 m(2) × n(2) 的矩阵。
    int M = 2, K = 3, N = 2;
    int MK = M * K, KN = K * N, MN = M * N;
    double *h_A = (double *)malloc(sizeof(double) * MK);
    double *h_B = (double *)malloc(sizeof(double) * KN);
    double *h_C = (double *)malloc(sizeof(double) * MN);
    // cuBLAS 库使用的是列主序（column-major order）存储矩阵数据，而 C/C++ 默认使用的是行主序（row-major order）存储矩阵数据。
    // 例如这里的矩阵 A 是 2 × 3 的矩阵，是按列主序存储的顺序： 0, 1, 2, 3, 4, 5，而不是按行主序存储的顺序：0, 2, 4, 1, 3, 5 。
    for (int i = 0; i < MK; i++) {
        h_A[i] = i;
    }
    print_matrix(M, K, h_A, "A");
    for (int i = 0; i < KN; i++) {
        h_B[i] = i;
    }
    print_matrix(K, N, h_B, "B");

    double *g_A, *g_B, *g_C;
    CHECK_CUDA_CALL(cudaMalloc((void **)&g_A, sizeof(double) * MK));
    CHECK_CUDA_CALL(cudaMalloc((void **)&g_B, sizeof(double) * KN));
    CHECK_CUDA_CALL(cudaMalloc((void **)&g_C, sizeof(double) * MN));

    // 调用 cublasSetVector 将矩阵 A, B, C 从主机内存（host memory）复制到设备内存（device memory）
    // cublasStatus_t cublasSetVector(int n, int elemSize, const void *x, int incx, void *y, int incy)
    // n: 要复制的元素个数
    // elemSize: 每个元素的字节大小
    // x: 指向主机内存中源数据的指针
    // incx: 源数据中元素之间的间隔（步长），通常为 1，表示连续存储
    // y: 指向设备内存中目标数据的指针
    // incy: 目标数据中元素之间的间隔（步长），通常为 1，表示连续存储
    cublasSetVector(MK, sizeof(double), h_A, 1, g_A, 1);
    cublasSetVector(KN, sizeof(double), h_B, 1, g_B, 1);

    // 使用 cuBLAS 三步骤，
    // 1. 调用 cublasCreate 创建 cuBLAS 库的上下文句柄，数据类型：cublasHandle_t
    // 2. 调用 cuBLAS 库的函数进行矩阵运算
    // 3. 调用 cublasDestroy 销毁 cuBLAS 库的上下文句柄，释放资源
    cublasHandle_t handle;
    cublasCreate(&handle);
    double alpha = 1.0;
    double beta = 0.0;
    // gemm 指 GEneral Matrix-Matrix multiplication，即“一般的矩阵-矩阵相乘”
    // cublasStatus_t cublasDgemm(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, 
    //                              const double *alpha, const double *A, int lda, 
    //                              const double *B, int ldb, const double *beta, 
    //                              double *C, int ldc)
    // 该函数实施如下计算： C = alpha * transa(A) * transb(B) + beta * C
    // handle: cuBLAS 库的上下文句柄
    // transa: 矩阵 A 是否转置，CUBLAS_OP_N 表示不转置，CUBLAS_OP_T 表示转置，CUBLAS_OP_C 表示共轭转置
    // transb: 矩阵 B 是否转置，CUBLAS_OP_N 表示不转置，CUBLAS_OP_T 表示转置，CUBLAS_OP_C 表示共轭转置
    // m: 矩阵 A 的行数
    // n: 矩阵 B 的列数
    // k: 矩阵 A 的列数（也是矩阵 B 的行数）
    // alpha: 标量 alpha 的指针，表示矩阵乘法的系数
    // A: 指向矩阵 A 的指针
    // lda: 矩阵 A 的 leading dimension，即矩阵 A 在内存中每列的元素个数，通常为矩阵 A 的行数
    // B: 指向矩阵 B 的指针
    // ldb: 矩阵 B 的 leading dimension，即矩阵 B 在内存中每列的元素个数，通常为矩阵 B 的行数
    // beta: 标量 beta 的指针，表示矩阵 C 的系数
    // C: 指向矩阵 C 的指针
    // ldc: 矩阵 C 的 leading dimension，即矩阵 C 在内存中每列的元素个数，通常为矩阵 C 的行数

    // 以这里为例，该函数实施如下计算： C = 1.0 * A * B + 0.0 * C，即 C = A * B，
    // 即 M × K 的矩阵 A 与 K × N 的矩阵 B 相乘，得到 M × N 的矩阵 C
    cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, M, N, K, &alpha, g_A, M, g_B, K, &beta, g_C, M);
    cublasDestroy(handle);

    // 调用 cublasGetVector 将矩阵 C 从设备内存（device memory）复制到主机内存（host memory）
    // cublasStatus_t cublasGetVector(int n, int elemSize, const void *x, int incx, void *y, int incy)
    // n: 要复制的元素个数
    // elemSize: 每个元素的字节大小
    // x: 指向设备内存中源数据的指针
    // incx: 源数据中元素之间的间隔（步长），通常为 1，表示连续存储
    // y: 指向主机内存中目标数据的指针
    // incy: 目标数据中元素之间的间隔（步长），通常为 1，表示连续存储
    cublasGetVector(MN, sizeof(double), g_C, 1, h_C, 1);
    print_matrix(M, N, h_C, "C = A * B");

    free(h_A);
    free(h_B);
    free(h_C);
    CHECK_CUDA_CALL(cudaFree(g_A));
    CHECK_CUDA_CALL(cudaFree(g_B));
    CHECK_CUDA_CALL(cudaFree(g_C));
    return 0;
}