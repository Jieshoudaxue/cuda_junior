#include <cmath>
#include <cstdlib>
#include <cstdio>

const double EPSILON = 1e-15;
const double a = 1.23;
const double b = 2.34;
const double c = 3.57;

__global__ void add(const double  *px, const double *py, double *pz, const int N) {
    const int tid = blockDim.x * blockIdx.x + threadIdx.x;
    // printf("tid = %d\n", tid);
    if (tid >= N) {
        return;
    }
    pz[tid] = px[tid] + py[tid];
}

void check(const double *pz, const int N) {
    bool has_error = false;
    for (int i = 0; i < N; ++i) {
        if (fabs(pz[i] - c) > EPSILON) {
            has_error = true;
        }
    }
    printf("%s\n", has_error ? "Has errors" : "No errors");
}

// 一个 CUDA 流指的是由主机发出的在一个设备中执行的 CUDA 操作序列，如主机－设备数据传输和核函数执行。
// 任何 CUDA 操作都存在于某个 CUDA 流中，要么是默认流（default stream），也称为空流（null stream），要么是明确指定的非空流，非空流由主机端负责创建和销毁。
// 一个 CUDA 流中各个操作的次序是由主机控制的，按照主机发布的次序执行，而且同一个 CUDA 流中的所有 CUDA 操作都是顺序执行的。然而，来自于两个不同 CUDA 流中的操作不一定按照某个次序执行，而有可能并发或交错地执行。

// 在这个样例里，如下四句将在默认的空流中按代码出现的顺序依次执行，
// 其中数据传输是同步的，也叫阻塞的，即主机调用 cudaMemcpy 后，会等待命令执行完毕再往前走。
// 而核函数的启动是异步的，也叫非阻塞的，即主机调用 add 后，不会等待核函数执行完毕，可以立即做别的事情。
// 但在当前样例中，第四句 cudaMemcpy 不会被立即执行，因为他属于默认的空流，必须等待前一个 CUDA 操作（add 执行）执行完毕才会执行。假如 add 后一句是某种主机计算任务，而不是 CUDA 操作，此时主机就会和设备同时并行计算。
// cudaMemcpy(d_px, h_px, M, cudaMemcpyHostToDevice);
// cudaMemcpy(d_py, h_py, M, cudaMemcpyHostToDevice);
// add<<<grid_size, block_size>>>(d_px, d_py, d_pz, N);
// cudaMemcpy(h_pz, d_pz, M, cudaMemcpyDeviceToHost);
int main(void) {
    const int N = 1e7;
    const int M = sizeof(double) * N;

    double *h_px = (double *)malloc(M);
    double *h_py = (double *)malloc(M);
    double *h_pz = (double *)malloc(M);

    for (int i = 0; i < N; ++i) {
        h_px[i] = a;
        h_py[i] = b;
    }

    double *d_px, *d_py, *d_pz;
    cudaMalloc((void **)&d_px, M);
    cudaMalloc((void **)&d_py, M);
    cudaMalloc((void **)&d_pz, M);
    cudaMemcpy(d_px, h_px, M, cudaMemcpyHostToDevice);
    cudaMemcpy(d_py, h_py, M, cudaMemcpyHostToDevice);

    const int block_size = 128;
    const int grid_size = (N % block_size == 0) ? (N / block_size) : (N / block_size + 1);
    printf("grid_size = %d, block_size = %d\n", grid_size, block_size);
    add<<<grid_size, block_size>>>(d_px, d_py, d_pz, N);

    cudaMemcpy(h_pz, d_pz, M, cudaMemcpyDeviceToHost);
    check(h_pz, N);

    free(h_px);
    free(h_py);
    free(h_pz);
    cudaFree(d_px);
    cudaFree(d_py);
    cudaFree(d_pz);
    return 0;
}
