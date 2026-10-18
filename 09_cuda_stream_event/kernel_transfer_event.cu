#include <cmath>
#include <cstdio>

#include "cuda_error.cuh"

typedef float real;
// typedef double real;

const int N = 1 << 22;
const int M = sizeof(real) * N;
const int NUM_STREAMS = 8;

cudaStream_t kernel_streams[NUM_STREAMS];
cudaStream_t copy_streams[NUM_STREAMS];
cudaEvent_t kernel_done[NUM_STREAMS];
cudaEvent_t copy_done[NUM_STREAMS];

__global__ void add(const real *x, const real *y, real *z, int N) {
    const int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < N) {
        z[tid] = x[tid] + y[tid];
    }
}

void check(const real *h_z, const int N) {
    bool has_error = false;
    for (int i = 0; i < N; ++i) {
        if (fabs(h_z[i] - 3.57) > 1e-6) {
            has_error = true;
            printf("h_z[%d] = %g\n", i, h_z[i]);
            break;
        }
    }
    printf("%s\n", has_error ? "Has errors" : "No errors");
}

// 这个例子我们演示流同步第二种之跨 stream 依赖： 
// 使用 Cuda Event（cudaEventRecord + cudaStreamWaitEvent），可以让一个 Stream 等待另一个 Stream 中的某个操作完成，而不必阻塞整个主机线程。 
int main(void) {
    real *h_x, *h_y, *h_z;
    CHECK_CUDA_CALL(cudaMallocHost(&h_x, M));
    CHECK_CUDA_CALL(cudaMallocHost(&h_y, M));
    CHECK_CUDA_CALL(cudaMallocHost(&h_z, M));
    for (int i = 0; i < N; i ++) {
        h_x[i] = 1.23;
        h_y[i] = 2.34;
    }

    real *d_x, *d_y, *d_z;
    CHECK_CUDA_CALL(cudaMalloc(&d_x, M));
    CHECK_CUDA_CALL(cudaMalloc(&d_y, M));
    CHECK_CUDA_CALL(cudaMalloc(&d_z, M));

    for (int i = 0; i < NUM_STREAMS; i++) {
        CHECK_CUDA_CALL(cudaStreamCreate(&kernel_streams[i]));
        CHECK_CUDA_CALL(cudaStreamCreate(&copy_streams[i]));
        CHECK_CUDA_CALL(cudaEventCreate(&kernel_done[i]));
        CHECK_CUDA_CALL(cudaEventCreate(&copy_done[i]));
    }

    printf("Using %d streams:\n", NUM_STREAMS);

    // 这个例子，我们将数组分为 NUM_STREAMS 份，每份数据在 kernel_streams[i] 中执行 H2D 和 KER，然后在 copy_streams[i] 中执行 D2H 拷贝。
    // 为了保证 D2H 拷贝在 KER 执行完成后再执行，我们使用 cudaEventRecord 在 kernel_streams[i] 中记录 kernel 完成事件，
    // 然后使用 cudaStreamWaitEvent 在 copy_streams[i] 中等待 kernel_done[i] 事件完成后再执行 D2H 拷贝。
    // cudaError_t cudaEventRecord(cudaEvent_t event, cudaStream_t stream = 0);
    // cudaError_t cudaStreamWaitEvent(cudaStream_t stream, cudaEvent_t event, unsigned int flags = 0);
    // flags 为 0 代表 cudaEventWaitDefault ，表示标准的等待行为，即让指定流等待事件完成。
    int N1 = N / NUM_STREAMS;
    int M1 = M / NUM_STREAMS;
    for (int i = 0; i < NUM_STREAMS; i++) {
        int offset = i * N1;
        CHECK_CUDA_CALL(cudaMemcpyAsync(d_x + offset, h_x + offset, M1, cudaMemcpyHostToDevice, kernel_streams[i]));
        CHECK_CUDA_CALL(cudaMemcpyAsync(d_y + offset, h_y + offset, M1, cudaMemcpyHostToDevice, kernel_streams[i]));

        int block_size = 128;
        int grid_size = (N1 - 1) / block_size + 1;
        add<<<grid_size, block_size, 0, kernel_streams[i]>>>(d_x + offset, d_y + offset, d_z + offset, N1);

        // 记录 kernel 完成事件
        CHECK_CUDA_CALL(cudaEventRecord(kernel_done[i], kernel_streams[i]));

        CHECK_CUDA_CALL(cudaStreamWaitEvent(copy_streams[i], kernel_done[i], 0));
        CHECK_CUDA_CALL(cudaMemcpyAsync(h_z + offset, d_z + offset, M1, cudaMemcpyDeviceToHost, copy_streams[i]));
        CHECK_CUDA_CALL(cudaEventRecord(copy_done[i], copy_streams[i]));
    }

    // 由于主机最后要进行结果检查，所以需要等待所有 D2H 拷贝完成，因此我们在主机线程中使用 cudaEventSynchronize 等待所有 copy_done[i] 事件完成。
    // cudaError_t cudaEventSynchronize(cudaEvent_t event);
    // 提示：这里使用的是 cudaEventSynchronize，而不是 cudaStreamSynchronize，前者粒度更细，用的更多
    for (int i = 0; i < NUM_STREAMS; ++i) {
        CHECK_CUDA_CALL(cudaEventSynchronize(copy_done[i]));
    }
    check(h_z, N);

    for (int i = 0; i < NUM_STREAMS; i ++) {
        CHECK_CUDA_CALL(cudaStreamDestroy(kernel_streams[i]));
        CHECK_CUDA_CALL(cudaStreamDestroy(copy_streams[i]));
        CHECK_CUDA_CALL(cudaEventDestroy(kernel_done[i]));
        CHECK_CUDA_CALL(cudaEventDestroy(copy_done[i]));
    }

    CHECK_CUDA_CALL(cudaFreeHost(h_x));
    CHECK_CUDA_CALL(cudaFreeHost(h_y));
    CHECK_CUDA_CALL(cudaFreeHost(h_z));
    CHECK_CUDA_CALL(cudaFree(d_x));
    CHECK_CUDA_CALL(cudaFree(d_y));
    CHECK_CUDA_CALL(cudaFree(d_z));
    return 0;
}