#include <cmath>
#include <cstdlib>
#include <cstdio>

const double EPSILON = 1e-15;
const double a = 1.23;
const double b = 2.34;
const double c = 3.57;

// CUDA 设备函数由 __device__ 修饰符修饰，表示这是一个在 GPU 上运行的函数，
// CUDA 设备函数由核函数或其他设备函数调用，不能由 CPU 端调用
// CUDA 设备函数可以有返回值
// CUDA 设备函数可以同时由 __host__ 和 __device__ 修饰符修饰，表示这个函数既可以在 CPU 上运行，也可以在 GPU 上运行，减少冗余代码
// CUDA 设备函数不能同时由 __global__ 和 __device__ 修饰符修饰
// CUDA 设备函数可以使用 __noinline__ 修饰符修饰，表示这个函数不进行内联展开（编译器不一定接受），减少编译器优化带来的问题
// CUDA 设备函数可以使用 __forceinline__ 修饰符修饰，表示这个函数强制进行内联展开，减少函数调用开销
__device__ double add1_device(const double x, const double y) {
    return x + y;
}

__global__ void add1(const double  *px, const double *py, double *pz, const int N) {
    const int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid >= N) {
        return;
    }
    pz[tid] = add1_device(px[tid], py[tid]);
}

__device__ void add2_device(const double x, const double y, double &rz) {
    rz = x + y;
}

__global__ void add2(const double  *px, const double *py, double *pz, const int N) {
    const int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid >= N) {
        return;
    }
    add2_device(px[tid], py[tid], pz[tid]);
}

// CUDA 核函数的返回值必须是 void, 不能有返回值，但核函数里面可以使用 return 语句提前结束函数的执行，return 后面不能跟任何值。
// CUDA 核函数必须使用 __global__ 修饰符修饰，表示这是一个在 GPU 上运行的函数，
// CUDA 核函数不能同时由 __host__ 和 __global__ 修饰符修饰
// CUDA 核函数支持 C++ 的重载（overload）
// CUDA 核函数的参数个数必须确定，不支持可变参数
// CUDA 核函数支持传递非指针变量，内容对每个线程可见
// CUDA 核函数不能是类的成员函数。
// CUDA 核函数由 CPU 端调用，由 GPU 端执行，CC 3.5 之前，核函数不能调用核函数，
// CUDA 核函数被调用时，必须使用 <<<...>>> 语法指定 grid 和 block 的尺寸
__global__ void add(const double  *px, const double *py, double *pz, const int N) {
    // 这个公式得到是线程在整个 Grid 中的 x 维度的坐标，
    // 但由于本样例中只使用了一维网格和一维线程块，因此 x 维度的坐标就是线程在整个 Grid 中的一维全局索引。
    // 这个一维全局索引值与数组长度一致，因此可以直接用这个索引值访问数组元素，
    // 从而实现每个线程处理一个数组元素的计算，所有数组元素并发执行计算。
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

int main(void) {
    // 踩坑记录：对于 GeForce MX450，nvidia-smi 显示显存大小为 1878MiB，测试发现，如果设为 1e8，则超过显存大小，程序失败，1e7 可以。
    const int N = 1e7;
    const int M = sizeof(double) * N;

    double *h_px = (double *)malloc(M);
    double *h_py = (double *)malloc(M);
    double *h_pz = (double *)malloc(M);

    for (int i = 0; i < N; ++i) {
        h_px[i] = a;
        h_py[i] = b;
    }

    // CUDA 编程的传统是将 CPU 端的变量命名为 h_ 开头，表示 host 端的变量；
    // 将 GPU 端的变量命名为 d_ 开头，表示 device 端的变量。
    double *d_px, *d_py, *d_pz;
    // 所有的 CUDA 运行时 API 都是以 cuda 开头
    // cudaMalloc() 函数在 GPU 显存上分配内存，API 接口：
    // cudaError_t cudaMalloc(void **address, size_t size);
    // address 是待分配内存的指针，是一个二重指针，size 是待分配内存的大小，单位是字节。
    // cudaError_t 是一个枚举类型，表示 CUDA 运行时 API 的返回值，cudaSuccess 表示成功，其他值表示失败。
    cudaMalloc((void **)&d_px, M);
    cudaMalloc((void **)&d_py, M);
    cudaMalloc((void **)&d_pz, M);
    // cudaMemcpy() 函数的作用是拷贝数据，API 接口：
    // cudaError_t cudaMemcpy(void *dst, const void *src, size_t count, enum cudaMemcpyKind kind);
    // dst 是目标地址，src 是源地址，count 是拷贝的字节数，
    // kind 是拷贝的方向，cudaMemcpyHostToDevice 表示从 CPU 端拷贝到 GPU 端，cudaMemcpyDeviceToHost 表示从 GPU 端拷贝到 CPU 端，cudaMemcpyDeviceToDevice 表示在 GPU 端拷贝。
    // cudaMemcpyDefault 表示由 CUDA 运行时 API 自动选择拷贝的方向，通常情况下，CUDA 运行时 API 会根据 dst 和 src 的地址判断拷贝的方向。
    // cudaMemcpyHostToHost 表示从 CPU 端拷贝到 CPU 端。
    cudaMemcpy(d_px, h_px, M, cudaMemcpyHostToDevice);
    cudaMemcpy(d_py, h_py, M, cudaMemcpyHostToDevice);

    const int block_size = 128;
    const int grid_size = (N % block_size == 0) ? (N / block_size) : (N / block_size + 1);
    printf("grid_size = %d, block_size = %d\n", grid_size, block_size);
    add<<<grid_size, block_size>>>(d_px, d_py, d_pz, N);
    // add1<<<grid_size, block_size>>>(d_px, d_py, d_pz, N);
    // add2<<<grid_size, block_size>>>(d_px, d_py, d_pz, N);

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
