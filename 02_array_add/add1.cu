#include <cmath>
#include <cstdlib>
#include <cstdio>

const double EPSILON = 1e-15;
const double a = 1.23;
const double b = 2.34;
const double c = 3.57;


__global__ void add(const double  *px, const double *py, double *pz) {
    // 这个公式得到是线程在整个 Grid 中的 x 维度的坐标，
    // 但由于本样例中只使用了一维网格和一维线程块，因此 x 维度的坐标就是线程在整个 Grid 中的一维全局索引。
    // 这个一维全局索引值与数组长度一致，因此可以直接用这个索引值访问数组元素，
    // 从而实现每个线程处理一个数组元素的计算，所有数组元素并发执行计算。
    const int tid = blockDim.x * blockIdx.x + threadIdx.x;
    // printf("tid = %d\n", tid);
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
    // 踩坑记录：这个值必须为 128 的整数倍，否则线程数会与数组长度不匹配，导致数组越界访问，程序失败。测试发现，1e7 可以，但 1e6 不可以。
    // 踩坑记录：对于 GeForce MX450，nvidia-smi 显示显存大小为 1878MiB，测试发现，如果设为 1e8，则超过显存大小，程序失败。
    const int N = 1e7;
    const int M = sizeof(double) * N;

    double *phx = (double *)malloc(M);
    double *phy = (double *)malloc(M);
    double *phz = (double *)malloc(M);

    for (int i = 0; i < N; ++i) {
        phx[i] = a;
        phy[i] = b;
    }

    double *pdx, *pdy, *pdz;
    cudaMalloc((void **)&pdx, M);
    cudaMalloc((void **)&pdy, M);
    cudaMalloc((void **)&pdz, M);
    cudaMemcpy(pdx, phx, M, cudaMemcpyHostToDevice);
    cudaMemcpy(pdy, phy, M, cudaMemcpyHostToDevice);

    const int block_size = 128;
    const int grid_size = N/block_size;
    printf("grid_size = %d, block_size = %d\n", grid_size, block_size);
    add<<<grid_size, block_size>>>(pdx, pdy, pdz);

    cudaMemcpy(phz, pdz, M, cudaMemcpyDeviceToHost);
    check(phz, N);

    free(phx);
    free(phy);
    free(phz);
    cudaFree(pdx);
    cudaFree(pdy);
    cudaFree(pdz);
    return 0;
}
