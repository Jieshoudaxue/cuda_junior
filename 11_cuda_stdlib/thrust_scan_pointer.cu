#include <thrust/execution_policy.h>
#include <thrust/scan.h>
#include <cstdio>

int main(void) {
    int N = 10;
    int *x, *y;
    cudaMalloc((void**)&x, N * sizeof(int));
    cudaMalloc((void**)&y, N * sizeof(int));
    int *h_x = (int *)malloc(sizeof(int) * N);
    for (int i = 0; i < N; ++i) {
        h_x[i] = i + 1;
        printf("%d  ", h_x[i]);
    }
    printf("\n");
    
    cudaMemcpy(x, h_x, N * sizeof(int), cudaMemcpyHostToDevice);

    thrust::inclusive_scan(thrust::device, x, x+ N, y);

    int *h_y = (int *)malloc(sizeof(int) * N);
    cudaMemcpy(h_y, y, N * sizeof(int), cudaMemcpyDeviceToHost);
    for (int i = 0; i < N; ++i) {
        printf("%d  ", h_y[i]);
    }
    printf("\n");

    cudaFree(x);
    cudaFree(y);
    free(h_x);
    free(h_y);
    return 0;
}