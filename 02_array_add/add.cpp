#include <cmath>
#include <cstdlib>
#include <cstdio>

const double EPSILON = 1e-15;
const double a = 1.23;
const double b = 2.34;
const double c = 3.57;

void add(const double *px, const double *py, double *pz, const int N) {
    for (int i = 0; i < N; ++i) {
        pz[i] = px[i] + py[i];
    }
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
    const int N = 1e7;
    const int M = sizeof(double) * N;

    double *px = (double *)malloc(M);
    double *py = (double *)malloc(M);
    double *pz = (double *)malloc(M);

    for (int i = 0; i < N; ++i) {
        px[i] = a;
        py[i] = b;
    }

    add(px, py, pz, N);
    check(pz, N);

    free(px);
    free(py);
    free(pz);
    return 0;
}
