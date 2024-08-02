#include <iostream>
#include <vector>
#include <cblas.h>
#include <chrono>

void matrixMultiply(const double* A, const double* B, double* C, int M) {
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                M, M, M,
                1.0, A, M, B, M,
                0.0, C, M);
}

int main(int argc, char** argv) {
    int M = 2048;
    if (argc >= 2) {
        M = atoi(argv[1]);
    }

    // Allocate aligned memory for better performance
    double* A = (double*)aligned_alloc(64, M * M * sizeof(double));
    double* B = (double*)aligned_alloc(64, M * M * sizeof(double));
    double* C = (double*)aligned_alloc(64, M * M * sizeof(double));

    // Initialize matrices
    std::fill_n(A, M * M, 1.0);
    std::fill_n(B, M * M, 1.0);

    // Warm-up run
    matrixMultiply(A, B, C, M);

    // Timed run
    auto start = std::chrono::high_resolution_clock::now();

    matrixMultiply(A, B, C, M);

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    std::cout << duration.count() << std::endl;

    // Clean up
    free(A);
    free(B);
    free(C);
    
    return 0;
}