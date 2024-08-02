#include <iostream>
#include <vector>
#include <chrono>
#include <cstring>
#include <arm_neon.h>
#include <omp.h>

constexpr int BLOCK_SIZE = 64;

void MultiplyBlock(const float* A, const float* B, float* C, int M, int i, int j, int k) {
    for (int ii = i; ii < std::min(i + BLOCK_SIZE, M); ++ii) {
        for (int jj = j; jj < std::min(j + BLOCK_SIZE, M); jj += 4) {
            float32x4_t sum = vld1q_f32(&C[ii * M + jj]); // Load existing values
            for (int kk = k; kk < std::min(k + BLOCK_SIZE, M); ++kk) {
                float32x4_t a = vdupq_n_f32(A[ii * M + kk]);
                float32x4_t b = vld1q_f32(&B[kk * M + jj]);
                sum = vaddq_f32(sum, vmulq_f32(a, b)); // Use vaddq_f32 for accumulation
            }
            vst1q_f32(&C[ii * M + jj], sum);
        }
    }
}

void MatrixMultiply(const float* A, const float* B, float* C, int M) {
    #pragma omp parallel for collapse(3)
    for (int i = 0; i < M; i += BLOCK_SIZE) {
        for (int j = 0; j < M; j += BLOCK_SIZE) {
            for (int k = 0; k < M; k += BLOCK_SIZE) {
                MultiplyBlock(A, B, C, M, i, j, k);
            }
        }
    }
}

int main(int argc, char** argv) {
    int M = 2048;
    if (argc >= 2) {
        M = atoi(argv[1]);
    }

    // Align memory to 32-byte boundary for better SIMD performance
    float* A = (float*)aligned_alloc(64, M * M * sizeof(float));
    float* B = (float*)aligned_alloc(64, M * M * sizeof(float));
    float* C = (float*)aligned_alloc(64, M * M * sizeof(float));

    // Initialize matrices
    std::fill_n(A, M * M, 1.0f);
    std::fill_n(B, M * M, 1.0f);
    std::fill_n(C, M * M, 0.0f); // Initialize C to zeros

    // Warm-up run
    MatrixMultiply(A, B, C, M);

    // Actual timed run
    auto start_time = std::chrono::high_resolution_clock::now();
    
    MatrixMultiply(A, B, C, M);
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

    std::cout << duration.count() << std::endl;

    // Clean up
    free(A);
    free(B);
    free(C);

    return 0;
}