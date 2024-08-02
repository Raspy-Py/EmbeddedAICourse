#include <iostream>
#include <vector>
#include <thread>
#include <chrono>
#include <arm_neon.h>
#include <cstring>

constexpr int BLOCK_SIZE = 64;

void MultiplyBlock(const float* A, const float* B, float* C, int M, int i_start, int i_end, int j_start, int j_end, int k_start, int k_end) {
    for (int i = i_start; i < i_end; ++i) {
        for (int j = j_start; j < j_end; j += 4) {
            float32x4_t sum = vld1q_f32(&C[i * M + j]); // Load existing values
            for (int k = k_start; k < k_end; ++k) {
                float32x4_t a = vdupq_n_f32(A[i * M + k]);
                float32x4_t b = vld1q_f32(&B[k * M + j]);
                sum = vaddq_f32(sum, vmulq_f32(a, b)); // Use vaddq_f32 for accumulation
            }
            vst1q_f32(&C[i * M + j], sum);
        }
    }
}

void MultiplyPart(const float* A, const float* B, float* C, int M, int from, int to) {
    for (int i = from; i < to; i += BLOCK_SIZE) {
        for (int j = 0; j < M; j += BLOCK_SIZE) {
            for (int k = 0; k < M; k += BLOCK_SIZE) {
                MultiplyBlock(A, B, C, M, i, std::min(i + BLOCK_SIZE, to), j, std::min(j + BLOCK_SIZE, M), k, std::min(k + BLOCK_SIZE, M));
            }
        }
    }
}

int main(int argc, char** argv) {
    int M = 2048;
    int numThreads = 4;

    if (argc >= 3) {
        M = atoi(argv[1]);
        numThreads = atoi(argv[2]);
    }

    // Allocate aligned memory
    float* A = (float*)aligned_alloc(64, M * M * sizeof(float));
    float* B = (float*)aligned_alloc(64, M * M * sizeof(float));
    float* C = (float*)aligned_alloc(64, M * M * sizeof(float));

    // Initialize matrices
    std::fill_n(A, M * M, 1.0f);
    std::fill_n(B, M * M, 1.0f);
    std::fill_n(C, M * M, 0.0f); // Initialize C to zeros

    auto start_time = std::chrono::high_resolution_clock::now();

    std::vector<std::thread> threads;
    int rowsPerThread = M / numThreads;

    for (int i = 0; i < numThreads; ++i) {
        int start = i * rowsPerThread;
        int end = (i == numThreads - 1) ? M : (i + 1) * rowsPerThread;
        threads.emplace_back(MultiplyPart, A, B, C, M, start, end);
    }

    for (auto& t : threads) {
        t.join();
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

    std::cout << duration.count() << std::endl;

    // Clean up
    free(A);
    free(B);
    free(C);

    return 0;
}