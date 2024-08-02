import cv2
import numpy as np
import time

def test_opencv(M: int = 2048, s: int = 512):
    # Create matrices filled with ones
    A = np.ones((M, M), dtype='float32')
    B = np.ones((M, M), dtype='float32')
    C = np.zeros((M, M), dtype='float32')

    if s > 0:
        start_time = time.time_ns()
        # Perform matrix multiplication in blocks
        for row in range(0, M, s):
            for col in range(0, M, s):
                # cv2.gemm(source1, source2, alpha, src3, beta, dst, flags)
                # alpha is the scale factor for source1*source2, beta is the scale factor for src3
                C[row:row+s, col:col+s] = cv2.gemm(A[row:row+s, :], B[:, col:col+s], 1, None, 0)
        end_time = time.time_ns()
    else:
        start_time = time.time_ns()
        # Full matrix multiplication if slice size is not positive
        C = cv2.gemm(A, B, 1, None, 0)
        end_time = time.time_ns()

    # Return the time taken in milliseconds
    return (end_time - start_time) / 1e6

print(test_opencv())