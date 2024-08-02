import numpy as np 
import torch
import time
import math
import cv2
from tqdm import tqdm

def test_numpy(M: int = 2048, s: int = 512):
    A = np.ones((M, M)).astype('float32')
    B = np.ones((M, M)).astype('float32')
    C = np.zeros(A.shape).astype('float32')

    if s > 0:
        start_time = time.time_ns()
        for row in range(0, M, s):
            for col in range(0, M, s):
                C[row:row+s, col:col+s] = A[row:row+s,:] @ B[:, col:col+s]
        end_time = time.time_ns()
    else:
        start_time = time.time_ns()
        C = A @ B
        end_time = time.time_ns()
    return (end_time-start_time)/1e6

def test_torch(M: int = 2048, s: int = 512):
    A = torch.ones((M, M), dtype=torch.float32, requires_grad=False)
    B = torch.ones((M, M), dtype=torch.float32, requires_grad=False)
    C = torch.zeros((M, M), dtype=torch.float32, requires_grad=False)

    if s > 0:
        start_time = time.time_ns()
        for row in range(0, M, s):
            for col in range(0, M, s):
                C[row:row+s, col:col+s] = A[row:row+s,:] @ B[:, col:col+s]
        end_time = time.time_ns()
    else:
        start_time = time.time_ns()
        C = A @ B
        end_time = time.time_ns()

    return (end_time-start_time)/1e6

def test_opencv(M: int = 2048, s: int = 512):
    A = np.ones((M, M), dtype='float32')
    B = np.ones((M, M), dtype='float32')
    C = np.zeros((M, M), dtype='float32')

    if s > 0:
        start_time = time.time_ns()
        for row in range(0, M, s):
            for col in range(0, M, s):
                C[row:row+s, col:col+s] = cv2.gemm(A[row:row+s, :], B[:, col:col+s], 1, None, 0)
        end_time = time.time_ns()
    else:
        start_time = time.time_ns()
        C = cv2.gemm(A, B, 1, None, 0)
        end_time = time.time_ns()

    return (end_time - start_time) / 1e6

def test_external(p1: int, p2: int, exec_path: str = './bin/openblas-matmul'):
    import subprocess

    result = subprocess.run([exec_path, str(p1), str(p2)], capture_output=True, text=True)

    if result.returncode == 0:
        time_result = float(result.stdout.strip())
        return time_result
    else:
        print(f"[{exec_path}] subproccess error: {result.stderr}")
        return -1

if __name__ == "__main__":
    import numpy as np
    from tqdm import tqdm

    M = 2048
    s = -1
    N = 10  # Number of iterations
    results = {}

    # Function to calculate mean execution time with tqdm progress bar
    def mean_time(func, *args, **kwargs):
        times = []
        for _ in tqdm(range(N), desc=f"Running {func.__name__}"):
            times.append(func(*args, **kwargs))
        return np.mean(times[1:])

    results['NumPy Python'] = mean_time(test_numpy, M, s)
    results['Torch Python'] = mean_time(test_torch, M, s)
    results['OpenCV Python'] = mean_time(test_opencv, M, s)
    results['OpenBLAS C++'] = mean_time(test_external, p1=M, p2=-1, exec_path='./bin/openblas-matmul')
    results['OpenMP+SIMD C++'] = mean_time(test_external, p1=M, p2=-1, exec_path='./bin/omp-matmul')
    results['Threads+SIMD C++'] = mean_time(test_external, p1=M, p2=8, exec_path='./bin/mt-simd-matmul')


    for method, time in results.items():
        print(f"{method:>16}: {math.floor(time):>4}ms")
