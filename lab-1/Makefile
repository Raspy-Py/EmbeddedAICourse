all: mt-simd-matmul openblas-matmul omp-matmul rknn-matmul

openblas-matmul: ./cpp/openblas-matmul.cpp
	g++ -o ./bin/openblas-matmul cpp/openblas-matmul.cpp -O3 -I/usr/include/aarch64-linux-gnu -L/usr/lib -lopenblas 

mt-simd-matmul: ./cpp/mt-simd-matmul.cpp
	g++ -o ./bin/mt-simd-matmul ./cpp/mt-simd-matmul.cpp -O3 -std=c++11 -pthread 

omp-matmul: ./cpp/omp-matmul.cpp
	g++ -o ./bin/omp-matmul ./cpp/omp-matmul.cpp -O3 -march=native -fopenmp
