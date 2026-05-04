#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// 每个线程计算 C 的一个元素 (row x col)
__global__ void matMulNaive(const float *A, const float *B, float *C,
                             int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

// CPU 矩阵乘法参考实现
void matMulCPU(const float *A, const float *B, float *C, int M, int N, int K) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

// Tiled 矩阵乘法 kernel：利用共享内存减少全局内存访问
// TILE_SIZE 为编译期常量（16 或 32），每个线程块加载 A 和 B 的两个 tile 到共享内存
// 然后线程块内各线程计算 C 中对应 tile 的一部分
// 此过程循环直到覆盖整个 K 维度

template <int TILE_SIZE>
__global__ void matMulTiled(const float *A, const float *B, float *C,
                             int M, int N, int K) {
    // 不需要 padding，没有争用
    __shared__ float tileA[TILE_SIZE][TILE_SIZE];
    __shared__ float tileB[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    float sum = 0.0f;
    int numTiles = (K + TILE_SIZE - 1) / TILE_SIZE;

    for (int t = 0; t < numTiles; t++) {
        // 协作加载 tileA 到共享内存
        if (row < M && t * TILE_SIZE + threadIdx.x < K)
            tileA[threadIdx.y][threadIdx.x] = A[row * K + t * TILE_SIZE + threadIdx.x];
        else
            tileA[threadIdx.y][threadIdx.x] = 0.0f;

        // 协作加载 tileB 到共享内存
        if (t * TILE_SIZE + threadIdx.y < K && col < N)
            tileB[threadIdx.y][threadIdx.x] = B[(t * TILE_SIZE + threadIdx.y) * N + col];
        else
            tileB[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();

        // 计算当前 tile 的局部累加
        #pragma unroll
        for (int i = 0; i < TILE_SIZE; i++) {
            sum += tileA[threadIdx.y][i] * tileB[i][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

// CPU time: 3242.239 ms
// GPU kernel time: 2.428 ms
// Speedup (CPU / GPU naive): 1335.16x
// Result check (naive): PASS
// GPU kernel (tiled) time: 1.542 ms
// Speedup (CPU / GPU tiled): 2102.60x
// Speedup (GPU tiled / GPU naive): 1.57x
// Result check (tiled): PASS
int main() {
    int M = 1024, N = 1024, K = 1024;  // 适当增大规模以测量更明显
    size_t size_A = M * K * sizeof(float);
    size_t size_B = K * N * sizeof(float);
    size_t size_C = M * N * sizeof(float);

    // Host 分配和初始化
    float *h_A = (float*)malloc(size_A);
    float *h_B = (float*)malloc(size_B);
    float *h_C = (float*)malloc(size_C);
    float *h_C_cpu = (float*)malloc(size_C);

    for (int i = 0; i < M * K; i++) h_A[i] = (float)(rand() % 100) / 10.0f;
    for (int i = 0; i < K * N; i++) h_B[i] = (float)(rand() % 100) / 10.0f;

    // --- CPU 计时 ---
    clock_t cpu_start = clock();
    matMulCPU(h_A, h_B, h_C_cpu, M, N, K);
    clock_t cpu_end = clock();
    double cpu_time_ms = (double)(cpu_end - cpu_start) / CLOCKS_PER_SEC * 1000.0;
    printf("CPU time: %.3f ms\n", cpu_time_ms);

    // Device 分配
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size_A);
    cudaMalloc(&d_B, size_B);
    cudaMalloc(&d_C, size_C);

    cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice);

    // 创建 CUDA Event
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // 二维执行配置: 16x16 线程块
    dim3 blockDim(16, 16);
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x,
                 (M + blockDim.y - 1) / blockDim.y);

    // 记录 GPU 启动事件并执行 kernel
    cudaEventRecord(start);
    matMulNaive<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);
    cudaEventRecord(stop);

    cudaMemcpy(h_C, d_C, size_C, cudaMemcpyDeviceToHost);

    // 等待 GPU 完成并计算耗时 (ms)
    cudaEventSynchronize(stop);
    float gpu_time_ms = 0.0f;
    cudaEventElapsedTime(&gpu_time_ms, start, stop);
    printf("GPU kernel time: %.3f ms\n", gpu_time_ms);
    printf("Speedup (CPU / GPU naive): %.2fx\n", cpu_time_ms / gpu_time_ms);

    // 简单验证结果一致性
    bool pass = true;
    for (int i = 0; i < M * N && pass; i++) {
        if (fabs(h_C[i] - h_C_cpu[i]) > 1e-1) {
            printf("Mismatch at [%d]: GPU=%.2f CPU=%.2f\n", i, h_C[i], h_C_cpu[i]);
            pass = false;
        }
    }
    if (pass) printf("Result check (naive): PASS\n");

    // ------------------- Tiled 矩阵乘法测试 -------------------
    const int TILE_SIZE = 32;
    dim3 blockDimTiled(TILE_SIZE, TILE_SIZE);
    dim3 gridDimTiled((N + TILE_SIZE - 1) / TILE_SIZE,
                      (M + TILE_SIZE - 1) / TILE_SIZE);

    cudaEventRecord(start);
    matMulTiled<TILE_SIZE><<<gridDimTiled, blockDimTiled>>>(d_A, d_B, d_C, M, N, K);
    cudaEventRecord(stop);

    cudaMemcpy(h_C, d_C, size_C, cudaMemcpyDeviceToHost);

    cudaEventSynchronize(stop);
    float gpu_tiled_ms = 0.0f;
    cudaEventElapsedTime(&gpu_tiled_ms, start, stop);
    printf("GPU kernel (tiled) time: %.3f ms\n", gpu_tiled_ms);
    printf("Speedup (CPU / GPU tiled): %.2fx\n", cpu_time_ms / gpu_tiled_ms);
    printf("Speedup (GPU tiled / GPU naive): %.2fx\n", gpu_time_ms / gpu_tiled_ms);

    // 验证 Tiled 结果一致性
    bool tiled_pass = true;
    for (int i = 0; i < M * N && tiled_pass; i++) {
        if (fabs(h_C[i] - h_C_cpu[i]) > 1e-1) {
            printf("Mismatch at [%d]: Tiled=%.2f CPU=%.2f\n", i, h_C[i], h_C_cpu[i]);
            tiled_pass = false;
        }
    }
    if (tiled_pass) printf("Result check (tiled): PASS\n");

    // 清理
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(h_A); free(h_B); free(h_C); free(h_C_cpu);
    return 0;
}