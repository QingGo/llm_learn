// matmul_tiled.cu
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define TILE_SIZE 32

// 朴素实现（层级1，用于对比）
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

// ====== Tiled 矩阵乘法（层级2核心） ======
__global__ void matMulTiled(const float *A, const float *B, float *C,
                              int M, int N, int K) {
    // 共享内存 tile
    __shared__ float tileA[TILE_SIZE][TILE_SIZE];
    __shared__ float tileB[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;  // C 的行
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;  // C 的列

    float sum = 0.0f;

    // 遍历 K 维度的所有 tile
    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; t++) {
        // ------ 协同加载 tileA ------
        int aRow = row;                        // A 的行
        int aCol = t * TILE_SIZE + threadIdx.x; // A 的列
        if (aRow < M && aCol < K) {
            tileA[threadIdx.y][threadIdx.x] = A[aRow * K + aCol];
        } else {
            tileA[threadIdx.y][threadIdx.x] = 0.0f;
        }

        // ------ 协同加载 tileB ------
        int bRow = t * TILE_SIZE + threadIdx.y; // B 的行
        int bCol = col;                         // B 的列
        if (bRow < K && bCol < N) {
            tileB[threadIdx.y][threadIdx.x] = B[bRow * N + bCol];
        } else {
            tileB[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();  // 确保整个 block 的 tile 加载完成

        // ------ 从共享内存读取并计算 ------
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += tileA[threadIdx.y][k] * tileB[k][threadIdx.x];
        }

        __syncthreads();  // 确保所有线程完成读取，再覆盖 tile
    }

    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

// ====== 带 Bank Conflict 避免的 Tiled 矩阵乘法 ======
#define TILE_SIZE_PAD (TILE_SIZE + 1)

__global__ void matMulTiledNoBankConflict(const float *A, const float *B, float *C,
                                            int M, int N, int K) {
    // 加 padding 避免 bank conflict（tileB 跨步访问时）
    __shared__ float tileA[TILE_SIZE][TILE_SIZE_PAD];
    __shared__ float tileB[TILE_SIZE][TILE_SIZE_PAD];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    float sum = 0.0f;

    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; t++) {
        int aCol = t * TILE_SIZE + threadIdx.x;
        if (row < M && aCol < K) {
            tileA[threadIdx.y][threadIdx.x] = A[row * K + aCol];
        } else {
            tileA[threadIdx.y][threadIdx.x] = 0.0f;
        }

        int bRow = t * TILE_SIZE + threadIdx.y;
        if (bRow < K && col < N) {
            tileB[threadIdx.y][threadIdx.x] = B[bRow * N + col];
        } else {
            tileB[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();

        for (int k = 0; k < TILE_SIZE; k++) {
            sum += tileA[threadIdx.y][k] * tileB[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

// 辅助：计时 + 正确性验证
float launchAndTime(void (*kernel)(const float*, const float*, float*, int, int, int),
                    dim3 grid, dim3 block,
                    const float *d_A, const float *d_B, float *d_C,
                    int M, int N, int K, const char *name,
                    float *h_C_ref) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaMemset(d_C, 0, M * N * sizeof(float));

    cudaEventRecord(start);
    kernel<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // 计算 TFLOPS
    float flops = 2.0f * M * N * K;
    float tflops = flops / (ms / 1000.0f) / 1e12f;
    printf("%-30s: %8.3f ms  |  %.3f TFLOPS\n", name, ms, tflops);

    return ms;
}

int main() {
    // 用较小的矩阵便于快速运行，也可增大观察差距
    int M = 1024, N = 1024, K = 1024;
    size_t size_A = M * K * sizeof(float);
    size_t size_B = K * N * sizeof(float);
    size_t size_C = M * N * sizeof(float);

    // Host 分配
    float *h_A = (float*)malloc(size_A);
    float *h_B = (float*)malloc(size_B);
    float *h_C = (float*)malloc(size_C);
    float *h_C_ref = (float*)malloc(size_C);  // 参考结果

    // 随机初始化
    srand(42);
    for (int i = 0; i < M * K; i++) h_A[i] = (float)(rand() % 100) / 100.0f;
    for (int i = 0; i < K * N; i++) h_B[i] = (float)(rand() % 100) / 100.0f;

    // Device 分配
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size_A);
    cudaMalloc(&d_B, size_B);
    cudaMalloc(&d_C, size_C);
    cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice);

    // 执行配置
    dim3 block(TILE_SIZE, TILE_SIZE);  // 16x16 = 256 线程/块
    dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE,
              (M + TILE_SIZE - 1) / TILE_SIZE);

    printf("矩阵维度: M=%d N=%d K=%d | Tile: %dx%d | Block: %dx%d | Grid: %dx%d\n\n",
           M, N, K, TILE_SIZE, TILE_SIZE, TILE_SIZE, TILE_SIZE,
           grid.x, grid.y);

    // 用朴素版本生成参考结果
    matMulNaive<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    cudaMemcpy(h_C_ref, d_C, size_C, cudaMemcpyDeviceToHost);

    // 预热
    matMulTiled<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    cudaDeviceSynchronize();

    // 测量
    printf("--- 性能对比 ---\n");
    launchAndTime(matMulNaive, grid, block, d_A, d_B, d_C, M, N, K, "Naive", h_C_ref);
    launchAndTime(matMulTiled, grid, block, d_A, d_B, d_C, M, N, K, "Tiled", h_C_ref);
    launchAndTime(matMulTiledNoBankConflict, grid, block, d_A, d_B, d_C,
                  M, N, K, "Tiled (No Bank Conflict)", h_C_ref);

    // 验证 tiled 版本正确性
    matMulTiled<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    cudaMemcpy(h_C, d_C, size_C, cudaMemcpyDeviceToHost);
    int errors = 0;
    for (int i = 0; i < M * N; i++) {
        if (fabs(h_C[i] - h_C_ref[i]) > 0.01f) {
            errors++;
            if (errors <= 3) printf("  错误[%d]: tiled=%f ref=%f\n", i, h_C[i], h_C_ref[i]);
        }
    }
    printf("\nTiled 版本正确性: %s (%d 个误差 > 0.01)\n",
           errors == 0 ? "通过" : "失败", errors);

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(h_A); free(h_B); free(h_C); free(h_C_ref);
    return 0;
}