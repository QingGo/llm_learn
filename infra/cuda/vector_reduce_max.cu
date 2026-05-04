#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// 核函数：使用线程块内规约（共享内存）求数组最大值
__global__ void vectorMax(const float *A, float *B, int N) {
    __shared__ float shared[256];
    
    int tid = threadIdx.x;
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    
    // 1. 从全局内存加载数据到共享内存（处理边界）
    if (i < N) {
        shared[tid] = A[i];
    } else {
        shared[tid] = -INFINITY;
    }
    __syncthreads();
    
    // 2. 树形规约（线程块内）
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            if (shared[tid + s] > shared[tid]) {
                shared[tid] = shared[tid + s];
            }
        }
        __syncthreads();
    }
    
    // 3. 线程 0 将当前块的最大值写入输出
    if (tid == 0) {
        B[blockIdx.x] = shared[0];
    }
}

int main() {
    int N = 1 << 10; // 1024 元素
    size_t size = N * sizeof(float);

    // 1. Host 内存分配与初始化
    float *h_A = (float*)malloc(size);

    srand(time(NULL));
    for (int i = 0; i < N; i++) {
        h_A[i] = (float)rand() / RAND_MAX;
    }

    // 2. Device 内存分配
    float *d_A, *d_B;
    cudaMalloc(&d_A, size);

    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    size_t sizeB = blocksPerGrid * sizeof(float);
    cudaMalloc(&d_B, sizeB);

    // 3. 拷贝数据 Host → Device
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);

    printf("Launching %d blocks with %d threads each\n", blocksPerGrid, threadsPerBlock);

    // 4. 启动核函数
    vectorMax<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, N);

    // 5. 将每个块的局部最大值拷贝回 Host
    float *h_B = (float*)malloc(sizeB);
    cudaMemcpy(h_B, d_B, sizeB, cudaMemcpyDeviceToHost);

    // 6. 在 host 端计算全局最大值
    float globalMax = -INFINITY;
    for (int i = 0; i < blocksPerGrid; i++) {
        if (h_B[i] > globalMax) {
            globalMax = h_B[i];
        }
    }
    printf("Global max = %f\n", globalMax);

    // 7. 验证：用 host 端简单扫描验证
    float expected = -INFINITY;
    for (int i = 0; i < N; i++) {
        if (h_A[i] > expected) expected = h_A[i];
    }
    printf("Expected max = %f\n", expected);
    printf("Result %s\n", (globalMax == expected) ? "PASS" : "FAIL");

    // 8. 释放内存
    cudaFree(d_A); cudaFree(d_B);
    free(h_A); free(h_B);

    return 0;
}