#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// 核函数：每个线程处理一个元素加法
__global__ void vectorAdd(const float *A, const float *B, float *C, int M, int N) {
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    if (col < N && row < M) {
        C[row * N + col] = A[row * N + col] + B[row * N + col];
    }
}

// 辅助：打印数组前几个值
void print_sample(const float *arr, int N, const char* name) {
    printf("%s (first 5): ", name);
    for (int i = 0; i < 5 && i < N; i++)
        printf("%f ", arr[i]);
    printf("\n");
}

int main() {
    int M = 256, N = 256;
    size_t size = M * N * sizeof(float);

    // 1. Host 内存分配与初始化
    float *h_A = (float*)malloc(size);
    float *h_B = (float*)malloc(size);
    float *h_C = (float*)malloc(size);

    srand(time(NULL));
    for (int i = 0; i < M * N; i++) {
        h_A[i] = (float)rand() / RAND_MAX;
        h_B[i] = (float)rand() / RAND_MAX;
    }

    // 2. Device 内存分配
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    // 3. 拷贝数据 Host → Device
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // 4. 设置执行配置
    dim3 blockDim(16, 16);
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x,
                 (M + blockDim.y - 1) / blockDim.y);

    // 5. 启动核函数
    vectorAdd<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N);
    // 6. 等待 kernel 结束，并将结果拷贝回 Host
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    // 7. 验证结果（只验证前几个）
    print_sample(h_A, N, "A");
    print_sample(h_B, N, "B");
    print_sample(h_C, N, "C=A+B");
    for (int i = 0; i < N; i++) {
        if (fabs(h_A[i] + h_B[i] - h_C[i]) > 1e-4) {
            printf("Error at index %d!\n", i);
            break;
        }
    }
    printf("Verification done.\n");

    // 8. 释放内存
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(h_A); free(h_B); free(h_C);

    return 0;
}