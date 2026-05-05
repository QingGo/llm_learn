// p2p_tensor_parallel.cu
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

// 核函数：每块计算矩阵×本切片（B_mat由存储在当前GPU上的部分列组成）
__global__ void partial_gemm(const float *A, const float *B_local,
                              float *C_partial, int M, int N_local, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < N_local) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B_local[k * N_local + col];
        }
        C_partial[row * N_local + col] = sum;
    }
}
// 汇总核函数：直接通过地址读取对等设备的局部结果并累加
__global__ void accumulate_peer(const float *C_local, const float *C_peer,
                                float *C_out, int M, int N_local) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = M * N_local;
    if (idx < total) {
        C_out[idx] = C_local[idx] + C_peer[idx];
    }
}

#define CUDA_CHECK(c) do { cudaError_t e = c; \
    if (e != cudaSuccess) { printf("%s:%d error %s\n", __FILE__, __LINE__, \
        cudaGetErrorString(e)); exit(1); } \
} while(0)

int main() {
    int M = 512, N = 512, K = 512;
    int N_per_gpu = N / 2;  // 每个GPU计算N/2列

    // 分配主机内存
    float *h_A = (float*)malloc(M * K * sizeof(float));
    float *h_B0 = (float*)malloc(K * N_per_gpu * sizeof(float));
    float *h_B1 = (float*)malloc(K * N_per_gpu * sizeof(float));
    for (int i = 0; i < M*K; i++) h_A[i] = (float)(rand()%100)/100.0f;
    for (int i = 0; i < K*N_per_gpu; i++) {
        h_B0[i] = (float)(rand()%100)/100.0f;
        h_B1[i] = (float)(rand()%100)/100.0f;
    }

    // 为两台GPU各自分配设备端显存
    float *d_A0, *d_B0, *d_C0, *d_A1, *d_B1, *d_C1;
    cudaSetDevice(0);
    CUDA_CHECK(cudaMalloc(&d_A0, M*K*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_B0, K*N_per_gpu*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_C0, M*N_per_gpu*sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_A0, h_A, M*K*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B0, h_B0, K*N_per_gpu*sizeof(float), cudaMemcpyHostToDevice));

    cudaSetDevice(1);
    CUDA_CHECK(cudaMalloc(&d_A1, M*K*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_B1, K*N_per_gpu*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_C1, M*N_per_gpu*sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_A1, h_A, M*K*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B1, h_B1, K*N_per_gpu*sizeof(float), cudaMemcpyHostToDevice));

    // 启用P2P
    int can;
    CUDA_CHECK(cudaDeviceCanAccessPeer(&can, 0, 1));
    if (!can) { printf("不支持P2P，退出\n"); exit(0); }
    cudaSetDevice(0); CUDA_CHECK(cudaDeviceEnablePeerAccess(1, 0));
    cudaSetDevice(1); CUDA_CHECK(cudaDeviceEnablePeerAccess(0, 0));

    // 计算
    dim3 block(16,16);
    dim3 grid((N_per_gpu+15)/16, (M+15)/16);
    cudaSetDevice(0);
    partial_gemm<<<grid, block>>>(d_A0, d_B0, d_C0, M, N_per_gpu, K);
    cudaSetDevice(1);
    partial_gemm<<<grid, block>>>(d_A1, d_B1, d_C1, M, N_per_gpu, K);
    cudaDeviceSynchronize();

    // P2P 读取对方结果并累加
    cudaSetDevice(0);
    accumulate_peer<<<grid, block>>>(d_C0, d_C1, d_C0, M, N_per_gpu);
    cudaSetDevice(1);
    accumulate_peer<<<grid, block>>>(d_C1, d_C0, d_C1, M, N_per_gpu);
    cudaDeviceSynchronize();

    printf("P2P 张量并行示例完成（无错误即成功）\n");

    // 清理
    cudaSetDevice(0); cudaFree(d_A0); cudaFree(d_B0); cudaFree(d_C0);
    cudaSetDevice(1); cudaFree(d_A1); cudaFree(d_B1); cudaFree(d_C1);
    free(h_A); free(h_B0); free(h_B1);
    return 0;
}