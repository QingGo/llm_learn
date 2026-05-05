// nccl_allreduce.cu
#include <cuda_runtime.h>
#include <nccl.h>
#include <stdio.h>
#include <stdlib.h>

#define CUDA_CHECK(c) do { cudaError_t e = c; \
    if (e != cudaSuccess) { printf("CUDA Error: %s\n", cudaGetErrorString(e)); exit(1); } \
} while(0)
#define NCCL_CHECK(c) do { ncclResult_t r = c; \
    if (r != ncclSuccess) { printf("NCCL Error: %s\n", ncclGetErrorString(r)); exit(1); } \
} while(0)

int main() {
    int nDev;
    CUDA_CHECK(cudaGetDeviceCount(&nDev));
    if (nDev < 2) { printf("需要至少2张GPU\n"); return 1; }
    printf("检测到 %d 张GPU\n", nDev);

    // ---------- 1. 初始化 NCCL ----------
    ncclComm_t *comms = (ncclComm_t*)malloc(nDev * sizeof(ncclComm_t));
    int *dev_list = (int*)malloc(nDev * sizeof(int));
    for (int i = 0; i < nDev; i++) dev_list[i] = i;
    NCCL_CHECK(ncclCommInitAll(comms, nDev, dev_list));

    // ---------- 2. 为每张GPU分配send/recv缓冲区 ----------
    const int N = 1024;
    size_t size = N * sizeof(float);
    float **h_send = (float**)malloc(nDev * sizeof(float*));
    float **h_recv = (float**)malloc(nDev * sizeof(float*));
    float **d_send = (float**)malloc(nDev * sizeof(float*));
    float **d_recv = (float**)malloc(nDev * sizeof(float*));
    cudaStream_t *streams = (cudaStream_t*)malloc(nDev * sizeof(cudaStream_t));

    for (int i = 0; i < nDev; i++) {
        CUDA_CHECK(cudaSetDevice(i));
        CUDA_CHECK(cudaHostAlloc(&h_send[i], size, cudaHostAllocDefault));
        CUDA_CHECK(cudaHostAlloc(&h_recv[i], size, cudaHostAllocDefault));
        CUDA_CHECK(cudaMalloc(&d_send[i], size));
        CUDA_CHECK(cudaMalloc(&d_recv[i], size));
        CUDA_CHECK(cudaStreamCreateWithFlags(&streams[i], cudaStreamNonBlocking));

        // 填充数据：每张GPU的贡献数据递增
        for (int j = 0; j < N; j++)
            h_send[i][j] = (float)(i + 1.0f);  // GPU0全1, GPU1全2, ...
        CUDA_CHECK(cudaMemcpyAsync(d_send[i], h_send[i], size,
                                    cudaMemcpyHostToDevice, streams[i]));
    }

    // ---------- 3. 执行 AllReduce ----------
    for (int i = 0; i < nDev; i++) {
        CUDA_CHECK(cudaSetDevice(i));
        NCCL_CHECK(ncclAllReduce(
            (const void*)d_send[i], (void*)d_recv[i], N,
            ncclFloat, ncclSum, comms[i], streams[i]));
    }

    // ---------- 4. 同步并取回结果 ----------
    for (int i = 0; i < nDev; i++) {
        CUDA_CHECK(cudaSetDevice(i));
        CUDA_CHECK(cudaStreamSynchronize(streams[i]));
        CUDA_CHECK(cudaMemcpyAsync(h_recv[i], d_recv[i], size,
                                    cudaMemcpyDeviceToHost, streams[i]));
        CUDA_CHECK(cudaStreamSynchronize(streams[i]));
    }

    // ---------- 5. 验证 ----------
    float expected = 0.0f;
    for (int i = 0; i < nDev; i++) expected += (float)(i + 1.0f);
    int errors = 0;
    for (int i = 0; i < nDev; i++) {
        for (int j = 0; j < N; j++) {
            if (fabsf(h_recv[i][j] - expected) > 0.01f) {
                errors++;
                if (errors <= 3) printf("GPU%d[%d]: got %f, expected %f\n",
                                         i, j, h_recv[i][j], expected);
            }
        }
    }
    printf("验证: %s (每GPU应收到 %.1f，共%d个元素中%d个错误)\n",
           errors == 0 ? "✓ 通过" : "✗ 失败", expected, nDev * N, errors);

    // 清理
    for (int i = 0; i < nDev; i++) {
        CUDA_CHECK(cudaSetDevice(i));
        CUDA_CHECK(cudaStreamDestroy(streams[i]));
        CUDA_CHECK(cudaFreeHost(h_send[i])); CUDA_CHECK(cudaFreeHost(h_recv[i]));
        CUDA_CHECK(cudaFree(d_send[i])); CUDA_CHECK(cudaFree(d_recv[i]));
        ncclCommDestroy(comms[i]);
    }
    free(comms); free(dev_list); free(h_send); free(h_recv);
    free(d_send); free(d_recv); free(streams);
    return 0;
}