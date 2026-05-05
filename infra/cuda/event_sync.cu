// event_sync.cu
#include <cuda_runtime.h>
#include <stdio.h>

__global__ void stage1(const float *in, float *tmp, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) tmp[idx] = in[idx] * 2.0f;
}

__global__ void stage2(const float *tmp, float *out, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) out[idx] = tmp[idx] * 3.0f; // 使用stage1的结果
}

#define CUDA_CHECK(c) do { cudaError_t e = c; \
    if (e != cudaSuccess) { printf("CUDA Error %s\n", cudaGetErrorString(e)); exit(1); } \
} while(0)

int main() {
    int N = 1 << 20;
    size_t size = N * sizeof(float);

    float *h_in = (float*)malloc(size);
    for (int i = 0; i < N; i++) h_in[i] = (float)i;

    float *d_in, *d_tmp, *d_out;
    CUDA_CHECK(cudaMalloc(&d_in, size));
    CUDA_CHECK(cudaMalloc(&d_tmp, size));
    CUDA_CHECK(cudaMalloc(&d_out, size));
    CUDA_CHECK(cudaMemcpy(d_in, h_in, size, cudaMemcpyHostToDevice));

    // 创建两个流
    cudaStream_t s1, s2;
    CUDA_CHECK(cudaStreamCreateWithFlags(&s1, cudaStreamNonBlocking));
    CUDA_CHECK(cudaStreamCreateWithFlags(&s2, cudaStreamNonBlocking));

    // 创建事件
    cudaEvent_t ev;
    CUDA_CHECK(cudaEventCreate(&ev));

    // 流1中执行阶段1
    stage1<<<4096, 256, 0, s1>>>(d_in, d_tmp, N);
    // 在流1中记录事件（标记阶段1完成的点）
    CUDA_CHECK(cudaEventRecord(ev, s1));
    // 流2中等待该事件，然后执行阶段2
    CUDA_CHECK(cudaStreamWaitEvent(s2, ev, 0));
    stage2<<<4096, 256, 0, s2>>>(d_tmp, d_out, N);

    CUDA_CHECK(cudaDeviceSynchronize());

    // 验证
    float *h_out = (float*)malloc(size);
    CUDA_CHECK(cudaMemcpy(h_out, d_out, size, cudaMemcpyDeviceToHost));
    int err = 0;
    for (int i = 0; i < N; i++) {
        if (fabsf(h_out[i] - i * 6.0f) > 0.01f) { err++; break; }
    }
    printf("Event同步验证: %s (结果应为 i*6)\n", err ? "失败" : "通过");

    CUDA_CHECK(cudaEventDestroy(ev));
    CUDA_CHECK(cudaStreamDestroy(s1));
    CUDA_CHECK(cudaStreamDestroy(s2));
    CUDA_CHECK(cudaFree(d_in)); CUDA_CHECK(cudaFree(d_tmp)); CUDA_CHECK(cudaFree(d_out));
    free(h_in); free(h_out);
    return 0;
}