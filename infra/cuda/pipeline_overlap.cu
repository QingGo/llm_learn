// pipeline_overlap.cu
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// ===== 模拟一个“计算密集型”kernel（如softmax或GEMM小分块） =====
__global__ void compute_kernel(const float *input, float *output, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        float x = input[idx];
        // 模拟一些计算：用几个乘加操作模拟softmax内部的指数/缩放
        float result = x;
        for (int i = 0; i < 64; i++) {
            result = 0.5f * (result + x / (1.0f + fabsf(result)));
        }
        output[idx] = result;
    }
}

// ===== 辅助宏：CUDA错误检查 =====
#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        printf("CUDA error at %s:%d — %s\n", __FILE__, __LINE__, \
               cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)

int main() {
    // -------- 参数 --------
    const int total_N = 16 * 1024 * 1024;  // 16M个float ≈ 64MB
    const int num_chunks = 4;
    const int chunk_N = total_N / num_chunks;
    const size_t chunk_bytes = chunk_N * sizeof(float);
    const int threads_per_block = 256;
    const int blocks_per_chunk = (chunk_N + threads_per_block - 1) / threads_per_block;
    
    // -------- 分配 pinned 内存（异步传输前提） --------
    float *h_input_pinned, *h_output_pinned, *h_output_ref;
    CUDA_CHECK(cudaHostAlloc(&h_input_pinned,  total_N * sizeof(float), cudaHostAllocDefault));
    CUDA_CHECK(cudaHostAlloc(&h_output_pinned, total_N * sizeof(float), cudaHostAllocDefault));
    h_output_ref = (float*)malloc(total_N * sizeof(float)); // 可换页参考值
    
    for (int i = 0; i < total_N; i++)
        h_input_pinned[i] = (float)(rand() % 1000) / 100.0f;
    
    // -------- 分配 device 内存（整块再分段） --------
    float *d_input, *d_output;
    CUDA_CHECK(cudaMalloc(&d_input,  total_N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output, total_N * sizeof(float)));
    
    // -------- 创建 4 条非阻塞流 --------
    const int num_streams = num_chunks;
    cudaStream_t streams[num_streams];
    for (int i = 0; i < num_streams; i++) {
        CUDA_CHECK(cudaStreamCreateWithFlags(&streams[i], cudaStreamNonBlocking));
    }
    
    // -------- 创建事件用于流间依赖 --------
    cudaEvent_t chunk_done[num_streams];
    for (int i = 0; i < num_streams; i++) {
        CUDA_CHECK(cudaEventCreate(&chunk_done[i]));
    }
    
    // ===== 模式 A：同步基准（全部顺序执行） =====
    cudaEvent_t start_sync, stop_sync;
    CUDA_CHECK(cudaEventCreate(&start_sync));
    CUDA_CHECK(cudaEventCreate(&stop_sync));
    
    CUDA_CHECK(cudaEventRecord(start_sync, 0));
    for (int i = 0; i < num_streams; i++) {
        int offset = i * chunk_N;
        // 同步传输 H2D
        CUDA_CHECK(cudaMemcpy(&d_input[offset], &h_input_pinned[offset],
                              chunk_bytes, cudaMemcpyHostToDevice));
        // 计算
        compute_kernel<<<blocks_per_chunk, threads_per_block>>>(
            &d_input[offset], &d_output[offset], chunk_N);
        // 同步传输 D2H
        CUDA_CHECK(cudaMemcpy(&h_output_ref[offset], &d_output[offset],
                              chunk_bytes, cudaMemcpyDeviceToHost));
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaEventRecord(stop_sync, 0));
    CUDA_CHECK(cudaEventSynchronize(stop_sync));
    float sync_time_ms;
    CUDA_CHECK(cudaEventElapsedTime(&sync_time_ms, start_sync, stop_sync));
    
    // ===== 模式 B：全异步流水线 =====
    cudaEvent_t start_async, stop_async;
    CUDA_CHECK(cudaEventCreate(&start_async));
    CUDA_CHECK(cudaEventCreate(&stop_async));
    
    CUDA_CHECK(cudaEventRecord(start_async, 0));
    for (int i = 0; i < num_streams; i++) {
        int offset = i * chunk_N;
        
        // ① 将 H2D、计算、D2H 连续打入同一流（流内顺序执行）
        CUDA_CHECK(cudaMemcpyAsync(&d_input[offset], &h_input_pinned[offset],
                                    chunk_bytes, cudaMemcpyHostToDevice, streams[i]));
        compute_kernel<<<blocks_per_chunk, threads_per_block, 0, streams[i]>>>(
            &d_input[offset], &d_output[offset], chunk_N);
        CUDA_CHECK(cudaMemcpyAsync(&h_output_pinned[offset], &d_output[offset],
                                    chunk_bytes, cudaMemcpyDeviceToHost, streams[i]));
        
        // ② 在该流中记录一个“本chunk完成”事件
        CUDA_CHECK(cudaEventRecord(chunk_done[i], streams[i]));
    }
    // ③ 等待所有 chunk 完成（事件是跨流的同步点）
    for (int i = 0; i < num_streams; i++) {
        CUDA_CHECK(cudaEventSynchronize(chunk_done[i]));
    }
    CUDA_CHECK(cudaEventRecord(stop_async, 0));
    CUDA_CHECK(cudaEventSynchronize(stop_async));
    float async_time_ms;
    CUDA_CHECK(cudaEventElapsedTime(&async_time_ms, start_async, stop_async));
    
    // ===== 结果对比 =====
    printf("========== 双流水线 vs 全异步性能对比 ==========\n");
    printf("数据大小: %d 个 float (%.0f MB) × %d chunk\n",
           total_N, (float)total_N * sizeof(float) / (1024*1024), num_chunks);
    printf("同步串行耗时:           %8.3f ms\n", sync_time_ms);
    printf("异步多流耗时:           %8.3f ms\n", async_time_ms);
    printf("加速比:                 %8.2f x\n", sync_time_ms / async_time_ms);
    
    // -------- 验证正确性（与Python对比思路一致） --------
    int errors = 0;
    for (int i = 0; i < total_N; i++) {
        if (fabsf(h_output_pinned[i] - h_output_ref[i]) > 1e-4f) {
            errors++;
            if (errors <= 3) printf("错误[%d]: async=%f ref=%f\n", i,
                                      h_output_pinned[i], h_output_ref[i]);
        }
    }
    printf("正确性验证: %s (共 %d 个误差 > 1e-4)\n",
           errors == 0 ? "✓ 通过" : "✗ 失败", errors);
    
    // -------- 清理 --------
    CUDA_CHECK(cudaFreeHost(h_input_pinned));
    CUDA_CHECK(cudaFreeHost(h_output_pinned));
    free(h_output_ref);
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
    for (int i = 0; i < num_streams; i++) {
        CUDA_CHECK(cudaStreamDestroy(streams[i]));
        CUDA_CHECK(cudaEventDestroy(chunk_done[i]));
    }
    CUDA_CHECK(cudaEventDestroy(start_sync));
    CUDA_CHECK(cudaEventDestroy(stop_sync));
    CUDA_CHECK(cudaEventDestroy(start_async));
    CUDA_CHECK(cudaEventDestroy(stop_async));
    
    return 0;
}