// cuda_graph_capture.cu
#include <cuda_runtime.h>
#include <stdio.h>

__global__ void gelu_fused(const float *in, float *out, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    float x = in[idx];
    // GELU tanh近似融合在一个kernel中（避免多次launch）
    float x3 = x * x * x;
    float inner = sqrtf(2.0f / 3.14159265f) * (x + 0.044715f * x3);
    out[idx] = 0.5f * x * (1.0f + tanhf(inner));
}
__global__ void add_residual(const float *in, const float *res, float *out, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    out[idx] = in[idx] + res[idx];
}
__global__ void layer_norm_fused(const float *in, float *out, int rows, int cols, float eps) {
    // 简化版：仅scatter add部分逻辑示意，实际需实现统计量规约
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= rows * cols) return;
    out[idx] = in[idx]; // placeholder
}

#define CUDA_CHECK(c) do { cudaError_t e = c; \
    if (e != cudaSuccess) { printf("CUDA Error %s\n", cudaGetErrorString(e)); exit(1); } \
} while(0)

int main() {
    int N = 1 << 20;
    size_t size = N * sizeof(float);
    
    float *h_in = (float*)malloc(size);
    for (int i = 0; i < N; i++) h_in[i] = (float)(rand() % 100) / 100.0f;
    float *d_in, *d_res, *d_tmp, *d_out;
    CUDA_CHECK(cudaMalloc(&d_in,  size)); CUDA_CHECK(cudaMalloc(&d_res,  size));
    CUDA_CHECK(cudaMalloc(&d_tmp, size)); CUDA_CHECK(cudaMalloc(&d_out, size));
    CUDA_CHECK(cudaMemcpy(d_in, h_in, size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_res, h_in, size, cudaMemcpyHostToDevice));
    
    // ---------- 创建流并开始捕获 ----------
    cudaStream_t cap_stream;
    CUDA_CHECK(cudaStreamCreate(&cap_stream));
    
    // cudaStreamCaptureModeGlobal 阻止捕获期间任何其他流的操作混入
    CUDA_CHECK(cudaStreamBeginCapture(cap_stream, cudaStreamCaptureModeGlobal));
    
    // 写入一系列操作（此时不执行，仅记录）
    gelu_fused<<<4096, 256, 0, cap_stream>>>(d_in, d_tmp, N);
    // gelu 输出写到 d_tmp，下一 kernel 读取 d_tmp
    add_residual<<<4096, 256, 0, cap_stream>>>(d_tmp, d_res, d_out, N);
    
    cudaGraph_t graph;
    CUDA_CHECK(cudaStreamEndCapture(cap_stream, &graph));
    
    // ---------- 实例化图 ----------
    cudaGraphExec_t instance;
    CUDA_CHECK(cudaGraphInstantiate(&instance, graph, nullptr, nullptr, 0));
    
    // ---------- 重放与测量 ----------
    // 重放三次进行时间测量（取第三次）
    CUDA_CHECK(cudaGraphLaunch(instance, 0)); // 预冷
    CUDA_CHECK(cudaDeviceSynchronize());
    
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start)); CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start, 0));
    const int repeats = 1000;
    for (int r = 0; r < repeats; r++) {
        CUDA_CHECK(cudaGraphLaunch(instance, 0));
    }
    CUDA_CHECK(cudaEventRecord(stop, 0));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float t_graph_ms;
    CUDA_CHECK(cudaEventElapsedTime(&t_graph_ms, start, stop));
    
    // ---------- 对比：不使用graph ----------
    CUDA_CHECK(cudaEventRecord(start, 0));
    for (int r = 0; r < repeats; r++) {
        gelu_fused<<<4096, 256>>>(d_in, d_tmp, N);
        add_residual<<<4096, 256>>>(d_tmp, d_res, d_out, N);
    }
    CUDA_CHECK(cudaEventRecord(stop, 0));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float t_normal_ms;
    CUDA_CHECK(cudaEventElapsedTime(&t_normal_ms, start, stop));
    
    printf("CUDA Graph 性能对比（%d 次重放）:\n", repeats);
    printf("  不使用 Graph:         %.3f ms  (平均每次 %.3f us)\n",
           t_normal_ms, t_normal_ms / repeats * 1000.0f);
    printf("  使用 Graph  :         %.3f ms  (平均每次 %.3f us)\n",
           t_graph_ms,   t_graph_ms   / repeats * 1000.0f);
    printf("  加速比:               %.2f x\n", t_normal_ms / t_graph_ms);
    
    CUDA_CHECK(cudaGraphExecDestroy(instance));
    CUDA_CHECK(cudaGraphDestroy(graph));
    CUDA_CHECK(cudaStreamDestroy(cap_stream));
    CUDA_CHECK(cudaFree(d_in)); CUDA_CHECK(cudaFree(d_res));
    CUDA_CHECK(cudaFree(d_tmp)); CUDA_CHECK(cudaFree(d_out));
    free(h_in);
    return 0;
}