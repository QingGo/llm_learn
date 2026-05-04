// coalescing_demo.cu
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

// 合并访问：按行读取，按行写入
__global__ void copyRowMajor(const float *in, float *out, int rows, int cols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < rows && col < cols) {
        out[row * cols + col] = in[row * cols + col];
    }
}

// 非合并访问：按行读取，按列写入（实际上写入是非合并的）
__global__ void copyColumnWrite(const float *in, float *out, int rows, int cols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < rows && col < cols) {
        // 写入 out 时按列索引，同一 warp 的线程写入到相距 rows*4 字节的地址
        out[col * rows + row] = in[row * cols + col];
    }
}

// 矩阵转置：读取合并，写入非合并（经典场景）
__global__ void transposeNaive(const float *in, float *out, int rows, int cols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < rows && col < cols) {
        out[col * rows + row] = in[row * cols + col];
    }
}

// 计时辅助
float measureKernel(void (*kernel)(const float*, float*, int, int),
                    dim3 grid, dim3 block,
                    const float *d_in, float *d_out, int rows, int cols) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    kernel<<<grid, block>>>(d_in, d_out, rows, cols);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return ms;
}

int main() {
    int rows = 4096, cols = 4096;
    size_t size = rows * cols * sizeof(float);

    float *h_in = (float*)malloc(size);
    float *h_out = (float*)malloc(size);

    for (int i = 0; i < rows * cols; i++)
        h_in[i] = (float)(rand() % 1000) / 1000.0f;

    float *d_in, *d_out;
    cudaMalloc(&d_in, size);
    cudaMalloc(&d_out, size);
    cudaMemcpy(d_in, h_in, size, cudaMemcpyHostToDevice);

    dim3 block(16, 16);
    dim3 grid((cols + 15) / 16, (rows + 15) / 16);

    // 预热
    copyRowMajor<<<grid, block>>>(d_in, d_out, rows, cols);
    cudaDeviceSynchronize();

    // 测量
    float t_row = measureKernel(copyRowMajor, grid, block, d_in, d_out, rows, cols);
    float t_col = measureKernel(copyColumnWrite, grid, block, d_in, d_out, rows, cols);
    float t_trans = measureKernel(transposeNaive, grid, block, d_in, d_out, rows, cols);

    printf("=== 合并访问性能对比 (4096x4096) ===\n");
    printf("CopyRowMajor    (读写均合并):    %.3f ms\n", t_row);
    printf("CopyColumnWrite (读合并, 写非合并): %.3f ms  (慢 %.1fx)\n",
           t_col, t_col / t_row);
    printf("TransposeNaive  (读合并, 写非合并): %.3f ms  (慢 %.1fx)\n",
           t_trans, t_trans / t_row);

    // 验证正确性
    copyRowMajor<<<grid, block>>>(d_in, d_out, rows, cols);
    cudaMemcpy(h_out, d_out, size, cudaMemcpyDeviceToHost);
    int errors = 0;
    for (int i = 0; i < rows * cols; i++) {
        if (fabs(h_in[i] - h_out[i]) > 1e-5) { errors++; break; }
    }
    printf("CopyRowMajor 正确性: %s\n", errors ? "失败" : "通过");

    cudaFree(d_in); cudaFree(d_out);
    free(h_in); free(h_out);
    return 0;
}