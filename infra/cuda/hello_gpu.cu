#include <cuda_runtime.h>
#include <stdio.h>

// __global__ 表明这是一个在 Device 上执行、由 Host 调用的函数
__global__ void hello_from_gpu() {
    // 使用内置变量打印当前线程的坐标
    printf("Hello from block (%d,%d,%d), thread (%d,%d,%d)!\n",
           blockIdx.x, blockIdx.y, blockIdx.z,
           threadIdx.x, threadIdx.y, threadIdx.z);
}

int main() {
    // 启动 2 个线程块，每个块 4 个线程
    // <<<网格, 块>>> 的语法称为"执行配置"
    dim3 grid(2, 1, 1);   // gridDim.x = 2
    dim3 block(4, 1, 1);  // blockDim.x = 4
    hello_from_gpu<<<grid, block>>>();

    // Host 等待 Device 完成所有 kernel 执行
    cudaDeviceSynchronize();

    // 检查是否有启动错误
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
    }

    return 0;
}