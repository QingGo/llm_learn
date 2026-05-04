// occupancy_demo.cu
#include <cuda_runtime.h>
#include <stdio.h>

__global__ void dummyKernel(float *data, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) data[idx] *= 2.0f;
}

int main() {
    int blockSize, minGridSize;
    int device;
    cudaGetDevice(&device);

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);

    // 使用 Occupancy API 自动计算推荐 block 大小
    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize,
                                        dummyKernel, 0, 0);

    printf("GPU: %s (Compute Capability %d.%d)\n",
           prop.name, prop.major, prop.minor);
    printf("每个 SM 最大线程数: %d\n", prop.maxThreadsPerMultiProcessor);
    printf("每个 SM 最大 warp 数: %d\n", prop.maxThreadsPerMultiProcessor / 32);
    printf("\n推荐 block 大小: %d\n", blockSize);
    printf("每个 SM 最大驻留 block 数: %d\n", prop.multiProcessorCount);

    return 0;
}