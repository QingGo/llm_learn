// layernorm_kernel.cu
#include <torch/extension.h>
#include <cuda_runtime.h>

// 融合 LayerNorm 前向传播
// 输入: [rows, cols]，沿最后一维（cols）归一化
__global__ void fused_layernorm_kernel(
    const float *input, const float *gamma, const float *beta,
    float *output, int rows, int cols, float eps) {

    // 每个线程块处理一行
    int row = blockIdx.x;
    if (row >= rows) return;

    // ---- 步骤 1: 计算均值（warp reduce）----
    float sum = 0.0f;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        sum += input[row * cols + i];
    }

    // 共享内存 warp reduce
    __shared__ float shared_sum[32];
    int lane = threadIdx.x % 32;
    int wid = threadIdx.x / 32;

    // warp 内规约
    for (int offset = 16; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }

    if (lane == 0) shared_sum[wid] = sum;
    __syncthreads();

    // 汇总各 warp 结果
    if (threadIdx.x < blockDim.x / 32) {
        sum = shared_sum[threadIdx.x];
        for (int offset = 16; offset > 0; offset /= 2) {
            sum += __shfl_down_sync(0xffffffff, sum, offset);
        }
    }

    // 广播均值到所有线程
    float mean = __shfl_sync(0xffffffff, sum, 0) / cols;

    // ---- 步骤 2: 计算方差 ----
    float var_sum = 0.0f;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        float diff = input[row * cols + i] - mean;
        var_sum += diff * diff;
    }

    for (int offset = 16; offset > 0; offset /= 2) {
        var_sum += __shfl_down_sync(0xffffffff, var_sum, offset);
    }

    if (lane == 0) shared_sum[wid] = var_sum;
    __syncthreads();

    if (threadIdx.x < blockDim.x / 32) {
        var_sum = shared_sum[threadIdx.x];
        for (int offset = 16; offset > 0; offset /= 2) {
            var_sum += __shfl_down_sync(0xffffffff, var_sum, offset);
        }
    }

    float variance = __shfl_sync(0xffffffff, var_sum, 0) / cols;
    float inv_std = rsqrtf(variance + eps);  // 1 / sqrt(var + eps)

    // ---- 步骤 3: 归一化 + 仿射变换 ----
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        float normalized = (input[row * cols + i] - mean) * inv_std;
        output[row * cols + i] = normalized * gamma[i] + beta[i];
    }
}

torch::Tensor fused_layernorm_forward(
    torch::Tensor input,
    torch::Tensor gamma,
    torch::Tensor beta,
    float eps = 1e-5f) {

    TORCH_CHECK(input.is_cuda(), "输入必须在 CUDA 设备上");
    TORCH_CHECK(input.dim() == 2, "输入必须是二维 [rows, cols]");

    int rows = input.size(0);
    int cols = input.size(1);

    auto output = torch::empty_like(input);

    // 每行一个 block，block 内线程协作处理该行
    int threads = min(256, ((cols + 31) / 32) * 32);  // 向上取整到 32 的倍数
    dim3 grid(rows);
    dim3 block(threads);

    fused_layernorm_kernel<<<grid, block>>>(
        input.data_ptr<float>(),
        gamma.data_ptr<float>(),
        beta.data_ptr<float>(),
        output.data_ptr<float>(),
        rows, cols, eps
    );

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &fused_layernorm_forward, "Fused LayerNorm forward (CUDA)");
}