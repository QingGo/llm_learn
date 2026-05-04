// fused_gelu_kernel.cu
#include <torch/extension.h>
#include <cuda_runtime.h>

// GELU 近似: x * 0.5 * (1.0 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
__global__ void fused_gelu_kernel(const float *input, float *output, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        float x = input[idx];
        // GELU 的 tanh 近似
        float x3 = x * x * x;
        float inner = sqrtf(2.0f / M_PI) * (x + 0.044715f * x3);
        output[idx] = 0.5f * x * (1.0f + tanhf(inner));
    }
}

// 前向传播
torch::Tensor fused_gelu_forward(torch::Tensor input) {
    TORCH_CHECK(input.device().is_cuda(), "输入必须在 CUDA 设备上");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "输入必须是 float32");

    auto output = torch::empty_like(input);
    int N = input.numel();

    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    // 调用 CUDA kernel
    fused_gelu_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        N
    );

    // 检查 kernel 启动错误
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel 启动失败: ", cudaGetErrorString(err));

    return output;
}

// 反向传播
torch::Tensor fused_gelu_backward(torch::Tensor grad_output, torch::Tensor input) {
    TORCH_CHECK(input.device().is_cuda(), "输入必须在 CUDA 设备上");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "输入必须是 float32");

    auto grad_input = torch::empty_like(input);
    int N = input.numel();

    // 在 CPU 上计算 grad_input（生产代码应也在 GPU 上，此处为简洁）
    auto h_input = input.cpu();
    auto h_grad = grad_output.cpu();
    auto h_grad_input = grad_input.cpu();

    auto h_input_ptr = h_input.data_ptr<float>();
    auto h_grad_ptr = h_grad.data_ptr<float>();
    auto h_grad_input_ptr = h_grad_input.data_ptr<float>();

    for (int i = 0; i < N; i++) {
        float x = h_input_ptr[i];
        float x3 = x * x * x;
        float inner = sqrtf(2.0f / M_PI) * (x + 0.044715f * x3);
        float tanh_inner = tanhf(inner);
        float sech2 = 1.0f - tanh_inner * tanh_inner;
        float grad = 0.5f * (1.0f + tanh_inner)
                   + 0.5f * x * sech2 * sqrtf(2.0f / M_PI) * (1.0f + 3.0f * 0.044715f * x * x);
        h_grad_input_ptr[i] = h_grad_ptr[i] * grad;
    }

    grad_input.copy_(h_grad_input);
    return grad_input;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &fused_gelu_forward, "Fused GELU forward (CUDA)");
    m.def("backward", &fused_gelu_backward, "Fused GELU backward (CUDA)");
}