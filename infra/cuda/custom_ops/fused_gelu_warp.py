# fused_gelu.py
import torch
import fused_gelu  # 通过 setup.py 构建后自动生成的模块
import layernorm_kernel

class FusedGELU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return fused_gelu.forward(input)

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        return fused_gelu.backward(grad_output, input)

def fused_gelu_function(input):
    return FusedGELU.apply(input)

class LayerNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, gamma, beta, eps = 1e-5):
        ctx.save_for_backward(input, gamma, beta, eps)
        return layernorm_kernel.forward(input, gamma, beta, eps)


def layer_norm_function(input, gamma, beta, eps = 1e-5):
    return LayerNorm.apply(input, gamma, beta, eps)
