import torch
import torch.nn.functional as F
from fused_gelu_warp import fused_gelu_function, layer_norm_function
import time

x = torch.randn(4096, 4096, device='cuda')

# 预热
for _ in range(10):
    _ = F.gelu(x, approximate='tanh')
    _ = fused_gelu_function(x)
torch.cuda.synchronize()

# PyTorch 原生
start = time.time()
for _ in range(100):
    _ = F.gelu(x, approximate='tanh')
torch.cuda.synchronize()
t_pytorch = time.time() - start

# 融合版本
start = time.time()
for _ in range(100):
    _ = fused_gelu_function(x)
torch.cuda.synchronize()
t_fused = time.time() - start

print(f"PyTorch GELU: {t_pytorch*10:.2f} ms")
print(f"Fused  GELU:  {t_fused*10:.2f} ms")
print(f"加速: {t_pytorch/t_fused:.2f}x")


x = torch.randn(4096, 4096, device='cuda')

# 预热
for _ in range(10):
    _ = F.layer_norm(x, normalized_shape=(4096,))
    _ = layer_norm_function(x, gamma=torch.ones(4096, device='cuda'), beta=torch.zeros(4096, device='cuda'))
torch.cuda.synchronize()

# PyTorch 原生
start = time.time()
for _ in range(100):
    _ = F.layer_norm(x, normalized_shape=(4096,))
torch.cuda.synchronize()
t_pytorch = time.time() - start

# 融合版本
start = time.time()
for _ in range(100):
    _ = layer_norm_function(x, gamma=torch.ones(4096, device='cuda'), beta=torch.zeros(4096, device='cuda'))
torch.cuda.synchronize()
t_fused = time.time() - start

print(f"PyTorch LayerNorm: {t_pytorch*10:.2f} ms")
print(f"Fused  LayerNorm:  {t_fused*10:.2f} ms")
print(f"加速: {t_pytorch/t_fused:.2f}x")
