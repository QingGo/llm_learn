import torch
from torch import nn
import torch.nn.functional as F
from typing import Tuple

class MLP(nn.Module):
    """
    对应官方代码中的 Expert/MLP 类。
    显式拆分 w1, w2, w3，符合 DeepSeek/LLaMA 的 SwiGLU 定义。
    """
    def __init__(self, dim: int, inter_dim: int):
        super().__init__()
        self.w1 = nn.Linear(dim, inter_dim, bias=False)  # Gate projection
        self.w2 = nn.Linear(inter_dim, dim, bias=False)  # Down projection
        self.w3 = nn.Linear(dim, inter_dim, bias=False)  # Up projection

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SwiGLU: w2( SiLU(w1(x)) * w3(x) )
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

class Gate(nn.Module):
    """
    对应官方代码中的 Gate 类。
    实现了 DeepSeek-V3 的 Sigmoid + Bias (Load Balancing) 逻辑。
    """
    def __init__(self, num_routed: int, dim: int, topk: int, route_scale: float = 1.0):
        super().__init__()
        self.num_routed = num_routed
        self.dim = dim
        self.topk = topk
        self.route_scale = route_scale
        
        # 路由权重矩阵
        self.weight = nn.Parameter(torch.empty(num_routed, dim))
        # 负载均衡 Bias (DeepSeek-V3 核心：用于调整选专家的倾向，但不直接加在权重上)
        self.bias = nn.Parameter(torch.zeros(num_routed))
        

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: [batch, seq, dim] -> [batch*seq, dim] 以简化计算
        # 1. 计算原始分数 (Sigmoid)
        logits = F.linear(x, self.weight)
        scores = logits.sigmoid()
        
        # 2. 备份原始分数用于最终权重计算
        original_scores = scores

        # 3. 加上 Bias 进行 TopK 选择 (Auxiliary-Loss-Free Load Balancing)
        # Bias 帮助负载均衡，但我们希望专家输出的加权系数反映真实的匹配度
        scores_for_selection = scores + self.bias
        
        # 4. TopK 选择
        indices = torch.topk(scores_for_selection, self.topk, dim=-1)[1] # [batch, seq, topk]
        
        # 5. 收集对应的权重 (从无 Bias 的 original_scores 中取)
        weights = original_scores.gather(2, indices)
        
        # 6. 归一化 (Renormalize)
        weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-20)
        
        # 7. 应用 Route Scale (DeepSeek 特有的缩放因子)
        weights = weights * self.route_scale
        
        return weights, indices

class SparseMoE(nn.Module):
    def __init__(
        self,
        dim: int,
        moe_inter_dim: int,   # 路由专家的中间维度
        shared_inter_dim: int, # 共享专家的中间维度 (通常很大)
        num_routed: int,
        topk: int,
        route_scale: float = 1.0,
    ):
        super().__init__()
        self.dim = dim
        self.num_routed = num_routed
        
        # 1. 共享专家 (Shared Experts)
        # 修改点：将多个共享专家合并为一个巨大的 MLP，效率更高
        self.shared_experts = MLP(dim, shared_inter_dim)
        
        # 2. 路由专家 (Routed Experts)
        self.experts = nn.ModuleList([
            MLP(dim, moe_inter_dim) for _ in range(num_routed)
        ])
        
        # 3. 门控 (Gate)
        self.gate = Gate(num_routed, dim, topk, route_scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [batch, seq_len, dim]
        """
        # 保存原始形状
        original_shape = x.shape
        identity = x  # 这里的 identity 是 (2, 8, 128)
        
        # 扁平化处理，方便索引 [batch*seq, dim]
        x_flat = x.view(-1, self.dim)
        
        # --- Shared Expert 路径 ---
        z_shared = self.shared_experts(x_flat)
        
        # --- Routed Expert 路径 ---
        weights, indices = self.gate(x) 
        
        # 扁平化路由参数
        weights = weights.view(-1, weights.size(-1)) 
        indices = indices.view(-1, indices.size(-1)) 
        
        y_routed = torch.zeros_like(x_flat)
        
        counts = torch.bincount(indices.flatten(), minlength=self.num_routed).tolist()
        
        for i in range(self.num_routed):
            if counts[i] == 0:
                continue
            
            expert = self.experts[i]
            row_idx, col_idx = torch.where(indices == i)
            
            if row_idx.numel() > 0:
                expert_input = x_flat[row_idx]
                expert_output = expert(expert_input)
                current_weights = weights[row_idx, col_idx].unsqueeze(1)
                y_routed.index_add_(0, row_idx, expert_output * current_weights)

        # --- 最终融合 ---
        # output 目前是扁平化的 (16, 128)
        output = z_shared + y_routed
        
        # 【修改点在此】：先将 output 变回 (2, 8, 128)，再与 identity 相加
        output = output.view(original_shape)
        
        return output + identity

if __name__ == "__main__":
    torch.manual_seed(42)
    
    # 超参数
    B, T, D = 2, 8, 128
    num_routed = 8
    topk = 2
    moe_inter_dim = 64
    # 假设有 2 个共享专家，每个维度 64 -> 合并为维度 128
    shared_inter_dim = 128 
    
    model = SparseMoE(
        dim=D,
        moe_inter_dim=moe_inter_dim,
        shared_inter_dim=shared_inter_dim,
        num_routed=num_routed,
        topk=topk,
        route_scale=1.0
    )
    
    x = torch.randn(B, T, D)
    out = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")
    print("Run successfully.")