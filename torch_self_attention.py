import torch
import torch.nn as nn
import math

class SingleHeadAttention(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        # d_model：输入输出特征维度（单头中d_k=d_model）
        self.d_model = d_model
        self.d_k = d_model  # 单头注意力中，K的维度等于d_model
        
        # Q、K、V的线性变换层（输出维度均为d_model）
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        
        # 可选：注意力权重dropout层
        self.dropout = nn.Dropout(p=0.1)
    
    def forward(self, q_input, k_input, v_input, mask=None):
        # 1. 线性变换得到Q、K、V
        # q_input/k_input/v_input: [batch_size, seq_len, d_model]
        Q = self.w_q(q_input)  # [bs, seq_len_q, d_k]
        K = self.w_k(k_input)  # [bs, seq_len_k, d_k]
        V = self.w_v(v_input)  # [bs, seq_len_v, d_k]
        
        # 2. 计算注意力分数：Q*K^T / √d_k
        attention_scores = torch.matmul(Q, K.transpose(-1, -2))  # [bs, seq_len_q, seq_len_k]
        attention_scores = attention_scores / math.sqrt(self.d_k)  # 缩放
        
        # 3. 应用掩码（可选）
        if mask is not None:
            # mask形状需与attention_scores匹配：[bs, 1, seq_len_k]或[bs, seq_len_q, seq_len_k]
            # 掩码值为-∞的位置，softmax后权重趋近于0
            attention_scores = attention_scores.masked_fill(mask == 0, -1e9)
        
        # 4. 计算注意力权重（softmax+dropout）
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)  # [bs, seq_len_q, seq_len_k]
        attention_probs = self.dropout(attention_probs)  # 应用dropout
        
        # 5. 加权求和得到注意力输出
        output = torch.matmul(attention_probs, V)  # [bs, seq_len_q, d_k]
        
        return output, attention_probs

if __name__ == "__main__":
    # 1. 初始化单头注意力（d_model=768)
    attention = SingleHeadAttention(d_model=768)

    # 2. 构造输入（batch_size=2, seq_len=5）
    q_input = torch.randn(2, 5, 768)  # 查询输入
    k_input = torch.randn(2, 5, 768)  # 键输入（seq_len与查询一致）
    v_input = torch.randn(2, 5, 768)  # 值输入（seq_len与键一致）

    # 3. 前向传播
    output, attn_probs = attention(q_input, k_input, v_input)

    # 4. 验证维度
    print("输出维度（应与输入一致：[2,5,768]）:", output.shape)
    print("注意力权重维度（应：[2,5,5]）:", attn_probs.shape)

    # 预期输出：
    # 输出维度（应与输入一致：[2,5,768]）: torch.Size([2, 5, 768])
    # 注意力权重维度（应：[2,5,5]）: torch.Size([2, 5, 5])