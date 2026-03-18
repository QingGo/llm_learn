from typing import Tuple

import torch
from torch import nn


class MLA(nn.Module):
    def __init__(
        self, input_dim: int, head_dim: int, n_heads: int, lora_rank: int, rope_dim: int
    ):
        """
        简化版的 MLA（多头潜在注意力），相较原版忽略了以下逻辑：
        - YaRN 外推上下文长度的 RoPE
        - `q_lora_rank == 0` 的分支，总是对 q 也进行投影
        - `attn_impl == "naive"` 的分支，总是存储压缩后的 kv_cache/pe_cache
        - 不包含并行运算和混合精度逻辑
        - qkv 的使用统一的维度大小
        """
        super().__init__()
        self.input_dim = input_dim
        self.head_dim = head_dim
        self.n_heads = n_heads
        self.lora_rank = lora_rank
        self.rope_dim = rope_dim

        # kv projection
        self.w_dkv_kr = nn.Linear(input_dim, lora_rank + rope_dim)
        self.w_uk = nn.Linear(lora_rank, n_heads * head_dim)
        self.w_uv = nn.Linear(lora_rank, n_heads * head_dim)
        self.kv_norm = nn.RMSNorm(lora_rank)
        # q projection
        self.w_dq = nn.Linear(input_dim, lora_rank)
        self.w_uq = nn.Linear(lora_rank, n_heads * head_dim)
        self.w_qr = nn.Linear(lora_rank, rope_dim)
        self.q_norm = nn.RMSNorm(lora_rank)
        # 输出
        self.w_o = nn.Linear(n_heads * head_dim, input_dim)

        self.max_batch_size = 10
        self.max_seq_len = 1024
        self.register_buffer(
            "kv_cache",
            torch.zeros(self.max_batch_size, self.max_seq_len, self.lora_rank),
            persistent=False,
        )
        self.register_buffer(
            "pe_cache",
            torch.zeros(self.max_batch_size, self.max_seq_len, self.rope_dim),
            persistent=False,
        )

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        rope_freqs: Tuple[torch.Tensor, torch.Tensor],
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        x: (batch_size, seq_len, input_dim)
        """
        batch_size, seq_len, _ = x.shape
        end_pos = start_pos + seq_len
        # 计算 kv
        dkv_kr = self.w_dkv_kr(x)  # (batch_size, seq_len, lora_rank + rope_dim)
        c_kv, k_r_pre = torch.split(dkv_kr, [self.lora_rank, self.rope_dim], dim=-1)
        c_kv = self.kv_norm(c_kv)
        self.kv_cache[:batch_size, start_pos:end_pos] = c_kv
        # k/v k_multi/v_multi 先不算出，后面用矩阵结合律计算
        # k =  self.w_uk(c_kv) # (batch_size, seq_len, n_heads * head_dim)
        # k_multi = k.view(batch_size, seq_len, self.n_heads, self.head_dim)
        k_r = apply_rotary_emb(k_r_pre, rope_freqs)  # (batch_size, seq_len, rope_dim)
        self.pe_cache[:batch_size, start_pos:end_pos] = k_r
        # k_multi = torch.cat([k_multi, k_rope.unsqueeze(2)], dim=-1) # (batch_size, seq_len, n_heads, head_dim + rope_dim)
        # v = self.w_uv(c_kv)
        # v_multi = v.view(batch_size, seq_len, self.n_heads, self.head_dim)
        # 计算 q
        c_q = self.w_dq(x)  # (batch_size, seq_len, lora_rank)
        c_q = self.q_norm(c_q)
        q = self.w_uq(c_q)  # (batch_size, seq_len, n_heads * head_dim)
        q_multi = q.view(
            batch_size, seq_len, self.n_heads, self.head_dim
        )  # (batch_size, seq_len, n_heads, head_dim)
        q_r = self.w_qr(c_q)  # (batch_size, seq_len, rope_dim)
        q_r = apply_rotary_emb(q_r, rope_freqs)
        w_uk_reshaped = self.w_uk.weight.view(self.n_heads, self.head_dim, self.lora_rank)
        qW = torch.einsum("bshd,hdr->bshr", q_multi, w_uk_reshaped)
        attenion_scores = torch.einsum("bshr,btr->bsht", qW, self.kv_cache[:batch_size, :end_pos])

        rope_scores = torch.einsum("bsd,btd->bst", q_r, self.pe_cache[:batch_size, :end_pos])
        attenion_scores = attenion_scores + rope_scores.unsqueeze(2)
        if mask is not None:
            attenion_scores += mask.unsqueeze(1)
        # 计算输出
        a_sum = torch.einsum("bsht,btr->bshr", attenion_scores, self.kv_cache[:batch_size, :end_pos])
        w_uv_reshaped = self.w_uv.weight.view(self.n_heads, self.head_dim, self.lora_rank)
        output = torch.einsum("bshr,hdr->bshd", a_sum, w_uv_reshaped)
        output = output.flatten(2)  # (batch_size, seq_len, n_heads * head_dim)
        output = self.w_o(output)  # (batch_size, seq_len, input_dim)
        return output


class MHA(nn.Module):
    def __init__(self, input_dim: int, head_dim: int, n_heads: int, rope_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.head_dim = head_dim
        self.n_heads = n_heads
        self.rope_dim = rope_dim

        self.w_q = nn.Linear(input_dim, n_heads * head_dim)
        self.w_k = nn.Linear(input_dim, n_heads * head_dim)
        self.w_v = nn.Linear(input_dim, n_heads * head_dim)
        self.w_o = nn.Linear(n_heads * head_dim, input_dim)

        self.max_batch_size = 10
        self.max_seq_len = 1024
        self.register_buffer(
            "kv_cache_k",
            torch.zeros(self.max_batch_size, self.max_seq_len, self.n_heads * self.head_dim),
            persistent=False,
        )
        self.register_buffer(
            "kv_cache_v",
            torch.zeros(self.max_batch_size, self.max_seq_len, self.n_heads * self.head_dim),
            persistent=False,
        )

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        rope_freqs: Tuple[torch.Tensor, torch.Tensor],
        mask: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        end_pos = start_pos + seq_len
        q = self.w_q(x)
        k = self.w_k(x)
        v = self.w_v(x)
        q_multi = q.view(batch_size, seq_len, self.n_heads, self.head_dim)
        k_multi = k.view(batch_size, seq_len, self.n_heads, self.head_dim)
        v_multi = v.view(batch_size, seq_len, self.n_heads, self.head_dim)
        if self.rope_dim > 0:
            q_head = q_multi[..., :self.rope_dim]
            q_tail = q_multi[..., self.rope_dim:]
            k_head = k_multi[..., :self.rope_dim]
            k_tail = k_multi[..., self.rope_dim:]
            bh = batch_size * self.n_heads
            q_head = q_head.reshape(bh, seq_len, self.rope_dim)
            k_head = k_head.reshape(bh, seq_len, self.rope_dim)
            q_head = apply_rotary_emb(q_head, rope_freqs)
            k_head = apply_rotary_emb(k_head, rope_freqs)
            q_head = q_head.view(batch_size, seq_len, self.n_heads, self.rope_dim)
            k_head = k_head.view(batch_size, seq_len, self.n_heads, self.rope_dim)
            q_multi = torch.cat([q_head, q_tail], dim=-1)
            k_multi = torch.cat([k_head, k_tail], dim=-1)
        self.kv_cache_k[:batch_size, start_pos:end_pos] = k_multi.flatten(2)
        self.kv_cache_v[:batch_size, start_pos:end_pos] = v_multi.flatten(2)
        k_cached = self.kv_cache_k[:batch_size, :end_pos].view(batch_size, end_pos, self.n_heads, self.head_dim)
        v_cached = self.kv_cache_v[:batch_size, :end_pos].view(batch_size, end_pos, self.n_heads, self.head_dim)
        attenion_scores = torch.einsum("bshd,bthd->bsht", q_multi, k_cached)
        if mask is not None:
            attenion_scores += mask.unsqueeze(1)
        a_sum = torch.einsum("bsht,bthd->bshd", attenion_scores, v_cached)
        output = a_sum.flatten(2)
        output = self.w_o(output)
        return output

def precompute_freqs_cis(
    dim: int,
    seq_len: int,
    theta: float = 10000.0,
    device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    预计算旋转位置编码的频率张量

    Args:
        dim: 嵌入维度
        seq_len: 序列长度
        theta: 旋转位置编码的基数，默认为10000.0

    Returns:
        freqs_cis: 预计算的频率张量，形状为 (seq_len, dim)
    """
    # 计算每个维度的theta值
    # dim必须是偶数，因为旋转是2D分组的
    assert dim % 2 == 0, "dim must be even"

    # 生成 dim/2 个不同的频率：theta_i = theta ^ (-2i/d)
    half = dim // 2
    freqs = 1.0 / (theta ** (torch.arange(0, half, device=device).float() / half))

    # 生成位置索引m = 0,1,...,seq_len-1
    t = torch.arange(seq_len, device=device)

    # 计算m * theta_i，形状为 (seq_len, dim/2)
    freqs = torch.outer(t, freqs)

    # 计算 cos(m*theta_i) 与 sin(m*theta_i)
    freqs_cos = torch.cos(freqs.float())
    freqs_sin = torch.sin(freqs.float())

    # 返回半维度 cos/sin，形状为 (seq_len, dim/2)
    return freqs_cos.to(dtype=dtype), freqs_sin.to(dtype=dtype)


def apply_rotary_emb(
    x: torch.Tensor, freqs_cis: Tuple[torch.Tensor, torch.Tensor],
) -> torch.Tensor:
    """
    应用旋转位置编码到输入张量，原论文使用 “偶/奇交错维度”配对旋转，
    这里使用 “前半/后半” 配对旋转。和 Hugging Face 对齐

    Args:
        x: 输入张量，形状为 (batch_size, groups/heads, seq_len, head_dim)
        freqs_cis: 预计算的频率 cos/sin 张量，形状为 ((seq_len//2, dim),(seq_len//2, dim)) 

    Returns:
        x_rotated: 应用旋转编码后的张量，形状与输入相同
    """
    # 确保输入维度是偶数
    cos_half, sin_half = freqs_cis
    assert x.shape[-1] % 2 == 0, "x dimension must be even"

    orig_dtype = x.dtype
    x_fp32 = x.float()
    cos_full = torch.cat([cos_half.float(), cos_half.float()], dim=-1)
    sin_full = torch.cat([sin_half.float(), sin_half.float()], dim=-1)

    # 旋转公式：x * cos + rotate_half(x) * sin
    half = x_fp32.shape[-1] // 2
    x1 = x_fp32[..., :half]
    x2 = x_fp32[..., half:]
    rotated = torch.cat([-x2, x1], dim=-1)
    x_rotated = x_fp32 * cos_full + rotated * sin_full

    return x_rotated.to(orig_dtype)

if __name__ == '__main__':
    seq_len=1024
    rope_dim=64
    input_dim=4096
    input = torch.rand(2, seq_len, input_dim)
    cos_freq_cis, sin_freq_cis = precompute_freqs_cis(dim=rope_dim, seq_len=seq_len)
    model = MLA(input_dim=input_dim, n_heads = 8, head_dim = 512, lora_rank = 512, rope_dim = rope_dim)
    output = model(input, start_pos = 0, rope_freqs = (cos_freq_cis, sin_freq_cis), mask = None)
    print(output.shape)

    mha = MHA(input_dim=input_dim, n_heads=8, head_dim=512, rope_dim=rope_dim)
    _ = mha(input, start_pos=0, rope_freqs=(cos_freq_cis, sin_freq_cis), mask=None)

    b = input.shape[0]
    s = seq_len
    mla_bytes = (
        model.kv_cache[:b, :s].numel() * model.kv_cache.element_size()
        + model.pe_cache[:b, :s].numel() * model.pe_cache.element_size()
    )
    mha_bytes = (
        mha.kv_cache_k[:b, :s].numel() * mha.kv_cache_k.element_size()
        + mha.kv_cache_v[:b, :s].numel() * mha.kv_cache_v.element_size()
    )
    def to_mib(x: int) -> float:
        return x / (1024 * 1024)
    print(f"MLA KV+PE Cache: {to_mib(mla_bytes):.2f} MiB")
    print(f"MHA K+V Cache: {to_mib(mha_bytes):.2f} MiB")
    if mha_bytes > 0:
        print(f"Ratio (MLA/MHA): {mla_bytes / mha_bytes:.4f}")
    '''
    torch.Size([2, 1024, 4096])
    MLA KV+PE Cache: 4.50 MiB
    MHA K+V Cache: 64.00 MiB
    Ratio (MLA/MHA): 0.0703
    '''