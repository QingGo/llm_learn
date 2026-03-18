import math
from typing import Optional

import torch
import torch.nn as nn

from transformer.model import AttentionUnit
from .config import GPTConfig


class GELUMLP(nn.Module):
    """使用 GELU 激活与 Dropout 的两层前馈网络"""
    def __init__(self, d_model: int, d_hidden: int, dropout: float):
        super().__init__()
        # GPT-1/GPT-2 使用 GELU 替代原始 Transformer 的 ReLU
        self.fc1 = nn.Linear(d_model, d_hidden)
        self.act = nn.GELU()
        self.drop1 = nn.Dropout(p=dropout)
        self.fc2 = nn.Linear(d_hidden, d_model)
        self.drop2 = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class GPTBlock(nn.Module):
    """Pre-LN 自注意力 + MLP 的 GPT 模块"""
    def __init__(self, d_model: int, seq_len: int, n_heads: int, d_hidden: int, dropout: float):
        super().__init__()
        # Pre-LN：LayerNorm 位于子块输入处（GPT-2），区别于原始 Transformer 的 Post-LN
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = AttentionUnit(d_model, seq_len, n_heads, use_mask=True)
        self.drop_attn = nn.Dropout(p=dropout)
        # Decoder-only：仅自注意力，且使用因果掩码保证自回归
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = GELUMLP(d_model, d_hidden, dropout)
        self.drop_ffn = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor, padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        a = self.attn(self.ln1(x), self.ln1(x), self.ln1(x), key_padding_mask=padding_mask)
        a = self.drop_attn(a)
        x = x + a
        m = self.mlp(self.ln2(x))
        m = self.drop_ffn(m)
        x = x + m
        return x


class GPT(nn.Module):
    """GPT-2 风格的 Decoder-only Transformer（学习式位置嵌入、Pre-LN、权重共享）"""
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        self.d_model = config.d_model
        self.seq_len = config.seq_len
        self.n_heads = config.n_heads
        self.d_hidden = config.d_hidden
        self.stack = config.stack
        # 学习式位置嵌入（GPT-1/GPT-2），区别于原始 Transformer 的正弦位置编码
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_embedding = nn.Embedding(config.seq_len, config.d_model)
        self.embed_dropout = nn.Dropout(p=config.dropout)
        self.blocks = nn.ModuleList([
            GPTBlock(config.d_model, config.seq_len, config.n_heads, config.d_hidden, config.dropout)
            for _ in range(config.stack)
        ])
        # 额外的最终 LayerNorm（GPT-2 在最后自注意力块后增加的 ln_f）
        self.ln_f = nn.LayerNorm(config.d_model)
        # 输出头与词嵌入权重共享（weight tying），与论文一致
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight
        self._init_weights()
        self._scale_residual_outputs()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Embedding)):
                # 论文中使用 N(0, 0.02) 的简单初始化
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def _scale_residual_outputs(self):
        if self.stack > 0:
            scale = 1.0 / math.sqrt(self.stack)
            for block in self.blocks:
                # GPT-2：残差路径的输出线性按 1/√N 缩放，以控制随深度的累积
                nn.init.constant_(block.attn.w_o.weight, 0.0)
                nn.init.normal_(block.attn.w_o.weight, mean=0.0, std=0.02)
                block.attn.w_o.weight.data.mul_((scale))
                nn.init.constant_(block.mlp.fc2.weight, 0.0)
                nn.init.normal_(block.mlp.fc2.weight, mean=0.0, std=0.02)
                block.mlp.fc2.weight.data.mul_(scale)

    def forward(self, x: torch.Tensor, y: torch.Tensor, x_padding_mask: Optional[torch.Tensor] = None, y_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """前向传播，保持与现有 Transformer 接口兼容，仅使用 y 与其 padding 掩码"""
        bsz, seqlen = y.size()
        pos_ids = torch.arange(0, seqlen, device=y.device).unsqueeze(0).expand(bsz, seqlen)
        x_emb = self.token_embedding(y) + self.pos_embedding(pos_ids)
        x_emb = self.embed_dropout(x_emb)
        for blk in self.blocks:
            x_emb = blk(x_emb, y_padding_mask)
        # GPT-2：在所有块之后再做一次 LayerNorm
        x_emb = self.ln_f(x_emb)
        logits = self.lm_head(x_emb)
        return logits

    def generate(self, input_ids: torch.Tensor, max_new_tokens: int, pad_token_id: int = 50256, eos_token_id: int = 50256, temperature: float = 1.0, top_k: Optional[int] = None) -> torch.Tensor:
        """自回归采样生成（支持温度与 Top-K）"""
        self.eval()
        with torch.no_grad():
            for _ in range(max_new_tokens):
                cur = input_ids[:, -self.seq_len:]
                # 使用因果掩码与 padding 掩码，保持自回归与合法注意力范围
                padding_mask = (cur == pad_token_id)
                logits = self.forward(cur, cur, None, padding_mask)
                logits = logits[:, -1, :] / max(temperature, 1e-5)
                if top_k is not None and top_k > 0:
                    values, _ = torch.topk(logits, top_k)
                    min_values = values[:, -1].unsqueeze(-1)
                    logits = torch.where(logits < min_values, torch.full_like(logits, -float('Inf')), logits)
                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                input_ids = torch.cat([input_ids, next_token], dim=1)
                if (next_token.squeeze(-1) == eos_token_id).all():
                    break
        return input_ids