import torch
import torch.nn as nn
import math
from typing import Optional


class AttentionUnit(nn.Module):
    def __init__(self, d_model: int, seq_len: int, n_heads: int = 1, use_mask=False):
        super().__init__()
        # d_model：输入输出特征维度（单头中d_k=d_model）
        self.d_model = d_model
        self.n_heads = n_heads
        self.use_mask = use_mask
        # seq_len 仅用于初始化时的参考，前向中改为动态长度
        self.seq_len = seq_len
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        # Scaled Dot-Product Attention，里 d_q 一定等于 d_k，Additive attention 则不一定。3.2.1
        self.d_q = d_model // n_heads
        self.d_k = d_model // n_heads
        # d_v 不一定等于 d_q/d_k. Table 3 rows (B),
        self.d_v = d_model // n_heads

        # Q、K、V的线性变换层（输出维度均为d_model），
        # 多头也是先算出一个总的，需要后续通过 view + transpose 拆分出各个头的 Q/K/V
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)

        self.w_o = nn.Linear(d_model, d_model)

        # 可选：注意力权重dropout层
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, q_input, k_input, v_input, key_padding_mask: Optional[torch.Tensor] = None):
        # 输入验证与动态长度
        batch_size, q_len, q_dim = q_input.shape
        k_len = k_input.size(1)
        v_len = v_input.size(1)
        # d_model 验证
        assert q_dim == self.d_model and k_input.size(-1) == self.d_model and v_input.size(-1) == self.d_model, "Input d_model must match module d_model"
        # K/V 长度匹配验证
        assert k_len == v_len, "Key and Value sequence lengths must match"

        # 1. 线性变换得到Q、K、V
        # q_input/k_input/v_input: [batch_size, seq_len, d_model]
        Q = self.w_q(q_input)  # [bs, seq_len, d_model]
        K = self.w_k(k_input)  # [bs, seq_len, d_model]
        V = self.w_v(v_input)  # [bs, seq_len, d_model]

        # 动态按当前序列长度拆分多头
        Q_multi = Q.view(batch_size, q_len, self.n_heads, self.d_q).transpose(
            1, 2
        )  # [bs, n_heads, seq_len, d_q]
        K_multi = K.view(batch_size, k_len, self.n_heads, self.d_k).transpose(
            1, 2
        )  # [bs, n_heads, seq_len, d_k]
        V_multi = V.view(batch_size, v_len, self.n_heads, self.d_v).transpose(
            1, 2
        )  # [bs, n_heads, seq_len, d_v]

        # 2. 计算注意力分数：Q*K^T / √d_k
        attention_scores = torch.matmul(
            Q_multi, K_multi.transpose(-1, -2)
        )  # [bs, n_heads, seq_len, seq_len]
        attention_scores = attention_scores / math.sqrt(self.d_k)  # 缩放

        # 3. 应用掩码，Masked Multi-Head Attention 中使用
        if self.use_mask:
            # 创建因果掩码，确保维度匹配（q_len x k_len）
            i = torch.arange(q_len, device=q_input.device)
            j = torch.arange(k_len, device=q_input.device)
            mask = (j.unsqueeze(0) > i.unsqueeze(1))  # [q_len, k_len]
            # 扩展到batch和head维度
            mask = mask.unsqueeze(0).unsqueeze(0)  # [1, 1, q_len, k_len]
            attention_scores = attention_scores.masked_fill(mask, float("-inf"))
            
        if key_padding_mask is not None:
            # key_padding_mask: [bs, k_len] -> [bs, 1, 1, k_len]
            key_padding_mask = key_padding_mask.unsqueeze(1).unsqueeze(2)
            attention_scores = attention_scores.masked_fill(key_padding_mask, float("-inf"))

        # 4. 计算注意力权重（softmax+dropout），对每个 Q 关联的所有 K 的注意力分数进行归一化
        # 数值稳定性：检查是否所有值都是-inf
        attention_probs = torch.where(
            torch.all(torch.isinf(attention_scores)), 
            torch.ones_like(attention_scores) / attention_scores.size(-1),
            torch.softmax(attention_scores, dim=-1)
        ) # [bs, n_heads, seq_len, seq_len] 
        attention_probs = self.dropout(attention_probs)  # 应用dropout

        # 5. 加权求和得到注意力输出
        output = torch.matmul(attention_probs, V_multi)  # [bs, n_heads, q_seq_len, d_v]

        # 6. 合并多头输出
        output = (
            output.transpose(1, 2).contiguous().view(batch_size, q_len, self.d_model)
        )  # [bs, q_len, d_model]

        # 7. 输出线性变换
        output = self.w_o(output)  # [bs, seq_len, d_model]

        return output


class FFN(nn.Module):
    def __init__(self, d_model, d_hidden):
        super().__init__()
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_hidden),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(d_hidden, d_model)
        )

    def forward(self, x):
        return self.ffn(x)


class Encoder(nn.Module):
    def __init__(self, d_model: int, seq_len: int, n_heads: int = 1, d_hidden=2048):
        super().__init__()
        self.self_attention = AttentionUnit(d_model, seq_len, n_heads)
        self.ffn = FFN(d_model, d_hidden)
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x, key_padding_mask=None):
        res_1 = x
        x = self.self_attention(x, x, x, key_padding_mask)
        x = self.dropout(x)
        x = x + res_1  # [bs, seq_len, d_model]
        x = self.layer_norm1(x)
        res_2 = x
        x = self.ffn(x)
        x = self.dropout(x)
        x = x + res_2  # [bs, seq_len, d_model]
        x = self.layer_norm2(x)
        return x


class Decoder(nn.Module):
    def __init__(self, d_model: int, seq_len: int, n_heads: int = 1, d_hidden=2048):
        super().__init__()
        self.masked_attention = AttentionUnit(d_model, seq_len, n_heads, use_mask=True)
        self.attention = AttentionUnit(d_model, seq_len, n_heads)
        self.ffn = FFN(d_model, d_hidden)
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)
        self.layer_norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x, k, v, tgt_padding_mask=None, src_padding_mask=None):
        res_1 = x
        # 自注意力使用目标序列的 padding 掩码
        x = self.masked_attention(x, x, x, tgt_padding_mask)
        x = self.dropout(x)
        x = x + res_1  # [bs, seq_len, d_model]
        x = self.layer_norm1(x)
        res_2 = x
        # 交叉注意力使用源序列的 padding 掩码
        x = self.attention(x, k, v, src_padding_mask)
        x = self.dropout(x)
        x = x + res_2
        x = self.layer_norm2(x)
        res_3 = x
        x = self.ffn(x)
        x = self.dropout(x)
        x = x + res_3  # [bs, seq_len, d_model]
        x = self.layer_norm3(x)
        return x


class Transformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        seq_len: int,
        n_heads: int = 1,
        d_hidden=2048,
        stack: int = 6,
        use_pos_encoding_cache = True
    ):
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.n_heads = n_heads
        self.d_hidden = d_hidden
        self.use_pos_encoding_cache = use_pos_encoding_cache
        self.encoder = nn.ModuleList(
            [Encoder(d_model, seq_len, n_heads, d_hidden) for _ in range(stack)]
        )
        self.decoder = nn.ModuleList(
            [Decoder(d_model, seq_len, n_heads, d_hidden) for _ in range(stack)]
        )
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.embed_scale = math.sqrt(d_model)
        # 使用线性层作为输出投影，并与 embedding 权重共享（weight tying）
        self.output_linear = nn.Linear(d_model, vocab_size, bias=False)
        self.output_linear.weight = self.embedding.weight
        
        # 位置编码缓存（默认开启；不持久化，避免写入state_dict）
        self.register_buffer('pos_encoding_cache', None, persistent=False)
        self.cached_seq_len = 0

        # embedding dropout
        self.decoder_embed_dropout = nn.Dropout(p=0.1)
        self.encoder_embed_dropout = nn.Dropout(p=0.1)

    def forward(self, x, y, x_padding_mask=None, y_padding_mask=None):
        x = self.embedding(x)  # [bs, seq_len, d_model]
        x = x * self.embed_scale
        # 使用实际序列长度的位置信息
        x = x + self._pos_encoding(x.size(1), device=x.device)
        x = self.encoder_embed_dropout(x)
        for encoder in self.encoder:
            x = encoder(x, x_padding_mask)
        y = self.embedding(y)  # [bs, seq_len, d_model]
        y = y * self.embed_scale
        # 使用实际序列长度的位置信息（修复设备问题）
        y = y + self._pos_encoding(y.size(1), device=y.device)
        y = self.decoder_embed_dropout(y)
        for decoder in self.decoder:
            y = decoder(y, x, x, tgt_padding_mask=y_padding_mask, src_padding_mask=x_padding_mask)  # [bs, seq_len, d_model]
        logits = self.output_linear(y)  # [bs, seq_len, vocab_size]
        return logits

    def _pos_encoding_no_cache(self, seq_len, device):
        # 位置编码计算
        pos = torch.arange(0, seq_len, dtype=torch.float32, device=device).unsqueeze(1)

        # 修复位置编码计算
        i = torch.arange(0, self.d_model, step=2, dtype=torch.float32, device=device)
        angle = pos / torch.pow(10000, 2 * i / self.d_model)  # 修复公式
        
        pos_encoding = torch.zeros(seq_len, self.d_model, device=device)
        pos_encoding[:, 0::2] = torch.sin(angle)
        
        # 处理奇数d_model的情况
        if self.d_model % 2 == 1:
            pos_encoding[:, 1::2] = torch.cos(angle[:, :-1])
        else:
            pos_encoding[:, 1::2] = torch.cos(angle)
        
        return pos_encoding

    def _pos_encoding(self, seq_len, device):
        if self.use_pos_encoding_cache:
            if seq_len > self.cached_seq_len:
                pos_encoding = self._pos_encoding_no_cache(seq_len, device)
                # 已注册为buffer，直接赋值更新其内容，保持非持久化属性
                self.pos_encoding_cache = pos_encoding
                self.cached_seq_len = seq_len
            return self.pos_encoding_cache[:seq_len]
        else:
            return self._pos_encoding_no_cache(seq_len, device)

if __name__ == "__main__":
    transformer = Transformer(
        vocab_size=10000, d_model=768, seq_len=5, n_heads=1, d_hidden=2048, stack=6
    )
    # 2. 构造输入（batch_size=2, seq_len=5）
    x = torch.randint(0, 10000, (2, 5))
    y = torch.randint(0, 10000, (2, 5))
    logits = transformer(x, y)
    print(logits.shape)  # torch.Size([2, 5, 10000])
    # 统计参数量
    total=sum(p.numel() for p in transformer.parameters())
    trainable=sum(p.numel() for p in transformer.parameters() if p.requires_grad)
    print('total=', total, 'trainable=', trainable) # total= 88031232 trainable= 88031232
    
    # 运行训练演示
    print("\n" + "="*60)
