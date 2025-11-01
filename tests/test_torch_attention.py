import torch
import pytest
import sys
import os

# 添加src目录到Python路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from transformer.model import AttentionUnit, FFN, Transformer


def test_attention_output_shapes_and_sum_to_one_in_eval():
    torch.manual_seed(0)
    d_model = 8
    seq_len = 2
    n_heads = 4
    attn = AttentionUnit(d_model, seq_len, n_heads)
    attn.eval()  # 评估模式关闭 Dropout，保证概率和为 1

    q = torch.randn(2, seq_len, d_model)
    k = torch.randn(2, seq_len, d_model)
    v = torch.randn(2, seq_len, d_model)

    out = attn(q, k, v)
    assert out.shape == (2, seq_len, d_model)


def test_attention_different_seq_lengths():
    """测试不同序列长度的处理"""
    torch.manual_seed(0)
    d_model = 8
    seq_len = 4
    n_heads = 2
    attn = AttentionUnit(d_model, seq_len, n_heads)
    
    q = torch.randn(2, 3, d_model)  # query长度为3
    k = torch.randn(2, 5, d_model)  # key长度为5
    v = torch.randn(2, 5, d_model)  # value长度为5
    
    out = attn(q, k, v)
    assert out.shape == (2, 3, d_model)  # 输出应该与query长度匹配


def test_attention_with_padding_mask():
    """测试padding mask的处理"""
    torch.manual_seed(0)
    d_model = 8
    seq_len = 4
    n_heads = 2
    attn = AttentionUnit(d_model, seq_len, n_heads)
    attn.eval()
    
    q = torch.randn(2, seq_len, d_model)
    k = torch.randn(2, seq_len, d_model)
    v = torch.randn(2, seq_len, d_model)
    
    # 创建padding mask，第一个样本的最后一个位置是padding
    key_padding_mask = torch.tensor([[False, False, False, True], 
                                   [False, False, False, False]])
    
    out = attn(q, k, v, key_padding_mask)
    assert out.shape == (2, seq_len, d_model)


def test_attention_with_causal_mask():
    """测试因果mask的处理"""
    torch.manual_seed(0)
    d_model = 8
    seq_len = 4
    n_heads = 2
    attn = AttentionUnit(d_model, seq_len, n_heads, use_mask=True)
    attn.eval()
    
    q = torch.randn(2, seq_len, d_model)
    k = torch.randn(2, seq_len, d_model)
    v = torch.randn(2, seq_len, d_model)
    
    out = attn(q, k, v)
    assert out.shape == (2, seq_len, d_model)


def test_positional_encoding_odd_d_model():
    """测试奇数d_model的位置编码"""
    vocab_size = 100
    d_model = 9  # 奇数
    seq_len = 5
    n_heads = 3
    
    transformer = Transformer(vocab_size, d_model, seq_len, n_heads)
    
    x = torch.randint(0, vocab_size, (2, seq_len))
    y = torch.randint(0, vocab_size, (2, seq_len))
    
    logits = transformer(x, y)
    assert logits.shape == (2, seq_len, vocab_size)


def test_numerical_stability_all_inf():
    """测试数值稳定性：所有attention score都是-inf的情况"""
    d_model = 4
    seq_len = 3
    n_heads = 1
    attn = AttentionUnit(d_model, seq_len, n_heads)
    attn.eval()
    
    # 创建会导致所有attention score为-inf的输入
    q = torch.zeros(1, seq_len, d_model)
    k = torch.zeros(1, seq_len, d_model)
    v = torch.randn(1, seq_len, d_model)
    
    # 使用极强的padding mask，所有位置都被mask
    key_padding_mask = torch.ones(1, seq_len, dtype=torch.bool)
    
    out = attn(q, k, v, key_padding_mask)
    assert out.shape == (1, seq_len, d_model)
    assert not torch.isnan(out).any(), "Output should not contain NaN values"


def test_input_validation():
    """测试输入验证功能"""
    d_model = 8
    seq_len = 4
    n_heads = 2
    attn = AttentionUnit(d_model, seq_len, n_heads)
    
    q = torch.randn(2, 3, d_model)
    k = torch.randn(2, 5, d_model)
    v = torch.randn(2, 4, d_model)  # v的长度与k不匹配
    
    # 应该抛出断言错误
    with pytest.raises(AssertionError, match="Key and Value sequence lengths must match"):
        attn(q, k, v)
    
    # 测试d_model不匹配
    q_wrong = torch.randn(2, 3, 6)  # 错误的d_model
    k_correct = torch.randn(2, 5, d_model)
    v_correct = torch.randn(2, 5, d_model)
    
    with pytest.raises(AssertionError, match="Input d_model"):
        attn(q_wrong, k_correct, v_correct)


def test_attention_backward_flows_gradients():
    torch.manual_seed(0)
    d_model = 8
    seq_len = 2
    n_heads = 4
    attn = AttentionUnit(d_model, seq_len, n_heads)
    attn.train()  # 开启 Dropout 也应当支持反向传播

    q = torch.randn(2, seq_len, d_model, requires_grad=True)
    k = torch.randn(2, seq_len, d_model, requires_grad=True)
    v = torch.randn(2, seq_len, d_model, requires_grad=True)

    out = attn(q, k, v)
    loss = out.sum()
    loss.backward()

    # 参数需要有梯度
    assert attn.w_q.weight.grad is not None
    assert attn.w_k.weight.grad is not None
    assert attn.w_v.weight.grad is not None
    # 输入也应获得梯度
    assert q.grad is not None and k.grad is not None and v.grad is not None


def test_ffn_output_shape_and_gradients():
    torch.manual_seed(0)
    d_model, d_hidden = 8, 16
    ffn = FFN(d_model, d_hidden)

    x = torch.randn(2, 5, d_model, requires_grad=True)
    out = ffn(x)
    assert out.shape == (2, 5, d_model)

    loss = out.sum()
    loss.backward()
    assert any(p.grad is not None for p in ffn.parameters())
    assert x.grad is not None


def test_positional_encoding_caching():
    """测试位置编码缓存机制"""
    vocab_size = 100
    d_model = 8
    seq_len = 10
    n_heads = 2
    
    transformer = Transformer(vocab_size, d_model, seq_len, n_heads)
    
    # 第一次调用
    x1 = torch.randint(0, vocab_size, (2, 5))
    y1 = torch.randint(0, vocab_size, (2, 5))
    _ = transformer(x1, y1)
    
    # 检查缓存是否创建
    assert transformer.pos_encoding_cache is not None
    assert transformer.cached_seq_len >= 5
    
    # 第二次调用相同长度，应该使用缓存
    x2 = torch.randint(0, vocab_size, (2, 5))
    y2 = torch.randint(0, vocab_size, (2, 5))
    _ = transformer(x2, y2)
    
    # 第三次调用更长序列，应该更新缓存
    x3 = torch.randint(0, vocab_size, (2, 8))
    y3 = torch.randint(0, vocab_size, (2, 8))
    _ = transformer(x3, y3)
    
    assert transformer.cached_seq_len >= 8


def test_transformer_with_different_input_output_lengths():
    """测试Transformer处理不同输入输出长度"""
    vocab_size = 100
    d_model = 8
    seq_len = 10
    n_heads = 2
    
    transformer = Transformer(vocab_size, d_model, seq_len, n_heads, stack=2)
    
    # 不同长度的输入
    x = torch.randint(0, vocab_size, (2, 6))  # encoder输入长度6
    y = torch.randint(0, vocab_size, (2, 4))  # decoder输入长度4
    
    logits = transformer(x, y)
    assert logits.shape == (2, 4, vocab_size)  # 输出应该与decoder输入长度匹配


def test_mask_with_different_seq_lengths():
    """测试不同序列长度下的mask处理"""
    d_model = 8
    seq_len = 6
    n_heads = 2
    attn = AttentionUnit(d_model, seq_len, n_heads, use_mask=True)
    attn.eval()
    
    # query长度3，key/value长度5
    q = torch.randn(1, 3, d_model)
    k = torch.randn(1, 5, d_model)
    v = torch.randn(1, 5, d_model)
    
    out = attn(q, k, v)
    assert out.shape == (1, 3, d_model)

