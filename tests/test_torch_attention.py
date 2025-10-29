import math
import torch

from torch_attention import AttentionUnit, FFN


def test_attention_output_shapes_and_sum_to_one_in_eval():
    torch.manual_seed(0)
    d_model = 8
    attn = AttentionUnit(d_model)
    attn.eval()  # 评估模式关闭 Dropout，保证概率和为 1

    q = torch.randn(2, 5, d_model)
    k = torch.randn(2, 5, d_model)
    v = torch.randn(2, 5, d_model)

    out, probs = attn(q, k, v)
    assert out.shape == (2, 5, d_model)
    assert probs.shape == (2, 5, 5)

    sums = probs.sum(dim=-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-7)


def test_attention_mask_zeroes_probabilities_for_masked_positions():
    torch.manual_seed(0)
    d_model = 8
    attn = AttentionUnit(d_model)
    attn.eval()  # 关闭 Dropout 以便于断言概率

    q = torch.randn(2, 5, d_model)
    k = torch.randn(2, 5, d_model)
    v = torch.randn(2, 5, d_model)

    # 掩码：最后一个键位置为 0（不可见），其余为 1
    mask = torch.ones(2, 5, 5, dtype=torch.bool)
    mask[..., -1] = 0

    _, probs = attn(q, k, v, mask=mask)
    # 被屏蔽的位置 softmax 后应为 0
    assert torch.allclose(probs[..., -1], torch.zeros_like(probs[..., -1]), atol=1e-7)


def test_attention_matches_manual_computation_with_identity_weights():
    torch.manual_seed(0)
    d_model = 4
    attn = AttentionUnit(d_model)
    attn.eval()

    # 将线性层设置为恒等映射，便于手工验证
    with torch.no_grad():
        attn.w_q.weight.copy_(torch.eye(d_model))
        attn.w_q.bias.zero_()
        attn.w_k.weight.copy_(torch.eye(d_model))
        attn.w_k.bias.zero_()
        attn.w_v.weight.copy_(torch.eye(d_model))
        attn.w_v.bias.zero_()

    q = torch.randn(1, 3, d_model)
    k = torch.randn(1, 3, d_model)
    v = torch.randn(1, 3, d_model)

    scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(d_model)
    probs = torch.softmax(scores, dim=-1)
    manual_out = torch.matmul(probs, v)

    out, _ = attn(q, k, v)
    torch.testing.assert_close(out, manual_out, rtol=1e-6, atol=1e-6)


def test_attention_backward_flows_gradients():
    torch.manual_seed(0)
    d_model = 8
    attn = AttentionUnit(d_model)
    attn.train()  # 开启 Dropout 也应当支持反向传播

    q = torch.randn(2, 5, d_model, requires_grad=True)
    k = torch.randn(2, 5, d_model, requires_grad=True)
    v = torch.randn(2, 5, d_model, requires_grad=True)

    out, _ = attn(q, k, v)
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

