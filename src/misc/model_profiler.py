"""
模型分析工具：参数/内存/FLOPs 估算与可视化

功能概览：
- 统计参数总数与可训练参数。
- 估算参数与激活内存（按张量 element_size 与 numel 计算）。
- 通过前向钩子在单次前向中累计 FLOPs（线性/注意力/归一化/激活）。
- 生成模块级结构图（Graphviz），并可选择 Autograd/ONNX 作为兜底。
- 命令行运行，可打印输入形状与关键超参、FLOPs 细分与单位化输出。
"""

import os
import sys
from typing import Any, Dict, Optional, Tuple
from dataclasses import dataclass

import torch
import torch.nn as nn

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from transformer.model import AttentionUnit, Transformer
from gpt.model import GPT, GPTConfig


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    """统计模型参数数量。
    total：全部参数个数；trainable：requires_grad=True 的参数个数。
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def estimate_param_memory(model: nn.Module) -> int:
    """估算参数与缓冲区的内存占用（字节数）。
    逐个张量按 numel * element_size 计算；不依赖 dtype 假设。
    """
    mem = 0
    for t in list(model.parameters()) + list(model.buffers()):
        mem += t.numel() * t.element_size()
    return mem


def _tensor_bytes(x: Any) -> int:
    """递归估算输出对象的字节数（仅支持 Tensor/list/tuple）。"""
    if isinstance(x, torch.Tensor):
        return x.numel() * x.element_size()
    if isinstance(x, (list, tuple)):
        return sum(_tensor_bytes(i) for i in x)
    return 0


def estimate_activation_memory(
    model: nn.Module, inputs: Tuple[Any, ...], kwargs: Optional[Dict[str, Any]] = None
) -> int:
    """通过 forward_hook 收集各层输出的内存占用（一次前向）。"""
    hooks = []
    sizes = []

    def hook(_m, _inp, out):
        sizes.append(_tensor_bytes(out))

    for m in model.modules():
        if m is model:
            continue
        hooks.append(m.register_forward_hook(hook))
    with torch.no_grad():
        if kwargs:
            model(*inputs, **kwargs)
        else:
            model(*inputs)
    for h in hooks:
        h.remove()
    return sum(sizes)


class FlopsCounter:
    """FLOPs 计数器（通过钩子在前向中累计）。

    约定：
    - nn.Linear 的 FLOPs 按 GEMM：2 * batch * tokens * in_features * out_features。
    - AttentionUnit：
      * QK^T：2 * bs * heads * q_len * k_len * d_k
      * 缩放与 softmax：近似逐元素加减乘除，合并为 ~6 * bs * heads * q_len * k_len
      * AttnV：2 * bs * heads * q_len * d_v * k_len
      * Q/K/V/WO 线性已由 linear 钩子统计。
    - LayerNorm：近似 4 * numel（均值/方差/归一化/仿射）。
    - GELU：近似 2 * numel（逐元素简化）。
    """
    def __init__(self) -> None:
        self.total = 0
        self.breakdown: Dict[str, int] = {
            "linear": 0,
            "attn_qk": 0,
            "attn_softmax": 0,
            "attn_av": 0,
            "layernorm": 0,
            "gelu": 0,
        }
        self._hooks: list[Any] = []

    def _add(self, v: int) -> None:
        self.total += int(v)

    def _linear_hook(
        self, m: nn.Linear, inp: Tuple[torch.Tensor, ...], out: torch.Tensor
    ) -> None:
        # GEMM: C = A @ B，FLOPs ≈ 2 * M * K * N（乘加各计一次）
        x = inp[0]
        if x.dim() == 3:
            bs, tokens, in_f = x.shape
            out_f = m.out_features
            val = 2 * bs * tokens * in_f * out_f
            self._add(val)
            self.breakdown["linear"] += val
        elif x.dim() == 2:
            bs, in_f = x.shape
            out_f = m.out_features
            val = 2 * bs * in_f * out_f
            self._add(val)
            self.breakdown["linear"] += val

    def _gelu_hook(
        self, m: nn.GELU, inp: Tuple[torch.Tensor, ...], out: torch.Tensor
    ) -> None:
        x = inp[0]
        val = 2 * x.numel()
        self._add(val)
        self.breakdown["gelu"] += val

    def _layernorm_hook(
        self, m: nn.LayerNorm, inp: Tuple[torch.Tensor, ...], out: torch.Tensor
    ) -> None:
        x = inp[0]
        val = 4 * x.numel()
        self._add(val)
        self.breakdown["layernorm"] += val

    def _attention_hook(
        self,
        m: AttentionUnit,
        inp: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        out: torch.Tensor,
    ) -> None:
        # 注意力主计算：QK^T、softmax、AttnV；线性层由线性钩子统计
        q, k, _ = inp[0], inp[1], inp[2]
        bs, q_len, _ = q.shape
        k_len = k.size(1)
        heads = m.n_heads
        d_k = m.d_k
        d_v = m.d_v
        v_qk = 2 * bs * heads * q_len * k_len * d_k
        v_scale = bs * heads * q_len * k_len  # 缩放近似逐元素乘法
        v_softmax = 5 * bs * heads * q_len * k_len  # softmax 经验近似
        v_av = 2 * bs * heads * q_len * d_v * k_len
        self._add(v_qk)
        self._add(v_scale)
        self._add(v_softmax)
        self._add(v_av)
        self.breakdown["attn_qk"] += v_qk
        self.breakdown["attn_softmax"] += v_scale + v_softmax
        self.breakdown["attn_av"] += v_av

    def install(self, model: nn.Module) -> None:
        for m in model.modules():
            if m is model:
                continue
            if isinstance(m, nn.Linear):
                self._hooks.append(m.register_forward_hook(self._linear_hook))
            elif isinstance(m, nn.GELU):
                self._hooks.append(m.register_forward_hook(self._gelu_hook))
            elif isinstance(m, nn.LayerNorm):
                self._hooks.append(m.register_forward_hook(self._layernorm_hook))
            elif isinstance(m, AttentionUnit):
                self._hooks.append(m.register_forward_hook(self._attention_hook))

    def remove(self) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks.clear()


def estimate_flops(
    model: nn.Module, inputs: Tuple[Any, ...], kwargs: Optional[Dict[str, Any]] = None
) -> Tuple[int, Dict[str, int]]:
    """一次前向中估算 FLOPs 与细分。

    返回：(总 FLOPs，细分 dict)
    """
    counter = FlopsCounter()
    counter.install(model)
    with torch.no_grad():
        if kwargs:
            model(*inputs, **kwargs)
        else:
            model(*inputs)
    counter.remove()
    return counter.total, counter.breakdown


def save_graph(
    model: nn.Module,
    inputs: Tuple[Any, ...],
    kwargs: Optional[Dict[str, Any]],
    out_path_no_ext: str,
    graph_mode: str = "module",
    depth: int = 0,
) -> Optional[str]:
    """生成模型图。
    - module：模块级结构图（默认），节点少、易读。
    - autograd：真实计算图（需梯度），节点多。
    - onnx：导出静态图（可用 Netron 查看）。
    """
    os.makedirs(os.path.dirname(out_path_no_ext), exist_ok=True)
    meta = _collect_meta(model, inputs)
    if graph_mode == "module":
        try:
            import graphviz

            g = graphviz.Digraph(name="model", format="png")
            g.attr(rankdir="LR", fontsize="10")
            if isinstance(model, Transformer):
                g.node(
                    "in_x",
                    f"Input x\n(batch={meta.get('batch_size')}, seq={meta.get('seq_len')})",
                    shape="box",
                )
                g.node(
                    "in_y",
                    f"Input y\n(batch={meta.get('batch_size')}, seq={meta.get('seq_len')})",
                    shape="box",
                )
                g.node(
                    "emb",
                    f"Embedding\n(vocab={meta.get('vocab_size')}, d_model={meta.get('d_model')})",
                    shape="box",
                )
                g.node("pos", "PosEncoding\n(sinusoidal)", shape="box")
                g.node(
                    "enc",
                    f"Encoder x {meta.get('stack')}\n[SelfAttn + FFN]",
                    shape="component",
                )
                g.node(
                    "dec",
                    f"Decoder x {meta.get('stack')}\n[MaskedSelfAttn + CrossAttn + FFN]",
                    shape="component",
                )
                g.node(
                    "out",
                    f"Output Linear (tied)\n(d_model->{meta.get('vocab_size')})",
                    shape="box",
                )
                g.edge("in_x", "emb")
                g.edge("in_y", "emb")
                g.edge("emb", "pos")
                g.edge("pos", "enc")
                g.edge("enc", "dec")
                g.edge("dec", "out")
            elif isinstance(model, GPT):
                g.node(
                    "in_y",
                    f"Input y\n(batch={meta.get('batch_size')}, seq={meta.get('seq_len')})",
                    shape="box",
                )
                g.node(
                    "tok_emb",
                    f"TokenEmbedding\n(vocab={meta.get('vocab_size')}, d_model={meta.get('d_model')})",
                    shape="box",
                )
                g.node(
                    "pos_emb",
                    f"PosEmbedding\n(seq_len={meta.get('seq_len')}, d_model={meta.get('d_model')})",
                    shape="box",
                )
                g.node("add", "Add(Embeddings)", shape="circle")
                g.node(
                    "blocks",
                    f"Blocks x {meta.get('stack')}\n[SelfAttn(masked) + GELU-MLP]",
                    shape="component",
                )
                g.node("lnf", "LayerNorm Final", shape="box")
                g.node(
                    "lm",
                    f"LMHead (tied)\n(d_model->{meta.get('vocab_size')})",
                    shape="box",
                )
                g.edge("in_y", "tok_emb")
                g.edge("in_y", "pos_emb")
                g.edge("tok_emb", "add")
                g.edge("pos_emb", "add")
                g.edge("add", "blocks")
                g.edge("blocks", "lnf")
                g.edge("lnf", "lm")
            else:
                g.node("model", model.__class__.__name__, shape="box")
            final_path = g.render(out_path_no_ext, cleanup=True)
            return final_path
        except Exception:
            pass
    if graph_mode == "autograd":
        try:
            import torchviz

            model.eval()
            with torch.enable_grad():
                if kwargs:
                    out = model(*inputs, **kwargs)
                else:
                    out = model(*inputs)
            dot = torchviz.make_dot(out, params=dict(model.named_parameters()))
            dot.format = "png"
            final_path = dot.render(out_path_no_ext, cleanup=True)
            return final_path
        except Exception:
            pass
    try:
        if any(i is None for i in inputs):
            return None
        torch.onnx.export(model, inputs, out_path_no_ext + ".onnx", opset_version=12)
        return out_path_no_ext + ".onnx"
    except Exception:
        return None


def bytes_to_mb(v: int) -> float:
    """字节转 MB（二进制）。"""
    return v / (1024.0 * 1024.0)


def _collect_meta(model: nn.Module, inputs: Tuple[Any, ...]) -> Dict[str, Any]:
    """收集输入形状与关键超参数。
    - 对输入：取第一个形状>=2的 Tensor 的 batch 与 seq。
    - 对模型：按类型提取 d_model/n_heads/stack/vocab_size/d_hidden。
    """
    bs = None
    seqlen = None
    for itm in inputs:
        if isinstance(itm, torch.Tensor) and itm.dim() >= 2:
            bs = int(itm.size(0))
            seqlen = int(itm.size(1))
            break
    meta: Dict[str, Any] = {"batch_size": bs, "seq_len": seqlen}
    if isinstance(model, Transformer):
        meta.update(
            {
                "d_model": getattr(model, "d_model", None),
                "n_heads": getattr(model, "n_heads", None),
                "stack": len(getattr(model, "encoder", [])),
                "vocab_size": getattr(model.embedding, "num_embeddings", None),
                "d_hidden": getattr(model, "d_hidden", None),
            }
        )
    elif isinstance(model, GPT):
        meta.update(
            {
                "d_model": getattr(model, "d_model", None),
                "n_heads": getattr(model, "n_heads", None),
                "stack": getattr(model, "stack", None),
                "vocab_size": getattr(model.token_embedding, "num_embeddings", None),
                "d_hidden": getattr(model, "d_hidden", None),
            }
        )
    return meta


@dataclass
class ProfileMeta:
    batch_size: Optional[int]
    seq_len: Optional[int]
    d_model: Optional[int]
    n_heads: Optional[int]
    stack: Optional[int]
    vocab_size: Optional[int]
    d_hidden: Optional[int]
    device: str
    dtype: str


@dataclass
class ProfileStats:
    meta: ProfileMeta
    params_total: int
    params_trainable: int
    param_memory_bytes: int
    activation_memory_bytes: int
    total_memory_bytes: int
    train_memory_bytes: int
    flops_forward: int
    flops_train: int
    flops_breakdown: Dict[str, int]
    graph_path: Optional[str]


def _make_meta(model: nn.Module, inputs: Tuple[Any, ...]) -> ProfileMeta:
    """构造 ProfileMeta，附带 device/dtype（取首个参数的属性）。"""
    m = _collect_meta(model, inputs)
    # 推断 device/dtype
    any_param = next(iter(model.parameters()))
    device = str(any_param.device)
    dtype = str(any_param.dtype).replace("torch.", "")
    return ProfileMeta(
        batch_size=m.get("batch_size"),
        seq_len=m.get("seq_len"),
        d_model=m.get("d_model"),
        n_heads=m.get("n_heads"),
        stack=m.get("stack"),
        vocab_size=m.get("vocab_size"),
        d_hidden=m.get("d_hidden"),
        device=device,
        dtype=dtype,
    )


def profile_model(
    model: nn.Module,
    inputs: Tuple[Any, ...],
    kwargs: Optional[Dict[str, Any]],
    name: str,
    graph_dir: str,
    train_factor: float = 3.0,
    graph_mode: str = "module",
    depth: int = 0,
    opt_state_factor: float = 0.0,
) -> ProfileStats:
    """统一执行一次前向并收集所有统计。

    - train_factor：训练步 FLOPs 近似为前向的倍数（GEMM 主导取 3.0 更合理）。
    - opt_state_factor：优化器状态内存按参数内存的倍数估算（AdamW 可选近似 2-3）。
    - graph_mode：控制图类型（模块级更简洁）。
    """
    total_params, trainable_params = count_parameters(model)
    param_mem = estimate_param_memory(model)
    # 单次前向：同时安装 FLOPs 与激活钩子
    flops_counter = FlopsCounter()
    flops_counter.install(model)
    act_hooks = []
    act_sizes = []

    def act_hook(_m, _inp, out):
        act_sizes.append(_tensor_bytes(out))

    for m in model.modules():
        if m is model:
            continue
        act_hooks.append(m.register_forward_hook(act_hook))
    with torch.no_grad():
        if kwargs:
            model(*inputs, **kwargs)
        else:
            model(*inputs)
    for h in act_hooks:
        h.remove()
    flops_counter.remove()
    act_mem = sum(act_sizes)
    flops_fwd = flops_counter.total
    flops_breakdown = flops_counter.breakdown
    total_mem = param_mem + act_mem
    train_mem = param_mem + act_mem * 2 + int(param_mem * opt_state_factor)
    graph_path = save_graph(
        model,
        inputs,
        kwargs,
        os.path.join(graph_dir, name),
        graph_mode=graph_mode,
        depth=depth,
    )
    meta = _make_meta(model, inputs)
    return ProfileStats(
        meta=meta,
        params_total=total_params,
        params_trainable=trainable_params,
        param_memory_bytes=param_mem,
        activation_memory_bytes=act_mem,
        total_memory_bytes=total_mem,
        train_memory_bytes=train_mem,
        flops_forward=flops_fwd,
        flops_train=int(flops_fwd * train_factor),
        flops_breakdown=flops_breakdown,
        graph_path=graph_path,
    )


def _print_result(title: str, res: ProfileStats) -> None:
    """统一的终端打印格式（含单位换算与细分）。"""
    f = res.flops_forward
    t = res.flops_train
    print(f" * {title}")
    m = res.meta
    if m.batch_size is not None and m.seq_len is not None:
        print(f" - 输入： batch_size={m.batch_size} ，seq_len={m.seq_len}")
    hp = []
    for k, v in {
        "d_model": m.d_model,
        "n_heads": m.n_heads,
        "stack": m.stack,
        "vocab_size": m.vocab_size,
        "d_hidden": m.d_hidden,
        "device": m.device,
        "dtype": m.dtype,
    }.items():
        if v is not None:
            hp.append(f"{k}={v}")
    if hp:
        print(" - 超参数： " + " ，".join(hp))
    print(
        f" - 参数总数： {res.params_total} ，参数内存： {bytes_to_mb(res.param_memory_bytes):.2f} MB"
    )
    print(f" - 激活内存： {bytes_to_mb(res.activation_memory_bytes):.2f} MB")
    print(f" - 总内存（推理）： {bytes_to_mb(res.total_memory_bytes):.2f} MB")
    print(f" - 训练内存估算： {bytes_to_mb(res.train_memory_bytes):.2f} MB")
    print(
        f" - 推理： {format(f, ',')} FLOPs， {f / 1e9:.2f} GFLOPs， {f / 1e12:.4f} TFLOPs"
    )
    print(
        f" - 训练： {format(t, ',')} FLOPs， {t / 1e9:.2f} GFLOPs， {t / 1e12:.4f} TFLOPs"
    )
    if res.flops_breakdown:
        parts = ", ".join(
            [f"{k}={format(v, ',')}" for k, v in res.flops_breakdown.items() if v > 0]
        )
        print(f" - FLOPs细分： {parts}")
    print(f" - 计算图： {res.graph_path}")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--seq", type=int, default=100)
    ap.add_argument(
        "--graph-mode",
        type=str,
        default="module",
        choices=["module", "autograd", "onnx"],
    )
    ap.add_argument("--depth", type=int, default=0)
    ap.add_argument("--train-factor", type=float, default=3.0)
    ap.add_argument("--opt-state-factor", type=float, default=0.0)
    ap.add_argument("--no-graph", action="store_true")
    args = ap.parse_args()

    bs = args.bs
    vocab_t = 10000
    d_model_t = 512
    seq_len_t = args.seq
    n_heads_t = 8
    d_hidden_t = 2048
    stack_t = 6
    transformer = Transformer(
        vocab_size=vocab_t,
        d_model=d_model_t,
        seq_len=seq_len_t,
        n_heads=n_heads_t,
        d_hidden=d_hidden_t,
        stack=stack_t,
    )
    x_t = torch.randint(0, vocab_t, (bs, seq_len_t))
    y_t = torch.randint(0, vocab_t, (bs, seq_len_t))
    inputs_t = (x_t, y_t)
    kwargs_t = {"x_padding_mask": None, "y_padding_mask": None}
    res_t = profile_model(
        transformer,
        inputs_t,
        kwargs_t,
        "transformer",
        os.path.join("graph", "transformer"),
        train_factor=args.train_factor,
        graph_mode=("module" if args.no_graph else args.graph_mode),
        depth=args.depth,
        opt_state_factor=args.opt_state_factor,
    )
    _print_result("Transformer", res_t)

    cfg = GPTConfig()
    cfg.seq_len = 128
    gpt = GPT(cfg)
    y_g = torch.randint(0, cfg.vocab_size, (bs, cfg.seq_len))
    x_g = y_g
    pad_mask_g = torch.zeros((bs, cfg.seq_len), dtype=torch.bool)
    inputs_g = (x_g, y_g, None, pad_mask_g)
    kwargs_g = None
    res_g = profile_model(
        gpt,
        inputs_g,
        kwargs_g,
        "gpt",
        os.path.join("graph", "gpt"),
        train_factor=args.train_factor,
        graph_mode=("module" if args.no_graph else args.graph_mode),
        depth=args.depth,
        opt_state_factor=args.opt_state_factor,
    )
    _print_result("GPT", res_g)
