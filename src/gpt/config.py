from dataclasses import dataclass
from typing import Optional


@dataclass
class GPTConfig:
    """GPT-2 模型配置"""
    # GPT-2 词表大小 50257（扩展自 GPT-1 的 40000 merges）
    vocab_size: int = 50257
    d_model: int = 768
    # GPT-2 将上下文长度扩大到 1024（GPT-1 为 512）
    seq_len: int = 1024
    n_heads: int = 12
    d_hidden: int = 3072
    stack: int = 12
    dropout: float = 0.1
    use_pos_encoding_cache: bool = False


@dataclass
class TrainConfig:
    """训练配置（对齐 GPT-1/2 超参数与调度）"""
    # GPT-2 大批量 512；epochs 100 参考 GPT-1 实验设置
    batch_size: int = 512
    num_epochs: int = 100
    # Adam 学习率 2.5e-4；这里采用 AdamW 的解耦权衰（现代实现）
    learning_rate: float = 2.5e-4
    weight_decay: float = 0.01
    # 线性预热 2000 步，随后余弦退火至 0
    warmup_steps: int = 2000
    total_steps: Optional[int] = None
    grad_clip: float = 1.0
    num_workers: int = 2
    pin_memory: bool = True
    buffer_size: int = 10000
    pad_token_id: int = 50256
    eos_token_id: int = 50256
    log_dir: str = './runs/gpt'
    enable_tensorboard: bool = True
    total_steps_for_epoch: Optional[int] = None
    loss_log_enabled: bool = True
    loss_log_path: str = './training_losses.txt'
    loss_log_max_steps: int = 10000