from typing import Optional
from dataclasses import dataclass

# 参考 openai_public.py 定义特殊标记
SPECIAL_TOKENS = {
    "<pad>": 100257,  # 填充标记，原本为 <|endoftext|>，这里不使用
    "<bos>": 100258,  # 句首标记（目标序列），原本为 <|fim_prefix|>，这里不使用
    "<eos>": 100259,  # 句尾标记（目标序列），原本为 <|fim_middle|>，这里不使用
    "<unk>": 0,  # 未知标记
}

@dataclass
class DataConfig:
    """数据处理配置类"""
    max_samples: int = 1000
    batch_size: int = 16
    en_max_len: Optional[int] = None
    zh_max_len: Optional[int] = None
    length_percentile: float = 0.95  # 用于自动确定最大长度的百分位数
    dataset_name: str = "Helsinki-NLP/opus-100"
    dataset_config: str = "en-zh"
    verbose: bool = True

