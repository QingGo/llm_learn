from typing import Tuple, Any, Dict, Iterator, List, Optional

import torch
from torch.utils.data import IterableDataset, DataLoader
from datasets import load_from_disk
from tokenizers import Tokenizer

from data.clean_cc_bc import (
    load_hf_datasets,
    create_merged_dataset,
    train_or_load_tokenizer,
    TextProcessingDataset,
    TextProcessingMapDataset,
)


class ShuffledIterableDataset(IterableDataset):
    """对 IterableDataset 进行缓冲随机化以提高训练稳定性与吞吐"""
    def __init__(self, base: IterableDataset, buffer_size: int = 10000):
        super().__init__()
        self.base = base
        self.buffer_size = buffer_size

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        import random
        # 迭代数据不支持 DataLoader 的 shuffle；通过缓冲区打散提升随机性与收敛稳定性
        buffer: List[Dict[str, torch.Tensor]] = []
        for item in self.base:
            buffer.append(item)
            if len(buffer) >= self.buffer_size:
                random.shuffle(buffer)
                while buffer:
                    yield buffer.pop()
        random.shuffle(buffer)
        while buffer:
            yield buffer.pop()

    def __len__(self) -> int:
        return len(self.base)


def create_gpt_datasets(seq_len: int, tokenizer_vocab_size: int = 50257, pad_token_id: int = 50256) -> Tuple[IterableDataset, IterableDataset, IterableDataset, Any]:
    """加载数据、合并、分词器准备，并构建 Train/Val/Test 的 TextProcessingDataset"""
    common_crawl_ds, book_corpus_ds = load_hf_datasets()
    merged = create_merged_dataset(common_crawl_ds, book_corpus_ds)
    tokenizer = train_or_load_tokenizer(common_crawl_ds, book_corpus_ds, vocab_size=tokenizer_vocab_size)
    train_valid = merged.train_test_split(test_size=0.02)
    valid_test = train_valid['test'].train_test_split(test_size=0.5)
    train_ds = train_valid['train']
    valid_ds = valid_test['train']
    test_ds = valid_test['test']
    train_pt = TextProcessingDataset(train_ds, tokenizer, max_seq_length=seq_len, pad_token_id=pad_token_id)
    valid_pt = TextProcessingDataset(valid_ds, tokenizer, max_seq_length=seq_len, pad_token_id=pad_token_id)
    test_pt = TextProcessingDataset(test_ds, tokenizer, max_seq_length=seq_len, pad_token_id=pad_token_id)
    return train_pt, valid_pt, test_pt, tokenizer


def collate_tokens(batch: List[Dict[str, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor]:
    """将 TextProcessingDataset 的字典批量化为 `(input_ids, attention_mask)`"""
    input_ids = torch.stack([item['input_ids'] for item in batch], dim=0)
    attention_mask = torch.stack([item['attention_mask'] for item in batch], dim=0)
    return input_ids, attention_mask


def create_gpt_dataloaders(
    seq_len: int,
    batch_size: int,
    buffer_size: int = 10000,
    num_workers: int = 2,
    pin_memory: bool = True,
    tokenizer_vocab_size: int = 50257,
    processed_path: Optional[str] = None,
    tokenizer_path: Optional[str] = "./data/tokenizer.json",
) -> Tuple[DataLoader, DataLoader, DataLoader, Any]:
    """构建 DataLoader

    - 若提供 `processed_path`，使用离线块级数据（随机索引Dataset，训练启用shuffle）
    - 否则回退到流式 `IterableDataset` 并用缓冲打散
    """
    if processed_path:
        ds = load_from_disk(processed_path)
        # 划分 Train/Val/Test
        train_valid = ds.train_test_split(test_size=0.02)
        valid_test = train_valid["test"].train_test_split(test_size=0.5)
        train_pt = TextProcessingMapDataset(train_valid["train"])
        valid_pt = TextProcessingMapDataset(valid_test["train"])
        test_pt = TextProcessingMapDataset(valid_test["test"])
        # 离线数据可使用 DataLoader 的 shuffle
        train_loader = DataLoader(train_pt, batch_size=batch_size, shuffle=True, collate_fn=collate_tokens, num_workers=num_workers, pin_memory=pin_memory)
        valid_loader = DataLoader(valid_pt, batch_size=batch_size, shuffle=False, collate_fn=collate_tokens, num_workers=num_workers, pin_memory=pin_memory)
        test_loader = DataLoader(test_pt, batch_size=batch_size, shuffle=False, collate_fn=collate_tokens, num_workers=num_workers, pin_memory=pin_memory)
        tokenizer = Tokenizer.from_file(tokenizer_path)
        return train_loader, valid_loader, test_loader, tokenizer

    # 流式回退：在线处理 + 迭代器缓冲打散
    train_pt, valid_pt, test_pt, tokenizer = create_gpt_datasets(seq_len, tokenizer_vocab_size)
    train_loader = DataLoader(ShuffledIterableDataset(train_pt, buffer_size=buffer_size), batch_size=batch_size, collate_fn=collate_tokens, num_workers=num_workers, pin_memory=pin_memory)
    valid_loader = DataLoader(valid_pt, batch_size=batch_size, collate_fn=collate_tokens, num_workers=num_workers, pin_memory=pin_memory)
    test_loader = DataLoader(test_pt, batch_size=batch_size, collate_fn=collate_tokens, num_workers=num_workers, pin_memory=pin_memory)
    return train_loader, valid_loader, test_loader, tokenizer