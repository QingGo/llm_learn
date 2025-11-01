import torch
import pandas as pd
import re
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset  # Hugging Face数据集加载工具
import tiktoken  # OpenAI分词器
from typing import Optional, Tuple, Dict, Any
from dataclasses import dataclass


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
    split: str = "train"
    verbose: bool = True


# 参考 openai_public.py 定义特殊标记
SPECIAL_TOKENS = {
    "<pad>": 100257,  # 填充标记，原本为 <|endoftext|>，这里不使用
    "<bos>": 100258,  # 句首标记（目标序列），原本为 <|fim_prefix|>，这里不使用
    "<eos>": 100259,  # 句尾标记（目标序列），原本为 <|fim_middle|>，这里不使用
    "<unk>": 0        # tiktoken默认未知标记ID
}


class TranslationDataset(Dataset):
    """翻译数据集类"""
    def __init__(self, df: pd.DataFrame):
        self.df = df
    
    def __len__(self) -> int:
        return len(self.df)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # 转换为长整型张量（模型输入要求）
        src = torch.tensor(self.df["en_processed"].iloc[idx], dtype=torch.long)
        tgt = torch.tensor(self.df["zh_processed"].iloc[idx], dtype=torch.long)
        return src, tgt


class TranslationDataProcessor:
    """翻译数据处理器 - 只能通过 create_translation_dataloader 创建"""
    
    def __init__(self, config: DataConfig):
        self.config = config
        self._tokenizer = tiktoken.get_encoding("cl100k_base")
        self._df = None
        self._stats = None
        # 用于收集被过滤的数据样本（用于verbose模式）
        self._filtered_samples = {
            'contains_original': [],  # 翻译中包含原文
            'too_short': [],         # 太短且无意义
            'not_translated': []     # 完全没有翻译
        }
    
    def _clean_text(self, text: str, is_english: bool = True) -> str:
        """轻量清洗：保留有效字符，去除冗余空格"""
        if not text:
            return ""
        # 保留核心字符（字母、数字、中文、常见标点）
        if is_english:
            # 英文：允许字母、常见标点（. , ! ? '）和空格
            text = re.sub(r"[^\w\s.,!?']", " ", text)
        else:
            # 中文：允许中文、字母、数字、常见标点
            text = re.sub(r"[^\u4e00-\u9fa5\w\s.,!?']", " ", text)
        # 合并连续空格并去除首尾空格
        text = re.sub(r"\s+", " ", text).strip()
        return text
    
    def _is_valid_translation_pair(self, en_text: str, zh_text: str) -> bool:
        """
        检查翻译对的质量，过滤低质量数据
        
        过滤规则：
        1. 翻译中包含原文（中文翻译中包含完整的英文句子）
        2. 太短且无明确含义的句子（已删除）
        3. 完全没有翻译（中英文基本相同）
        """
        if not en_text or not zh_text:
            return False
        
        # 规则1: 检查中文翻译中是否包含完整的英文原句
        # 移除标点符号进行比较
        en_clean_for_check = re.sub(r'[^\w\s]', '', en_text.lower()).strip()
        zh_clean_for_check = re.sub(r'[^\u4e00-\u9fa5\w\s]', '', zh_text.lower()).strip()
        
        # 如果英文句子长度超过5个字符，且完整出现在中文翻译中，则过滤
        if len(en_clean_for_check) > 5 and en_clean_for_check in zh_clean_for_check:
            if self.config.verbose and len(self._filtered_samples['contains_original']) < 10:
                self._filtered_samples['contains_original'].append((en_text, zh_text))
            return False
        
        # 规则2: 过滤太短且无明确含义的句子
        # 计算有效字符数（排除标点和空格）
        # en_meaningful_chars = len(re.sub(r'[^\w\u4e00-\u9fa5]', '', en_text))
        # zh_meaningful_chars = len(re.sub(r'[^\w\u4e00-\u9fa5]', '', zh_text))
        
        # # 如果任一语言的有效字符数少于3个，则过滤
        # if en_meaningful_chars < 3 or zh_meaningful_chars < 3:
        #     if self.config.verbose and len(self._filtered_samples['too_short']) < 10:
        #         self._filtered_samples['too_short'].append((en_text, zh_text))
        #     return False
        
        # 规则3: 检查是否完全没有翻译（相似度过高）
        # 移除所有非字母数字字符，转为小写进行比较
        en_normalized = re.sub(r'[^\w]', '', en_text.lower())
        zh_normalized = re.sub(r'[^\w\u4e00-\u9fa5]', '', zh_text.lower())
        
        # 如果两个文本完全相同或中文中英文字符占比过高，则过滤
        if en_normalized == zh_normalized:
            if self.config.verbose and len(self._filtered_samples['not_translated']) < 10:
                self._filtered_samples['not_translated'].append((en_text, zh_text))
            return False
        
        # 检查中文翻译中英文字符的比例
        # 计算英文字符数（只包含英文字母和数字）
        zh_english_chars = len(re.sub(r'[^a-zA-Z0-9]', '', zh_text))
        # 计算总的有效字符数（排除空格和标点）
        zh_total_chars = len(re.sub(r'[^\w\u4e00-\u9fa5]', '', zh_text))
        
        # 如果中文翻译中英文字符占比超过70%，则认为翻译质量不佳
        if zh_total_chars > 0 and zh_english_chars / zh_total_chars > 0.7:
            if self.config.verbose and len(self._filtered_samples['not_translated']) < 10:
                self._filtered_samples['not_translated'].append((en_text, zh_text))
            return False
        
        return True
    
    def _print_filtered_samples(self):
        """打印被各种规则过滤掉的样本数据"""
        if not self.config.verbose:
            return
        
        print("\n=== 质量过滤详细信息 ===")
        
        # 规则1: 翻译中包含原文
        if self._filtered_samples['contains_original']:
            print(f"\n规则1 - 翻译中包含原文 (前{len(self._filtered_samples['contains_original'])}条):")
            for i, (en, zh) in enumerate(self._filtered_samples['contains_original'], 1):
                print(f"  {i}. EN: {en}")
                print(f"     ZH: {zh}")
        
        # 规则2: 太短且无意义
        if self._filtered_samples['too_short']:
            print(f"\n规则2 - 太短且无明确含义 (前{len(self._filtered_samples['too_short'])}条):")
            for i, (en, zh) in enumerate(self._filtered_samples['too_short'], 1):
                print(f"  {i}. EN: {en}")
                print(f"     ZH: {zh}")
        
        # 规则3: 完全没有翻译
        if self._filtered_samples['not_translated']:
            print(f"\n规则3 - 完全没有翻译 (前{len(self._filtered_samples['not_translated'])}条):")
            for i, (en, zh) in enumerate(self._filtered_samples['not_translated'], 1):
                print(f"  {i}. EN: {en}")
                print(f"     ZH: {zh}")
        
        print("=" * 40)
    
    def _process_sequence_tokens(self, token_ids: list, is_target: bool = False, max_len: int = 30) -> list:
        """
        处理逻辑：
        - 目标序列：添加<bos>（开头）和<eos>（结尾）
        - 统一长度到max_len（超长截断，不足填充）
        """
        if is_target:
            processed = [SPECIAL_TOKENS["<bos>"]] + token_ids + [SPECIAL_TOKENS["<eos>"]]
        else:
            processed = token_ids  # 源序列无需首尾标记
        
        # 截断
        if len(processed) > max_len:
            processed = processed[:max_len]
        # 填充
        else:
            pad_len = max_len - len(processed)
            processed += [SPECIAL_TOKENS["<pad>"]] * pad_len
        
        return processed
    
    def create_padding_mask(self, sequences: torch.Tensor, pad_token_id: int) -> torch.Tensor:
        """创建padding掩码"""
        # sequences: [batch_size, seq_len]
        # 返回: [batch_size, seq_len] 其中True表示padding位置
        return sequences == pad_token_id
    
    def decode_tokens(self, token_ids: list, remove_special_tokens: bool = True) -> str:
        """将token_ids还原成文本"""
        if not token_ids:
            return ""
        
        # 如果需要移除特殊标记
        if remove_special_tokens:
            # 过滤掉特殊标记
            special_token_values = set(SPECIAL_TOKENS.values())
            filtered_tokens = [token_id for token_id in token_ids if token_id not in special_token_values]
        else:
            filtered_tokens = token_ids
        
        # 使用分词器解码
        try:
            text = self._tokenizer.decode(filtered_tokens)
            return text.strip()
        except Exception as e:
            print(f"解码失败: {e}")
            return ""
    
    def _load_dataset(self) -> pd.DataFrame:
        """加载数据集"""
        print("正在加载数据集...")
        dataset = load_dataset(self.config.dataset_name, self.config.dataset_config)
        
        # 提取英文和中文句子
        en_sentences = [item["translation"]["en"] for item in dataset[self.config.split]]
        zh_sentences = [item["translation"]["zh"] for item in dataset[self.config.split]]
        
        # 构建DataFrame
        df = pd.DataFrame({
            "en": en_sentences[:self.config.max_samples],
            "zh": zh_sentences[:self.config.max_samples]
        })
        
        print(f"原始数据量：{len(df)}条")
        return df
    
    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """清洗数据"""
        initial_count = len(df)
        
        # 重置过滤样本收集器
        self._filtered_samples = {
            'contains_original': [],
            'too_short': [],
            'not_translated': []
        }
        
        # 应用清洗
        df["en_clean"] = df["en"].apply(lambda x: self._clean_text(x, is_english=True))
        df["zh_clean"] = df["zh"].apply(lambda x: self._clean_text(x, is_english=False))
        
        # 过滤空句子
        df = df[(df["en_clean"] != "") & (df["zh_clean"] != "")]
        empty_filtered_count = len(df)
        
        # 应用质量检查，过滤低质量翻译对
        quality_mask = df.apply(lambda row: self._is_valid_translation_pair(row["en_clean"], row["zh_clean"]), axis=1)
        df = df[quality_mask]
        final_count = len(df)
        
        if self.config.verbose:
            print("数据清洗统计：")
            print(f"  原始数据量：{initial_count}条")
            print(f"  过滤空句子后：{empty_filtered_count}条 (过滤了{initial_count - empty_filtered_count}条)")
            print(f"  质量检查后：{final_count}条 (过滤了{empty_filtered_count - final_count}条低质量数据)")
            # 打印被过滤的样本
            self._print_filtered_samples()
            print(f"  总过滤率：{((initial_count - final_count) / initial_count * 100):.1f}%")
            
            # 把最多 1000 对数据写入临时文件，确认清洗逻辑正确
            df.head(1000).to_csv("temp_cleaned_data.csv", index=False, encoding="utf-8", sep="\t")
        return df
    
    def _tokenize_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """分词处理"""
        # 分词
        df["en_token_ids"] = df["en_clean"].apply(lambda text: self._tokenizer.encode(text))
        df["zh_token_ids"] = df["zh_clean"].apply(lambda text: self._tokenizer.encode(text))
        return df

    
    
    def _compute_statistics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """计算统计信息"""
        # 计算token序列长度
        df["en_token_length"] = df["en_token_ids"].apply(len)
        df["zh_token_length"] = df["zh_token_ids"].apply(len)
        
        # 动态计算所需的百分位数
        percentiles = [0.95, 0.99, 0.995, 0.999]
        if self.config.length_percentile not in percentiles:
            percentiles.append(self.config.length_percentile)
        percentiles.sort()
        
        # 使用pandas agg统计长度分布
        agg_funcs = [lambda x, p=p: x.quantile(p) for p in percentiles] + ['max']
        
        en_stats = df["en_token_length"].agg(agg_funcs)
        zh_stats = df["zh_token_length"].agg(agg_funcs)
        
        # 创建索引名称
        index_names = [f'p{int(p*100)}' for p in percentiles] + ['max']
        en_stats.index = index_names
        zh_stats.index = index_names
        
        # 确定用于最大长度的百分位数键
        percentile_key = f'p{int(self.config.length_percentile * 100)}'
        
        stats = {
            'english': en_stats,
            'chinese': zh_stats,
            'en_max_len': int(en_stats[percentile_key]),
            'zh_max_len': int(zh_stats[percentile_key])
        }
        
        return stats
    
    def _process_sequences(self, df: pd.DataFrame, en_max_len: int, zh_max_len: int) -> pd.DataFrame:
        """处理序列长度"""
        # 处理源序列（英文）和目标序列（中文）
        df["en_processed"] = df["en_token_ids"].apply(
            lambda x: self._process_sequence_tokens(x, is_target=False, max_len=en_max_len)
        )
        df["zh_processed"] = df["zh_token_ids"].apply(
            lambda x: self._process_sequence_tokens(x, is_target=True, max_len=zh_max_len)
        )
        return df
    
    def _process_data(self) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """完整的数据处理流程"""
        # 1. 加载数据集
        df = self._load_dataset()
        
        # 2. 清洗数据
        df = self._clean_data(df)
        
        # 3. 分词
        df = self._tokenize_data(df)
        
        # 4. 计算统计信息
        stats = self._compute_statistics(df)
        
        # 5. 确定最大长度
        en_max_len = self.config.en_max_len or stats['en_max_len']
        zh_max_len = self.config.zh_max_len or stats['zh_max_len']
        
        # 6. 处理序列
        df = self._process_sequences(df, en_max_len, zh_max_len)

        # 7. 把最多 1000 对处理序列进行还原，写入临时文件，确认处理逻辑正确
        if self.config.verbose:
            # decode_tokens
            df["en_decoded"] = df["en_processed"].apply(lambda x: self.decode_tokens(x))
            df["zh_decoded"] = df["zh_processed"].apply(lambda x: self.decode_tokens(x))
            # 把最多 1000 对处理序列进行还原，写入临时文件，确认处理逻辑正确
            df[["en_decoded", "zh_decoded"]].head(1000).to_csv("temp_processed_data.csv", index=False, encoding="utf-8", sep="\t")
        
        # 更新统计信息
        stats.update({
            'final_en_max_len': en_max_len,
            'final_zh_max_len': zh_max_len
        })
        
        self._df = df
        self._stats = stats
        
        return df, stats
    
    def create_dataloader(self) -> DataLoader:
        """创建数据加载器（公共方法）"""
        df, stats = self._process_data()
        
        dataset = TranslationDataset(self._df)
        dataloader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            drop_last=True
        )
        if self.config.verbose:
            print(stats)
            sample_data = self._get_sample_data()
            print("\n数据示例：")
            print(f"英文原句：{sample_data['en_clean']}")
            print(f"英文ID序列：{sample_data['en_token_ids']}")
            print(f"中文原句：{sample_data['zh_clean']}")
            print(f"中文ID序列：{sample_data['zh_token_ids']}")
            print(f"处理后英文长度：{len(sample_data['en_processed'])}")
            print(f"处理后中文长度：{len(sample_data['zh_processed'])}")
            df.head()
            
        return dataloader
    
    def _get_sample_data(self) -> Dict[str, Any]:
        """获取样本数据用于展示"""

        df = self._df
        sample_data = {
            'en_original': df["en"].iloc[0],
            'zh_original': df["zh"].iloc[0],
            'en_clean': df["en_clean"].iloc[0],
            'zh_clean': df["zh_clean"].iloc[0],
            'en_token_ids': df["en_token_ids"].iloc[0],
            'zh_token_ids': df["zh_token_ids"].iloc[0],
            'en_processed': df["en_processed"].iloc[0],
            'zh_processed': df["zh_processed"].iloc[0]
        }
        return sample_data
    

# 便捷函数，用于快速创建数据处理器和数据加载器
def create_translation_dataloader(
    max_samples: int = 1000,
    batch_size: int = 16,
    en_max_len: Optional[int] = None,
    zh_max_len: Optional[int] = None,
    length_percentile: float = 0.95,
    verbose=True
) -> Tuple[DataLoader, TranslationDataProcessor]:
    """
    工厂方法：创建翻译数据加载器
    
    Returns:
        dataloader: 数据加载器
        processor: 数据处理器实例
    """
    config = DataConfig(
        max_samples=max_samples,
        batch_size=batch_size,
        en_max_len=en_max_len,
        zh_max_len=zh_max_len,
        length_percentile=length_percentile,
        verbose=verbose
    )
    # 使用内部创建标志创建处理器实例
    processor = TranslationDataProcessor(config)
    return processor.create_dataloader(), processor


def main():
    # 使用便捷函数快速创建数据加载器
    dataloader, _ = create_translation_dataloader(
        max_samples=100,
        batch_size=8,
        length_percentile=0.9,  # 使用p90作为最大长度
        verbose=True
    )
    
    # 获取一个批次的数据
    src_batch, tgt_batch = next(iter(dataloader))
    print(f"批次形状：{src_batch.shape}, {tgt_batch.shape}")
    
    return dataloader

if __name__ == '__main__':
    main()