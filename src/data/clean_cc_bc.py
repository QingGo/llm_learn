import re
import os
from typing import List, Dict, Any, Optional, Iterator, Union, Tuple

import torch
from torch.utils.data import IterableDataset
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing
from datasets import load_dataset, concatenate_datasets


def clean_text(text: str) -> str:
    """
    清洗文本，去除特殊符号，保留有意义的标点符号和基本内容

    Args:
        text: 原始文本

    Returns:
        str: 清洗后的标准化文本

    处理步骤:
    1. 移除HTML标签
    2. 标准化空白字符（连续空格、制表符、换行符替换为单个空格）
    3. 过滤特殊字符，仅保留英文、数字、常见标点符号和空格
    4. 去除首尾空白字符
    """
    # 移除HTML标签
    text = re.sub(r"<[^>]+>", "", text)

    # 移除连续的空格、制表符、换行符，替换为单个空格
    text = re.sub(r"\s+", " ", text)

    # 保留英文、数字、常见标点符号（,.!?;:'"()[]{}）和空格
    # 移除其他特殊符号
    text = re.sub(r'[^A-Za-z0-9,.!?;:\'"()\[\]{} \n]', "", text)

    # 去除首尾空格
    text = text.strip()

    return text


# 测试文本清洗函数
def split_into_sentences(text: str) -> List[str]:
    """
    将文本分割成句子

    Args:
        text: 清洗后的文本

    Returns:
        句子列表
    """
    # 首先按段落分割（两个或多个换行符）
    paragraphs = re.split(r"\n\s*\n", text)

    sentences = []
    for paragraph in paragraphs:
        # 跳过空段落
        if not paragraph.strip():
            continue

        # 使用正则表达式分割句子，考虑常见的句子结束符号
        # 句子结束符后跟空格或换行符或文本结束
        para_sentences = re.split(r"(?<=[.!?;])\s+", paragraph)

        # 过滤空句子
        para_sentences = [s.strip() for s in para_sentences if s.strip()]
        sentences.extend(para_sentences)

    return sentences


def split_long_text(text: str, max_length: int = 5000) -> List[str]:
    """
    智能分割长文本，优先在句子边界处切分，保持语义完整性
    适用于处理Book Corpus等包含长段落的数据集

    Args:
        text: 原始长文本字符串
        max_length: 每个文本块的最大长度，默认为5000字符

    Returns:
        List[str]: 分割后的文本块列表，每个块长度不超过max_length

    算法说明:
    1. 如果文本长度小于等于max_length，直接返回原文本
    2. 尝试将文本分割为句子，然后按句子组合成块
    3. 对于超长句子（超过max_length），进行强制分割并递归处理剩余部分
    """
    if len(text) <= max_length:
        return [text]

    chunks = []

    # 尝试在句子边界处分割
    sentences = split_into_sentences(text)
    current_chunk = ""

    for sentence in sentences:
        # 如果添加当前句子会超过最大长度，则保存当前块并开始新块
        if len(current_chunk) + len(sentence) + 1 > max_length:
            if current_chunk:  # 确保当前块不为空
                chunks.append(current_chunk)
                current_chunk = sentence
            else:
                # 如果单个句子就超过最大长度，直接分割该句子
                chunks.append(sentence[:max_length])
                remaining = sentence[max_length:]
                # 递归处理剩余部分
                chunks.extend(split_long_text(remaining, max_length))
        else:
            if current_chunk:  # 不是第一个句子，添加空格
                current_chunk += " " + sentence
            else:  # 第一个句子
                current_chunk = sentence

    # 添加最后一个块
    if current_chunk:
        chunks.append(current_chunk)

    return chunks


def create_bpe_tokenizer(vocab_size: int = 30000) -> Tuple[Tokenizer, BpeTrainer]:
    """
    创建Byte-Pair Encoding (BPE)分词器及其训练器

    Args:
        vocab_size: 词汇表大小，默认为30000

    Returns:
        Tuple[Tokenizer, BpeTrainer]: 初始化的BPE分词器和训练器对象

    配置说明:
    - 使用Whitespace作为预分词器
    - 设置UNK、CLS、SEP、PAD、MASK等特殊标记
    - 最小频率阈值设为2，过滤极低频词汇
    """
    # 创建BPE模型
    tokenizer = Tokenizer(BPE(unk_token="<UNK>"))

    # 设置预分词器（按空格分词）
    tokenizer.pre_tokenizer = Whitespace()

    # 设置特殊标记
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["<UNK>", "<CLS>", "<SEP>", "<PAD>", "<MASK>"],
        min_frequency=2,
    )

    return tokenizer, trainer


def train_tokenizer(
    tokenizer: Tokenizer, trainer: BpeTrainer, texts: List[str]
) -> None:
    """
    训练BPE分词器

    Args:
        tokenizer: 要训练的分词器
        trainer: BPE训练器
        texts: 用于训练的文本列表
    """

    # 创建一个文本迭代器来训练分词器
    def text_iterator():
        for text in texts:
            yield text

    # 训练分词器
    tokenizer.train_from_iterator(text_iterator(), trainer=trainer)

    # 设置后处理器，添加CLS和SEP标记
    tokenizer.post_processor = TemplateProcessing(
        single="<CLS> $A <SEP>",
        pair="<CLS> $A <SEP> $B:1 <SEP>:1",
        special_tokens=[
            ("<CLS>", tokenizer.token_to_id("<CLS>")),
            ("<SEP>", tokenizer.token_to_id("<SEP>")),
        ],
    )


def tokenize_text(tokenizer: Tokenizer, text: str) -> Dict[str, List[int]]:
    """
    使用分词器对文本进行tokenize

    Args:
        tokenizer: 训练好的分词器
        text: 要分词的文本

    Returns:
        包含token IDs的字典
    """
    # 对文本进行分词
    encoding = tokenizer.encode(text)

    return {
        "input_ids": encoding.ids,
        "attention_mask": encoding.attention_mask,
        "token_type_ids": encoding.type_ids,
    }


def convert_to_tensors(
    token_dict: Dict[str, List[int]],
    max_length: Optional[int] = None,
    padding: bool = False,
    pad_token_id: int = 0,
) -> Dict[str, torch.Tensor]:
    """
    将token字典转换为PyTorch张量

    Args:
        token_dict: 包含token IDs的字典
        max_length: 最大序列长度，超过将被截断
        padding: 是否进行填充到max_length
        pad_token_id: 填充使用的token ID，默认为0

    Returns:
        包含PyTorch张量的字典
    """
    result = {}

    # 获取最长序列长度
    seq_length = None
    if max_length is not None:
        seq_length = max_length
    else:
        # 如果没有指定max_length，使用实际长度
        for key in token_dict:
            if seq_length is None or len(token_dict[key]) > seq_length:
                seq_length = len(token_dict[key])

    # 转换每个字段为张量
    for key, values in token_dict.items():
        # 截断
        if len(values) > seq_length:
            values = values[:seq_length]
        # 填充
        elif padding and len(values) < seq_length:
            # 对于attention_mask，填充0；对于其他，使用指定的pad_token_id
            pad_value = 0 if key == "attention_mask" else pad_token_id
            values = values + [pad_value] * (seq_length - len(values))

        # 转换为张量
        result[key] = torch.tensor(values, dtype=torch.long)

    return result


def check_data_consistency(
    data: Dict[str, Any],
    required_fields: List[str] = None,
    max_seq_length: Optional[int] = None,
    min_seq_length: int = 1,
) -> Dict[str, bool]:
    """
    检查数据格式一致性

    Args:
        data: 要检查的数据字典
        required_fields: 必需的字段列表，默认为["input_ids", "attention_mask"]
        max_seq_length: 最大序列长度限制
        min_seq_length: 最小序列长度限制，默认为1

    Returns:
        包含检查结果的字典
    """
    if required_fields is None:
        required_fields = ["input_ids", "attention_mask"]

    results = {
        "all_fields_present": True,
        "all_fields_valid": True,
        "sequence_length_valid": True,
        "special_tokens_valid": True,
    }

    # 检查必需字段是否存在
    for field in required_fields:
        if field not in data:
            results["all_fields_present"] = False
            print(f"错误: 缺少必需字段 {field}")
        elif not isinstance(data[field], (list, torch.Tensor)):
            results["all_fields_valid"] = False
            print(f"错误: 字段 {field} 不是有效的列表或张量类型")

    # 检查序列长度
    if "input_ids" in data:
        seq_length = (
            len(data["input_ids"])
            if isinstance(data["input_ids"], list)
            else data["input_ids"].shape[0]
        )

        if max_seq_length is not None and seq_length > max_seq_length:
            results["sequence_length_valid"] = False
            print(f"错误: 序列长度 {seq_length} 超过最大限制 {max_seq_length}")

        if seq_length < min_seq_length:
            results["sequence_length_valid"] = False
            print(f"错误: 序列长度 {seq_length} 小于最小限制 {min_seq_length}")

    # 检查字段长度一致性
    if "input_ids" in data:
        main_length = (
            len(data["input_ids"])
            if isinstance(data["input_ids"], list)
            else data["input_ids"].shape[0]
        )
        for field in data:
            if field != "input_ids":
                field_length = (
                    len(data[field])
                    if isinstance(data[field], list)
                    else data[field].shape[0]
                )
                if field_length != main_length:
                    results["all_fields_valid"] = False
                    print(
                        f"错误: 字段 {field} 长度 {field_length} 与input_ids长度 {main_length} 不一致"
                    )

    # 检查特殊token（如果有input_ids）
    if "input_ids" in data and isinstance(data["input_ids"], list):
        # 检查是否包含非法负值
        if any(id < 0 for id in data["input_ids"]):
            results["special_tokens_valid"] = False
            print("错误: input_ids 包含非法负值")

    # 整体一致性
    results["is_consistent"] = (
        results["all_fields_present"]
        and results["all_fields_valid"]
        and results["sequence_length_valid"]
        and results["special_tokens_valid"]
    )

    return results


class TextProcessingDataset(IterableDataset):
    """
    文本处理可迭代数据集，集成了完整的文本处理流水线
    
    该类实现了从原始文本到PyTorch张量的完整转换过程，包括清洗、分句、
    分词和数据一致性检查，适用于大规模语言模型的训练数据准备。

    Args:
        dataset: Hugging Face datasets对象
        tokenizer: 预训练或新训练的BPE分词器
        text_column: 文本列名，默认为'text'
        max_seq_length: 最大序列长度，超过将被截断，默认1024
        min_seq_length: 最小序列长度，低于将被过滤，默认5
        book_threshold: 长文本分割阈值，超过将被分割为多个块，默认2000
        pad_token_id: 填充token ID，默认50256
        skip_invalid: 是否跳过无效数据，默认True
    """

    def __init__(
        self,
        dataset,
        tokenizer: Tokenizer,
        text_column: str = "text",
        max_seq_length: Optional[int] = 1024,
        min_seq_length: int = 5,
        book_threshold: int = 2000,
        pad_token_id: int = 50256,
        skip_invalid: bool = True,
    ):
        # 使用with_format方法将Hugging Face数据集直接转换为PyTorch格式
        # 这样可以在数据迭代时直接获得PyTorch张量，提高处理效率
        self.dataset = dataset.with_format("torch")
        self.tokenizer = tokenizer
        self.text_column = text_column
        self.max_seq_length = max_seq_length
        self.min_seq_length = min_seq_length
        self.book_threshold = book_threshold
        self.pad_token_id = pad_token_id
        self.skip_invalid = skip_invalid

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        """迭代器实现，处理每条数据并返回PyTorch张量"""
        for example in self.dataset:
            # 检查文本字段是否存在
            if self.text_column not in example:
                if not self.skip_invalid:
                    raise ValueError(f"文本字段 {self.text_column} 不存在")
                continue

            text = example[self.text_column]

            # 1. 清洗文本
            cleaned_text = clean_text(text)

            # 2. 分句（根据文本长度决定是否需要分割）
            if len(cleaned_text) > self.book_threshold:
                # 长文本分割为多个块
                text_blocks = split_long_text(cleaned_text, max_length=2000)
            else:
                # 短文本作为单个块
                text_blocks = [cleaned_text]

            # 处理每个文本块
            for text_block in text_blocks:
                # 3. 分词
                tokenized = tokenize_text(self.tokenizer, text_block)

                # 4. 转换为张量
                tensors = convert_to_tensors(
                    tokenized,
                    max_length=self.max_seq_length,
                    padding=True,
                    pad_token_id=self.pad_token_id,
                )

                # 5. 检查数据一致性
                consistency = check_data_consistency(
                    tensors,
                    max_seq_length=self.max_seq_length,
                    min_seq_length=self.min_seq_length,
                )

                # 如果数据有效，则返回
                if consistency["is_consistent"]:
                    yield tensors
                elif not self.skip_invalid:
                    raise ValueError(f"数据一致性检查失败: {consistency}")
                # 否则跳过无效数据


def load_hf_datasets() -> Tuple[Any, Any]:
    """
    加载Common Crawl和Book Corpus替代数据集

    Returns:
        tuple: (common_crawl_ds, book_corpus_ds)

    数据集说明:
    - Common Crawl: 使用Creative Commons许可的子集，包含网络爬虫文本
    - Book Corpus: 使用the_pile_books3_minus_gutenberg作为替代，避免版权问题
    
    注意事项:
    - 设置verification_mode="no_checks"以避免单文件下载时的NonMatchingSplitsSizesError
    - 数据集缓存在./data目录下以避免重复下载
    """
    print("加载数据集...")
    common_crawl_ds = load_dataset(
        "BramVanroy/CommonCrawl-CreativeCommons",
        "CC-MAIN-2019-30-eng",
        data_files="data/CC-MAIN-2019-30/eng/001_00004.parquet",
        split="train",
        cache_dir="./data",
    )

    # 大部分 Book 数据集都因为版权问题不可用了，这里使用 the_pile_books3_minus_gutenberg 数据集
    book_corpus_ds = load_dataset(
        "SaylorTwift/the_pile_books3_minus_gutenberg",
        name="default",
        data_files="data/train-00000-of-00213-312fd8d7a3c58a63.parquet",
        split="train",
        cache_dir="./data",
        # 只下载单文件会报 NonMatchingSplitsSizesError，也许是因为最终数据量和 README.md 里指定的不符
        verification_mode="no_checks",
    )

    print(f"实际加载的Common Crawl记录数: {len(common_crawl_ds)}")
    print(f"实际加载的Book Corpus记录数: {len(book_corpus_ds)}")

    return common_crawl_ds, book_corpus_ds


def test_text_processing(common_crawl_ds, book_corpus_ds):
    """
    测试文本清洗和长文本分割功能

    Args:
        common_crawl_ds: Common Crawl数据集
        book_corpus_ds: Book Corpus数据集
    """
    # 对样例数据进行清洗并显示结果
    print("\n清洗后的样例数据:")
    sample_cc = common_crawl_ds[0]["text"][:1000]  # 取前1000字符进行测试
    sample_bc = book_corpus_ds[0]["text"][:1000]

    cleaned_cc = clean_text(sample_cc)
    cleaned_bc = clean_text(sample_bc)

    print(f"\nCommon Crawl 清洗前长度: {len(sample_cc)}, 清洗后长度: {len(cleaned_cc)}")
    print(f"Common Crawl 清洗后前200字符: {cleaned_cc[:200]}...")

    print(f"\nBook Corpus 清洗前长度: {len(sample_bc)}, 清洗后长度: {len(cleaned_bc)}")
    print(f"Book Corpus 清洗后前200字符: {cleaned_bc[:200]}...")

    # 测试对实际Book Corpus数据的长文本分割
    print("\n测试实际Book Corpus数据的长文本分割:")
    book_sample = book_corpus_ds[0]["text"][:10000]  # 取前10000字符进行测试
    book_chunks = split_long_text(book_sample, max_length=2000)
    print(f"原始长度: {len(book_sample)}")
    print(f"分割后的块数量: {len(book_chunks)}")
    print(f"各块长度: {[len(chunk) for chunk in book_chunks]}")


def prepare_training_data(common_crawl_ds, book_corpus_ds):
    """
    准备分词器训练数据

    Args:
        common_crawl_ds: Common Crawl数据集
        book_corpus_ds: Book Corpus数据集

    Returns:
        List[str]: 训练文本列表
    """
    print("\n合并数据集并准备分词器训练数据:")

    # 从两个数据集中获取清洗后的文本
    training_texts = []

    # 使用全量数据
    for example in common_crawl_ds:
        # 确保example是字典并且包含'text'键
        cleaned = clean_text(example["text"])
        # 将长文本分割成小块
        chunks = split_long_text(cleaned, max_length=2000)
        training_texts.extend(chunks)

    # 处理Book Corpus数据
    # 直接从book_corpus_ds中提取文本并分割
    for example in book_corpus_ds:
        cleaned = clean_text(example["text"])
        chunks = split_long_text(cleaned, max_length=2000)
        training_texts.extend(chunks)

    print(f"训练文本总数: {len(training_texts)}")

    return training_texts


def train_or_load_tokenizer(
    common_crawl_ds, book_corpus_ds, tokenizer_path="./data/tokenizer.json", vocab_size: int = 30000
):
    """
    训练新分词器或加载已有的预训练分词器

    Args:
        common_crawl_ds: Common Crawl数据集
        book_corpus_ds: Book Corpus数据集
        tokenizer_path: 分词器保存路径，默认"./data/tokenizer.json"
        vocab_size: 词汇表大小，当需要训练新分词器时使用，默认30000

    Returns:
        Tokenizer: 训练好的分词器

    工作流程:
    1. 检查是否存在已保存的分词器文件
    2. 如果存在，直接加载并返回
    3. 如果不存在，准备训练数据，创建并训练新的分词器
    4. 保存训练好的分词器供后续使用
    """
    # 尝试加载已有的tokenizer，如果不存在则训练新的
    if os.path.exists(tokenizer_path):
        print(f"从 {tokenizer_path} 加载已训练的分词器...")
        tokenizer = Tokenizer.from_file(tokenizer_path)
    else:
        print("训练新的分词器...")
        training_texts = prepare_training_data(common_crawl_ds, book_corpus_ds)
        tokenizer, trainer = create_bpe_tokenizer(vocab_size=vocab_size)
        train_tokenizer(tokenizer, trainer, training_texts)

        # 保存分词器, 确保目录存在
        os.makedirs(os.path.dirname(tokenizer_path), exist_ok=True)
        tokenizer.save(tokenizer_path)
        print(f"分词器已保存到 {tokenizer_path}")

    # 测试分词器
    sample_text = "Hello world! This is a test sentence."
    encoding = tokenizer.encode(sample_text)
    print(f"\n原始文本: {sample_text}")
    print(f"Token IDs: {encoding.ids[:10]}...")
    print(f"Tokens: {[tokenizer.id_to_token(id) for id in encoding.ids[:10]]}...")

    return tokenizer


def create_merged_dataset(common_crawl_ds, book_corpus_ds):
    """
    创建合并数据集

    Args:
        common_crawl_ds: Common Crawl数据集
        book_corpus_ds: Book Corpus数据集

    Returns:
        Dataset: 合并后的数据集
    """
    print("\n创建合并数据集:")

    # 使用全量数据合并
    merged_dataset = concatenate_datasets([common_crawl_ds, book_corpus_ds])
    print(f"合并后数据集大小: {len(merged_dataset)}")

    return merged_dataset


def create_pt_dataset(
    max_seq_length: int, skip_invalid: bool = True, tokenizer_vocab_size: int = 30000
) -> Union[TextProcessingDataset, IterableDataset, IterableDataset, Tokenizer]:
    """
    合并处理逻辑，执行数据清洗和预处理流程，返回处理后的 PyTorch 数据集

    Args:
        max_seq_length: 最大序列长度，超过将被截断
        skip_invalid: 是否跳过无效数据，默认True
        tokenizer_vocab_size: 分词器词汇表大小，仅在需要训练新分词器时使用，默认30000

    Returns:
        Tuple[TextProcessingDataset, Dataset, Dataset, Tokenizer]: 
            处理后的PyTorch数据集、原始Common Crawl数据集、原始Book Corpus数据集和分词器
    """
    # 1. 加载数据集
    common_crawl_ds, book_corpus_ds = load_hf_datasets()

    # 2. 创建合并数据集
    merged_dataset = create_merged_dataset(common_crawl_ds, book_corpus_ds)

    # 3. 训练或加载分词器
    tokenizer = train_or_load_tokenizer(common_crawl_ds, book_corpus_ds)

    return (
        TextProcessingDataset(
            dataset=merged_dataset,
            tokenizer=tokenizer,
            max_seq_length=max_seq_length,
            skip_invalid=skip_invalid,
        ),
        common_crawl_ds,
        book_corpus_ds,
        tokenizer,
    )


def main():
    """
    主函数，展示整个数据处理流程并进行测试
    
    功能:
    1. 创建PyTorch数据集
    2. 测试文本处理功能
    3. 迭代数据集并展示样本
    """
    dataset, common_crawl_ds, book_corpus_ds, tokenizer = create_pt_dataset(max_seq_length=200)
    print(f"字典大小：{tokenizer.get_vocab_size()}")
    test_text_processing(common_crawl_ds, book_corpus_ds)

    # 迭代数据集
    print("\n迭代合并数据集结果:")
    for i, data in enumerate(dataset):
        print(f"\n数据样本 {i + 1}:")
        print(f"input_ids 形状: {data['input_ids'].shape}")
        print(f"input_ids 值: {data['input_ids']}")
        print(f"attention_mask 形状: {data['attention_mask'].shape}")
        print(f"token_type_ids 形状: {data['token_type_ids'].shape}")
        # 解码
        decoded_text = tokenizer.decode(data["input_ids"].tolist())
        print(f"解码后的文本: {decoded_text}")

        if i >= 2:
            break


if __name__ == "__main__":
    """
    主入口，执行数据清洗与预处理流程
    
    本模块用于处理Common Crawl和Book Corpus数据集，
    为LLM训练准备高质量的文本数据。
    """
    main()
    '''
    加载数据集...
    实际加载的Common Crawl记录数: 230818
    实际加载的Book Corpus记录数: 905

    创建合并数据集:
    合并后数据集大小: 231723
    从 ./data/tokenizer.json 加载已训练的分词器...

    原始文本: Hello world! This is a test sentence.
    Token IDs: [1, 8675, 761, 5, 374, 94, 53, 1200, 7477, 11]...
    Tokens: ['<CLS>', 'Hello', 'world', '!', 'This', 'is', 'a', 'test', 'sentence', '.']...
    字典大小：30000

    清洗后的样例数据:

    Common Crawl 清洗前长度: 1000, 清洗后长度: 999
    Common Crawl 清洗后前200字符: Pushkar town is one of the most popular places in Rajasthan among the tourists, both domestic and international. Pushkar Camel Fair or Pushkar Mela is synonymous with this town. Even though there are ...

    Book Corpus 清洗前长度: 1000, 清洗后长度: 950
    Book Corpus 清洗后前200字符: Table of Contents Title Page Dedication Part 1 : OVERVIEW Introduction How Did I Get Here? About the Book How to Use This Book What Is a Triathlon? The History of Triathlon Triathlon Distances Triathl...

    测试实际Book Corpus数据的长文本分割:
    原始长度: 10000
    分割后的块数量: 6
    各块长度: [1968, 1919, 1998, 1888, 1970, 164]

    迭代合并数据集结果:

    数据样本 1:
    input_ids 形状: torch.Size([200])
    input_ids 值: tensor([    1, 17398, 10699,  1854,    94,   200,   102,    86,   470,  2848,
            3143,    82, 11884,   100,   429,  1676,    86, 16667,    10,   834,
            6941,   101,  3304,    11, 17398, 10699,  2545,   145,  6261,    90,
            17398, 10699,  4290,    53,    94, 28955,   153,   208,  1854,    11,
            2639,   910,   388,   148,   635,   325, 14426,    58,  2508,   220,
            4031,   102,   369,    94,   100,  2848,   100,   208,   200,    11,
            33,   222,   384,  8795, 17398, 10699,    85,    53,  2959,  4201,
            1048,    86,   895,   998,   730,   101,   222,   415,  8310,   208,
            4797,  1741,    11,    33,  1196,   208,  1854,   120,   635,  1013,
            99,  4537,    11,  7877,   333,  4219, 12531,  1676,    86, 16667,
            101, 14274,    10,    33,  3138,    97,  1712,   257,  4876,    82,
            208, 17398, 10699,  4290,    53,  1720,    11,   150,    27,  2415,
            6261,    94,  8310,   190,    53,   998,  3269,  5410,  7722,   636,
            469,    11,   150,   750,   102, 28391, 27116,   202,   844,    86,
            761,  8795,    86, 10783,  1854,  1048,   208,  1787,  1993,  5641,
            82,  5998,    11,  8461,    97,   393,   352, 17398, 10699,     5,
            17398, 10699,    94,    53, 10783,  1854,   120, 13742,   141,  1582,
            53, 11415,  5225,  1133, 17398, 10699,  6621,  1504,   153,    86,
            4797, 18515,   245,  8982,    11,   713,  2838,   635, 17730,   102,
            18515,   245,    82,  3522,    11,  2639,   910,   635,   532,  1550])
    attention_mask 形状: torch.Size([200])
    token_type_ids 形状: torch.Size([200])
    解码后的文本: Push kar town is one of the most popular places in Raj as than among the tourists , both domestic and international . Push kar Cam el Fair or Push kar Mel a is synonymous with this town . Even though there are many other cattle f airs but none of them is as popular as this one . I have been visiting Push kar on a regular basis during the last few years and have also attended this famous event . I love this town for many things it offers . Given its huge popularity among the tourists and bloggers , I decided to share my opinion in this Push kar Mel a blog . The C attle Fair is attended by a few hundred thousand visitors every year . The number of acclaimed photographers from around the world visiting the holy town during this period itself runs in thousands . Everything to know about Push kar ! Push kar is a holy town for Hind us having a sacred lake called Push kar Lake along with the famous Brah ma temple . There arent many temples of Brah ma in India . Even though many people claim
    '''