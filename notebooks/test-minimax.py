from transformers import AutoTokenizer
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from safetensors import safe_open
from huggingface_hub import hf_hub_download
from functools import lru_cache
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass

plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica', 'Noto Sans CJK JP', 'WenQuanYi Micro Hei']
plt.rcParams['axes.unicode_minus'] = False


@dataclass
class ModelConfig:
    """
    模型配置类
    
    存储模型相关的配置信息，包括模型名称、权重路径等
    """
    model_name: str = "MiniMaxAI/MiniMax-M2.5"
    weight_path_base: str = "/root/.cache/huggingface/hub/models--MiniMaxAI--MiniMax-M2.5/snapshots/f710177d938eff80b684d42c5aa84b382612f21f/"
    lm_head_file: str = "model-00124-of-00126.safetensors"
    embedding_file: str = "model-00000-of-00126.safetensors"


@dataclass
class AnalysisConfig:
    """
    分析配置类
    
    存储分析相关的配置参数，包括异常token ID、阈值等
    """
    anomaly_token_id: int = 190467
    study_token_ids: List[int] = None
    special_token_threshold: int = 200000
    max_plot_tokens: int = 1000
    neighbor_k: int = 220
    similarity_threshold: float = 0.18
    default_bins: int = 50
    default_figsize: Tuple[float, float] = (8.6, 8.2)
    
    def __post_init__(self):
        if self.study_token_ids is None:
            self.study_token_ids = [177085, 190468, 183969]


@lru_cache
def bytes_to_unicode() -> Dict[int, str]:
    """
    创建字节到Unicode字符的映射
    
    用于GPT-2风格的字节级BPE编码，将字节值映射到可打印的Unicode字符
    这样可以确保所有256个字节值都有对应的可打印字符
    
    Returns:
        Dict[int, str]: 字节值到Unicode字符的映射字典
    """
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, map(chr, cs)))


@lru_cache
def unicode_to_bytes() -> Dict[str, int]:
    """
    创建Unicode字符到字节的映射
    
    bytes_to_unicode()的反向映射
    
    Returns:
        Dict[str, int]: Unicode字符到字节值的映射字典
    """
    btou = bytes_to_unicode()
    return {v: k for k, v in btou.items()}


def bytelevel_encode(text: str) -> str:
    """
    将文本编码为字节级表示
    
    Args:
        text: 要编码的文本
        
    Returns:
        str: 编码后的Unicode字符串
    """
    btou = bytes_to_unicode()
    return "".join(btou[b] for b in text.encode("utf-8"))


def bytelevel_decode_bytes(encoded: str) -> bytes:
    """
    将字节级编码解码回字节
    
    Args:
        encoded: 字节级编码的字符串
        
    Returns:
        bytes: 解码后的字节数组
    """
    utob = unicode_to_bytes()
    return bytes(utob[ch] for ch in encoded)


def bytelevel_decode(encoded: str, errors: str = "strict") -> str:
    """
    将字节级编码解码回文本
    
    Args:
        encoded: 字节级编码的字符串
        errors: 错误处理策略
        
    Returns:
        str: 解码后的文本
    """
    return bytelevel_decode_bytes(encoded).decode("utf-8", errors=errors)


def safe_token_text(token: str) -> str:
    """
    安全地解码token文本
    
    尝试解码token，如果失败则返回原始token
    
    Args:
        token: 要解码的token
        
    Returns:
        str: 解码后的文本或原始token
    """
    try:
        return bytelevel_decode(token)
    except (UnicodeDecodeError, KeyError):
        return token


def build_reverse_vocab(tokenizer: AutoTokenizer) -> Dict[int, str]:
    """
    构建反向词表
    
    将token ID映射到可读的token文本
    
    Args:
        tokenizer: 分词器对象
        
    Returns:
        Dict[int, str]: token ID到token文本的映射
    """
    vocab = tokenizer.get_vocab()
    reverse_vocab = {v: safe_token_text(k) for k, v in vocab.items()}
    for token, token_id in tokenizer.get_added_vocab().items():
        reverse_vocab[token_id] = token
    for token_id in getattr(tokenizer, "all_special_ids", []):
        reverse_vocab.setdefault(token_id, tokenizer.convert_ids_to_tokens(token_id))
    return reverse_vocab


def token_text(row_idx: int, reverse_vocab: Dict[int, str], tokenizer: AutoTokenizer) -> str:
    """
    获取指定token ID对应的文本
    
    Args:
        row_idx: token ID
        reverse_vocab: 反向词表
        tokenizer: 分词器对象
        
    Returns:
        str: token对应的文本
    """
    if row_idx in reverse_vocab:
        return reverse_vocab[row_idx]
    token = tokenizer.convert_ids_to_tokens(row_idx)
    if token is None:
        return f"<missing:{row_idx}>"
    decoded = safe_token_text(token)
    reverse_vocab[row_idx] = decoded
    return decoded


def load_lm_head(weight_path_base: str, lm_head_file: str = "model-00124-of-00126.safetensors") -> torch.Tensor:
    """
    加载lm_head权重
    
    Args:
        weight_path_base: 权重文件基础路径
        lm_head_file: lm_head权重文件名
        
    Returns:
        torch.Tensor: lm_head权重张量
    """
    with safe_open(weight_path_base + lm_head_file, framework="pt", device="cpu") as f:
        return f.get_tensor("lm_head.weight")


def load_embedding(weight_path_base: str, embedding_file: str = "model-00000-of-00126.safetensors") -> torch.Tensor:
    """
    加载embedding权重
    
    Args:
        weight_path_base: 权重文件基础路径
        embedding_file: embedding权重文件名
        
    Returns:
        torch.Tensor: embedding权重张量
    """
    with safe_open(weight_path_base + embedding_file, framework="pt", device="cpu") as f:
        return f.get_tensor("model.embed_tokens.weight")


def compute_norms(tensor: torch.Tensor) -> torch.Tensor:
    """
    计算张量沿最后一维的模长
    
    Args:
        tensor: 输入张量，形状为 (n, d)
        
    Returns:
        torch.Tensor: 模长向量，形状为 (n,)
    """
    return tensor.norm(dim=1)


def get_top_bottom_indices(norms: torch.Tensor, top_n: int = 10, bottom_n: int = 10) -> Dict[str, Any]:
    """
    获取模长最大和最小的索引
    
    Args:
        norms: 模长向量
        top_n: 返回的最大模长数量
        bottom_n: 返回的最小模长数量
        
    Returns:
        Dict[str, Any]: 包含排序后的索引和模长的字典
    """
    sorted_norms, sorted_indices = torch.sort(norms, descending=True)
    return {
        'top_indices': sorted_indices[:top_n],
        'bottom_indices': sorted_indices[-bottom_n:],
        'sorted_norms': sorted_norms,
        'sorted_indices': sorted_indices
    }


def print_top_bottom_norms(norms: torch.Tensor, top_n: int = 10, bottom_n: int = 10) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    打印模长最大和最小的行索引
    
    Args:
        norms: 模长向量
        top_n: 打印的最大模长数量
        bottom_n: 打印的最小模长数量
        
    Returns:
        Tuple[torch.Tensor, torch.Tensor]: 排序后的模向量和索引
    """
    result = get_top_bottom_indices(norms, top_n, bottom_n)
    print(f"前 {top_n} 行的 norm 排序:")
    for i in result['top_indices']:
        print(f"行索引: {i.item()}, 模长: {result['sorted_norms'][i].item()}")
    print("...")
    print(f"后 {bottom_n} 行的 norm 排序:")
    for i in result['bottom_indices']:
        print(f"行索引: {i.item()}, 模长: {result['sorted_norms'][i].item()}")
    return result['sorted_norms'], result['sorted_indices']


def get_row_rank(norms: torch.Tensor, sorted_indices: torch.Tensor, row_idx: int) -> int:
    """
    获取指定行的排名
    
    Args:
        norms: 模长向量
        sorted_indices: 排序后的索引
        row_idx: 要查询的行索引
        
    Returns:
        int: 排名（从1开始）
    """
    rank = torch.where(sorted_indices == row_idx)[0][0].item()
    return rank + 1


def print_row_rank(norms: torch.Tensor, sorted_indices: torch.Tensor, row_idx: int):
    """
    打印指定行的模长和排名
    
    Args:
        norms: 模长向量
        sorted_indices: 排序后的索引
        row_idx: 要打印的行索引
    """
    rank = get_row_rank(norms, sorted_indices, row_idx)
    print(f"lm_head {row_idx} 行的模长: {norms[row_idx].item()}, 排名: {rank}")


def plot_vector_distribution(vector: np.ndarray, title: str, bins: int = 50):
    """
    绘制向量的数值分布直方图
    
    Args:
        vector: 要绘制的向量
        title: 图表标题
        bins: 直方图的箱数
    """
    plt.hist(vector, bins=bins, density=True)
    plt.title(title)
    plt.xlabel("Value")
    plt.ylabel("Density")
    plt.show()


def compute_similarity(tensor: torch.Tensor, target_vector: torch.Tensor) -> torch.Tensor:
    """
    计算张量与目标向量的余弦相似度
    
    Args:
        tensor: 输入张量
        target_vector: 目标向量
        
    Returns:
        torch.Tensor: 相似度向量
    """
    return torch.cosine_similarity(tensor, target_vector.unsqueeze(0), dim=1)


def get_similarity_analysis(similarities: torch.Tensor, target_row: int, reverse_vocab: Dict[int, str], 
                           tokenizer: AutoTokenizer, top_n: int = 20, bottom_n: int = 10) -> Dict[str, Any]:
    """
    获取相似度分析结果
    
    Args:
        similarities: 相似度向量
        target_row: 目标行索引
        reverse_vocab: 反向词表
        tokenizer: 分词器
        top_n: 返回的最相似数量
        bottom_n: 返回的最不相似数量
        
    Returns:
        Dict[str, Any]: 包含相似度分析结果的字典
    """
    target_token = token_text(target_row, reverse_vocab, tokenizer)
    top_indices = similarities.argsort(descending=True)[:top_n]
    bottom_indices = similarities.argsort(descending=False)[:bottom_n]
    average_similarity = similarities.mean().item()
    
    return {
        'target_token': target_token,
        'target_row': target_row,
        'top_indices': top_indices,
        'bottom_indices': bottom_indices,
        'average_similarity': average_similarity,
        'similarities': similarities
    }


def print_similarity_analysis(similarities: torch.Tensor, target_row: int, reverse_vocab: Dict[int, str], 
                             tokenizer: AutoTokenizer, top_n: int = 20, bottom_n: int = 10):
    """
    打印相似度分析结果
    
    Args:
        similarities: 相似度向量
        target_row: 目标行索引
        reverse_vocab: 反向词表
        tokenizer: 分词器
        top_n: 打印的最相似数量
        bottom_n: 打印的最不相似数量
    """
    result = get_similarity_analysis(similarities, target_row, reverse_vocab, tokenizer, top_n, bottom_n)
    print(f"target 行索引: {result['target_row']}, token: {repr(result['target_token'])}")
    
    print(f"top {top_n}:")
    for i in result['top_indices']:
        print(f"行索引: {i.item()}, token: {repr(token_text(i.item(), reverse_vocab, tokenizer))}, 相似度: {result['similarities'][i].item()}")
    
    print(f"bottom {bottom_n}:")
    for i in result['bottom_indices']:
        print(f"行索引: {i.item()}, token: {repr(token_text(i.item(), reverse_vocab, tokenizer))}, 相似度: {result['similarities'][i].item()}")
    
    print(f"平均相似度: {result['average_similarity']}")


def lm_topk_neighbors(lm_head: torch.Tensor, row_idx: int, k: int = 64) -> Tuple[List[int], torch.Tensor]:
    """
    获取lm_head中指定行的top-k邻居
    
    Args:
        lm_head: lm_head权重张量
        row_idx: 目标行索引
        k: 邻居数量
        
    Returns:
        Tuple[List[int], torch.Tensor]: 邻居ID列表和相似度向量
    """
    sims = torch.cosine_similarity(lm_head, lm_head[row_idx].unsqueeze(0), dim=1)
    order = sims.argsort(descending=True)
    order = order[order != row_idx][:k]
    return [idx.item() for idx in order], sims


def normalize_token_text(token: str) -> str:
    """
    标准化token文本
    
    将特殊格式的token转换为标准格式
    
    Args:
        token: 原始token文本
        
    Returns:
        str: 标准化后的token文本
    """
    token = str(token)
    if token.startswith("]<]") and token.endswith("[>["):
        return f"<{token[3:-3]}>"
    return token


def classify_family(token: str) -> str:
    """
    对token进行分类
    
    根据token的特征将其归类到不同的家族
    
    Args:
        token: token文本
        
    Returns:
        str: token家族类别
    """
    token = normalize_token_text(token)
    lowered = token.lower()
    if any(key in lowered for key in ['speech', 'image', 'video', 'vision', 'code_', 'review_', 'pr_', 'source_', 'file', 'interpreter', 'fim_']):
        return 'Tooling / Multimodal Specials'
    if token.startswith('<') and token.endswith('>'):
        return 'Tooling / Multimodal Specials'
    if any(ch in token for ch in ['|', '\\', '/', '=', '+']) or token.isascii() and len(token) >= 8:
        return 'Noisy Latin / Encoded Fragments'
    if any('\u4e00' <= ch <= '\u9fff' for ch in token):
        return 'Study Tokens / CJK Targets / Controls'
    if any(ord(ch) > 127 for ch in token):
        return 'Multilingual Boilerplate / Reference Fragments'
    return 'Noisy Latin / Encoded Fragments'


def classical_mds(distance_matrix: np.ndarray, n_components: int = 2) -> np.ndarray:
    """
    经典多维标度分析（MDS）
    
    将距离矩阵转换为低维坐标
    
    Args:
        distance_matrix: 距离矩阵
        n_components: 降维后的维度
        
    Returns:
        np.ndarray: 降维后的坐标矩阵
    """
    d2 = distance_matrix ** 2
    n = d2.shape[0]
    j = np.eye(n) - np.ones((n, n)) / n
    b = -0.5 * j @ d2 @ j
    eigvals, eigvecs = np.linalg.eigh(b)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    pos = np.maximum(eigvals[:n_components], 0)
    coords = eigvecs[:, :n_components] * np.sqrt(pos)
    return coords


def get_embedding_lm_head_similarity(embedding: torch.Tensor, lm_head: torch.Tensor) -> Dict[str, Any]:
    """
    计算embedding和lm_head的相似度
    
    Args:
        embedding: embedding权重张量
        lm_head: lm_head权重张量
        
    Returns:
        Dict[str, Any]: 包含相似度信息的字典
    """
    similarities = torch.cosine_similarity(embedding, lm_head, dim=1)
    sorted_indices = similarities.argsort(descending=True)
    sorted_similarities = similarities[sorted_indices]
    
    return {
        'similarities': similarities,
        'sorted_indices': sorted_indices,
        'sorted_similarities': sorted_similarities
    }


def print_embedding_lm_head_similarity(embedding: torch.Tensor, lm_head: torch.Tensor):
    """
    打印embedding和lm_head的相似度分析结果
    
    Args:
        embedding: embedding权重张量
        lm_head: lm_head权重张量
    """
    result = get_embedding_lm_head_similarity(embedding, lm_head)
    
    for i in range(10):
        print(f"行索引: {result['sorted_indices'][i].item()}, 相似度: {result['sorted_similarities'][i].item()}")
    for i in range(10):
        print(f"行索引: {result['sorted_indices'][-i-1].item()}, 相似度: {result['sorted_similarities'][-i-1].item()}")
    print(f"第 190467 行的相似性分析:")
    print(f"行索引: {190467}, 相似度: {result['similarities'][190467].item()}, 排名: {result['sorted_indices'].tolist().index(190467)+1}")


def get_embedding_lm_head_norm_diff(embedding: torch.Tensor, lm_head: torch.Tensor) -> Dict[str, Any]:
    """
    计算embedding和lm_head的模长差异
    
    Args:
        embedding: embedding权重张量
        lm_head: lm_head权重张量
        
    Returns:
        Dict[str, Any]: 包含模长差异信息的字典
    """
    embedding_norms = torch.norm(embedding, dim=1)
    lm_head_norms = torch.norm(lm_head, dim=1)
    norm_diffs = embedding_norms - lm_head_norms
    
    sorted_indices = norm_diffs.argsort(descending=True)
    sorted_diffs = norm_diffs[sorted_indices]
    
    return {
        'embedding_norms': embedding_norms,
        'lm_head_norms': lm_head_norms,
        'norm_diffs': norm_diffs,
        'sorted_indices': sorted_indices,
        'sorted_diffs': sorted_diffs
    }


def print_embedding_lm_head_norm_diff(embedding: torch.Tensor, lm_head: torch.Tensor):
    """
    打印embedding和lm_head的模长差异分析结果
    
    Args:
        embedding: embedding权重张量
        lm_head: lm_head权重张量
    """
    result = get_embedding_lm_head_norm_diff(embedding, lm_head)
    
    for i in range(10):
        print(f"行索引: {result['sorted_indices'][i].item()}, 差异: {result['sorted_diffs'][i].item()}")
    for i in range(10):
        print(f"行索引: {result['sorted_indices'][-i-1].item()}, 差异: {result['sorted_diffs'][-i-1].item()}")
    print(f"第 190467 行的模长差异分析:")
    print(f"行索引: {190467}, 差异: {result['norm_diffs'][190467].item()}, 排名: {result['sorted_indices'].tolist().index(190467)+1}, embedding 模长: {result['embedding_norms'][190467].item()}, lm_head 模长: {result['lm_head_norms'][190467].item()}")


def test_tokenizer(model_config: ModelConfig):
    """
    测试tokenizer的编码和解码功能
    
    Args:
        model_config: 模型配置对象
        
    Returns:
        AutoTokenizer: 加载的tokenizer对象
    """
    tokenizer = AutoTokenizer.from_pretrained(model_config.model_name)
    
    print("Minimax M2.5词表大小:", tokenizer.vocab_size)

    text = "马嘉祺马嘉棋马嘉|祺马јект马嘉诚马佳琦马星驰"
    tokens = tokenizer(text, return_tensors="pt")
    print("Token IDs:", tokens['input_ids'])

    print("单个Token解码结果:")
    for token_id in tokens['input_ids'][0]:
        decoded = tokenizer.decode([token_id])
        print(f"ID {token_id}: {decoded}")

    detokenized_text = tokenizer.decode(tokens['input_ids'][0])
    print("完整解码结果:", detokenized_text)
    
    return tokenizer


def test_bytelevel_encoding():
    """
    测试字节级编码解码功能
    """
    text = "嘉祺"
    utf8_bytes = text.encode("utf-8")
    btou = bytes_to_unicode()
    encoded = "".join(btou[b] for b in utf8_bytes)

    print("utf8 bytes hex:", utf8_bytes.hex(" "))
    print("bytelevel encoded:", encoded)

    decoded_bytes = bytelevel_decode_bytes(encoded)
    decoded_text = decoded_bytes.decode("utf-8")

    print("decoded bytes hex:", decoded_bytes.hex(" "))
    print("decoded text:", decoded_text)


def analyze_lm_head(model_config: ModelConfig, analysis_config: AnalysisConfig, tokenizer: AutoTokenizer):
    """
    分析lm_head权重
    
    包括模长分析、相似度分析和可视化
    
    Args:
        model_config: 模型配置对象
        analysis_config: 分析配置对象
        tokenizer: 分词器对象
        
    Returns:
        Tuple[torch.Tensor, Dict[int, str]]: lm_head权重和反向词表
    """
    reverse_vocab = build_reverse_vocab(tokenizer)
    print(reverse_vocab[analysis_config.anomaly_token_id])

    config_path = hf_hub_download(repo_id=model_config.model_name, filename="model.safetensors.index.json")

    lm_head = load_lm_head(model_config.weight_path_base, model_config.lm_head_file)
    print(f"lm_head 形状: {lm_head.shape}")

    lm_head_norms = compute_norms(lm_head)
    sorted_norms, sorted_indices = print_top_bottom_norms(lm_head_norms)
    print_row_rank(lm_head_norms, sorted_indices, analysis_config.anomaly_token_id)

    vector_anomaly = lm_head[analysis_config.anomaly_token_id].float().cpu().numpy()
    plot_vector_distribution(vector_anomaly, "Numerical Density Distribution Histogram - Row 190467")

    vector_144803 = lm_head[144803].float().cpu().numpy()
    plot_vector_distribution(vector_144803, "Numerical Density Distribution Histogram - Row 144803")

    for i in (list(range(55260, 55271))):
        print(f"行索引: {sorted_indices[i].item()}, 模长: {sorted_norms[i].item()}，排名: {i}")

    jq_rank_177085 = torch.where(sorted_indices == 177085)
    jq_norm_177085 = sorted_norms[jq_rank_177085]
    print(f"177085 行的模长: {jq_norm_177085[0]}, 排名: {jq_rank_177085[0][0]}")

    target_row = analysis_config.anomaly_token_id
    target_vector = lm_head[target_row]
    similarities = compute_similarity(lm_head, target_vector)
    print_similarity_analysis(similarities, target_row, reverse_vocab, tokenizer)

    print(lm_head.shape)
    
    return lm_head, reverse_vocab


def analyze_embedding(model_config: ModelConfig, analysis_config: AnalysisConfig, tokenizer: AutoTokenizer, reverse_vocab: Dict[int, str]):
    """
    分析embedding权重
    
    包括相似度分析和可视化
    
    Args:
        model_config: 模型配置对象
        analysis_config: 分析配置对象
        tokenizer: 分词器对象
        reverse_vocab: 反向词表
        
    Returns:
        torch.Tensor: embedding权重张量
    """
    embedding = load_embedding(model_config.weight_path_base, model_config.embedding_file)
    print(f"Embedding 形状: {embedding.shape}")

    target_vector = embedding[analysis_config.anomaly_token_id]
    similarities = compute_similarity(embedding, target_vector)
    print_similarity_analysis(similarities, analysis_config.anomaly_token_id, reverse_vocab, tokenizer, top_n=20, bottom_n=10)

    similarity_177085 = similarities[177085].item()
    print(f"177085 行的相似度: {similarity_177085}")

    emb_vector_190467 = embedding[196954].float().cpu().numpy()
    plot_vector_distribution(emb_vector_190467, "Numerical Density Distribution Histogram - Embedding Row 190467")

    emb_vector_190468 = embedding[190468].float().cpu().numpy()
    plot_vector_distribution(emb_vector_190468, "Numerical Density Distribution Histogram - Embedding Row 190468")

    emb_vector_183969 = embedding[183969].float().cpu().numpy()
    plot_vector_distribution(emb_vector_183969, "Numerical Density Distribution Histogram - Embedding Row 183969")
    
    return embedding


def compare_embedding_lm_head(embedding: torch.Tensor, lm_head: torch.Tensor):
    """
    对比embedding和lm_head
    
    Args:
        embedding: embedding权重张量
        lm_head: lm_head权重张量
    """
    print_embedding_lm_head_similarity(embedding, lm_head)
    print_embedding_lm_head_norm_diff(embedding, lm_head)


def select_plot_tokens(analysis_config: AnalysisConfig, lm_head: torch.Tensor) -> Tuple[List[int], torch.Tensor, torch.Tensor]:
    """
    选择要绘制的token ID
    
    根据配置选择特殊token、异常token的邻居等
    
    Args:
        analysis_config: 分析配置对象
        lm_head: lm_head权重张量
        
    Returns:
        Tuple[List[int], torch.Tensor, torch.Tensor]: 
            - plot_ids: 要绘制的token ID列表
            - center_sims_all: 所有token与聚类中心的相似度
            - lm_head_unit: 归一化后的lm_head
    """
    MANDATORY_HIGH_IDS = [idx for idx in range(lm_head.shape[0]) if idx >= analysis_config.special_token_threshold]
    MANDATORY_IDS = set(MANDATORY_HIGH_IDS + [analysis_config.anomaly_token_id] + analysis_config.study_token_ids)

    anomaly_neighbor_ids, anomaly_sims = lm_topk_neighbors(lm_head, analysis_config.anomaly_token_id, k=analysis_config.neighbor_k)

    cluster_seed_ids = sorted(set(
        [analysis_config.anomaly_token_id] +
        MANDATORY_HIGH_IDS +
        anomaly_neighbor_ids[:120]
    ))

    lm_head_unit = torch.nn.functional.normalize(lm_head.float(), dim=1)
    cluster_centroid = torch.nn.functional.normalize(lm_head_unit[cluster_seed_ids].mean(dim=0, keepdim=True), dim=1)[0]
    center_sims_all = torch.matmul(lm_head_unit, cluster_centroid)

    selected_ids = set(MANDATORY_IDS)
    selected_ids.update(cluster_seed_ids)
    selected_ids.update(anomaly_neighbor_ids[:180])

    non_special_ids = [
        idx for idx in center_sims_all.argsort(descending=True).tolist()
        if idx not in selected_ids
    ]
    selected_ids.update(non_special_ids[:260])

    remaining_budget = analysis_config.max_plot_tokens - len(selected_ids)
    if remaining_budget > 0:
        far_ids = [
            idx for idx in center_sims_all.argsort(descending=False).tolist()
            if idx not in selected_ids
        ]
        stride = max(1, len(far_ids) // remaining_budget)
        selected_ids.update(far_ids[::stride][:remaining_budget])

    plot_ids = sorted(selected_ids)
    if len(plot_ids) > analysis_config.max_plot_tokens:
        must_keep = sorted(MANDATORY_IDS | set(anomaly_neighbor_ids[:180]))
        must_keep = [idx for idx in must_keep if idx in selected_ids]
        leftovers = [idx for idx in plot_ids if idx not in set(must_keep)]
        plot_ids = must_keep + leftovers[:analysis_config.max_plot_tokens - len(must_keep)]
        plot_ids = sorted(set(plot_ids))

    plot_ids = plot_ids[:analysis_config.max_plot_tokens]
    plot_id_set = set(plot_ids)
    plot_cluster_seed_ids = [idx for idx in cluster_seed_ids if idx in plot_id_set]
    
    return plot_ids, center_sims_all, lm_head_unit


def create_plot_dataframe(plot_ids: List[int], center_sims_all: torch.Tensor, 
                         analysis_config: AnalysisConfig, tokenizer: AutoTokenizer, 
                         reverse_vocab: Dict[int, str], plot_cluster_seed_ids: List[int]) -> pd.DataFrame:
    """
    创建绘图用的DataFrame
    
    为每个选中的token创建包含分类信息的记录
    
    Args:
        plot_ids: 要绘制的token ID列表
        center_sims_all: 所有token与聚类中心的相似度
        analysis_config: 分析配置对象
        tokenizer: 分词器对象
        reverse_vocab: 反向词表
        plot_cluster_seed_ids: 聚类种子token ID列表
        
    Returns:
        pd.DataFrame: 包含token信息的DataFrame
    """
    plot_rows = []
    for idx in plot_ids:
        token = normalize_token_text(token_text(idx, reverse_vocab, tokenizer))
        family = classify_family(token)
        is_special = idx >= analysis_config.special_token_threshold
        is_study = idx in {analysis_config.anomaly_token_id} | set(analysis_config.study_token_ids)
        is_control = (not is_special) and (center_sims_all[idx].item() > float(center_sims_all[analysis_config.anomaly_token_id].item()) - analysis_config.similarity_threshold)
        point_type = 'ordinary token'
        if is_special:
            point_type = 'special token'
        if is_control:
            point_type = 'matched control'
        if is_study:
            point_type = 'study / highlighted'
        plot_rows.append({
            'token_id': idx,
            'token': token,
            'family': family,
            'point_type': point_type,
            'center_sim': float(center_sims_all[idx].item()),
            'is_special_like': bool(is_special),
            'is_cluster_seed': idx in plot_cluster_seed_ids,
        })

    left_plot_df = pd.DataFrame(plot_rows)
    tier_labels = ['edge', 'mid_ring', 'inner_ring', 'core']
    left_plot_df['tier'] = pd.qcut(left_plot_df['center_sim'], q=4, labels=tier_labels, duplicates='drop')
    left_plot_df['tier'] = left_plot_df['tier'].astype(str)
    
    return left_plot_df


def compute_mds_coordinates(left_plot_df: pd.DataFrame, lm_head_unit: torch.Tensor, plot_ids: List[int]) -> pd.DataFrame:
    """
    计算MDS坐标
    
    使用经典MDS算法将高维向量降维到2D
    
    Args:
        left_plot_df: 包含token信息的DataFrame
        lm_head_unit: 归一化后的lm_head
        plot_ids: 要绘制的token ID列表
        
    Returns:
        pd.DataFrame: 添加了x, y坐标的DataFrame
    """
    plot_vectors = lm_head_unit[plot_ids]
    dist = torch.cdist(plot_vectors, plot_vectors, p=2).cpu().numpy()

    coords = classical_mds(dist, n_components=2)
    left_plot_df['x'] = coords[:, 0]
    left_plot_df['y'] = coords[:, 1]
    
    return left_plot_df


def get_cluster_boundary(left_plot_df: pd.DataFrame) -> Tuple[float, float, float, float, float, float]:
    """
    计算聚类边界
    
    基于聚类种子token的坐标计算边界框
    
    Args:
        left_plot_df: 包含token信息的DataFrame
        
    Returns:
        Tuple[float, float, float, float, float, float]: 
            - x0, x1: x轴边界
            - y0, y1: y轴边界
            - pad_x, pad_y: 边界内边距
    """
    cluster_points = left_plot_df[left_plot_df['is_cluster_seed']].copy()
    x0, x1 = cluster_points['x'].quantile([0.05, 0.95]).tolist()
    y0, y1 = cluster_points['y'].quantile([0.05, 0.95]).tolist()
    pad_x = max(0.08, (x1 - x0) * 0.12)
    pad_y = max(0.08, (y1 - y0) * 0.12)
    return x0, x1, y0, y1, pad_x, pad_y


def get_plot_styles() -> Tuple[Dict[str, str], Dict[str, str], Dict[str, int], Dict[str, str]]:
    """
    获取绘图样式配置
    
    Returns:
        Tuple[Dict[str, str], Dict[str, str], Dict[str, int], Dict[str, str]]: 
            - family_colors: token家族颜色映射
            - point_markers: 点类型标记映射
            - point_sizes: 点大小映射
            - point_edge: 点边框映射
    """
    family_colors = {
        'Tooling / Multimodal Specials': '#2b8cbe',
        'Noisy Latin / Encoded Fragments': '#e08214',
        'Multilingual Boilerplate / Reference Fragments': '#66a61e',
        'Study Tokens / CJK Targets / Controls': '#525252',
    }
    point_markers = {
        'ordinary token': 'o',
        'special token': 's',
        'matched control': '^',
        'study / highlighted': 'o',
    }
    point_sizes = {
        'ordinary token': 18,
        'special token': 26,
        'matched control': 24,
        'study / highlighted': 42,
    }
    point_edge = {
        'ordinary token': 'none',
        'special token': '#999999',
        'matched control': 'none',
        'study / highlighted': '#222222',
    }
    return family_colors, point_markers, point_sizes, point_edge


def create_mds_plot(left_plot_df: pd.DataFrame, analysis_config: AnalysisConfig, 
                   x0: float, x1: float, y0: float, y1: float, pad_x: float, pad_y: float,
                   family_colors: Dict[str, str], point_markers: Dict[str, str], 
                   point_sizes: Dict[str, int], point_edge: Dict[str, str]):
    """
    创建MDS可视化图表
    
    Args:
        left_plot_df: 包含token信息的DataFrame
        analysis_config: 分析配置对象
        x0, x1: x轴边界
        y0, y1: y轴边界
        pad_x, pad_y: 边界内边距
        family_colors: token家族颜色映射
        point_markers: 点类型标记映射
        point_sizes: 点大小映射
        point_edge: 点边框映射
    """
    fig, ax = plt.subplots(figsize=analysis_config.default_figsize)

    shell_order = ['edge', 'mid_ring', 'inner_ring', 'core']
    shell_alpha = {'edge': 0.05, 'mid_ring': 0.08, 'inner_ring': 0.10, 'core': 0.13}
    for tier in shell_order:
        shell_df = left_plot_df[left_plot_df['tier'] == tier]
        if len(shell_df) < 8:
            continue
        sx0, sx1 = shell_df['x'].quantile([0.12, 0.88]).tolist()
        sy0, sy1 = shell_df['y'].quantile([0.12, 0.88]).tolist()
        ax.add_patch(Rectangle(
            (sx0 - 0.04, sy0 - 0.04),
            (sx1 - sx0) + 0.08,
            (sy1 - sy0) + 0.08,
            facecolor='#f6d9b8',
            edgecolor='none',
            alpha=shell_alpha[tier],
            zorder=0,
        ))

    ax.add_patch(Rectangle(
        (x0 - pad_x, y0 - pad_y),
        (x1 - x0) + 2 * pad_x,
        (y1 - y0) + 2 * pad_y,
        fill=False,
        linestyle='--',
        linewidth=1.1,
        edgecolor='#444444',
        zorder=1,
    ))

    for point_type, marker in point_markers.items():
        sub = left_plot_df[left_plot_df['point_type'] == point_type]
        if sub.empty:
            continue
        colors = sub['family'].map(family_colors)
        ax.scatter(
            sub['x'],
            sub['y'],
            c=colors,
            s=point_sizes[point_type],
            marker=marker,
            alpha=0.88 if point_type != 'ordinary token' else 0.72,
            edgecolors=point_edge[point_type],
        )

    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#525252', markersize=8, label='Study Tokens / CJK Targets / Controls'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#2b8cbe', markersize=8, label='Tooling / Multimodal Specials'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#e08214', markersize=8, label='Noisy Latin / Encoded Fragments'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#66a61e', markersize=8, label='Multilingual Boilerplate / Reference Fragments'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.02, 1), framealpha=0.95, fontsize=9)

    ax.set_xlabel('MDS Dimension 1', fontsize=10)
    ax.set_ylabel('MDS Dimension 2', fontsize=10)
    ax.set_title('Token Embedding Space (MDS Projection)', fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.15, linestyle='--')

    plt.tight_layout()
    plt.show()


def create_mds_visualization(model_config: ModelConfig, analysis_config: AnalysisConfig, 
                             tokenizer: AutoTokenizer, reverse_vocab: Dict[int, str], lm_head: torch.Tensor):
    """
    创建MDS可视化
    
    协调各个子函数完成完整的MDS可视化流程
    
    Args:
        model_config: 模型配置对象
        analysis_config: 分析配置对象
        tokenizer: 分词器对象
        reverse_vocab: 反向词表
        lm_head: lm_head权重张量
    """
    plot_ids, center_sims_all, lm_head_unit = select_plot_tokens(analysis_config, lm_head)
    
    cluster_seed_ids = sorted(set(
        [analysis_config.anomaly_token_id] +
        [idx for idx in range(lm_head.shape[0]) if idx >= analysis_config.special_token_threshold] +
        lm_topk_neighbors(lm_head, analysis_config.anomaly_token_id, k=analysis_config.neighbor_k)[0][:120]
    ))
    plot_cluster_seed_ids = [idx for idx in cluster_seed_ids if idx in set(plot_ids)]
    
    left_plot_df = create_plot_dataframe(plot_ids, center_sims_all, analysis_config, tokenizer, reverse_vocab, plot_cluster_seed_ids)
    left_plot_df = compute_mds_coordinates(left_plot_df, lm_head_unit, plot_ids)
    
    x0, x1, y0, y1, pad_x, pad_y = get_cluster_boundary(left_plot_df)
    family_colors, point_markers, point_sizes, point_edge = get_plot_styles()
    
    create_mds_plot(left_plot_df, analysis_config, x0, x1, y0, y1, pad_x, pad_y, 
                    family_colors, point_markers, point_sizes, point_edge)


def main():
    """
    主函数
    
    执行完整的分析流程
    """
    model_config = ModelConfig()
    analysis_config = AnalysisConfig()
    
    tokenizer = test_tokenizer(model_config)
    test_bytelevel_encoding()
    lm_head, reverse_vocab = analyze_lm_head(model_config, analysis_config, tokenizer)
    embedding = analyze_embedding(model_config, analysis_config, tokenizer, reverse_vocab)
    compare_embedding_lm_head(embedding, lm_head)
    create_mds_visualization(model_config, analysis_config, tokenizer, reverse_vocab, lm_head)


if __name__ == "__main__":
    main()
