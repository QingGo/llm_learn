import torch
import time
from typing import List, Tuple
from .model import Transformer

class TranslationInference:
    """翻译推理器"""
    
    def __init__(self, model: Transformer, processor, device: str = 'cpu'):
        self.model = model.to(device)
        self.device = device
        self.processor = processor
        self.model.eval()
    
    def translate(self, english_text: str, max_length: int = 50, beam_size: int = 1) -> Tuple[str, float]:
        """将英文翻译成中文，返回翻译结果和推理时间"""
        start_time = time.time()
        
        # 预处理输入文本
        cleaned_text = self.processor._clean_text(english_text, is_english=True)
        
        # 分词
        token_ids = self.processor._tokenizer.encode(cleaned_text)
        
        # 处理序列（添加padding等）
        src_tokens = self.processor._process_sequence_tokens(token_ids, is_target=False, max_len=self.model.seq_len)
        
        # 转换为tensor
        src_tensor = torch.tensor([src_tokens], dtype=torch.long, device=self.device)  # [1, seq_len]
        
        # 创建padding mask
        src_padding_mask = src_tensor == 100257  # pad token
        
        if beam_size == 1:
            translation = self._greedy_decode(src_tensor, src_padding_mask, max_length)
        else:
            translation = self._beam_search_decode(src_tensor, src_padding_mask, max_length, beam_size)
        
        end_time = time.time()
        inference_time = end_time - start_time
        
        return translation, inference_time
    
    def _greedy_decode(self, src_tensor: torch.Tensor, src_padding_mask: torch.Tensor, max_length: int) -> str:
        """贪心解码"""
        batch_size = src_tensor.size(0)
        
        # 初始化decoder输入，从<bos> token开始
        decoder_input = torch.tensor([[100258]], dtype=torch.long, device=self.device)  # <bos> token
        
        with torch.no_grad():
            for _ in range(max_length):
                # 创建decoder padding mask
                tgt_padding_mask = decoder_input == 100257
                
                # 前向传播
                logits = self.model(src_tensor, decoder_input, src_padding_mask, tgt_padding_mask)
                
                # 获取最后一个位置的预测
                next_token_logits = logits[:, -1, :]  # [batch_size, vocab_size]
                
                # 贪心选择概率最大的token
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)  # [batch_size, 1]
                
                # 如果生成了<eos> token，停止生成
                if next_token.item() == 100259:  # <eos> token
                    break
                
                # 将新token添加到decoder输入
                decoder_input = torch.cat([decoder_input, next_token], dim=1)
        
        # 解码生成的token序列
        generated_tokens = decoder_input[0].tolist()
        
        # 移除<bos>和<eos> token
        if generated_tokens[0] == 100258:  # <bos>
            generated_tokens = generated_tokens[1:]
        if generated_tokens and generated_tokens[-1] == 100259:  # <eos>
            generated_tokens = generated_tokens[:-1]
        
        # 转换为文本
        translated_text = self.processor.decode_tokens(generated_tokens, remove_special_tokens=True)
        return translated_text
    
    def _beam_search_decode(self, src_tensor: torch.Tensor, src_padding_mask: torch.Tensor, 
                           max_length: int, beam_size: int) -> str:
        """束搜索解码"""
        # 简化版束搜索实现
        batch_size = src_tensor.size(0)
        
        # 初始化beam
        beams = [{'tokens': [100258], 'score': 0.0}]  # 从<bos>开始
        
        with torch.no_grad():
            for step in range(max_length):
                candidates = []
                
                for beam in beams:
                    if beam['tokens'][-1] == 100259:  # 如果已经结束，直接添加到候选
                        candidates.append(beam)
                        continue
                    
                    # 准备decoder输入
                    decoder_input = torch.tensor([beam['tokens']], dtype=torch.long, device=self.device)
                    tgt_padding_mask = decoder_input == 100257
                    
                    # 前向传播
                    logits = self.model(src_tensor, decoder_input, src_padding_mask, tgt_padding_mask)
                    next_token_logits = logits[:, -1, :]  # [1, vocab_size]
                    
                    # 获取top-k候选
                    log_probs = torch.log_softmax(next_token_logits, dim=-1)
                    top_k_probs, top_k_indices = torch.topk(log_probs, beam_size, dim=-1)
                    
                    # 为每个候选创建新的beam
                    for i in range(beam_size):
                        new_token = top_k_indices[0, i].item()
                        new_score = beam['score'] + top_k_probs[0, i].item()
                        
                        candidates.append({
                            'tokens': beam['tokens'] + [new_token],
                            'score': new_score
                        })
                
                # 选择最佳的beam_size个候选
                candidates.sort(key=lambda x: x['score'], reverse=True)
                beams = candidates[:beam_size]
                
                # 如果所有beam都结束了，提前停止
                if all(beam['tokens'][-1] == 100259 for beam in beams):
                    break
        
        # 选择得分最高的beam
        best_beam = max(beams, key=lambda x: x['score'])
        generated_tokens = best_beam['tokens']
        
        # 移除特殊token
        if generated_tokens[0] == 100258:  # <bos>
            generated_tokens = generated_tokens[1:]
        if generated_tokens and generated_tokens[-1] == 100259:  # <eos>
            generated_tokens = generated_tokens[:-1]
        
        # 转换为文本
        translated_text = self.processor.decode_tokens(generated_tokens, remove_special_tokens=True)
        return translated_text
    
    def translate_batch(self, english_texts: List[str], max_length: int = 50) -> List[Tuple[str, float]]:
        """批量翻译，返回翻译结果和推理时间的列表"""
        results = []
        for text in english_texts:
            translation, inference_time = self.translate(text, max_length)
            results.append((translation, inference_time))
        return results
