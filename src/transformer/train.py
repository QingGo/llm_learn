import torch
import torch.nn as nn
import torch.optim as optim
import math
import os
import json
import time
import typer
from typing import List, Tuple, Optional
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from .model import Transformer
from .prepare_data import create_translation_dataloaders

def get_device() -> str:
    """智能选择设备：优先级 mps > cuda > cpu"""
    if torch.backends.mps.is_available():
        return 'mps'
    elif torch.cuda.is_available():
        return 'cuda'
    else:
        return 'cpu'


class TranslationTrainer:
    """翻译模型训练器"""
    
    def __init__(self, model: Transformer, device: str = None, log_dir: str = None, enable_tensorboard: bool = True, verbose: bool = True):
        if device is None:
            device = get_device()
        self.model = model.to(device)
        self.device = device
        self.optimizer = None
        self.criterion = None
        self.train_losses = []
        self.val_losses = []
        self.train_perplexities = []
        self.val_perplexities = []
        self.best_val_loss = float('inf')
        self.step_count = 0  # 添加步数计数器
        self.verbose = verbose
        
        # TensorBoard 设置
        self.enable_tensorboard = enable_tensorboard
        self.writer = None
        if self.enable_tensorboard:
            self.writer = SummaryWriter(log_dir=log_dir)
            print(f"TensorBoard 日志将保存到: {log_dir}")
        
        if verbose:
            # 初始化training_losses.txt文件
            self._init_loss_file()
    
    def _init_loss_file(self):
        """初始化logs/training_losses.txt文件并写入表头"""
        if not os.path.exists("logs"):
            os.mkdir("logs")
        with open('logs/training_losses.txt', 'w', encoding='utf-8') as f:
            f.write('Step,Loss\n')
    
    def setup_training(self, learning_rate: float = 1e-4, weight_decay: float = 1e-5):
        """设置训练参数"""
        self.optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        # 使用标签平滑的交叉熵损失，忽略padding token
        self.criterion = nn.CrossEntropyLoss(ignore_index=100257, label_smoothing=0.1)  # 100257是pad token
        
    def train_epoch(self, dataloader, epoch: int) -> Tuple[float, float]:
        """训练一个epoch，返回损失和perplexity"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        progress_bar = tqdm(dataloader, desc=f'Epoch {epoch+1} Training')
        
        for batch_idx, (src_batch, tgt_batch) in enumerate(progress_bar):
            src_batch = src_batch.to(self.device)  # [batch_size, src_seq_len]
            tgt_batch = tgt_batch.to(self.device)  # [batch_size, tgt_seq_len]
            
            # 准备decoder输入和目标
            # decoder输入：去掉最后一个token
            decoder_input = tgt_batch[:, :-1]  # [batch_size, tgt_seq_len-1]
            # 目标：去掉第一个token（<bos>）
            targets = tgt_batch[:, 1:]  # [batch_size, tgt_seq_len-1]
            
            # 创建padding mask
            src_padding_mask = self._create_padding_mask(src_batch, 100257)  # pad token id
            tgt_padding_mask = self._create_padding_mask(decoder_input, 100257)
            
            # 前向传播
            self.optimizer.zero_grad()
            logits = self.model(src_batch, decoder_input, src_padding_mask, tgt_padding_mask)
            
            # 计算损失
            # logits: [batch_size, seq_len, vocab_size]
            # targets: [batch_size, seq_len]
            loss = self.criterion(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
            
            # 反向传播
            loss.backward()
            
            # 梯度裁剪防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # 参数更新
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1

            self.step_count += 1
            if self.verbose:
                # 将每步的loss写入文件
                with open('srcs/training_losses.txt', 'a', encoding='utf-8') as f:
                    f.write(f'{self.step_count},{loss.item():.6f}\n')
            
            # 记录到 TensorBoard
            if self.writer is not None:
                self.writer.add_scalar('Loss/Train_Step', loss.item(), self.step_count)
                if batch_idx % 10 == 0:  # 每10个batch记录一次学习率
                    current_lr = self.optimizer.param_groups[0]['lr']
                    self.writer.add_scalar('Learning_Rate', current_lr, self.step_count)

            # 更新进度条
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
            
        avg_loss = total_loss / num_batches
        perplexity = math.exp(avg_loss)
        
        self.train_losses.append(avg_loss)
        self.train_perplexities.append(perplexity)
        
        # 记录到 TensorBoard
        if self.writer is not None:
            self.writer.add_scalar('Loss/Train_Epoch', avg_loss, epoch)
            self.writer.add_scalar('Perplexity/Train_Epoch', perplexity, epoch)
        
        return avg_loss, perplexity
    
    def validate_epoch(self, dataloader, epoch: int) -> Tuple[float, float]:
        """验证一个epoch，返回损失和perplexity"""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            progress_bar = tqdm(dataloader, desc=f'Epoch {epoch+1} Validation')
            
            for batch_idx, (src_batch, tgt_batch) in enumerate(progress_bar):
                src_batch = src_batch.to(self.device)
                tgt_batch = tgt_batch.to(self.device)
                
                # 准备decoder输入和目标
                decoder_input = tgt_batch[:, :-1]
                targets = tgt_batch[:, 1:]
                
                # 创建padding mask
                src_padding_mask = self._create_padding_mask(src_batch, 100257)
                tgt_padding_mask = self._create_padding_mask(decoder_input, 100257)
                
                # 前向传播
                logits = self.model(src_batch, decoder_input, src_padding_mask, tgt_padding_mask)
                
                # 计算损失
                loss = self.criterion(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
                
                total_loss += loss.item()
                num_batches += 1
                
                # 更新进度条
                progress_bar.set_postfix({'val_loss': f'{loss.item():.4f}'})
        
        avg_loss = total_loss / num_batches
        perplexity = math.exp(avg_loss)
        
        self.val_losses.append(avg_loss)
        self.val_perplexities.append(perplexity)
        
        # 记录到 TensorBoard
        if self.writer is not None:
            self.writer.add_scalar('Loss/Val_Epoch', avg_loss, epoch)
            self.writer.add_scalar('Perplexity/Val_Epoch', perplexity, epoch)
        
        return avg_loss, perplexity
    
    def _create_padding_mask(self, sequences: torch.Tensor, pad_token_id: int) -> torch.Tensor:
        """创建padding掩码"""
        return sequences == pad_token_id
    
    def _log_model_stats(self, epoch: int):
        """记录模型参数和梯度统计信息到 TensorBoard"""
        if self.writer is None:
            return
        
        # 记录模型参数的统计信息
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                # 参数值的统计
                self.writer.add_histogram(f'Parameters/{name}', param.data, epoch)
                self.writer.add_scalar(f'Parameters/{name}_mean', param.data.mean().item(), epoch)
                self.writer.add_scalar(f'Parameters/{name}_std', param.data.std().item(), epoch)
                
                # 梯度的统计（如果存在）
                if param.grad is not None:
                    self.writer.add_histogram(f'Gradients/{name}', param.grad.data, epoch)
                    self.writer.add_scalar(f'Gradients/{name}_mean', param.grad.data.mean().item(), epoch)
                    self.writer.add_scalar(f'Gradients/{name}_std', param.grad.data.std().item(), epoch)
                    self.writer.add_scalar(f'Gradients/{name}_norm', param.grad.data.norm().item(), epoch)
    
    def evaluate_test_set(self, test_dataloader) -> Tuple[float, float]:
        """评估测试集，返回损失和perplexity"""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            progress_bar = tqdm(test_dataloader, desc='Testing')
            
            for batch_idx, (src_batch, tgt_batch) in enumerate(progress_bar):
                src_batch = src_batch.to(self.device)
                tgt_batch = tgt_batch.to(self.device)
                
                # 准备decoder输入和目标
                decoder_input = tgt_batch[:, :-1]
                targets = tgt_batch[:, 1:]
                
                # 创建padding mask
                src_padding_mask = self._create_padding_mask(src_batch, 100257)
                tgt_padding_mask = self._create_padding_mask(decoder_input, 100257)
                
                # 前向传播
                logits = self.model(src_batch, decoder_input, src_padding_mask, tgt_padding_mask)
                
                # 计算损失
                loss = self.criterion(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
                
                total_loss += loss.item()
                num_batches += 1
                
                # 更新进度条
                progress_bar.set_postfix({'test_loss': f'{loss.item():.4f}'})
        
        avg_loss = total_loss / num_batches
        perplexity = math.exp(avg_loss)
        
        return avg_loss, perplexity
    
    def train(self, train_dataloader, val_dataloader, test_dataloader, num_epochs: int, save_dir: str = './checkpoints'):
        """完整训练流程"""
        os.makedirs(save_dir, exist_ok=True)
        
        print(f"开始训练，共 {num_epochs} 个epoch")
        print(f"设备: {self.device}")
        print(f"模型参数量: {sum(p.numel() for p in self.model.parameters()):,}")
        
        for epoch in range(num_epochs):
            print(f"\n=== Epoch {epoch+1}/{num_epochs} ===")
            
            # 训练
            train_loss, train_ppl = self.train_epoch(train_dataloader, epoch)
            print(f"训练损失: {train_loss:.4f}, 训练困惑度: {train_ppl:.2f}")
            
            # 验证
            val_loss, val_ppl = self.validate_epoch(val_dataloader, epoch)
            print(f"验证损失: {val_loss:.4f}, 验证困惑度: {val_ppl:.2f}")
            
            # 记录模型参数和梯度统计信息（每5个epoch记录一次以节省空间）
            if (epoch + 1) % 5 == 0:
                self._log_model_stats(epoch)
            
            # 保存最佳模型
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_model(os.path.join(save_dir, 'best_model.pt'))
                print(f"保存最佳模型，验证损失: {val_loss:.4f}")
                
                # 记录最佳验证损失到 TensorBoard
                if self.writer is not None:
                    self.writer.add_scalar('Loss/Best_Val', val_loss, epoch)
            
            # 每5个epoch保存一次检查点
            if (epoch + 1) % 5 == 0:
                checkpoint_path = os.path.join(save_dir, f'checkpoint_epoch_{epoch+1}.pt')
                self.save_model(checkpoint_path)
                print(f"保存检查点: {checkpoint_path}")
        
        # 训练结束后评估测试集
        print("\n=== 测试集评估 ===")
        test_loss, test_ppl = self.evaluate_test_set(test_dataloader)
        print(f"测试损失: {test_loss:.4f}, 测试困惑度: {test_ppl:.2f}")
        
        # 记录测试集结果到 TensorBoard
        if self.writer is not None:
            self.writer.add_scalar('Loss/Test', test_loss, num_epochs - 1)
            self.writer.add_scalar('Perplexity/Test', test_ppl, num_epochs - 1)
        
        # 保存训练历史
        self.save_training_history(save_dir)
        
        # 绘制损失曲线
        self.plot_training_history(save_dir)
        
        # 关闭 TensorBoard writer
        if self.writer is not None:
            self.writer.close()
            print(f"TensorBoard 日志已保存，可以运行 'tensorboard --logdir={self.writer.log_dir}' 查看")
        
        print("\n训练完成！")
    
    def save_model(self, filepath: str):
        """保存模型"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict() if self.optimizer else None,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': self.best_val_loss,
            'model_config': {
                'vocab_size': self.model.embedding.num_embeddings,
                'd_model': self.model.d_model,
                'seq_len': self.model.seq_len,
                'n_heads': self.model.n_heads,  # 这里需要根据实际情况调整
                'stack': len(self.model.encoder)
            }
        }, filepath)
    
    def load_model(self, filepath: str):
        """加载模型"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        if self.optimizer and checkpoint['optimizer_state_dict']:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        
        print(f"模型加载成功: {filepath}")
    
    def save_training_history(self, save_dir: str):
        """保存训练历史"""
        history = {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_perplexities': self.train_perplexities,
            'val_perplexities': self.val_perplexities,
            'best_val_loss': self.best_val_loss
        }
        
        with open(os.path.join(save_dir, 'training_history.json'), 'w') as f:
            json.dump(history, f, indent=2)
    
    def plot_training_history(self, save_dir: str):
        """绘制训练历史曲线"""
        if not self.train_losses or not self.val_losses:
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        epochs = range(1, len(self.train_losses) + 1)
        
        # 损失曲线
        ax1.plot(epochs, self.train_losses, 'b-', label='训练损失')
        ax1.plot(epochs, self.val_losses, 'r-', label='验证损失')
        ax1.set_title('训练和验证损失曲线')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # 困惑度曲线
        if self.train_perplexities and self.val_perplexities:
            ax2.plot(epochs, self.train_perplexities, 'b-', label='训练困惑度')
            ax2.plot(epochs, self.val_perplexities, 'r-', label='验证困惑度')
            ax2.set_title('训练和验证困惑度曲线')
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Perplexity')
            ax2.legend()
            ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'training_history.png'), dpi=300, bbox_inches='tight')
        plt.close()


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


def create_model_and_trainer(vocab_size: int = 100260, d_model: int = 512, seq_len: int = 128, 
                           n_heads: int = 8, d_hidden: int = 2048, stack: int = 6, 
                           device: str = None, log_dir: str = 'runs/transformer', 
                           enable_tensorboard: bool = True) -> Tuple[Transformer, TranslationTrainer]:
    """创建模型和训练器的工厂函数"""
    if device is None:
        device = get_device()
    
    # 创建模型
    model = Transformer(
        vocab_size=vocab_size,
        d_model=d_model,
        seq_len=seq_len,
        n_heads=n_heads,
        d_hidden=d_hidden,
        stack=stack
    )
    
    # 创建训练器
    trainer = TranslationTrainer(model, device, log_dir, enable_tensorboard)
    
    return model, trainer


app = typer.Typer(help="训练翻译模型")

@app.command()
def main(
    max_samples: int = typer.Option(1000, help="最大样本数量"),
    batch_size: int = typer.Option(16, help="批次大小"),
    length_percentile: float = typer.Option(0.9, help="长度百分位数"),
    num_epochs: int = typer.Option(10, help="训练轮数"),
    log_dir: Optional[str] = typer.Option('./runs/translation_training', help="TensorBoard 日志目录"),
    disable_tensorboard: bool = typer.Option(False, help="禁用 TensorBoard 记录")
):
    """演示训练流程"""
    print("=== 翻译模型训练演示 ===")
    
    # 1. 准备数据
    print("1. 加载数据...")
    train_dataloader, val_dataloader, test_dataloader, processor = create_translation_dataloaders(
        max_samples=max_samples,
        batch_size=batch_size,
        length_percentile=length_percentile,
        verbose=True
    )
    
    # 2. 创建模型和训练器
    print("2. 创建模型...")
    model, trainer = create_model_and_trainer(
        vocab_size=100260,  # tiktoken词汇表大小 + 特殊token
        d_model=512,
        seq_len=64,  # 根据数据统计调整
        n_heads=8,
        d_hidden=2048,
        stack=6,
        log_dir=log_dir,
        enable_tensorboard=not disable_tensorboard
    )
    
    # 3. 设置训练参数
    trainer.setup_training(learning_rate=1e-4, weight_decay=1e-5)
    
    # 4. 开始训练
    print("3. 开始训练...")
    trainer.train(
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        test_dataloader=test_dataloader,
        num_epochs=num_epochs,
        save_dir='./translation_checkpoints'
    )
    
    # 5. 测试推理
    print("4. 测试翻译...")
    inference = TranslationInference(model, processor, trainer.device)
    
    test_sentences = [
        "Hello, how are you?",
        "I love machine learning.",
        "The weather is nice today.",
        "Thank you for your help."
    ]
    
    for sentence in test_sentences:
        translation, inference_time = inference.translate(sentence, max_length=30)
        print(f"EN: {sentence}")
        print(f"ZH: {translation}")
        print(f"推理时间: {inference_time:.3f} 秒")
        print("-" * 50)

if __name__ == "__main__":
    app()