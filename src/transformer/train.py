import torch
import torch.nn as nn
import torch.optim as optim
import math
import os
import json
import typer
from typing import Tuple, Optional
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler

from transformer.model import Transformer
from transformer.prepare_data import create_translation_dataloaders
from transformer.inference import TranslationInference
from util.ddp_helper import EnvConfig, get_env_config, get_device_for_ddp, print_env_info, \
        init_process, log_with_rank


class TranslationTrainer:
    """翻译模型训练器"""
    
    def __init__(self, model: Transformer, env_config: EnvConfig, log_dir: str = None, enable_tensorboard: bool = True, verbose: bool = True):
        # DDP 配置
        self.env_config = env_config
        self.device, _, _ = get_device_for_ddp(env_config.rank, env_config.world_size)
        self.rank = env_config.rank if env_config.rank is not None else 0
        self.world_size = env_config.world_size if env_config.world_size is not None else 1

        log_with_rank(f"初始化翻译模型训练器，device: {self.device}，DDP 配置: {self.env_config}", self.rank)
        # DDP 模型封装
        model = model.to(self.device)
        if self.world_size > 1:
            # 在DDP中，需要指定find_unused_parameters=True，因为Transformer的decoder的因果掩码会使得某些参数在某些前向传播中不被使用
            self.model = DistributedDataParallel(model, device_ids=[self.device] if self.device.type == 'cuda' else None, find_unused_parameters=True)
        else:
            self.model = model

       # TensorBoard 设置
        self.enable_tensorboard = enable_tensorboard
        self.writer = None
        if self.enable_tensorboard and self.rank == 0:
            self.writer = SummaryWriter(log_dir=log_dir)
            # 添加模型图，服务器上运行这段会静默退出，也捕获不到 Exception，原因未知，先注释掉
            # self.writer.add_graph(model, (
            #     torch.zeros((16, model.seq_len), dtype=torch.long, device=self.device),
            #     torch.ones((16, model.seq_len), dtype=torch.long, device=self.device),
            #     torch.ones((16, model.seq_len), dtype=torch.bool, device=self.device),
            #     torch.ones((16, model.seq_len), dtype=torch.bool, device=self.device),
            #     ))
            print(f"TensorBoard 日志将保存到: {log_dir}")

        # 需要 setup_training 初始化的优化器和损失函数
        self.optimizer = None
        self.criterion = None

        # 日志记录
        self.train_losses = []
        self.val_losses = []
        self.train_perplexities = []
        self.val_perplexities = []
        self.best_val_loss = float('inf')
        self.step_count = 0  # 添加步数计数器
        self.verbose = verbose and self.rank == 0
        
        if self.verbose:
            # 初始化training_losses.txt文件
            self._init_loss_file()
    
    def _init_loss_file(self):
        """初始化logs/training_losses.txt文件并写入表头"""
        if not os.path.exists("logs"):
            os.mkdir("logs")
        with open('logs/training_losses.txt', 'w', encoding='utf-8') as f:
            f.write('Step,Loss\n')
    
    def _create_padding_mask(self, sequences: torch.Tensor, pad_token_id: int) -> torch.Tensor:
        """创建padding掩码"""
        # sequences: [batch_size, seq_len]
        # 返回: [batch_size, seq_len] 其中True表示padding位置
        return sequences == pad_token_id

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
                with open('logs/training_losses.txt', 'a', encoding='utf-8') as f:
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
    
    def validate_epoch(self, val_dataloader, epoch: int) -> Tuple[float, float]:
        """验证一个epoch"""
        self.model.eval()
        total_loss = 0
        total_tokens = 0
        with torch.no_grad():
            for src, tgt, tgt_y, src_mask, tgt_mask in val_dataloader:
                src, tgt, tgt_y, src_mask, tgt_mask = src.to(self.device), tgt.to(self.device), \
                                                     tgt_y.to(self.device), src_mask.to(self.device), \
                                                     tgt_mask.to(self.device)
                output = self.model(src, tgt, src_mask, tgt_mask)
                loss = self.criterion(output.contiguous().view(-1, output.size(-1)), tgt_y.contiguous().view(-1))
                total_loss += loss.item() * tgt_y.numel()
                total_tokens += tgt_y.numel()

        # DDP: 聚合所有进程的损失和token数量
        if dist.is_initialized():
            total_loss_tensor = torch.tensor(total_loss, device=self.device)
            total_tokens_tensor = torch.tensor(total_tokens, device=self.device)
            dist.all_reduce(total_loss_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(total_tokens_tensor, op=dist.ReduceOp.SUM)
            total_loss = total_loss_tensor.item()
            total_tokens = total_tokens_tensor.item()

        avg_loss = total_loss / total_tokens
        ppl = math.exp(avg_loss)
        
        if self.writer is not None and self.rank == 0:
            self.writer.add_scalar('Loss/Validation', avg_loss, epoch)
            self.writer.add_scalar('Perplexity/Validation', ppl, epoch)

        return avg_loss, ppl
    
    def _log_model_stats(self, epoch: int):
        """记录模型参数和梯度的统计信息"""
        if self.rank == 0:
            if self.writer is not None:
                for name, param in self.model.named_parameters():
                    if param.requires_grad:
                        self.writer.add_histogram(f'params/{name}', param.data, epoch)
                        if param.grad is not None:
                            self.writer.add_histogram(f'grads/{name}', param.grad, epoch)

    def evaluate_test_set(self, test_dataloader) -> Tuple[float, float]:
        """评估测试集，返回损失和perplexity"""
        self.model.eval()
        total_loss = 0
        total_tokens = 0
        with torch.no_grad():
            # 只有主进程显示进度条
            dataloader_iter = tqdm(test_dataloader, desc='Testing') if self.rank == 0 else test_dataloader
            
            for src, tgt, tgt_y, src_mask, tgt_mask in dataloader_iter:
                src, tgt, tgt_y, src_mask, tgt_mask = src.to(self.device), tgt.to(self.device), \
                                                     tgt_y.to(self.device), src_mask.to(self.device), \
                                                     tgt_mask.to(self.device)
                output = self.model(src, tgt, src_mask, tgt_mask)
                loss = self.criterion(output.contiguous().view(-1, output.size(-1)), tgt_y.contiguous().view(-1))
                total_loss += loss.item() * tgt_y.numel()
                total_tokens += tgt_y.numel()

        # DDP: 聚合所有进程的损失和token数量
        if dist.is_initialized():
            total_loss_tensor = torch.tensor(total_loss, device=self.device)
            total_tokens_tensor = torch.tensor(total_tokens, device=self.device)
            dist.all_reduce(total_loss_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(total_tokens_tensor, op=dist.ReduceOp.SUM)
            total_loss = total_loss_tensor.item()
            total_tokens = total_tokens_tensor.item()

        avg_loss = total_loss / total_tokens
        ppl = math.exp(avg_loss)
        
        return avg_loss, ppl
    
    def train(self, train_dataloader, val_dataloader, test_dataloader, num_epochs: int, 
              save_dir: str = './checkpoints', resume_from_checkpoint: str = None):
        """完整训练流程"""
        if self.rank == 0:
            os.makedirs(save_dir, exist_ok=True)
        
        # 确定起始epoch
        start_epoch = 0
        if resume_from_checkpoint and os.path.exists(resume_from_checkpoint):
            if self.rank == 0:
                print(f"从检查点恢复训练: {resume_from_checkpoint}")
            start_epoch = self.load_model(resume_from_checkpoint, load_optimizer=True)
        
        if self.rank == 0:
            print(f"开始训练，共 {num_epochs} 个epoch")
            if start_epoch > 0:
                print(f"从第 {start_epoch + 1} 个epoch继续训练")
            print(f"设备: {self.device}")
            print(f"模型参数量: {sum(p.numel() for p in self.model.parameters()):,}")
        
        for epoch in range(start_epoch, num_epochs):
            # DDP: 设置sampler的epoch，确保每个epoch数据shuffle不同
            if isinstance(train_dataloader.sampler, DistributedSampler):
                train_dataloader.sampler.set_epoch(epoch)

            if self.rank == 0:
                print(f"\n=== Epoch {epoch+1}/{num_epochs} ===")
            
            # 训练
            train_loss, train_ppl = self.train_epoch(train_dataloader, epoch)
            if self.rank == 0:
                print(f"训练损失: {train_loss:.4f}, 训练困惑度: {train_ppl:.2f}")
            
            # 验证
            val_loss, val_ppl = self.validate_epoch(val_dataloader, epoch)
            if self.rank == 0:
                print(f"验证损失: {val_loss:.4f}, 验证困惑度: {val_ppl:.2f}")
            
            # 记录模型参数和梯度统计信息（每5个epoch记录一次以节省空间）
            if (epoch + 1) % 5 == 0:
                self._log_model_stats(epoch)
            
            # 保存最佳模型
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_model(os.path.join(save_dir, 'best_model.pt'), epoch)
                if self.rank == 0:
                    print(f"保存最佳模型，验证损失: {val_loss:.4f}")
                
                # 记录最佳验证损失到 TensorBoard
                if self.writer is not None:
                    self.writer.add_scalar('Loss/Best_Val', val_loss, epoch)
            
            # 每5个epoch保存一次检查点
            if (epoch + 1) % 5 == 0:
                checkpoint_path = os.path.join(save_dir, f'checkpoint_epoch_{epoch+1}.pt')
                self.save_model(checkpoint_path, epoch)
                if self.rank == 0:
                    print(f"保存检查点: {checkpoint_path}")
        
        # 训练结束后评估测试集
        if self.rank == 0:
            print("\n=== 测试集评估 ===")
        test_loss, test_ppl = self.evaluate_test_set(test_dataloader)
        if self.rank == 0:
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
    
    def save_model(self, filepath: str, epoch: int = None):
        """保存模型和训练状态"""
        if self.rank != 0:
            return # 只有主进程保存模型

        # 如果模型是DDP包装的，则保存其module属性
        model_to_save = self.model.module if isinstance(self.model, DistributedDataParallel) else self.model

        save_dict = {
            'model_state_dict': model_to_save.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict() if self.optimizer else None,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_perplexities': self.train_perplexities,
            'val_perplexities': self.val_perplexities,
            'best_val_loss': self.best_val_loss,
            'step_count': self.step_count,
            'model_config': {
                'vocab_size': model_to_save.embedding.num_embeddings,
                'd_model': model_to_save.d_model,
                'seq_len': model_to_save.seq_len,
                'n_heads': model_to_save.n_heads,
                'd_hidden': model_to_save.d_hidden,
                'stack': len(model_to_save.encoder)
            }
        }
        
        # 如果提供了epoch信息，也保存
        if epoch is not None:
            save_dict['epoch'] = epoch
            
        torch.save(save_dict, filepath)
    
    def load_model(self, filepath: str, load_optimizer: bool = True):
        """加载模型和训练状态"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        if load_optimizer and self.optimizer and checkpoint.get('optimizer_state_dict'):
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # 恢复训练历史
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])
        self.train_perplexities = checkpoint.get('train_perplexities', [])
        self.val_perplexities = checkpoint.get('val_perplexities', [])
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        self.step_count = checkpoint.get('step_count', 0)
        
        # 返回epoch信息（如果有的话）
        start_epoch = checkpoint.get('epoch', 0)
        
        print(f"模型加载成功: {filepath}")
        if start_epoch > 0:
            print(f"将从第 {start_epoch + 1} 个epoch开始继续训练")
        
        return start_epoch
    
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

def create_model_and_trainer(vocab_size: int = 100260, d_model: int = 512, seq_len: int = 100, 
                           n_heads: int = 8, d_hidden: int = 2048, stack: int = 6, 
                           env_config: EnvConfig = None, log_dir: str = 'runs/transformer', 
                           enable_tensorboard: bool = True) -> Tuple[Transformer, TranslationTrainer]:    
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
    trainer = TranslationTrainer(model, env_config, log_dir, enable_tensorboard)
    
    return model, trainer


app = typer.Typer(help="训练翻译模型")

@app.command()
def main(
    max_samples: int = typer.Option(1000, help="最大样本数量"),
    batch_size: int = typer.Option(16, help="批次大小"),
    length_percentile: float = typer.Option(0.9, help="长度百分位数"),
    max_len: int = typer.Option(100, help="固定序列长度"),
    num_epochs: int = typer.Option(10, help="训练轮数"),
    log_dir: Optional[str] = typer.Option('./runs/translation_training', help="TensorBoard 日志目录"),
    disable_tensorboard: bool = typer.Option(False, help="禁用 TensorBoard 记录"),
    resume_checkpoint: Optional[str] = typer.Option(None, help="从指定检查点继续训练")
):
    """演示训练流程"""
    print("=== 翻译模型训练演示 ===")
    
    # 获取环境配置
    env_config = get_env_config()
    print_env_info(env_config)
    rank = env_config.rank if env_config.rank is not None else None

    # 如果是DDP模式，初始化进程组
    if env_config.world_size > 1:
        init_process(env_config)
        log_with_rank("DDP process group initialized.", env_config.rank)

    # 1. 准备数据
    log_with_rank("1. 加载数据...", rank, True)
    train_dataloader, val_dataloader, test_dataloader, processor = create_translation_dataloaders(
        max_samples=max_samples,
        batch_size=batch_size,
        # length_percentile=length_percentile,
        en_max_len=max_len,
        zh_max_len=max_len,
        verbose=True,
        env_config=env_config
    )
    
    # 2. 创建模型和训练器
    log_with_rank("2. 创建模型...", rank, True)
    
    # 如果有检查点，从检查点加载模型配置
    if resume_checkpoint and os.path.exists(resume_checkpoint):
        log_with_rank(f"从检查点加载模型配置: {resume_checkpoint}", rank, True)
        checkpoint = torch.load(resume_checkpoint, map_location='cpu')
        model_config = checkpoint.get('model_config', {})
        
        model, trainer = create_model_and_trainer(
            vocab_size=model_config.get('vocab_size', 100260),
            d_model=model_config.get('d_model', 512),
            seq_len=model_config.get('seq_len', 100),
            n_heads=model_config.get('n_heads', 8),
            d_hidden=model_config.get('d_hidden', 2048),
            stack=model_config.get('stack', 6),
            env_config=env_config,
            log_dir=log_dir,
            enable_tensorboard=not disable_tensorboard
        )
    else:
        # 使用默认配置创建新模型
        model, trainer = create_model_and_trainer(
            vocab_size=100260,  # tiktoken词汇表大小 + 特殊token
            d_model=512,
            seq_len=100,  # 根据数据统计调整
            n_heads=8,
            d_hidden=2048,
            stack=6,
            env_config=env_config,
            log_dir=log_dir,
            enable_tensorboard=not disable_tensorboard
        )
    
    # 3. 设置训练参数
    trainer.setup_training(learning_rate=1e-4, weight_decay=1e-5)
    
    # 4. 开始训练
    log_with_rank("3. 开始训练...", rank, True)
    trainer.train(
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        test_dataloader=test_dataloader,
        num_epochs=num_epochs,
        save_dir='./translation_checkpoints',
        resume_from_checkpoint=resume_checkpoint
    )
    
    # 5. 测试推理
    log_with_rank("4. 测试翻译...", rank, True)
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

    # DDP模式下，所有进程同步后退出
    if env_config.world_size > 1:
        dist.destroy_process_group()
        log_with_rank("DDP process group destroyed.", rank, True)

if __name__ == "__main__":
    app()