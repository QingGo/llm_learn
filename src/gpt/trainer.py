import math
import os
from typing import Optional, Tuple, List

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from tqdm import tqdm

from util.ddp_helper import EnvConfig, get_device_for_ddp
from .config import TrainConfig, GPTConfig
from .model import GPT


class GPTTrainer:
    """GPT 模型训练器：支持 AdamW、Warmup+Cosine 调度、完整 Train/Val/Test 流程"""
    def __init__(self, model: GPT, env_config: EnvConfig, log_dir: Optional[str] = None, enable_tensorboard: bool = True, verbose: bool = True):
        self.env_config = env_config
        self.device, _, _ = get_device_for_ddp(env_config.rank, env_config.world_size)
        self.rank = env_config.rank if env_config.rank is not None else 0
        self.world_size = env_config.world_size if env_config.world_size is not None else 1
        model = model.to(self.device)
        if self.world_size > 1:
            self.model = DistributedDataParallel(model, device_ids=[self.device] if self.device.type == 'cuda' else None, find_unused_parameters=True)
        else:
            self.model = model
        self.enable_tensorboard = enable_tensorboard
        self.writer = None
        if self.enable_tensorboard and self.rank == 0:
            self.writer = SummaryWriter(log_dir=log_dir)
        self.optimizer = None
        self.scheduler = None
        self.criterion = None
        self.train_losses: List[float] = []
        self.val_losses: List[float] = []
        self.train_ppls: List[float] = []
        self.val_ppls: List[float] = []
        self.best_val_loss = float('inf')
        self.step_count = 0
        self.verbose = verbose and self.rank == 0
        if self.verbose:
            self._init_loss_file()

    def _init_loss_file(self):
        if not os.path.exists('logs'):
            os.mkdir('logs')
        with open('logs/training_losses.txt', 'w', encoding='utf-8') as f:
            f.write('Step,Loss\n')

    def _create_padding_mask(self, sequences: torch.Tensor, pad_token_id: int) -> torch.Tensor:
        return sequences == pad_token_id

    def _build_optimizer(self, train_config: TrainConfig):
        decay_params = []
        no_decay_params = []
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            # GPT-1 的“modified L2”针对非 bias/gain 权重；这里用 AdamW 的参数分组排除 bias 与 LayerNorm
            if name.endswith('bias') or 'ln' in name or 'layer_norm' in name:
                no_decay_params.append(param)
            else:
                decay_params.append(param)
        param_groups = [
            {'params': decay_params, 'weight_decay': train_config.weight_decay},
            {'params': no_decay_params, 'weight_decay': 0.0},
        ]
        self.optimizer = optim.AdamW(param_groups, lr=train_config.learning_rate)

    def _build_scheduler(self, train_config: TrainConfig, total_steps: int):
        warmup = max(train_config.warmup_steps, 0)
        def lr_lambda(step: int):
            # 线性 warmup（论文 2000 步）
            if step < warmup:
                return float(step) / float(max(1, warmup))
            progress = float(step - warmup) / float(max(1, total_steps - warmup))
            # 余弦退火至 0（GPT-1/2）
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
        self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

    def setup_training(self, train_config: TrainConfig):
        self._build_optimizer(train_config)
        # 交叉熵损失，忽略 padding token（与数据管线对齐）
        self.criterion = nn.CrossEntropyLoss(ignore_index=train_config.pad_token_id)
        if train_config.total_steps is not None:
            self._build_scheduler(train_config, train_config.total_steps)

    def train_epoch(self, dataloader: DataLoader, epoch: int, train_config: TrainConfig) -> Tuple[float, float, int]:
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        iterator = tqdm(dataloader, desc=f'Epoch {epoch+1} Training') if self.rank == 0 else dataloader
        for (y_batch, attn_mask) in iterator:
            y_batch = y_batch.to(self.device)
            attn_mask = attn_mask.to(self.device)
            decoder_input = y_batch[:, :-1]
            targets = y_batch[:, 1:]
            tgt_padding_mask = self._create_padding_mask(decoder_input, train_config.pad_token_id)
            self.optimizer.zero_grad()
            logits = self.model(y_batch, decoder_input, None, tgt_padding_mask)
            loss = self.criterion(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=train_config.grad_clip)
            self.optimizer.step()
            if self.scheduler is not None:
                self.scheduler.step()
            total_loss += loss.item()
            num_batches += 1
            self.step_count += 1
            if self.verbose:
                with open('logs/training_losses.txt', 'a', encoding='utf-8') as f:
                    f.write(f'{self.step_count},{loss.item():.6f}\n')
            if self.writer is not None:
                self.writer.add_scalar('Loss/Train_Step', loss.item(), self.step_count)
                self.writer.add_scalar('LR', self.optimizer.param_groups[0]['lr'], self.step_count)
            if self.rank == 0:
                try:
                    iterator.set_postfix({'loss': f'{loss.item():.4f}'})
                except Exception:
                    pass

        avg_loss = total_loss / max(1, num_batches)
        ppl = math.exp(avg_loss)
        self.train_losses.append(avg_loss)
        self.train_ppls.append(ppl)
        if self.writer is not None:
            self.writer.add_scalar('Loss/Train_Epoch', avg_loss, epoch)
            self.writer.add_scalar('Perplexity/Train_Epoch', ppl, epoch)
        return avg_loss, ppl, num_batches

    def validate_epoch(self, dataloader: DataLoader, epoch: int, train_config: TrainConfig) -> Tuple[float, float]:
        self.model.eval()
        total_loss = 0.0
        total_tokens = 0
        with torch.no_grad():
            iterator = tqdm(dataloader, desc='Validation') if self.rank == 0 else dataloader
            for (y_batch, attn_mask) in iterator:
                y_batch = y_batch.to(self.device)
                decoder_input = y_batch[:, :-1]
                targets = y_batch[:, 1:]
                tgt_padding_mask = self._create_padding_mask(decoder_input, train_config.pad_token_id)
                logits = self.model(y_batch, decoder_input, None, tgt_padding_mask)
                loss = self.criterion(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
                total_loss += loss.item() * targets.numel()
                total_tokens += targets.numel()
                if self.rank == 0:
                    try:
                        iterator.set_postfix({'loss': f'{(total_loss/max(1,total_tokens)):.6f}'})
                    except Exception:
                        pass
        if dist.is_initialized():
            tl = torch.tensor(total_loss, device=self.device)
            tt = torch.tensor(total_tokens, device=self.device)
            dist.all_reduce(tl, op=dist.ReduceOp.SUM)
            dist.all_reduce(tt, op=dist.ReduceOp.SUM)
            total_loss = tl.item()
            total_tokens = tt.item()
        avg_loss = total_loss / max(1, total_tokens)
        ppl = math.exp(avg_loss)
        self.val_losses.append(avg_loss)
        self.val_ppls.append(ppl)
        if self.writer is not None and self.rank == 0:
            self.writer.add_scalar('Loss/Val', avg_loss, epoch)
            self.writer.add_scalar('Perplexity/Val', ppl, epoch)
        return avg_loss, ppl

    def evaluate_test_set(self, dataloader: DataLoader, train_config: TrainConfig) -> Tuple[float, float]:
        self.model.eval()
        total_loss = 0.0
        total_tokens = 0
        with torch.no_grad():
            iterator = tqdm(dataloader, desc='Testing') if self.rank == 0 else dataloader
            for (y_batch, attn_mask) in iterator:
                y_batch = y_batch.to(self.device)
                decoder_input = y_batch[:, :-1]
                targets = y_batch[:, 1:]
                tgt_padding_mask = self._create_padding_mask(decoder_input, train_config.pad_token_id)
                logits = self.model(y_batch, decoder_input, None, tgt_padding_mask)
                loss = self.criterion(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
                total_loss += loss.item() * targets.numel()
                total_tokens += targets.numel()
        if dist.is_initialized():
            tl = torch.tensor(total_loss, device=self.device)
            tt = torch.tensor(total_tokens, device=self.device)
            dist.all_reduce(tl, op=dist.ReduceOp.SUM)
            dist.all_reduce(tt, op=dist.ReduceOp.SUM)
            total_loss = tl.item()
            total_tokens = tt.item()
        avg_loss = total_loss / max(1, total_tokens)
        ppl = math.exp(avg_loss)
        return avg_loss, ppl

    def save_model(self, filepath: str, epoch: Optional[int] = None):
        if self.rank != 0:
            return
        model_to_save = self.model.module if isinstance(self.model, DistributedDataParallel) else self.model
        save_dict = {
            'model_state_dict': model_to_save.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict() if self.optimizer else None,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_ppls': self.train_ppls,
            'val_ppls': self.val_ppls,
            'best_val_loss': self.best_val_loss,
            'step_count': self.step_count,
            'model_config': {
                'vocab_size': model_to_save.config.vocab_size,
                'd_model': model_to_save.config.d_model,
                'seq_len': model_to_save.config.seq_len,
                'n_heads': model_to_save.config.n_heads,
                'd_hidden': model_to_save.config.d_hidden,
                'stack': model_to_save.config.stack,
            }
        }
        if epoch is not None:
            save_dict['epoch'] = epoch
        torch.save(save_dict, filepath)

    def load_model(self, filepath: str, load_optimizer: bool = True) -> int:
        checkpoint = torch.load(filepath, map_location=self.device)
        model_to_load = self.model.module if isinstance(self.model, DistributedDataParallel) else self.model
        model_to_load.load_state_dict(checkpoint['model_state_dict'], strict=False)
        if load_optimizer and self.optimizer and checkpoint.get('optimizer_state_dict'):
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])
        self.train_ppls = checkpoint.get('train_ppls', [])
        self.val_ppls = checkpoint.get('val_ppls', [])
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        self.step_count = checkpoint.get('step_count', 0)
        start_epoch = checkpoint.get('epoch', 0)
        return start_epoch

    def train(self, train_loader: DataLoader, val_loader: DataLoader, test_loader: DataLoader, train_config: TrainConfig, save_dir: str = './gpt_checkpoints', resume_from_checkpoint: Optional[str] = None):
        if self.rank == 0:
            os.makedirs(save_dir, exist_ok=True)
        start_epoch = 0
        if resume_from_checkpoint and os.path.exists(resume_from_checkpoint):
            start_epoch = self.load_model(resume_from_checkpoint, load_optimizer=True)
        if self.rank == 0:
            print(f"开始训练，共 {train_config.num_epochs} 个epoch")
        steps_per_epoch_estimate = None
        for epoch in range(start_epoch, train_config.num_epochs):
            train_loss, train_ppl, steps_this_epoch = self.train_epoch(train_loader, epoch, train_config)
            if steps_per_epoch_estimate is None:
                steps_per_epoch_estimate = steps_this_epoch
                if self.scheduler is None and train_config.total_steps is None and steps_per_epoch_estimate > 0:
                    total_steps = train_config.num_epochs * steps_per_epoch_estimate
                    self._build_scheduler(train_config, total_steps)
            val_loss, val_ppl = self.validate_epoch(val_loader, epoch, train_config)
            if self.rank == 0:
                print(f"Epoch {epoch+1}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, train_ppl={train_ppl:.2f}, val_ppl={val_ppl:.2f}")
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_model(os.path.join(save_dir, 'best_model.pt'), epoch)
            if (epoch + 1) % 5 == 0:
                self.save_model(os.path.join(save_dir, f'checkpoint_epoch_{epoch+1}.pt'), epoch)
        if self.rank == 0:
            print("测试集评估")
        test_loss, test_ppl = self.evaluate_test_set(test_loader, train_config)
        if self.rank == 0:
            print(f"测试损失: {test_loss:.4f}, 测试困惑度: {test_ppl:.2f}")


def create_model_and_trainer(gpt_config: Optional[GPTConfig] = None, env_config: Optional[EnvConfig] = None, log_dir: Optional[str] = './runs/gpt', enable_tensorboard: bool = True) -> Tuple[GPT, GPTTrainer]:
    """工厂方法：创建 GPT 模型与训练器"""
    cfg = gpt_config or GPTConfig()
    model = GPT(cfg)
    trainer = GPTTrainer(model, env_config or EnvConfig(), log_dir, enable_tensorboard)
    return model, trainer