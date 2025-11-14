import typer
import torch.distributed as dist
from typing import Optional

from util.ddp_helper import get_env_config, init_process, log_with_rank, print_env_info
from .config import GPTConfig, TrainConfig
from .trainer import create_model_and_trainer
from .data import create_gpt_dataloaders


app = typer.Typer(help="GPT-2 风格语言模型训练")


@app.command()
def train(
    batch_size: int = typer.Option(512, help="批次大小"),
    seq_len: int = typer.Option(1024, help="上下文长度"),
    num_epochs: int = typer.Option(100, help="训练轮数"),
    learning_rate: float = typer.Option(2.5e-4, help="最大学习率"),
    warmup_steps: int = typer.Option(2000, help="线性预热步数"),
    weight_decay: float = typer.Option(0.01, help="权重衰减"),
    buffer_size: int = typer.Option(10000, help="迭代器打散缓冲大小"),
    log_dir: Optional[str] = typer.Option('./runs/gpt', help="TensorBoard 日志目录"),
    disable_tensorboard: bool = typer.Option(False, help="禁用 TensorBoard"),
    resume_checkpoint: Optional[str] = typer.Option(None, help="从检查点恢复"),
    processed_path: Optional[str] = typer.Option(None, help="离线块级数据集路径（datasets.save_to_disk 的输出）"),
    tokenizer_path: Optional[str] = typer.Option('./data/tokenizer.json', help="分词器文件路径"),
):
    env_config = get_env_config()
    print_env_info(env_config)
    if env_config.world_size > 1:
        init_process(env_config)
        log_with_rank("DDP process group initialized.", env_config.rank)
    train_loader, val_loader, test_loader, tokenizer = create_gpt_dataloaders(seq_len=seq_len, batch_size=batch_size, buffer_size=buffer_size, processed_path=processed_path, tokenizer_path=tokenizer_path)
    gpt_cfg = GPTConfig(vocab_size=tokenizer.get_vocab_size(), d_model=768, seq_len=seq_len, n_heads=12, d_hidden=3072, stack=12, dropout=0.1)
    model, trainer = create_model_and_trainer(gpt_cfg, env_config, log_dir, not disable_tensorboard)
    tcfg = TrainConfig(batch_size=batch_size, num_epochs=num_epochs, learning_rate=learning_rate, weight_decay=weight_decay, warmup_steps=warmup_steps, buffer_size=buffer_size)
    trainer.setup_training(tcfg)
    trainer.train(train_loader, val_loader, test_loader, tcfg, save_dir='./gpt_checkpoints', resume_from_checkpoint=resume_checkpoint)
    if env_config.world_size > 1:
        dist.destroy_process_group()
        log_with_rank("DDP process group destroyed.", env_config.rank)


if __name__ == '__main__':
    app()