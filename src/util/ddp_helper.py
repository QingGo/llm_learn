import os
import torch
from typing import NamedTuple, Optional
import torch.distributed as dist
from contextlib import contextmanager

class EnvConfig(NamedTuple):
    world_size: int
    rank: int
    local_rank: int
    master_addr: str
    master_port: str

def get_env_config() -> EnvConfig:
    """获取当前进程的环境配置"""
    # 尝试从环境变量获取torchrun参数       
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    rank = int(os.environ.get('RANK', 0))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    master_addr = os.environ.get('MASTER_ADDR', '127.0.0.1')
    master_port = os.environ.get('MASTER_PORT', '23456')
    env_config = EnvConfig(world_size, rank, local_rank, master_addr, master_port)
    return env_config

def print_env_info(env_config):
    """打印环境变量信息"""
    print(f'''Environment Variables:
    WORLD_SIZE: {env_config.world_size}
    RANK: {env_config.rank}
    LOCAL_RANK: {env_config.local_rank}
    MASTER_ADDR: {env_config.master_addr}
    MASTER_PORT: {env_config.master_port}\n''')

def get_device_for_ddp(rank, world_size):
    """为DDP获取合适的设备和backend，处理MPS兼容性问题"""
    # 检测可用设备
    if torch.cuda.is_available():
        local_rank = int(os.environ.get('LOCAL_RANK', rank))
        device = torch.device(f'cuda:{local_rank}')
        backend = 'nccl'
        device_type = 'cuda'
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        if world_size > 1:
            # MPS在多进程DDP时有兼容性问题，回退到CPU
            if rank == 0:
                print("Warning: MPS doesn't fully support DDP operations, falling back to CPU for multi-process training")
            device, backend, device_type = _get_cpu_config()
        else:
            # 单进程时可以使用MPS
            device = torch.device('mps')
            backend = 'gloo'
            device_type = 'mps'
    else:
        device, backend, device_type = _get_cpu_config()
    
    return device, device_type, backend


def _get_cpu_config():
    """获取CPU配置的辅助函数"""
    return torch.device('cpu'), 'gloo', 'cpu'


def init_process(env_config: EnvConfig):
    """初始化分布式进程组"""
    rank, world_size = env_config.rank, env_config.world_size
    _, _, backend = get_device_for_ddp(rank, world_size)

    # 初始化进程组，使用环境变量中的 MASTER_ADDR / MASTER_PORT
    dist.init_process_group(
        backend=backend,
        init_method='env://',
        rank=rank,
        world_size=world_size
    )

    if rank == 0:
        print(f"Initialized DDP with backend: {backend}, world_size: {world_size}")

def log_with_rank(msg: str, rank: Optional[int] = None, only_rank0: bool = False):
    if only_rank0:
        if (rank is not None and rank == 0):
            print(msg)
        else:
            return
    elif rank is None:
        print(msg)
    else:
        print(f'[Rank {rank}] {msg}')

@contextmanager
def ddp_sync_section(world_size: int):
    """在多进程 DDP 下，进入/退出时各做一次 barrier 保持同步"""
    if world_size > 1:
        dist.barrier()
    try:
        yield
    finally:
        if world_size > 1:
            dist.barrier()
