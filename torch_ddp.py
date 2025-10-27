import os
import time
from contextlib import contextmanager
from typing import NamedTuple
import torch
import torch.distributed as dist
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import Dataset


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


def _get_cifar10_transform():
    """获取CIFAR10数据预处理transform"""
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])


def _create_dataloader(dataset: Dataset, rank, world_size, batch_size, is_train=True):
    """创建数据加载器的通用函数"""
    # 根据设备类型调整参数
    _, device_type, _ = get_device_for_ddp(rank, world_size)
    num_workers = 0 if device_type == 'mps' else 2  # MPS在多进程时可能有问题
    pin_memory = (device_type == 'cuda')
    
    # 配置采样器和shuffle
    if world_size > 1:
        # 训练/评估均使用分布式采样器（评估不打乱）
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=is_train)
        shuffle = False
    else:
        # 单进程时保持训练打乱、评估不打乱
        sampler = None
        shuffle = is_train  # 训练时shuffle，测试时不shuffle
    
    return DataLoader(
        dataset, batch_size=batch_size, sampler=sampler, shuffle=shuffle,
        num_workers=num_workers, pin_memory=pin_memory
    )


def build_cifar10_dataloader(split: str, rank: int, world_size: int, batch_size: int):
    """构建 CIFAR10 的通用 dataloader，统一训练/测试逻辑"""
    is_train = split == 'train'
    transform = _get_cifar10_transform()
    dataset = torchvision.datasets.CIFAR10(
        root='./data', train=is_train, download=True, transform=transform
    )
    return _create_dataloader(dataset, rank, world_size, batch_size, is_train=is_train)


def get_dataloader(rank, world_size, batch_size=32):
    """获取分布式训练数据加载器"""
    return build_cifar10_dataloader('train', rank, world_size, batch_size)


def get_test_dataloader(rank, world_size, batch_size=32):
    """获取测试数据加载器"""
    return build_cifar10_dataloader('test', rank, world_size, batch_size)


def evaluate_model(model, test_dataloader, device, rank):
    """评估模型性能（分布式切分与聚合）"""
    model.eval()  # 设置为评估模式
    criterion = torch.nn.CrossEntropyLoss()
    classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    # 本地统计
    loss_sum = 0.0  # 加权loss：sum(loss * batch_size)
    correct = 0
    total = 0
    class_correct = [0] * 10
    class_total = [0] * 10

    with torch.no_grad():
        for data, target in test_dataloader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            loss = criterion(outputs, target)
            batch_size = target.size(0)

            loss_sum += loss.item() * batch_size
            _, predicted = torch.max(outputs, 1)

            total += batch_size
            correct += (predicted == target).sum().item()

            # 每类准确率
            c = (predicted == target).squeeze()
            for i in range(batch_size):
                label = int(target[i].item())
                class_correct[label] += int(c[i].item())
                class_total[label] += 1

    # 分布式聚合
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    if world_size > 1:
        loss_sum_t = torch.tensor(loss_sum, dtype=torch.float32, device=device)
        total_t = torch.tensor(total, dtype=torch.long, device=device)
        correct_t = torch.tensor(correct, dtype=torch.long, device=device)
        class_correct_t = torch.tensor(class_correct, dtype=torch.long, device=device)
        class_total_t = torch.tensor(class_total, dtype=torch.long, device=device)

        # 聚合到 rank 0
        dist.reduce(loss_sum_t, dst=0, op=dist.ReduceOp.SUM)
        dist.reduce(total_t, dst=0, op=dist.ReduceOp.SUM)
        dist.reduce(correct_t, dst=0, op=dist.ReduceOp.SUM)
        dist.reduce(class_correct_t, dst=0, op=dist.ReduceOp.SUM)
        dist.reduce(class_total_t, dst=0, op=dist.ReduceOp.SUM)

        if rank == 0:
            loss_sum = float(loss_sum_t.item())
            total = int(total_t.item())
            correct = int(correct_t.item())
            class_correct = class_correct_t.tolist()
            class_total = class_total_t.tolist()
        else:
            # 非主进程返回占位，避免误用
            return None, None

    # 计算总体指标（仅单进程或主进程）
    avg_loss = (loss_sum / total) if total > 0 else 0.0
    accuracy = (100.0 * correct / total) if total > 0 else 0.0

    # 只在主进程打印结果
    if rank == 0:
        print('\nTest Results:')
        print(f'Test Loss: {avg_loss:.4f}')
        print(f'Test Accuracy: {accuracy:.2f}% ({correct}/{total})')
        print('\nPer-class Accuracy:')
        for i in range(10):
            if class_total[i] > 0:
                class_acc = 100.0 * class_correct[i] / class_total[i]
                print(f'  {classes[i]}: {class_acc:.2f}% ({class_correct[i]}/{class_total[i]})')

    return accuracy, avg_loss


def build_model(rank, world_size):
    """构建分布式模型"""
    device, device_type, _ = get_device_for_ddp(rank, world_size)

    # 创建模型
    model = torchvision.models.resnet18(weights=None, num_classes=10).to(device)

    # 只有在多进程时才使用DDP
    if world_size > 1:
        if device_type == 'cuda':
            local_rank = int(os.environ.get('LOCAL_RANK', rank))
            ddp_model = DistributedDataParallel(model, device_ids=[local_rank])
        else:
            ddp_model = DistributedDataParallel(model)
    else:
        ddp_model = model  # 单进程时不需要DDP包装

    return ddp_model, device

def log(rank: int, msg: str):
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


def train(epochs, batch_size, env_config: EnvConfig, test_every_epoch=True):
    """分布式训练函数"""
    rank, world_size = env_config.rank, env_config.world_size
    
    # 只有在多进程时才初始化进程组
    if world_size > 1:
        init_process(env_config)
    
    # 获取数据加载器和模型
    dataloader = get_dataloader(rank, world_size, batch_size)
    test_dataloader = get_test_dataloader(rank, world_size, batch_size) if test_every_epoch else None
    model, device = build_model(rank, world_size)
    
    # 定义损失函数和优化器
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    
    # 记录总训练时间
    total_start_time = time.time()
    
    log(rank, f"Starting training on {device} for {epochs} epochs...")
    log(rank, f"Batches per epoch: {len(dataloader)}")
    if rank == 0:
        print(f"Dataset size: {len(dataloader.dataset)}, Batch size: {batch_size}") # type: ignore
        print(f"World size: {world_size}")
    
    for epoch in range(epochs):
        epoch_start_time = time.time()
        
        # 重要：设置sampler的epoch，确保数据拆分一致（仅在多进程时需要）
        if world_size > 1:
            if isinstance(dataloader.sampler, DistributedSampler):
                dataloader.sampler.set_epoch(epoch)
        
        # 训练模式
        model.train()
        epoch_loss = 0.0
        num_batches = 0
        
        for batch_idx, (data, target) in enumerate(dataloader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
            
            # 打印训练进度（减少打印频率），所有 Rank 前缀标注
            if batch_idx % 100 == 0:
                log(rank, f'Epoch {epoch+1}/{epochs}, Batch {batch_idx}/{len(dataloader)}, Loss: {loss.item():.4f}')
        
        # 计算并打印epoch统计信息
        epoch_time = time.time() - epoch_start_time
        avg_loss = epoch_loss / num_batches
        
        log(rank, f'Epoch {epoch+1}/{epochs} completed in {epoch_time:.2f}s, Average Loss: {avg_loss:.4f}')
        if rank == 0:
            print(f'Throughput: {len(dataloader.dataset) / epoch_time:.2f} samples/sec') # type: ignore
        
        # 每个epoch后进行测试（可选）：所有Rank参与评估，聚合后仅在Rank 0记录
        if test_every_epoch and test_dataloader is not None:
            with ddp_sync_section(world_size):
                log(rank, f'\nEvaluating after epoch {epoch+1}...')
                accuracy, test_loss = evaluate_model(model, test_dataloader, device, rank)
                if rank == 0:
                    log(rank, f'Epoch {epoch+1} - Test Accuracy: {accuracy:.2f}%, Test Loss: {test_loss:.4f}')
    
    # 计算总训练时间
    total_time = time.time() - total_start_time
    
    if rank == 0:
        print('\nTraining completed!')
        print(f'Total training time: {total_time:.2f}s')
        print(f'Average time per epoch: {total_time/epochs:.2f}s')
        print(f'Overall throughput: {len(dataloader.dataset) * epochs / total_time:.2f} samples/sec') # type: ignore
    
    # 最终测试（如果没有每个epoch都测试）：所有Rank参与评估，聚合后仅在Rank 0输出
    if not test_every_epoch:
        with ddp_sync_section(world_size):
            print('\nFinal evaluation...')
            test_dataloader = get_test_dataloader(rank, world_size, batch_size)
            final_accuracy, final_test_loss = evaluate_model(model, test_dataloader, device, rank)
            if rank == 0:
                print(f'Final Test Results - Accuracy: {final_accuracy:.2f}%, Loss: {final_test_loss:.4f}')

    # 仅在 Rank 0 保存模型（兼容 DDP 包裹），并在进入/退出时同步
    with ddp_sync_section(world_size):
        if rank == 0:
            save_path = 'resnet18_cifar10_ddp.pth'
            state_dict = model.module.state_dict() if isinstance(model, DistributedDataParallel) else model.state_dict()
            torch.save(state_dict, save_path)
            print(f'Model saved to {save_path}')
    
    # 清理环境（仅在多进程时需要）
    if world_size > 1:
        dist.destroy_process_group()
    
    return model


'''
本机 uv run torchrun --nproc_per_node=1 torch_ddp.py，mps 单机训练
Total training time: 73.91s
Average time per epoch: 73.91s
Overall throughput: 676.50 samples/sec
本机 uv run torchrun --nproc_per_node=2 torch_ddp.py，cpu 双进程训练
Training completed!
Total training time: 526.98s
Average time per epoch: 526.98s
Overall throughput: 94.88 samples/sec
CPU 服务器 uv run torchrun --nproc_per_node=4 torch_ddp.py，cpu 四进程训练
'''

if __name__ == "__main__":
    env_config = get_env_config()
    print_env_info(env_config)
    train(epochs=1, batch_size=32, env_config=env_config, test_every_epoch=True)
    