import os
import time
import torch
import torch.distributed as dist
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import Dataset

# uv run --env-file .env torchrun --nproc_per_node=1 src/misc/torch_ddp.py
# uv run --env-file .env src/misc/torch_ddp.py
from util.ddp_helper import EnvConfig, get_env_config, get_device_for_ddp, \
    print_env_info, init_process, log_with_rank, ddp_sync_section

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
    
    log_with_rank(f"Starting training on {device} for {epochs} epochs...", rank)
    log_with_rank(f"Batches per epoch: {len(dataloader)}", rank)
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
                log_with_rank(f'Epoch {epoch+1}/{epochs}, Batch {batch_idx}/{len(dataloader)}, Loss: {loss.item():.4f}', rank)
        
        # 计算并打印epoch统计信息
        epoch_time = time.time() - epoch_start_time
        avg_loss = epoch_loss / num_batches
        
        log_with_rank(f'Epoch {epoch+1}/{epochs} completed in {epoch_time:.2f}s, Average Loss: {avg_loss:.4f}', rank)
        if rank == 0:
            print(f'Throughput: {len(dataloader.dataset) / epoch_time:.2f} samples/sec') # type: ignore
        
        # 每个epoch后进行测试（可选）：所有Rank参与评估，聚合后仅在Rank 0记录
        if test_every_epoch and test_dataloader is not None:
            with ddp_sync_section(world_size):
                log_with_rank(f'\nEvaluating after epoch {epoch+1}...', rank)
                accuracy, test_loss = evaluate_model(model, test_dataloader, device, rank)
                if rank == 0:
                    log_with_rank(f'Epoch {epoch+1} - Test Accuracy: {accuracy:.2f}%, Test Loss: {test_loss:.4f}', rank)
    
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
Total training time: 66.00s                                                               │请执行以下命令开启系统代理: proxy_on
Average time per epoch: 66.00s                                                            │
Overall throughput: 757.52 samples/sec
GPU 服务器 uv run torchrun --nproc_per_node=8 torch_ddp.py，8块3090训练
Training completed!
Total training time: 8.13s
Average time per epoch: 8.13s
Overall throughput: 6149.27 samples/sec
GPU 服务器 uv run torchrun --nproc_per_node=4 torch_ddp.py，4块3090训练
Training completed!
Total training time: 13.18s
Average time per epoch: 13.18s
Overall throughput: 3792.22 samples/sec
GPU 服务器 uv run torchrun --nproc_per_node=1 torch_ddp.py，1块3090训练
Training completed!
Total training time: 34.80s
Average time per epoch: 34.80s
Overall throughput: 1436.94 samples/sec
'''

if __name__ == "__main__":
    env_config = get_env_config()
    print_env_info(env_config)
    train(epochs=1, batch_size=32, env_config=env_config, test_every_epoch=True)
    