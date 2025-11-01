import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# 设备选择：优先 MPS(Apple) > CUDA > CPU
def get_device():
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")

# 随机种子：提高可复现性
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# 两层 FCN
class TwoLayerFCN(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=256, output_dim=10, dropout_rate=0.5):
        super().__init__()
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, output_dim)
        )
    def forward(self, x):
        return self.model(x)

# 数据加载与标准化
def make_loaders(batch_size=64):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    train_ds = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    test_ds  = datasets.MNIST(root="./data", train=False, download=True, transform=transform)
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True),
        DataLoader(test_ds,  batch_size=batch_size, shuffle=False)
    )

# 训练与评估
def train_one_epoch(model, loader, criterion, optimizer, device, epoch):
    # 训练模式，启用训练时行为，如随机丢弃（Dropout）和统计更新（BatchNorm）
    model.train()
    total_loss, total_samples = 0.0, len(loader.dataset)
    for batch_idx, (data, target) in enumerate(loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()
        # CrossEntropyLoss 默认参数 reduction='mean'，计算每个样本的损失后取平均。
        # 这里手动乘以样本数，用于计算总损失。
        total_loss += loss.item() * data.size(0)
        if (batch_idx + 1) % 100 == 0:
            print(f"Epoch [{epoch}/5] Batch [{batch_idx+1}/{len(loader)}] Loss: {loss.item():.4f}")
    # 计算并返回当前 epoch 的平均损失
    return total_loss / total_samples

def evaluate(model, loader, criterion, device):
    # 评估模式，关闭训练时的随机性，用“估计好的”统计值进行前向。它不冻结参数、也不关闭梯度。
    model.eval()
    total_loss, correct = 0.0, 0
    total_samples = len(loader.dataset)
    # 评估时不计算梯度，节省内存，提升速度
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            loss = criterion(outputs, target)
            total_loss += loss.item() * data.size(0)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == target).sum().item()
    return total_loss / total_samples, 100.0 * correct / total_samples

# 主流程
def main():
    set_seed(42)
    device = get_device()
    print("使用设备:", device)
    train_loader, test_loader = make_loaders()
    # 维度验证（一次）
    x, y = next(iter(train_loader))
    print("单批次数据维度:", x.shape, "标签维度:", y.shape)

    model = TwoLayerFCN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-3)

    # 初始化 TensorBoard 记录器
    writer = SummaryWriter(log_dir="runs/mnist_fcn")
    writer.add_graph(model, x.to(device))  # 记录计算图

    for epoch in range(1, 6):
        print(f"\n===== 开始训练第 {epoch} 轮 =====")
        avg_train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch)
        avg_test_loss, test_accuracy = evaluate(model, test_loader, criterion, device)

        print(f"第 {epoch} 轮完成 | 训练损失: {avg_train_loss:.4f} | 测试损失: {avg_test_loss:.4f} | 准确率: {test_accuracy:.2f}%")

        # TensorBoard 标量与参数分布
        writer.add_scalar("Loss/train", avg_train_loss, epoch)
        writer.add_scalar("Loss/test", avg_test_loss, epoch)
        writer.add_scalar("Accuracy/test", test_accuracy, epoch)
        for name, param in model.named_parameters():
            writer.add_histogram(f"Params/{name}", param, epoch)

    writer.flush()
    writer.close()
    print("\n训练完成。可运行：uv run tensorboard --logdir runs --port 6006 查看曲线")

if __name__ == "__main__":
    main()