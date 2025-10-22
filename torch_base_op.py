import torch
import numpy as np

# 1. 张量创建与设备/数据类型
# 1.1 以 3 种方式创建形状为 (2, 3, 4) 的 float32 张量；1.2 设备迁移；1.3 标记梯度需求
t1 = torch.tensor(np.random.randn(2,3,4), dtype=torch.float32)
t2 = torch.zeros((2,3,4), dtype=torch.float32)
t3 = torch.ones((2,3,4), dtype=torch.float32)
print(f"张量 t1/t2/t3 形状分别为: {t1.shape}, {t2.shape}, {t3.shape}")
# 1.2 设备迁移（CPU/MPS）
# 说明：Apple MPS 仅支持 float32，不支持 float64；运算需设备一致
t1 = t1.to("mps")
print(f"t1 设备（迁移到 MPS 后）: {t1.device}")
# 注意：不同设备上的张量不能直接运算（例如 t1 + t3），需在同一设备
t1 = t1.to("cpu")
print(f"t1 设备（迁移回 CPU 后）: {t1.device}")
# 1.3 标记张量是否需要梯度
print(f"t1 是否需要梯度（初始）: {t1.requires_grad}")
t1.requires_grad_(True)
print(f"t1 是否需要梯度（启用后）: {t1.requires_grad}")
t4 = t1 * 2
print(f"t4 是否需要梯度（从 t1 计算得到）: {t4.requires_grad}")
t5 = torch.tensor([[2,1], [1,2]], dtype=torch.float32, requires_grad=True)
print(f"t5 是否需要梯度: {t5.requires_grad}")

# 2. 张量索引、切片与维度变换
t6 = t1.view(-1, 12)
print(f"t6 形状（view 重排）: {t6.shape}")
print(f"t1 是否为连续内存: {t1.is_contiguous()}")
# 等价于切片 t1[:, 1:3, :]
t7 = torch.index_select(t1, dim=1, index=torch.tensor([1,2]))
print(f"t7 是否为连续内存（index_select 结果）: {t7.is_contiguous()}")
# 非连续张量使用 reshape 或 permute 改变形状，view 会报错
# 建议采用 reshape 或 permute 安全变换形状
# t8 = t7.view(-1, 4) 会报错，因为 t7 不是连续内存布局
# 可以使用 .reshape() 或 .permute() 等方法来改变维度
t8 = t7.reshape(-1, 4)
print(f"t8 形状（reshape 后）: {t8.shape}")
t9 = t1.permute(2, 0, 1)
print(f"t9 形状（permute 后）: {t9.shape}")
t10 = torch.zeros((2,1,4), dtype=torch.float32).squeeze(1)
print(f"t10 形状（squeeze 后）: {t10.shape}")
# 3. 张量核心运算（元素级、矩阵运算、广播）
# 创建元素值为 2 的张量
t11 = torch.full((2,3), 2, dtype=torch.float32)
t12 = torch.full((1,3), 1, dtype=torch.float32)
print("广播加法结果 t11 + t12:\n", t11 + t12)
t13 = torch.full((3,4), 3, dtype=torch.float32)
t14 = torch.full((4,5), 4, dtype=torch.float32)
# 等价于 torch.mm(t13, t14)
print("矩阵乘法结果 t13 @ t14:\n", t13 @ t14)
# 4. autograd 动态图与梯度求解流程
w = torch.tensor(2, dtype=torch.float32, requires_grad=True)
x = torch.tensor(3, dtype=torch.float32)
b = torch.tensor(4, dtype=torch.float32, requires_grad=True)
y = w * w * x + b
print(f"y = w*w*x + b 的数值: {y.item()}")
# 调用 backward 以计算梯度；否则后续 grad 将为 None
y.backward()
# dy/dw = 2*w*x
print(f"梯度 w.grad（2*w*x）: {w.grad.item()}")
# x 未设置 requires_grad，故梯度为 None
print(f"x.grad（未跟踪梯度）: {x.grad}")
# dy/db = 1
print(f"梯度 b.grad（常数 1）: {b.grad.item()}")
with torch.no_grad():
    z = w * w * x + b
    # z.backward() 报错：不含 grad_fn（未跟踪梯度）
    
# 5. 梯度清零与参数更新
linear = torch.nn.Linear(3, 2)
print("线性层初始权重 W:\n", linear.weight)
print("线性层初始偏置 b:\n", linear.bias)
x = torch.randn(4, 3)
y_true = torch.ones(4, 2)
# 前向传播
y_pred = linear(x)
print("预测值 y_pred:\n", y_pred)
loss = torch.nn.MSELoss()(y_pred, y_true)  # 均方误差损失
print(f"均方误差损失 MSE: {loss.item():.6f}")

# 反向传播
loss.backward()

# 查看梯度
print("线性层权重 W 的梯度:\n", linear.weight.grad)
print("线性层偏置 b 的梯度:\n", linear.bias.grad)

# 更新参数
learning_rate = 0.01
for param in linear.parameters():
    param.data -= learning_rate * param.grad
print("更新后权重 W:\n", linear.weight)
print("更新后偏置 b:\n", linear.bias)

# 梯度清零
linear.zero_grad()
