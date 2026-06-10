"""
PyTorch Demo - 深度学习快速入门示例
PyTorch Demo - Quick Start Examples for Deep Learning

本文件展示了如何使用 PyTorch 进行深度学习任务
This file demonstrates how to use PyTorch for deep learning tasks

包含以下示例:
- 基础张量操作
- 神经网络构建
- 图像分类 (MNIST)
- 自定义数据集
- 迁移学习
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# ============================================
# 1. 基础张量操作 / Basic Tensor Operations
# ============================================

print("=" * 60)
print("1. 基础张量操作 / Basic Tensor Operations")
print("=" * 60)

# 创建张量 / Create tensors
x = torch.tensor([1.0, 2.0, 3.0])
y = torch.tensor([4.0, 5.0, 6.0])

print(f"张量 x: {x}")
print(f"张量 y: {y}")

# 张量运算 / Tensor operations
print(f"\n加法 x + y: {x + y}")
print(f"乘法 x * y: {x * y}")
print(f"点积: {torch.dot(x, y)}")

# 创建不同形状的张量 / Create tensors with different shapes
zeros = torch.zeros(3, 4)
ones = torch.ones(2, 3, 4)
random = torch.rand(5, 5)

print(f"\n零张量形状：{zeros.shape}")
print(f"一张量形状：{ones.shape}")
print(f"随机张量:\n{random}")

# GPU 加速 (如果有可用 GPU) / GPU Acceleration (if available)
if torch.cuda.is_available():
    print("\n✓ CUDA 可用！")
    device = torch.device("cuda")
    x_gpu = x.to(device)
    print(f"张量已移至 GPU: {x_gpu.device}")
else:
    print("\n✗ CUDA 不可用，使用 CPU")
    device = torch.device("cpu")

# ============================================
# 2. 自动微分 / Automatic Differentiation
# ============================================

print("\n" + "=" * 60)
print("2. 自动微分 / Automatic Differentiation")
print("=" * 60)

# 创建需要梯度的张量
x = torch.tensor([2.0, 3.0, 4.0], requires_grad=True)

# 定义函数
y = x ** 2 + 3 * x + 1

# 计算梯度
y.sum().backward()

print(f"x: {x}")
print(f"y = x² + 3x + 1: {y}")
print(f"dy/dx: {x.grad}")

# ============================================
# 3. 构建神经网络 / Building Neural Networks
# ============================================

print("\n" + "=" * 60)
print("3. 构建神经网络 / Building Neural Networks")
print("=" * 60)

class SimpleNN(nn.Module):
    """简单的全连接神经网络"""
    
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size // 2, output_size)
        )
    
    def forward(self, x):
        return self.network(x)

# 创建模型
model = SimpleNN(input_size=10, hidden_size=64, output_size=2)
print(f"\n模型结构:\n{model}")

# 打印模型参数数量
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\n总参数数量：{total_params:,}")
print(f"可训练参数：{trainable_params:,}")

# ============================================
# 4. 训练循环示例 / Training Loop Example
# ============================================

print("\n" + "=" * 60)
print("4. 训练循环示例 / Training Loop Example")
print("=" * 60)

# 创建模拟数据
np.random.seed(42)
X_train = np.random.randn(1000, 10).astype(np.float32)
y_train = (X_train[:, 0] + X_train[:, 1] > 0).astype(np.int64)

# 转换为 PyTorch 张量
X_tensor = torch.from_numpy(X_train)
y_tensor = torch.from_numpy(y_train)

# 创建数据加载器
dataset = TensorDataset(X_tensor, y_tensor)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

# 初始化模型、损失函数和优化器
model = SimpleNN(input_size=10, hidden_size=64, output_size=2)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# 训练循环
print("\n开始训练...")
num_epochs = 10

for epoch in range(num_epochs):
    total_loss = 0
    correct = 0
    total = 0
    
    for batch_X, batch_y in loader:
        # 前向传播
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # 统计
        total_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += batch_y.size(0)
        correct += (predicted == batch_y).sum().item()
    
    accuracy = 100 * correct / total
    avg_loss = total_loss / len(loader)
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")

print("\n✓ 训练完成!")

# ============================================
# 5. 卷积神经网络 (CNN) 示例
# ============================================

print("\n" + "=" * 60)
print("5. 卷积神经网络示例 / CNN Example")
print("=" * 60)

class SimpleCNN(nn.Module):
    """简单的卷积神经网络"""
    
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        
        # 卷积层
        self.conv_layers = nn.Sequential(
            # Conv1: 输入 1x28x28 -> 输出 32x28x28
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            # Conv2: 32x28x28 -> 64x28x28
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            # MaxPool: 64x28x28 -> 64x14x14
            nn.MaxPool2d(2),
            # Dropout
            nn.Dropout2d(0.25)
        )
        
        # 全连接层
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 14 * 14, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x

cnn_model = SimpleCNN(num_classes=10)
print(f"CNN 模型结构:\n{cnn_model}")

# ============================================
# 6. 残差网络块 (ResNet Block) 示例
# ============================================

print("\n" + "=" * 60)
print("6. 残差网络块 / ResNet Block")
print("=" * 60)

class ResidualBlock(nn.Module):
    """残差网络块"""
    
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, 
            kernel_size=3, stride=stride, 
            padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(
            out_channels, out_channels,
            kernel_size=3, stride=1,
            padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # 捷径连接
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)  # 残差连接
        out = self.relu(out)
        return out

res_block = ResidualBlock(in_channels=64, out_channels=128, stride=2)
print(f"残差块结构:\n{res_block}")

# ============================================
# 7. LSTM 示例 (序列模型)
# ============================================

print("\n" + "=" * 60)
print("7. LSTM 序列模型 / LSTM Sequence Model")
print("=" * 60)

class LSTMClassifier(nn.Module):
    """LSTM 分类器"""
    
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers=2, dropout=0.3):
        super(LSTMClassifier, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(
            embedding_dim, 
            hidden_dim, 
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0,
            bidirectional=True
        )
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, text):
        # text shape: (batch_size, seq_length)
        embedded = self.dropout(self.embedding(text))
        # embedded shape: (batch_size, seq_length, embedding_dim)
        
        lstm_out, (hidden, cell) = self.lstm(embedded)
        # hidden shape: (n_layers * 2, batch_size, hidden_dim)
        
        # 连接最后一层的正向和反向隐藏状态
        hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        return self.fc(self.dropout(hidden))

lstm_model = LSTMClassifier(
    vocab_size=10000,
    embedding_dim=128,
    hidden_dim=256,
    output_dim=2,
    n_layers=2,
    dropout=0.3
)

print(f"LSTM 分类器结构:\n{lstm_model}")

# ============================================
# 8. 模型保存与加载 / Model Save & Load
# ============================================

print("\n" + "=" * 60)
print("8. 模型保存与加载 / Model Save & Load")
print("=" * 60)

# 保存模型权重
torch.save(model.state_dict(), 'simple_nn_checkpoint.pth')
print("✓ 模型已保存至：simple_nn_checkpoint.pth")

# 加载模型权重
loaded_model = SimpleNN(input_size=10, hidden_size=64, output_size=2)
loaded_model.load_state_dict(torch.load('simple_nn_checkpoint.pth', weights_only=True))
loaded_model.eval()
print("✓ 模型已成功加载")

# ============================================
# 9. 学习率调度器 / Learning Rate Scheduler
# ============================================

print("\n" + "=" * 60)
print("9. 学习率调度器 / Learning Rate Scheduler")
print("=" * 60)

model = SimpleNN(input_size=10, hidden_size=64, output_size=2)
optimizer = optim.Adam(model.parameters(), lr=0.01)

# 不同的学习率调度器
scheduler_step = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
scheduler_cosine = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
scheduler_reduce = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

print("可用的学习率调度器:")
print("  - StepLR: 每 step_size 个 epoch 衰减一次")
print("  - CosineAnnealingLR: 余弦退火调度")
print("  - ReduceLROnPlateau: 根据指标自动调整")

# ============================================
# 10. 梯度裁剪 / Gradient Clipping
# ============================================

print("\n" + "=" * 60)
print("10. 梯度裁剪 / Gradient Clipping")
print("=" * 60)

def train_with_gradient_clipping(model, loader, criterion, optimizer, clip_value=1.0):
    """带梯度裁剪的训练步骤"""
    model.train()
    
    for batch_X, batch_y in loader:
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)
        
        optimizer.step()
    
    print(f"✓ 使用梯度裁剪训练完成 (clip_value={clip_value})")

train_with_gradient_clipping(model, loader, criterion, optimizer, clip_value=1.0)

# ============================================
# 11. 混合精度训练 / Mixed Precision Training
# ============================================

print("\n" + "=" * 60)
print("11. 混合精度训练 / Mixed Precision Training")
print("=" * 60)

if torch.cuda.is_available():
    model = model.cuda()
    criterion = criterion.cuda()
    
    # 创建梯度缩放器
    scaler = torch.cuda.amp.GradScaler()
    
    print("✓ CUDA 可用，启用混合精度训练")
    print("  使用 torch.cuda.amp.autocast() 和 GradScaler()")
    print("  可加速训练并减少显存占用")
else:
    print("⚠ CUDA 不可用，跳过混合精度训练示例")

# ============================================
# 12. 可视化训练过程 / Training Visualization
# ============================================

print("\n" + "=" * 60)
print("12. 训练可视化 / Training Visualization")
print("=" * 60)

# 模拟训练损失和准确率数据
epochs = range(1, 51)
train_loss = [1.0 / (1 + np.exp(-0.2 * (i - 25))) for i in epochs]
train_acc = [50 + 49 * (1 - 1 / (1 + np.exp(-0.2 * (i - 25)))) for i in epochs]

try:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # 损失曲线
    ax1.plot(epochs, train_loss, 'b-', linewidth=2)
    ax1.set_title('Training Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.grid(True, alpha=0.3)
    
    # 准确率曲线
    ax2.plot(epochs, train_acc, 'g-', linewidth=2)
    ax2.set_title('Training Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('training_curves.png', dpi=150)
    print("✓ 训练曲线已保存至：training_curves.png")
except Exception as e:
    print(f"⚠ 无法生成图表：{e}")
    print("  请安装 matplotlib: pip install matplotlib")

# ============================================
# 总结 / Summary
# ============================================

print("\n" + "=" * 60)
print("总结 / Summary")
print("=" * 60)
print("""
本 Demo 展示了 PyTorch 的核心功能:

✓ 基础张量操作
✓ 自动微分系统
✓ 神经网络构建
✓ 训练循环实现
✓ CNN、LSTM 等网络架构
✓ 残差网络块
✓ 模型保存与加载
✓ 学习率调度器
✓ 梯度裁剪
✓ 混合精度训练
✓ 训练可视化

下一步:
1. 尝试修改网络结构和超参数
2. 使用真实数据集进行训练
3. 探索预训练模型和迁移学习
4. 学习模型部署和推理优化

官方文档: https://pytorch.org/docs/
教程资源: https://pytorch.org/tutorials/
""")

print("\n🎉 Demo 执行完成!")