# PyTorch-Wrap

<div align="center">

**PyTorch 深度学习封装 - Python 动态神经网络框架**

[PyTorch](https://github.com/pytorch/pytorch) 的封装包装，提供简洁易用的 API 接口

[![License](https://img.shields.io/badge/license-BSD--3--Clause-blue.svg)](https://github.com/pytorch/pytorch/blob/main/LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/pytorch-2.0%2B-red)](https://pytorch.org/)

[English](#english) | [中文](#中文)

</div>

---

## 📖 简介 / Introduction

**PyTorch-Wrap** 是对 [PyTorch](https://github.com/pytorch/pytorch) 深度学习框架的高级封装，旨在提供更简洁、更直观的 API 接口，让深度学习的入门和实践变得更加简单。

**PyTorch-Wrap** is a high-level wrapper around the [PyTorch](https://github.com/pytorch/pytorch) deep learning framework, designed to provide a cleaner and more intuitive API, making deep learning easier to learn and practice.

### 核心特性 / Core Features

- 🚀 **开箱即用** - 预配置的模型、训练器和数据加载器
- 🎯 **简洁 API** - 大幅减少样板代码，专注于核心逻辑
- 🔧 **灵活扩展** - 完全兼容原生 PyTorch，可随时切换
- 📦 **丰富组件** - 内置常用网络模型和工具函数
- 🌐 **双语支持** - 完善的中文和英文文档

---

## 💻 快速开始 / Quick Start

### 安装 / Installation

```bash
# 从源码安装
pip install pytorch-wrap

# 或使用 conda
conda install -c conda-forge pytorch-wrap
```

### 基础示例 / Basic Example

```python
import torch
from pytorch_wrap import SimpleNN, Trainer, DataLoader

# 1. 准备数据 / Prepare Data
train_data = DataLoader(x_train, y_train, batch_size=32, shuffle=True)
test_data = DataLoader(x_test, y_test, batch_size=32)

# 2. 定义模型 / Define Model
model = SimpleNN(
    input_size=784,
    hidden_sizes=[512, 256],
    output_size=10,
    dropout=0.3
)

# 3. 配置训练器 / Configure Trainer
trainer = Trainer(
    model=model,
    loss_fn='cross_entropy',
    optimizer='adam',
    lr=0.001,
    device='auto'  # 自动使用 GPU
)

# 4. 训练模型 / Train Model
trainer.fit(train_data, epochs=10, val_data=test_data)

# 5. 评估与预测 / Evaluate and Predict
accuracy = trainer.evaluate(test_data)
predictions = trainer.predict(x_test)
```

---

## 📚 详细文档 / Documentation

### 1. 模型构建 / Model Building

```python
from pytorch_wrap import ModelBuilder

# 快速构建 CNN
cnn = ModelBuilder.create_cnn(
    input_channels=3,
    layers=[64, 128, 256],
    kernel_size=3,
    pool_size=2,
    num_classes=10
)

# 快速构建 RNN/LSTM
rnn = ModelBuilder.create_lstm(
    input_size=128,
    hidden_size=256,
    num_layers=2,
    num_classes=10,
    bidirectional=True
)

# 快速构建 Transformer
transformer = ModelBuilder.create_transformer(
    d_model=512,
    nhead=8,
    num_encoder_layers=6,
    dim_feedforward=2048,
    num_classes=1000
)
```

### 2. 数据加载 / Data Loading

```python
from pytorch_wrap import Dataset, DataLoader

# 自动数据增强
train_dataset = Dataset(
    data_path='./data/train',
    augment=True,
    resize=(224, 224),
    normalize=True
)

# 高效数据加载
train_loader = DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True,
    num_workers=4,
    pin_memory=True
)
```

### 3. 训练配置 / Training Configuration

```python
from pytorch_wrap import Trainer, EarlyStopping, LearningRateScheduler

# 高级训练配置
trainer = Trainer(
    model=model,
    loss_fn='focal_loss',
    optimizer='adamw',
    lr=0.001,
    weight_decay=1e-4,
    device='cuda:0',
    mixed_precision=True,  # 混合精度训练
    grad_clip=1.0,
    callbacks=[
        EarlyStopping(patience=10, monitor='val_loss'),
        LearningRateScheduler('cosine', T_max=50),
        TensorBoardLogger('./logs'),
        ModelCheckpoint('./checkpoints', save_best=True)
    ]
)

# 开始训练
trainer.fit(
    train_loader,
    epochs=100,
    val_loader=val_loader,
    resume_from='./checkpoints/best.pth'
)
```

---

## 🎯 实际案例 / Real-World Examples

### 图像分类 / Image Classification

```python
from pytorch_wrap import ImageClassifier

# 一键图像分类
clf = ImageClassifier(
    backbone='resnet50',
    pretrained=True,
    num_classes=100,
    freeze_backbone=False
)

clf.train(
    train_dir='./data/train',
    val_dir='./data/val',
    epochs=50,
    batch_size=32
)

# 预测
results = clf.predict('./test_images/')
```

### 文本分类 / Text Classification

```python
from pytorch_wrap import TextClassifier

# 文本分类
clf = TextClassifier(
    model_type='bert',
    num_classes=5,
    max_length=128
)

clf.fit(
    texts=train_texts,
    labels=train_labels,
    val_size=0.2,
    epochs=10
)
```

### 时间序列预测 / Time Series Forecasting

```python
from pytorch_wrap import TimeSeriesPredictor

predictor = TimeSeriesPredictor(
    model_type='lstm',
    input_window=60,
    output_window=10
)

predictor.fit(time_series_data)
forecast = predictor.predict(steps=30)
```

---

## 🔧 高级用法 / Advanced Usage

### 自定义模型 / Custom Models

```python
from pytorch_wrap import BaseModel
import torch.nn as nn

class MyCustomModel(BaseModel):
    def __init__(self, input_size, num_classes):
        super().__init__()
        
        self.features = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3)
        )
        
        self.classifier = nn.Linear(256, num_classes)
    
    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)

# 使用自定义模型
model = MyCustomModel(input_size=1000, num_classes=10)
trainer = Trainer(model)
trainer.fit(train_loader, epochs=50)
```

### 分布式训练 / Distributed Training

```python
from pytorch_wrap import DistributedTrainer

# 多 GPU 训练
trainer = DistributedTrainer(
    model=model,
    strategy='ddp',
    gpus=4,
    sync_bn=True
)

trainer.fit(train_loader, epochs=100)
```

### 模型导出 / Model Export

```python
# 导出为 ONNX
trainer.export_onnx('model.onnx', input_shape=(1, 3, 224, 224))

# 导出为 TorchScript
traced_model = trainer.to_torchscript()
traced_model.save('model.pt')

# 导出为 OpenVINO
trainer.export_openvino('model_xml')
```

---

## 📊 性能对比 / Performance Comparison

| 框架 / Framework | 代码行数 / Lines | 训练速度 / Speed | 易用性 / Ease |
|-----------------|----------------|-----------------|--------------|
| 原生 PyTorch | 150+ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |
| PyTorch-Wrap | 30-50 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| FastAI | 40-60 | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| PyTorch Lightning | 80-100 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |

---

## 🛠️ 安装说明 / Installation Guide

### 系统要求 / System Requirements

- Python 3.10 或更高版本
- PyTorch 2.0 或更高版本
- CUDA 11.7+ (可选，用于 GPU 加速)
- 操作系统：Linux / macOS / Windows

### 安装步骤 / Installation Steps

```bash
# 1. 创建虚拟环境
python -m venv pytorch-env
source pytorch-env/bin/activate  # Linux/macOS
# 或
pytorch-env\Scripts\activate     # Windows

# 2. 安装 PyTorch (根据系统选择)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 3. 安装 PyTorch-Wrap
pip install pytorch-wrap

# 4. 验证安装
python -c "import pytorch_wrap; print(pytorch_wrap.__version__)"
```

### 从源码构建 / Build from Source

```bash
git clone https://github.com/q15004040209-creator/pytorch-wrap.git
cd pytorch-wrap
pip install -e .
```

---

## 📖 API 参考 / API Reference

详细 API 文档请查看：[https://pytorch-wrap.readthedocs.io](https://pytorch-wrap.readthedocs.io)

### 核心模块 / Core Modules

- `pytorch_wrap.models` - 预定义模型
- `pytorch_wrap.data` - 数据加载与处理
- `pytorch_wrap.train` - 训练器与回调
- `pytorch_wrap.utils` - 工具函数
- `pytorch_wrap.metrics` - 评估指标
- `pytorch_wrap.callbacks` - 训练回调

---

## 🤝 贡献 / Contributing

我们欢迎所有形式的贡献！

我们欢迎所有形式的贡献！

1. Fork 本仓库
2. 创建特性分支：`git checkout -b feature/amazing-feature`
3. 提交更改：`git commit -m 'Add amazing feature'`
4. 推送到分支：`git push origin feature/amazon-feature`
5. 开启 Pull Request

请阅读 [CONTRIBUTING.md](CONTRIBUTING.md) 了解更多细节。

---

## 📝 许可证 / License

本项目采用 BSD-3-Clause 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情。

This project is licensed under the BSD-3-Clause License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 致谢 / Acknowledgments

- [PyTorch](https://pytorch.org/) - 强大的深度学习框架
- 所有贡献者和用户

---

## 📬 联系方式 / Contact

- 问题反馈：[GitHub Issues](https://github.com/q15004040209-creator/pytorch-wrap/issues)
- 讨论交流：[Discussion Forum](https://github.com/q15004040209-creator/pytorch-wrap/discussions)

---

<div align="center">

**Made with ❤️ by the PyTorch-Wrap Team**

⭐ 如果这个项目对你有帮助，请给一个星标！

</div>