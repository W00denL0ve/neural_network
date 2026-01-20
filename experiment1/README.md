实验1: 基于 PyTorch 和 MNIST 的神经网络训练示例

包含文件:
- train_bp.py: MLP (BP 神经网络) 训练脚本
- train_cnn.py: 简单 CNN 训练脚本

使用方法示例:
  python train_bp.py --epochs 10 --batch-size 128 --lr 1e-3
  python train_cnn.py --epochs 10 --batch-size 128 --lr 1e-3

数据会自动下载到 experiment1/data，模型保存在 experiment1/models
