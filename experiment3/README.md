# 实验三：生成对抗网络（GAN）实验代码与说明

目录结构：
- code/models.py    # 模型实现（MLP GAN, DCGAN, WGAN-GP 的 critic）
- code/train_gan.py # 训练脚本，支持 --model {gan,dcgan,wgangp}
- results/          # 训练输出：samples, models, logs

快速运行示例（PowerShell）：
  python .\experiment3\code\train_gan.py --model dcgan --epochs 20 --batch-size 128 --out-dir experiment3/results

说明：
- 默认使用 Fashion-MNIST 数据，数据会从 experiment1/data 下载并使用。
- 训练产生的样本网格保存在 experiment3/results/samples，模型权重保存在 experiment3/results/models。
