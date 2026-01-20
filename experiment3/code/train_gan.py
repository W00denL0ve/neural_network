"""
训练脚本骨架：支持 --model {gan,dcgan,wgangp}
保存样本、模型与日志
"""
import os
import argparse
import random
import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, utils
from torch.utils.data import DataLoader

from models import MLPGenerator, MLPDiscriminator, DCGenerator, DCDiscriminator, Critic, weights_init_normal


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_samples(generator, z_fixed, out_path, device):
    generator.eval()
    with torch.no_grad():
        imgs = generator(z_fixed.to(device))
        imgs = (imgs + 1) / 2.0
        grid = utils.make_grid(imgs, nrow=8, normalize=False)
        utils.save_image(grid, out_path)
    generator.train()


def gradient_penalty(critic, real, fake, device):
    batch_size = real.size(0)
    alpha = torch.rand(batch_size, 1, 1, 1, device=device)
    interpolates = (alpha * real + (1 - alpha) * fake).requires_grad_(True)
    d_interpolates = critic(interpolates)
    grads = torch.autograd.grad(outputs=d_interpolates, inputs=interpolates,
                                grad_outputs=torch.ones_like(d_interpolates), create_graph=True,
                                retain_graph=True, only_inputs=True)[0]
    grads = grads.view(batch_size, -1)
    gp = ((grads.norm(2, dim=1) - 1) ** 2).mean()
    return gp


def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
    set_seed(args.seed)

    # data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    dataset = datasets.FashionMNIST(root=str(Path(__file__).resolve().parents[2] / 'experiment1' / 'data'), train=True, download=True, transform=transform)
    # 预检查：在创建 DataLoader 前尝试访问第一个样本，若 transform/读取有问题会尽早抛出异常
    try:
        _img, _label = dataset[0]
    except Exception as e:
        print('数据集读取/预处理出错：', e)
        raise
    # 在 Windows 上建议使用 num_workers=0 避免子进程相关的挂起/卡住问题；
    # 若使用 GPU，可启用 pin_memory 提升数据拷贝效率
    pin_memory = True if (device.type == 'cuda') else False
    torch.backends.cudnn.benchmark = True if (device.type == 'cuda') else False
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=pin_memory)
    print(f'已创建 DataLoader，batch_size={args.batch_size}, num_workers=0, device={device}')

    # models
    if args.model == 'gan':
        netG = MLPGenerator(z_dim=args.z_dim).to(device)
        netD = MLPDiscriminator().to(device)
    elif args.model == 'dcgan':
        netG = DCGenerator(z_dim=args.z_dim).to(device)
        netD = DCDiscriminator().to(device)
    elif args.model == 'wgangp':
        netG = DCGenerator(z_dim=args.z_dim).to(device)
        netD = Critic().to(device)
    else:
        raise ValueError('unknown model')

    netG.apply(weights_init_normal)
    netD.apply(weights_init_normal)

    # optimizers and losses
    if args.model == 'wgangp':
        optimizerD = optim.Adam(netD.parameters(), lr=args.lr_d, betas=(args.beta1_d, args.beta2_d))
        optimizerG = optim.Adam(netG.parameters(), lr=args.lr_g, betas=(args.beta1_g, args.beta2_g))
    else:
        optimizerD = optim.Adam(netD.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
        optimizerG = optim.Adam(netG.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
        criterion = nn.BCELoss()

    out_dir = Path(args.out_dir)
    (out_dir / 'samples').mkdir(parents=True, exist_ok=True)
    (out_dir / 'models').mkdir(parents=True, exist_ok=True)

    z_fixed = torch.randn(64, args.z_dim)

    # 日志和画图准备
    (out_dir / 'logs').mkdir(parents=True, exist_ok=True)
    (out_dir / 'figures').mkdir(parents=True, exist_ok=True)
    # 将每个 epoch 的平均损失记录到列表，用于绘图
    history = {
        'epoch': [],
        'lossD': [],
        'lossG': [],
        'd_real': [],
        'd_fake': [],
        'gp': []
    }

    # training loop skeleton，增加 KeyboardInterrupt 捕获以便安全停止并保存当前模型
    try:
        for epoch in range(1, args.epochs + 1):
            epoch_lossD = 0.0
            epoch_lossG = 0.0
            epoch_d_real = 0.0
            epoch_d_fake = 0.0
            epoch_gp = 0.0
            iters = 0

            for i, (imgs, _) in enumerate(loader):
                imgs = imgs.to(device)
                bs = imgs.size(0)
                iters += 1

                # train discriminator / critic
                if args.model == 'wgangp':
                    # update critic n_critic times per generator update
                    for _ in range(args.n_critic):
                        z = torch.randn(bs, args.z_dim, device=device)
                        fake = netG(z)
                        real = imgs
                        d_real = netD(real).mean()
                        d_fake = netD(fake.detach()).mean()
                        gp = gradient_penalty(netD, real, fake.detach(), device)
                        lossD = d_fake - d_real + args.lambda_gp * gp

                        optimizerD.zero_grad()
                        lossD.backward()
                        optimizerD.step()

                        epoch_lossD += lossD.item()
                        epoch_d_real += d_real.item() if isinstance(d_real, torch.Tensor) else float(d_real)
                        epoch_d_fake += d_fake.item() if isinstance(d_fake, torch.Tensor) else float(d_fake)
                        epoch_gp += gp.item()
                else:
                    # standard GAN / DCGAN discriminator update
                    real_labels = torch.ones(bs, device=device)
                    fake_labels = torch.zeros(bs, device=device)
                    optimizerD.zero_grad()
                    outputs_real = netD(imgs)
                    loss_real = criterion(outputs_real, real_labels)
                    z = torch.randn(bs, args.z_dim, device=device)
                    fake = netG(z)
                    outputs_fake = netD(fake.detach())
                    loss_fake = criterion(outputs_fake, fake_labels)
                    lossD = (loss_real + loss_fake) * 0.5
                    lossD.backward()
                    optimizerD.step()

                    epoch_lossD += lossD.item()
                    # for reporting, approximate d_real/d_fake as means of outputs
                    try:
                        epoch_d_real += outputs_real.mean().item()
                        epoch_d_fake += outputs_fake.mean().item()
                    except Exception:
                        pass

                # train generator
                optimizerG.zero_grad()
                z = torch.randn(bs, args.z_dim, device=device)
                fake = netG(z)
                if args.model == 'wgangp':
                    lossG = -netD(fake).mean()
                else:
                    outputs = netD(fake)
                    lossG = criterion(outputs, real_labels)
                lossG.backward()
                optimizerG.step()

                epoch_lossG += lossG.item()

            # 记录本代平均值并保存到 history
            avg_lossD = epoch_lossD / max(1, iters)
            avg_lossG = epoch_lossG / max(1, iters)
            avg_d_real = epoch_d_real / max(1, iters)
            avg_d_fake = epoch_d_fake / max(1, iters)
            avg_gp = epoch_gp / max(1, iters)

            history['epoch'].append(epoch)
            history['lossD'].append(avg_lossD)
            history['lossG'].append(avg_lossG)
            history['d_real'].append(avg_d_real)
            history['d_fake'].append(avg_d_fake)
            history['gp'].append(avg_gp)

            # 保存每代样本与模型
            save_samples(netG, z_fixed, str(out_dir / 'samples' / f"{args.model}_epoch_{epoch:03d}.png"), device)
            torch.save(netG.state_dict(), out_dir / 'models' / f"generator_{args.model}_epoch{epoch:03d}.pth")
            torch.save(netD.state_dict(), out_dir / 'models' / f"discriminator_{args.model}_epoch{epoch:03d}.pth")

            # 保存历史为 CSV
            import csv
            csv_path = out_dir / 'logs' / f"{args.model}_history.csv"
            write_header = not csv_path.exists()
            with open(csv_path, 'a', newline='') as f:
                writer = csv.writer(f)
                if write_header:
                    writer.writerow(['epoch', 'lossD', 'lossG', 'd_real', 'd_fake', 'gp'])
                writer.writerow([epoch, avg_lossD, avg_lossG, avg_d_real, avg_d_fake, avg_gp])

            # 绘制并保存损失曲线
            try:
                import matplotlib.pyplot as plt
                plt.figure()
                plt.plot(history['epoch'], history['lossD'], label='lossD')
                plt.plot(history['epoch'], history['lossG'], label='lossG')
                plt.xlabel('epoch')
                plt.ylabel('loss')
                plt.legend()
                plt.grid(True)
                plt.tight_layout()
                plt.savefig(out_dir / 'figures' / f"loss_{args.model}.png")
                plt.close()
            except Exception as e:
                print('绘图失败：', e)
    except KeyboardInterrupt:
        print('\nKeyboardInterrupt detected — saving current models and exiting gracefully...')
        try:
            torch.save(netG.state_dict(), out_dir / 'models' / f"generator_{args.model}_interrupted.pth")
            torch.save(netD.state_dict(), out_dir / 'models' / f"discriminator_{args.model}_interrupted.pth")
            save_samples(netG, z_fixed, str(out_dir / 'samples' / f"{args.model}_interrupted.png"), device)
            print('已保存中断时的模型和样本。')
        except Exception as e:
            print('保存失败：', e)
        raise

    print('训练完成')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='dcgan', choices=['gan', 'dcgan', 'wgangp'])
    parser.add_argument('--out-dir', type=str, default='experiment3/results')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--z-dim', type=int, default=100)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--beta1', type=float, default=0.5)
    parser.add_argument('--beta2', type=float, default=0.999)
    # wgangp specific
    parser.add_argument('--lr-g', type=float, default=1e-4)
    parser.add_argument('--lr-d', type=float, default=1e-4)
    parser.add_argument('--beta1-g', type=float, default=0.0)
    parser.add_argument('--beta2-g', type=float, default=0.9)
    parser.add_argument('--beta1-d', type=float, default=0.0)
    parser.add_argument('--beta2-d', type=float, default=0.9)
    parser.add_argument('--n-critic', type=int, default=5)
    parser.add_argument('--lambda-gp', type=float, default=10.0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--no-cuda', action='store_true')
    args = parser.parse_args()
    train(args)
