"""
训练基于 PyTorch 的 MLP（BP 神经网络）用于 MNIST
用法示例：
  python train_bp.py --epochs 10 --batch-size 128 --lr 1e-3
"""
import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 10)
        )

    def forward(self, x):
        return self.net(x)


def train(model, device, train_loader, optimizer, criterion, epoch):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    loop = tqdm(train_loader, desc=f"Train Epoch {epoch}")
    for data, target in loop:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * data.size(0)
        preds = output.argmax(dim=1)
        correct += preds.eq(target).sum().item()
        total += data.size(0)
        loop.set_postfix(loss=total_loss / total, acc=100.0 * correct / total)

    return total_loss / total, 100.0 * correct / total


def test(model, device, test_loader, criterion):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            total_loss += loss.item() * data.size(0)
            preds = output.argmax(dim=1)
            correct += preds.eq(target).sum().item()
            total += data.size(0)
    return total_loss / total, 100.0 * correct / total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--no-cuda', action='store_true')
    parser.add_argument('--save-dir', type=str, default='experiment1/models')
    args = parser.parse_args()

    use_cuda = (not args.no_cuda) and torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = datasets.MNIST(root='experiment1/data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='experiment1/data', train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    model = MLP().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    os.makedirs(args.save_dir, exist_ok=True)

    best_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train(model, device, train_loader, optimizer, criterion, epoch)
        test_loss, test_acc = test(model, device, test_loader, criterion)
        print(f"Epoch {epoch}: train_loss={train_loss:.4f} train_acc={train_acc:.2f}% | test_loss={test_loss:.4f} test_acc={test_acc:.2f}%")
        # 保存最佳模型
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), os.path.join(args.save_dir, 'mlp_mnist.pth'))

    print(f"训练完成，最佳测试准确率: {best_acc:.2f}%")


if __name__ == '__main__':
    main()
