import argparse
import copy
import sys
from tqdm import tqdm
import torch
import timm
from torch.utils.data import DataLoader
from datasets import LodaData
from torch import nn

def train_model(dataloader, model, loss_fn, optimizer, opt):
    size_train = len(dataloader.dataset)
    model.train()
    train_loss = 0
    correct = 0
    pbar = tqdm(desc='train', total=size_train//opt.bs, colour='green', file=sys.stdout)
    for batch, (inputs, labels) in enumerate(dataloader):
        inputs, labels = inputs.to(device), labels.to(device)  # 放入显卡
        pred = model(inputs)
        loss = loss_fn(pred, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        _, preds = torch.max(pred, 1)
        train_loss += loss.item()
        correct += torch.sum(preds == labels.data)
        pbar.update(1)
    pbar.close()
    train_loss = train_loss / size_train
    correct = correct / size_train
    print(f"Accyracy:{(100 * correct):0.1f}%, val_loss:{train_loss:8f}")

def test_model(dataloader, model, best_acc, opt):
    size_val = len(dataloader.dataset)
    model.eval()
    test_loss = 0
    correct = 0
    pbar = tqdm(desc='val', total=size_val // opt.bs, colour='green', file=sys.stdout)
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)  # 放入显卡
            pred = model(inputs)
            _, preds = torch.max(pred, 1)
            test_loss += loss_fn(pred, labels).item()
            correct += torch.sum(preds == labels.data)
            pbar.update(1)
    pbar.close()
    test_loss = test_loss / size_val
    correct = correct / size_val
    print(f"Accyracy:{(100*correct):0.1f}%, val_loss:{test_loss:8f}")

    if (correct > best_acc):
        print(f"the last best_acc is {(100*best_acc):0.1f}%,the correct is {(100*correct):0.1f}%")
        best_acc = correct
        print(f"the correct is {(100*best_acc):0.1f}%, save the best model")
        torch.save(model, 'model.pt')
        return best_acc
    return best_acc

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='empty', help='initial weights path')
    parser.add_argument('--model', type=str, default='resnet18', help='model name')
    parser.add_argument('--numclasses', type=int, default=102)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--bs', type=int, default=16, help='total batch size for all GPUs')
    parser.add_argument('--img-size', nargs='+', type=int, default=[224, 224], help='[train, test] image sizes')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--workers', type=int, default=0, help='maximum number of dataloader workers')
    opt = parser.parse_args()

    # 加载数据
    train_data = LodaData("train.txt")
    val_data = LodaData("valid.txt", False)
    train_dataloader = DataLoader(dataset=train_data, num_workers=opt.workers, pin_memory=True, batch_size=opt.bs,
                                  shuffle=True)
    val_dataloader = DataLoader(dataset=val_data, num_workers=opt.workers, pin_memory=True, batch_size=opt.bs)
    # 选择设备
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("using {} device".format(device))
    # 定义模型
    if opt.weights is 'empty':
        print("create model")
        print(f"{opt.numclasses} classes")
        model = timm.create_model(opt.model, pretrained=True, pretrained_cfg_overlay=dict(file='pytorch_model.bin'), num_classes=opt.numclasses)
    else:
        model = torch.load(opt.weights)
    # 定义损失函数
    loss_fn = nn.CrossEntropyLoss()
    # 定义优化器
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

    best_acc = 0
    # 训练模型
    model.to(device)  # 放入显卡
    for epoch in range(opt.epochs):
        print("Epoch {}/{}".format(epoch + 1, opt.epochs))
        train_model(train_dataloader, model, loss_fn, optimizer, opt)
        best_acc = test_model(val_dataloader, model, best_acc, opt)
