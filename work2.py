import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

from model.uav import Train
from Learn1 import train_data_size, learning_rate, optimizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 定义数据集路径
train_dir = './data/dataset1/train'
val_dir = './data/dataset1/val'
test_dir = './data/dataset1/test'


# 自定义数据集
class ImageDataset(Dataset):
    def __init__(self, uav_dir, background_dir, transform=None):
        self.uav_images = [os.path.join(uav_dir, img) for img in os.listdir(uav_dir)]
        self.background_images = [os.path.join(background_dir, img) for img in os.listdir(background_dir)]
        self.images = self.uav_images + self.background_images
        self.labels = [1] * len(self.uav_images) + [0] * len(self.background_images)
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label, img_path

# 数据增强和转换
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
])
# 尝试添加PCA降维


# 创建数据集和数据加载器
train_dataset = ImageDataset(os.path.join(train_dir, 'UAV'), os.path.join(train_dir, 'background'), transform=transform)
val_dataset = ImageDataset(os.path.join(val_dir, 'UAV'), os.path.join(val_dir, 'background'), transform=transform)
test_dataset = ImageDataset(os.path.join(test_dir, 'UAV'), os.path.join(test_dir, 'background'), transform=transform)

print(f"train_data.size is {len(train_dataset)}")
print(f"test_data.size is {len(test_dataset)}")


train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=100, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=100, shuffle=True)

prac = Train()
if torch.cuda.is_available():
    prac = prac.to(device)
# 04 损失函数
loss_fn = nn.CrossEntropyLoss()
if torch.cuda.is_available():
    loss_fn = loss_fn.to(device)

learning_rate = 1e-2
optimizer = torch.optim.SGD(prac.parameters(), lr=learning_rate)

# 06 设置训练网络的一些参数
# 记录训练次数
total_train_step = 0
# 记录测试次数
total_test_step = 0
# 训练的轮数
epoch = 20

# 添加tensorboard
writer = SummaryWriter("./logs_train")

# 07 设置轮数，多次训练
for i in range(epoch):
    print(f"---------第{i}轮训练开始-----------")
    # 08 训练步骤开始
    # 开启训练，某些网络需要
    prac.train()
    for data in train_loader:
        imgs,targets = data
        if torch.cuda.is_available():
            imgs = imgs.to(device)
            targets = targets.to(device)
        output = prac(imgs)
        loss = loss_fn(output,targets)
        # 优化器优化模型
        # 梯度清零
        optimizer.zero_grad()
        # 优化参数
        loss.backward()
        optimizer.step()
        total_train_step += 1
        if total_train_step % 100 == 0:
            print(f"训练次数：{total_train_step},Loss:{loss.item()}")
            writer.add_scalar("train_loss",loss.item(),total_train_step)
    # 09 测试步骤
    # 开启测试，某些网络需要
    prac.eval()
    total_test_loss = 0
    total_accuracy = 0
    # 没有梯度调优
    with torch.no_grad():
        for data in test_loader:
            imgs,targets = data
            if torch.cuda.is_available():
                imgs = imgs.to(device)
                targets = targets.to(device)
            outputs = prac(imgs)
            loss = loss_fn(outputs,targets)
            total_test_loss += loss
            accuracy = (outputs.argmax(1) == targets).sum()
            total_accuracy += accuracy
    # 10 展示
    print(f"整体测试集上的loss:{total_test_loss}")
    print(f"整体测试集上的正确率：{total_accuracy/len(test_dataset)}")
    writer.add_scalar("test_loss",total_test_loss,total_test_step)
    writer.add_scalar("test_accuracy",total_accuracy/len(test_dataset),total_test_step)
    total_test_step += 1

    torch.save(prac,f"./logs_prac/prac_{i}.pth")
    # torch.save(prac.state_dict(),f"prac_{i}.pth")
    print("------模型已保存---------")

writer.close()
