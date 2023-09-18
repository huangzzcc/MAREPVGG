import torch
from torchvision import datasets, transforms
from repvgg import RepVGG
import torch.nn as nn
import torch.optim as optim
import time
import numpy as np
from ResNet18 import RestNet18
from efficientnet_pytorch import EfficientNet
import torchvision.models as models
from MARepVGG import MARepVGG
from torch.utils.tensorboard import SummaryWriter

# 将torch.tensor分配到的设备的对象 GPU
device = torch.device("cuda:0")

# 训练集路径，路径前面加一个'r',是为了保持路径在读取时不被漏读，错读
# train_path = r"E:\python project\MyRepVGG\data\real_and_fake_face"  # 本地路径
# test_path = r"E:\python project\MyRepVGG\data\real_and_fake_face_detection\real_and_fake_face"
train_path = r"/tmp/pycharm_project_167/data/real_and_fake_face"
test_path = r"/tmp/pycharm_project_167/data/real_and_fake_face_detection/real_and_fake_face"

# 训练批次中数据个数
batch_size = 32

# datasets.ImageFolder这个函数读取训练集的数据
# train_dataset = datasets.ImageFolder(train_path, transforms.Compose([
#             transforms.RandomResizedCrop(224),
#             transforms.RandomHorizontalFlip(),
#             transforms.ToTensor(),
#             transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
#         ]))
train_dataset = datasets.ImageFolder(train_path, transform = transforms.Compose([
        transforms.Resize(299),
        transforms.CenterCrop(299),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ]))
# test_dataset = datasets.ImageFolder(test_path, transforms.Compose([
#             transforms.RandomResizedCrop(224),
#             transforms.RandomHorizontalFlip(),
#             transforms.ToTensor(),
#             transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
#         ]))
test_dataset = datasets.ImageFolder(test_path, transform = transforms.Compose([
        transforms.Resize(299),
        transforms.CenterCrop(299),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ]))
# 对训练数据集进行 batch 的划分
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

# 加载模型
# model = RepVGG(num_blocks=[2, 4, 14, 1],
#                num_classes=1000,
#                width_multiplier=[0.75, 0.75, 0.75, 2.5],
#                override_groups_map=None, deploy=False)
# model = RestNet18()
# model = EfficientNet.from_name('efficientnet-b4')
model = models.inception_v3(pretrained=False)
# model = MARepVGG(num_classes=2, M=8)


# 修改repvgg神经网络的线性层,in_features是输入的神经元个数,out_features是输出神经元个数
model.fc = nn.Linear(model.fc.in_features, out_features=2)
print(model)
# 将模型加载到指定设备上
model.to(device)

# 优化器
# optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
# optimizer = optim.Adam(model.parameters(), lr=1e-3)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, betas=(0.9, 0.999), weight_decay=1e-6)

# 每间隔几个epoch，调整学习率
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 5, 0.1)

# 交叉熵损失函数，用于解决多分类问题，也可用于解决二分类问题。
criteration = nn.CrossEntropyLoss()

# eval的预测
best_acc = 0
total_epochs = 50
for epoch in range(total_epochs):
    model.train()
    for i, (x, y) in enumerate(train_loader):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        output = model(x)
        loss = criteration(output.logits, y)
        # loss = output['ensemble_loss'] + output['aux_loss']  # 1
        loss.backward()
        optimizer.step()
        if i % 63 == 0 and i != 0:
            print("Epoch %d/%d, iter %d/%d, loss=%.4f" % (
                epoch, total_epochs, i, len(train_loader), loss.item()))
    # model.eval()就是帮我们一键搞定将Dropout层和batch normalization层设置到预测模式
    model.eval()
    total = 0
    correct = 0
    # torch.no_grad()用于神经网络的推理阶段, 表示张量的计算过程中无需计算梯度
    with torch.no_grad():
        for i, (img, target) in enumerate(test_loader):
            img = img.to(device)
            out = model(img)
            pred = out.max(1)[1].detach().cpu().numpy()
            target = target.cpu().numpy()
            correct += (pred == target).sum()
            total += len(target)
        acc = correct / total
        print("\tValidation : Acc=%.4f" % acc, "({}/{})".format(correct, len(test_loader.dataset)))
    if acc > best_acc:
        best_acc = acc
        # 训练后权重路径
        torch.save(model.state_dict(), "weights/inceptionV3.pth")
    scheduler.step()
# print("第%d个epoch的学习率：%f" % (epoch, optimizer.param_groups[0]['lr']))
print("Best Acc=%.4f" % best_acc)
