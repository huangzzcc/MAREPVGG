import torch
from torch import nn
from torchvision import datasets, transforms
from repvgg import RepVGG

device = torch.device("cuda:0")

test_path = r"E:\python project\MyRepVGG\data\real_and_fake_face_detection\real_and_fake_face"

test_dataset = datasets.ImageFolder(test_path, transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ]))

print('测试集图像数量', len(test_dataset))
print('类别个数', len(test_dataset.classes))
print('各类别名称', test_dataset.classes)

test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=True)

model = RepVGG(num_blocks=[2, 4, 14, 1],
               num_classes=1000,
               width_multiplier=[0.75, 0.75, 0.75, 2.5],
               override_groups_map=None, deploy=False)

model.linear = nn.Linear(model.linear.in_features, out_features=2)

state_dict = torch.load("weights/best_real_fake.pth", map_location='cpu')
model.load_state_dict(state_dict)

model.to(device)

model.eval()

with torch.no_grad():
    for i, (img, target) in enumerate(test_loader):
        img = img.to(device)
        out = model(img)
        print(out)
        print(out.argmax(1))
        print(target)
        print("-------------------------"+str(i)+"-------------------------")