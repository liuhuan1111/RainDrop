import torch
from NetModel import Net
from unet import UNet
from DataTest import MyTestDataset
from torch.utils.data import DataLoader
from torchvision.utils import save_image

# 测试图像的路径
input_path = 'data/test/input'

net = UNet(3, 3)
net.load_state_dict(torch.load('./model.pth'))  # 加载训练好的模型参数
net.eval()

cnt = 0

dataloader = DataLoader(MyTestDataset(input_path))
for input in dataloader:
    cnt += 1
    input = input
    print(cnt)
    with torch.no_grad():
        output_image = net(input)  # 输出的是张量
        print(output_image)
        save_image(output_image, 'data/test/output/'+str(cnt).zfill(4)+'.jpg')  # 直接保存张量图片，自动转换
