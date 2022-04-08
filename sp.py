import math
import matplotlib.pyplot as plt
import torchvision
from torch.autograd import Variable
import pytorch_ssim
import numpy
import torch
import cv2
from DataTrain import MyDataset
from unet import UNet
from DataTest import MyTestDataset
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import numpy as np


def psnr(img1, img2):
    mse = numpy.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 255
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))
    # return 10 * np.log(255 * 2 / (np.mean(np.square(img1 - img2))))


def ssim(img1, img2):
    img1 = torch.from_numpy(np.rollaxis(img1, 2)).float().unsqueeze(0) / 255
    img2 = torch.from_numpy(np.rollaxis(img2, 2)).float().unsqueeze(0) / 255
    img1 = Variable(img1, requires_grad=False)  # torch.Size([256, 256, 3])
    img2 = Variable(img2, requires_grad=False)
    ssim_value = pytorch_ssim.ssim(img1, img2).item()
    return ssim_value
    # img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    # img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    # u_true = np.mean(img1)
    # u_pred = np.mean(img2)
    # var_true = np.var(img1)
    # var_pred = np.var(img2)
    # std_true = np.sqrt(img1)
    # std_pred = np.sqrt(img2)
    # c1 = np.square(0.01 * 7)
    # c2 = np.square(0.03 * 7)
    # ssim = (2 * u_true * u_pred + c1) * (2 * std_pred * std_true + c2)
    # denom = (u_true ** 2 + u_pred ** 2 + c1) * (var_pred + var_true + c2)
    # return ssim / denom


input_path = r'data/train/image/'
label_path = r'data/train/label/'

sum_ssim = 0
sum_psnr = 0

# net = UNet(3, 3).cuda()
net = UNet(3, 3)
net.load_state_dict(torch.load('./model.pth', map_location='cpu'))  # 加载训练好的模型参数
net.eval()


dataset_train = MyDataset(input_path, label_path)
trainloader = DataLoader(dataset=dataset_train, batch_size=1, shuffle=True)

for i in range(1):
    # 这个循环一般在 epoch 循环下表示一次训练，x，y 对应前述返回的 input，label
    for j, (x, y) in enumerate(trainloader):  # 加载训练数据
        input = Variable(x)
        label = Variable(y)

        output = net(input)

        output = torchvision.transforms.ToPILImage()(output[0])
        label1 = torchvision.transforms.ToPILImage()(label[0])

        output = np.array(output, np.float32)
        label1 = np.array(label1, np.float32)

        p = psnr(output, label1)
        s = ssim(output, label1)
        print("psnr: ", p)
        print("ssim: ", s)
        sum_ssim = sum_ssim + s
        sum_psnr = sum_psnr + p

print("avg_ssim:", sum_ssim / 1800)
print("avg_psnr", sum_psnr / 1800)
