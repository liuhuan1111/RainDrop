import json
import numpy as np
import torch
import torch.optim as optim
from unet import UNet
import torch.nn as nn
import os
from DataTrain import MyDataset
from torch.utils.data import DataLoader
from torch.autograd import Variable

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import matplotlib.pyplot as plt

## matplotlib显示图片中显示汉字
plt.rcParams['font.sans-serif'] = ['SimSun']
plt.rcParams['axes.unicode_minus'] = False

# 训练图像的路径
input_path = 'data/train/image/'
label_path = 'data/train/label/'
# net = UNet(3, 3).cuda()
net = UNet(3, 3)

learning_rate = 1e-3
batch_size = 8  # 分批训练数据，每批数据量
epoch = 10  # 训练次数
Loss_list = []  # 简单的显示损失曲线列表，反注释后训练完显示曲线

optimizer = optim.Adam(net.parameters(), lr=learning_rate)
loss_f = nn.SmoothL1Loss()

net.train()

if os.path.exists('./model.pth'):  # 判断模型有没有提前训练过
    print("continue")
    net.load_state_dict(torch.load('./model.pth'))  # 加载训练过的模型
else:
    print("start")

dataset_train = MyDataset(input_path, label_path)
trainloader = DataLoader(dataset=dataset_train, batch_size=batch_size, shuffle=True)

for i in range(epoch):
    # 这个循环一般在 epoch 循环下表示一次训练，x，y 对应前述返回的 input，label
    for j, (x, y) in enumerate(trainloader):  # 加载训练数据
        # input = Variable(x).cuda()
        # label = Variable(y).cuda()

        input = Variable(x)
        label = Variable(y)

        net.zero_grad()
        optimizer.zero_grad()

        output = net(input)
        loss = loss_f(output, label)

        optimizer.zero_grad()
        loss.backward()  # 反向传播
        optimizer.step()

        # Loss_list.append(loss)

        # print(i+1, '     ', loss)
        Loss_list.append(loss.detach().numpy())
    if i % 9 == 0:
        torch.save(net.state_dict(), 'model.pth')  # 保存训练模型
    print('epoch: %d | loss: %.4f' % (i, loss.data.cpu()))
    # Loss_list.append(loss.detach().numpy())


# np.savetxt("./MSELoss_list.txt", Loss_list)
out = np.loadtxt("./MSELoss_list.txt")

plt.figure(dpi=500)
x = range(0, 2250)
y = out
plt.plot(x, y, linewidth=0.1)
plt.ylabel('当前损失/1')
plt.xlabel('批训练次数/次数')
plt.savefig('./loss.jpg')
plt.show()
