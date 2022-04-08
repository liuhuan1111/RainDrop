import torchvision
from torch.utils.data import Dataset
import os
from PIL import Image


class MyDataset(Dataset):  # 继承 Dataset 类
    def __init__(self, input_path, label_path):
        self.input_path = input_path  # 受污染图片所在文件夹
        self.input_path_image = os.listdir(input_path)  # 文件夹下的所有图片对象

        self.label_path = label_path  # 干净图片所在文件夹
        self.label_path_image = os.listdir(label_path)

        # 定义要对图片进行的变换
        self.transforms = torchvision.transforms.Compose([
            # 中心裁剪64*64大小作为pacth
            torchvision.transforms.CenterCrop([128, 128]),
            # 将读入的数据归一化[0, 1]之间并变为张量类型
            torchvision.transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.input_path_image)  # 返回长度

    def __getitem__(self, index):
        # index 索引对应的受污染图片完整路径
        input_image_path = os.path.join(self.input_path, self.input_path_image[index])
        # 利用PIL.Image 读入图片数据并转换通道结构
        input_image = Image.open(input_image_path).convert('RGB')

        label_image_path = os.path.join(self.label_path, self.label_path_image[index])
        label_image = Image.open(label_image_path).convert('RGB')

        # 对读入的图片进行固定的变换
        input = self.transforms(input_image)
        label = self.transforms(label_image)

        return input, label  # 返回适合在网络中训练的图片数据
