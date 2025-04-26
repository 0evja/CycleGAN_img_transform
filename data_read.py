import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import os
from PIL import Image
import random


# [-1, 1] -> [0, 1], 调整张量形状
def to_img(x):
    out = (x + 1) / 2
    out = out.clamp(0, 1)
    out = out.view(-1, 3, 256, 256)
    return out


data_path = r'data'
img_size = 256
batch_size = 1
transforms = transforms.Compose([transforms.Resize(int(img_size * 1.12), Image.BICUBIC),
                                 transforms.RandomCrop(img_size),
                                 transforms.RandomHorizontalFlip(),
                                 transforms.ToTensor(),
                                 transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

# train_a_filepath = data_path + '/trainA'
# train_b_filepath = data_path + '/trainB'

# train_a_list = os.listdir(train_a_filepath)
# train_b_list = os.listdir(train_b_filepath)
# print(train_a_list)


# 读取训练数据集  a表示艺术风格图像 b表示艺术风格图像
def get_train_data(batch_size):
    train_a_filepath = data_path + '/trainA'
    train_b_filepath = data_path + '/trainB'

    train_a_list = os.listdir(train_a_filepath)
    train_b_list = os.listdir(train_b_filepath)

    train_a_result = []
    train_b_result = []

    # 随机选择 batch_size 个样本
    numlist = random.sample(range(0, len(train_a_list)), batch_size)
    for i in numlist:
        # 加载 trainA 
        a_filename = train_a_list[i]
        a_img = Image.open(os.path.join(train_a_filepath, a_filename)).convert('RGB')
        res_a_img = transforms(a_img)
        train_a_result.append(torch.unsqueeze(res_a_img, dim=0))

        # 加载 trainB
        b_filename = train_b_list[i]
        b_img = Image.open(os.path.join(train_b_filepath, b_filename)).convert('RGB')
        res_b_img = transforms(b_img)
        train_b_result.append(torch.unsqueeze(res_b_img, dim=0))

    # 将 train_a_result 和 train_b_result 拼接成一个 batch
    return torch.cat(train_a_result, dim=0), torch.cat(train_b_result, dim=0)

                                
