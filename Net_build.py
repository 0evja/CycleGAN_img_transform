import torch 
import torch.nn as nn
import torch.nn.functional as F
import random
from torch.autograd import Variable


'''
CycleGAN网络结构主体包括：
1. 两个生成器G和F, G:(A->B) F:(B->A)
2. 两个判别器D_A和D_B
3. 残差块ResidualBlock
4. 卷积块ConvBlock
'''

# ----------------------基础模块----------------------
# 残差网络
class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()
        # 残差模块中的网络层
        self.block_layer = nn.Sequential(
            nn.ReflectionPad2d(1),  # 边界填充，降低边界效应
            nn.Conv2d(in_features, in_features, kernel_size=3),
            nn.InstanceNorm2d(in_features),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, kernel_size=3),
            nn.InstanceNorm2d(in_features),
        )
    
    def forward(self, x):
        # 残差连接
        return x + self.block_layer(x)
    
# 生成器
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(3, 64, 7),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True)]

        in_features = 64
        out_features = in_features * 2

        for _ in range(2):
            # 添加卷积块  逐步减少图像分辨率， 提取高层次特征
            model += [nn.Conv2d(in_features, out_features, kernel_size=3, stride=2, padding=1),
                      nn.InstanceNorm2d(out_features),
                      nn.ReLU(inplace=True)]
            in_features = out_features
            out_features = in_features * 2

        for _ in range(9):
            # 添加9个残差块   捕捉图像细节特征
            model += [ResidualBlock(in_features)]

        out_features = in_features // 2

        for _ in range(2):
            # 添加2个反卷积块   逐步恢复图像分辨率， 还原图像细节  
            model += [nn.ConvTranspose2d(in_features, out_features, kernel_size=3, stride=2, padding=1, output_padding=1),
                      nn.InstanceNorm2d(out_features),
                      nn.ReLU(inplace=True)
               ]
            in_features = out_features
            out_features = in_features // 2

        model += [nn.ReflectionPad2d(3),
                    nn.Conv2d(64, 3, 7),
                    nn.Tanh()]
        
        self.gen = nn.Sequential( * model)

    def forward(self, x):
        return self.gen(x)
    
# 判别器
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator,self).__init__()
        self.dis = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(256, 512, 4, padding=1),
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(512, 1, 4, padding=1)
        )
    
    def forward(self, x):
        x = self.dis(x)
        return F.avg_pool2d(x, x.size()[2:]).view(x.size()[0], -1) # 平均池化，展平


# 定义缓存队列
'''
训练网络时，每个周期都会读取一定批量大小的数据用于训练，
需要注意，艺术风格和现实风格是成对读取的，固定搭配可能会被网络学习到
因此 需要将输入的数据进行打乱
'''

class ReplayBuffer():
    # 缓存队列，不足则新增，否则随机替换
    def __init__(self, max_size=50):
        self.max_size = max_size
        self.buffer = []

    def push_and_pop(self, data):
        to_return = []
        for element in data.data:
            element = torch.unsqueeze(element, 0)
            if len(self.buffer) < self.max_size:
                self.buffer.append(element)
                to_return.append(element)
            else:
                if random.uniform(0, 1) > 0.5:
                    # 随机替换
                    i = random.randint(0, self.max_size - 1)
                    to_return.append(self.buffer[i].clone())
                    self.buffer[i] = element
                else:
                    to_return.append(element)

        return Variable(torch.cat(to_return))