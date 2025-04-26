
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import itertools
import os

from Net_build import Generator, Discriminator
from Net_build import ReplayBuffer
from data_read import get_train_data, to_img
from torchvision.utils import save_image

fake_A_buffer = ReplayBuffer()
fake_B_buffer = ReplayBuffer()

# 定义生成器和判别器
G_A2B = Generator()
G_B2A = Generator()
D_A = Discriminator()
D_B = Discriminator()

# 定义损失函数
criterion_GAN = nn.MSELoss()
criterion_cycle = nn.L1Loss()
criterion_identity = nn.L1Loss()

# 优化器参数
d_lrt = 3e-4
g_lrt = 3e-4
optim_betas = (0.5, 0.999)

# 定义优化器
g_optim = optim.Adam(itertools.chain(G_A2B.parameters(), G_B2A.parameters()), lr=d_lrt)

da_optim = optim.Adam(D_A.parameters(), lr=d_lrt)
db_optim = optim.Adam(D_B.parameters(), lr=d_lrt)

num_epochs = 4000
batch_size = 1

# 训练网络
for epoch in range(num_epochs):
    real_a, real_b = get_train_data(batch_size)
    target_real = torch.full((batch_size, 1), 1).float()  # (batch_size, )=1
    target_fake = torch.full((batch_size, 1), 0).float()  # (batch_size, )=0

    g_optim.zero_grad()

    # 训练生成器 
    '''
    这里是CycleGAN 中身份损失的一部分
    确保生成器在输入图像已经属于目标域时不会改变其内容
    '''
    same_B = G_A2B(real_b).float()
    loss_identity_B = criterion_identity(same_B, real_b) * 5.0
    same_A = G_B2A(real_a).float()
    loss_identity_A = criterion_identity(same_A, real_a) * 5.0

    fake_B = G_A2B(real_a).float()
    pred_fake = D_B(fake_B).float()
    loss_GAN_A2B = criterion_GAN(pred_fake, target_real)
    fake_A = G_B2A(real_b).float()
    pred_fake = D_A(fake_A)
    loss_GAN_B2A = criterion_GAN(pred_fake, target_real)
    recovered_A = G_B2A(fake_B).float()
    loss_cycle_ABA = criterion_cycle(recovered_A, real_a) * 10.0
    recovered_B = G_A2B(fake_A).float()
    loss_cycle_BAB = criterion_cycle(recovered_B, real_b) * 10.0
    loss_GAN = (loss_identity_A + loss_identity_B + loss_GAN_A2B + loss_GAN_B2A 
               +loss_cycle_ABA + loss_cycle_BAB)
    
    loss_GAN.backward()
    g_optim.step() #g_optim 包含两个生成器的参数

    # 训练判别器A
    da_optim.zero_grad()
    pred_real = D_A(real_a).float()
    loss_D_real = criterion_GAN(pred_real, target_real)
    fake_A = fake_A_buffer.push_and_pop(fake_A)
    pred_fake = D_A(fake_A.detach()).float()
    loss_D_fake= criterion_GAN(pred_fake, target_fake)
    loss_DA = (loss_D_fake + loss_D_real) * 0.5
    loss_DA.backward()
    da_optim.step()

    # 训练判别器B
    db_optim.zero_grad()
    pred_real = D_B(real_b)
    loss_D_real = criterion_GAN(pred_real, target_real)
    fake_B = fake_B_buffer.push_and_pop(fake_B)
    pred_fake = D_B(fake_B.detach()).float()
    loss_D_fake = criterion_GAN(pred_fake, target_real)
    loss_DB = (loss_D_real + loss_D_fake) * 0.5
    loss_DB.backward()
    db_optim.step()

    # 输出损失，存储伪造图像
    print('Epoch[{}],loss_G:{:.6f} ,loss_D_A:{:.6f},loss_D_B:{:.6f}'
      .format(epoch, loss_GAN.data.item(), loss_DA.data.item(), loss_DB.data.item()))
    if (epoch + 1) % 20 == 0 or epoch == 0:  
        b_fake = to_img(fake_B.data)
        a_fake = to_img(fake_A.data)
        a_real = to_img(real_a.data)
        b_real = to_img(real_b.data)
        save_image(a_fake, 'runs/a_fake.png') 
        save_image(b_fake, 'runs/b_fake.png') 
        save_image(a_real, 'runs/a_real.png')
        save_image(b_real, 'runs/b_real.png') 


# 创建保存模型的文件夹
model_dir = 'model'
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

# 保存模型
torch.save(G_A2B.state_dict(), os.path.join(model_dir, 'G_A2B.pth'))
torch.save(G_B2A.state_dict(), os.path.join(model_dir, 'G_B2A.pth'))
torch.save(D_A.state_dict(), os.path.join(model_dir, 'D_A.pth'))
torch.save(D_B.state_dict(), os.path.join(model_dir, 'D_B.pth'))

print("模型已保存到 'model' 文件夹")
        







    




