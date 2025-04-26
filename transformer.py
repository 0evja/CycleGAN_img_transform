import torch
import torchvision.transforms as transforms
from torchvision.utils import save_image
from PIL import Image
import os
from Net_build import Generator
from data_read import to_img

class StyleTransformer:
    def __init__(self, model_path='model'):
        # 初始化生成器
        self.G_A2B = Generator()
        self.G_B2A = Generator()
        
        # 加载模型权重
        self.G_A2B.load_state_dict(torch.load(os.path.join(model_path, 'G_A2B.pth')))
        self.G_B2A.load_state_dict(torch.load(os.path.join(model_path, 'G_B2A.pth')))
        
        # 设置为评估模式
        self.G_A2B.eval()
        self.G_B2A.eval()
        
        # 定义图像预处理
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
    
    def transform_image_A2B(self, image_path, save_path=None):
        """
        将图片从风格A转换为风格B
        Args:
            image_path: 输入图片路径
            save_path: 保存路径，如果为None则不保存
        Returns:
            转换后的图片张量
        """
        # 加载并预处理图片
        img = Image.open(image_path).convert('RGB')
        img_tensor = self.transform(img).unsqueeze(0)

        # 使用生成器A2B进行转换
        with torch.no_grad():
            fake_img = self.G_A2B(img_tensor)

        # 处理生成的图像
        result_img = to_img(fake_img.data)
        
        # 如果指定了保存路径，则保存图像
        if save_path:
            save_image(result_img, save_path)
            print(f"A->B 风格转换后的图片已保存到: {save_path}")
            
        return result_img

    def transform_image_B2A(self, image_path, save_path=None):
        """
        将图片从风格B转换为风格A
        Args:
            image_path: 输入图片路径
            save_path: 保存路径，如果为None则不保存
        Returns:
            转换后的图片张量
        """
        # 加载并预处理图片
        img = Image.open(image_path).convert('RGB')
        img_tensor = self.transform(img).unsqueeze(0)

        # 使用生成器B2A进行转换
        with torch.no_grad():
            fake_img = self.G_B2A(img_tensor)

        # 处理生成的图像
        result_img = to_img(fake_img.data)
        
        # 如果指定了保存路径，则保存图像
        if save_path:
            save_image(result_img, save_path)
            print(f"B->A 风格转换后的图片已保存到: {save_path}")
            
        return result_img
    

def main():
    # 创建输出目录
    if not os.path.exists('output'):
        os.makedirs('output')
    
    # 初始化转换器
    transformer = StyleTransformer()
    
    # 处理 A->B 转换
    test_dir_A = 'test_img_A'
    if os.path.exists(test_dir_A):
        print(f"\n正在处理 A->B 转换...")
        # 创建 A->B 输出子目录
        output_dir_A2B = os.path.join('output', 'A2B')
        if not os.path.exists(output_dir_A2B):
            os.makedirs(output_dir_A2B)
            
        for img_name in os.listdir(test_dir_A):
            if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(test_dir_A, img_name)
                output_path = os.path.join(output_dir_A2B, f"A2B_{img_name}")
                
                print(f"正在处理图片: {img_name}")
                transformer.transform_image_A2B(img_path, output_path)
    else:
        print(f"警告：找不到测试图片目录 '{test_dir_A}'")
    
    # 处理 B->A 转换
    test_dir_B = 'test_img_B'
    if os.path.exists(test_dir_B):
        print(f"\n正在处理 B->A 转换...")
        # 创建 B->A 输出子目录
        output_dir_B2A = os.path.join('output', 'B2A')
        if not os.path.exists(output_dir_B2A):
            os.makedirs(output_dir_B2A)
            
        for img_name in os.listdir(test_dir_B):
            if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(test_dir_B, img_name)
                output_path = os.path.join(output_dir_B2A, f"B2A_{img_name}")
                
                print(f"正在处理图片: {img_name}")
                transformer.transform_image_B2A(img_path, output_path)
    else:
        print(f"警告：找不到测试图片目录 '{test_dir_B}'")

if __name__ == '__main__':
    main()

