# 图像风格转换项目使用指南

## 项目结构
```plaintext
root/
├── data/                 # 训练数据集（无需关注）
├── runs/                 # 训练结果（无需关注）
├── utils/                # 空目录（无需关注）
├── model/
│   ├── 1000/            # 1000轮训练权重
│   ├── 2000/            # 2000轮训练权重
│   ├── 3000/            # 3000轮训练权重
│   └── 其他权重/         # 4000轮训练权重
├── test_img_A/           # A类风格测试图像
├── test_img_B/           # B类风格测试图像
├── data_read.py          # 数据读取模块
├── Net_build.py          # 模型构建模块
├── train.py              # 训练模块
├── transformer.py        # 图像风格转换模块（输出到output目录）
├── requirements.txt      # 依赖库列表
└── output/
    ├── A2B/              # A→B风格转换结果
    └── B2A/              # B→A风格转换结果

## 快速开始

### 环境安装
```bash
pip install -r requirements.txt

```bash
streamlit run app.py