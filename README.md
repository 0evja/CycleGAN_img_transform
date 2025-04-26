"""
       A: 艺术风格，          B: 现实风格
       
文件目录结构

root--|--- data              # 项目训练数据集  ，无需关注
      |
      |--- runs              # 训练结果 ，无需关注
      |
      |--- utils             # 空， 无需关注
      |
      |--- model -|--- 1000  # 训练一千轮的权重
      |           |--- 2000  # 训练两千轮的权重
      |           |--- 3000  # 训练三千轮的权重
      |           |--- 其他是训练4000轮的权重
      |
      |         # 模型权重保存点 
      |
      |--- test_img_A        # A类风格测试集图像
      |
      |--- test_img_B        # B类风格测试集图像
      |
      |--- data_read.py      # 数据读取文件, 无需关注
      |
      |--- Net_build.py      # 模型搭建文件, 无需关注
      |
      |--- train.py          # 训练文件 ，无需关注
      |
      |--- transformer.py    # 加载保存好的模型权重，图像风格转换文件, 保存路径为output
      |
      |--- requirements.txt  # 项目依赖库
      |
      |--- output-|--- A2B   # A -> B
      |           |--- B2A   # B -> A

"""


Run
pip install -r requirements.txt

streamlit run app.py 