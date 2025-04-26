import streamlit as st
from transformer import StyleTransformer
import torch
from PIL import Image
import io
import os
import numpy as np

def load_transformer():
    """加载模型转换器"""
    return StyleTransformer()

def process_image(transformer, image, direction):
    """处理图片转换"""
    # 保存上传的图片到临时文件
    temp_path = "temp_upload.jpg"
    with open(temp_path, "wb") as f:
        f.write(image.getvalue())
    
    try:
        # 根据选择的方向进行转换
        if direction == "A_to_B":
            result_tensor = transformer.transform_image_A2B(temp_path)
        else:
            result_tensor = transformer.transform_image_B2A(temp_path)
        
        # 删除临时文件
        os.remove(temp_path)

        # 调整张量维度并转换为PIL图像
        if result_tensor.dim() == 4:  # 如果是批处理格式 (B,C,H,W)
            result_tensor = result_tensor.squeeze(0)  # 移除批处理维度
        
        # 确保张量格式为 (C,H,W)
        result_array = result_tensor.permute(1, 2, 0).cpu().numpy()
        
        # 将像素值从[-1,1]范围转换到[0,255]范围
        result_array = ((result_array + 1) * 127.5).clip(0, 255).astype('uint8')
        
        # 转换为PIL图像
        result_image = Image.fromarray(result_array)
        
        return result_image
    except Exception as e:
        st.error(f"转换过程中出现错误：{str(e)}")
        if os.path.exists(temp_path):
            os.remove(temp_path)
        return None

def show_A_to_B_page():
    """A->B 转换页面"""
    st.title("艺术风格转换到现实风格")
    
    # 上传图片
    uploaded_file = st.file_uploader("上传风格A的图片", type=["png", "jpg", "jpeg"], key="upload_A")
    
    if uploaded_file is not None:
        col1, col2 = st.columns(2)
        
        # 显示原图
        with col1:
            st.subheader("原始图片")
            original_image = Image.open(uploaded_file)
            st.image(original_image, use_column_width=True)
        
        # 转换按钮
        if st.button("开始转换成风格B", key="convert_to_B"):
            with st.spinner("正在进行风格转换..."):
                result = process_image(transformer, uploaded_file, "A_to_B")
                if result is not None:
                    with col2:
                        st.subheader("转换结果")
                        st.image(result, use_column_width=True)

def show_B_to_A_page():
    """B->A 转换页面"""
    st.title("现实风格转换到艺术风格")
    
    # 上传图片
    uploaded_file = st.file_uploader("上传风格B的图片", type=["png", "jpg", "jpeg"], key="upload_B")
    
    if uploaded_file is not None:
        col1, col2 = st.columns(2)
        
        # 显示原图
        with col1:
            st.subheader("原始图片")
            original_image = Image.open(uploaded_file)
            st.image(original_image, use_column_width=True)
        
        # 转换按钮
        if st.button("开始转换成风格A", key="convert_to_A"):
            with st.spinner("正在进行风格转换..."):
                result = process_image(transformer, uploaded_file, "B_to_A")
                if result is not None:
                    with col2:
                        st.subheader("转换结果")
                        st.image(result, use_column_width=True)

def main():
    st.set_page_config(
        page_title="CycleGAN 风格转换",
        page_icon="🎨",
        layout="wide"
    )

    # 添加侧边栏
    st.sidebar.title("导航")
    selection = st.sidebar.radio(
        "选择转换方向",
        ["艺术风格 -> 现实风格", "现实风格 → 艺术风格"]
    )

    # 加载模型（使用缓存避免重复加载）
    global transformer
    transformer = st.cache_resource(load_transformer)()

    # 根据选择显示对应页面
    if selection == "艺术风格 -> 现实风格":
        show_A_to_B_page()
    else:
        show_B_to_A_page()

    # 添加页脚信息
    st.sidebar.markdown("---")
    st.sidebar.info("CycleGAN 图像风格转换演示")

if __name__ == "__main__":
    main()