import os
import json
from typing import Optional, Union
import torch
from PIL import Image
import numpy as np

class SaveTextNode:
    def __init__(self):
        self.output_dir = ""
        
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"default": "", "multiline": True}),
                "file_type": (["文本", "图像(可选)"], {"default": "文本"}),
                "output_path": ("STRING", {"default": "", "multiline": False}),
                "file_name": ("STRING", {"default": "output", "multiline": False}),
                "file_extension": (["txt", "json", "csv", "md", "html", "xml"], {"default": "txt"}),
                "overwrite": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "image": ("IMAGE",),
            }
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("status",)
    FUNCTION = "save_file"
    CATEGORY = "file/save"
    
    def save_text_file(self, text: str, file_path: str, overwrite: bool) -> str:
        """保存文本文件"""
        try:
            # 检查文件是否存在且不允许覆盖
            if os.path.exists(file_path) and not overwrite:
                return f"错误: 文件已存在且不允许覆盖 - {file_path}"
            
            # 确保目录存在
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            # 根据文件扩展名选择编码
            encoding = 'utf-8'
            if file_path.lower().endswith(('.csv', '.txt')):
                encoding = 'utf-8'
            elif file_path.lower().endswith('.json'):
                # JSON文件需要特殊处理
                try:
                    data = json.loads(text) if text.strip() else {}
                    with open(file_path, 'w', encoding=encoding) as f:
                        json.dump(data, f, ensure_ascii=False, indent=2)
                    return f"成功保存JSON文件: {file_path}"
                except json.JSONDecodeError:
                    return f"错误: 无效的JSON格式"
            
            # 保存普通文本文件
            with open(file_path, 'w', encoding=encoding) as f:
                f.write(text)
            
            return f"成功保存文本文件: {file_path}"
            
        except Exception as e:
            return f"保存文件时出错: {str(e)}"
    
    def save_image_file(self, image: torch.Tensor, file_path: str, overwrite: bool) -> str:
        """保存图像文件"""
        try:
            # 检查文件是否存在且不允许覆盖
            if os.path.exists(file_path) and not overwrite:
                return f"错误: 文件已存在且不允许覆盖 - {file_path}"
            
            # 确保目录存在
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            # 转换tensor为PIL图像
            if isinstance(image, torch.Tensor):
                # 如果是批次图像，取第一张
                if image.dim() == 4:
                    image = image[0]
                
                # 确保值在0-255范围内
                if image.max() <= 1.0:
                    image = (image * 255).clamp(0, 255)
                
                # 转换为numpy数组
                image_np = image.cpu().numpy().astype(np.uint8)
                
                # 如果是灰度图像，转换为RGB
                if image_np.shape[-1] == 1:
                    image_np = np.repeat(image_np, 3, axis=-1)
                
                pil_image = Image.fromarray(image_np)
            else:
                pil_image = image
            
            # 保存图像
            pil_image.save(file_path)
            
            return f"成功保存图像文件: {file_path}"
            
        except Exception as e:
            return f"保存图像时出错: {str(e)}"
    
    def save_file(self, text: str, file_type: str, output_path: str, file_name: str, 
                  file_extension: str, overwrite: bool, image: Optional[torch.Tensor] = None):
        """主要的保存函数"""
        
        # 构建完整的文件路径
        if not output_path:
            output_path = os.getcwd()  # 使用当前工作目录
        
        # 根据文件类型选择扩展名
        if file_type == "图像(可选)":
            if file_extension in ["txt", "json", "csv", "md", "html", "xml"]:
                file_extension = "png"  # 图像文件使用png扩展名
        
        # 构建文件名
        if not file_name:
            file_name = "output"
        
        # 确保文件名有正确的扩展名
        if not file_name.endswith(f".{file_extension}"):
            file_name = f"{file_name}.{file_extension}"
        
        full_path = os.path.join(output_path, file_name)
        
        # 根据文件类型保存
        if file_type == "文本":
            return (self.save_text_file(text, full_path, overwrite),)
        elif file_type == "图像(可选)":
            if image is not None:
                return (self.save_image_file(image, full_path, overwrite),)
            else:
                return ("错误: 选择了图像模式但没有提供图像",)
        else:
            return ("错误: 不支持的文件类型",)

# 节点注册
NODE_CLASS_MAPPINGS = {
    "SaveText": SaveTextNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SaveText": "保存文本"
}