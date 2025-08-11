import os
import cv2
import numpy as np
from typing import List, Tuple, Optional
import torch
from PIL import Image

class VideoLoaderNode:
    def __init__(self):
        self.video_files = []
        self.current_video_index = 0
        self.current_frame_index = 0
        self.current_video = None
        self.current_video_frames = []
        self.video_fps = 0
        self.video_width = 0
        self.video_height = 0
        self.last_directory = ""
        self.last_mode = "single"
        
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "directory": ("STRING", {"default": "", "multiline": False}),
                "mode": (["single", "next"], {"default": "single"}),
                "video_index": ("INT", {"default": 0, "min": 0, "max": 9999}),
                "frames_per_video": ("INT", {"default": 30, "min": 1, "max": 1000}),
                "custom_width": ("INT", {"default": 512, "min": 64, "max": 2048}),
                "custom_height": ("INT", {"default": 512, "min": 64, "max": 2048}),
                "target_fps": ("FLOAT", {"default": 30.0, "min": 1.0, "max": 120.0}),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("video", "filename_text")
    FUNCTION = "load_video"
    CATEGORY = "video"
    
    def IS_CHANGED(self, directory, mode, video_index, frames_per_video, custom_width, custom_height, target_fps):
        """处理状态变化，确保next模式能正确触发重新执行"""
        # 如果目录改变，重置索引
        if directory != self.last_directory:
            self.current_video_index = 0
            self.last_directory = directory
        
        # 如果模式改变，重置索引
        if mode != self.last_mode:
            self.current_video_index = 0
            self.last_mode = mode
        
        # 对于next模式，每次执行都应该触发重新计算
        if mode == "next":
            return True
        
        # 对于single模式，只有当参数改变时才重新计算
        return None
    
    def get_video_files(self, directory: str) -> List[str]:
        """从目录中获取所有视频文件"""
        if not os.path.exists(directory):
            return []
        
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm', '.m4v']
        video_files = []
        
        for file in os.listdir(directory):
            if any(file.lower().endswith(ext) for ext in video_extensions):
                video_files.append(os.path.join(directory, file))
        
        # 使用自然数字排序，正确处理包含数字的文件名
        def natural_sort_key(filepath):
            import re
            # 将数字部分转换为整数进行比较
            filename = os.path.basename(filepath)
            return [int(text) if text.isdigit() else text.lower()
                   for text in re.split('([0-9]+)', filename)]
        
        video_files.sort(key=natural_sort_key)
        return video_files
    
    def load_video_frames(self, video_path: str, max_frames: int, target_width: int, target_height: int) -> Tuple[List[np.ndarray], float, int, int]:
        """加载视频帧"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return [], 0, 0, 0
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        frames = []
        frame_count = 0
        
        while frame_count < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            # 转换为RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # 调整尺寸
            if frame_rgb.shape[:2] != (target_height, target_width):
                frame_rgb = cv2.resize(frame_rgb, (target_width, target_height))
            
            frames.append(frame_rgb)
            frame_count += 1
        
        cap.release()
        return frames, fps, width, height
    
    def load_video(self, directory: str, mode: str, video_index: int, frames_per_video: int, 
                   custom_width: int, custom_height: int, target_fps: float):
        """主要的视频加载函数"""
        # 获取视频文件列表
        self.video_files = self.get_video_files(directory)
        
        if not self.video_files:
            # 返回空视频和错误信息
            empty_frame = np.zeros((custom_height, custom_width, 3), dtype=np.uint8)
            return (torch.from_numpy(empty_frame).unsqueeze(0), "错误: 目录中没有找到视频文件")
        
        # 确定要加载的视频索引
        if mode == "single":
            current_index = min(video_index, len(self.video_files) - 1)
        else:  # next mode
            current_index = self.current_video_index % len(self.video_files)
            # 更新索引为下一个视频
            self.current_video_index = (self.current_video_index + 1) % len(self.video_files)
        
        # 加载视频
        video_path = self.video_files[current_index]
        frames, fps, orig_width, orig_height = self.load_video_frames(
            video_path, frames_per_video, custom_width, custom_height
        )
        
        if not frames:
            # 返回空视频和错误信息
            empty_frame = np.zeros((custom_height, custom_width, 3), dtype=np.uint8)
            return (torch.from_numpy(empty_frame).unsqueeze(0), f"错误: 无法加载视频 {video_path}")
        
        # 转换为tensor格式
        frames_tensor = torch.from_numpy(np.array(frames)).float() / 255.0
        
        # 获取文件名
        filename = os.path.basename(video_path)
        
        return (frames_tensor, filename)


class ImageLoaderNode:
    def __init__(self):
        self.image_files = []
        self.current_image_index = 0
        self.last_directory = ""
        self.last_mode = "single"
        
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "directory": ("STRING", {"default": "", "multiline": False}),
                "mode": (["single", "next"], {"default": "single"}),
                "image_index": ("INT", {"default": 0, "min": 0, "max": 9999}),
                "custom_width": ("INT", {"default": 512, "min": 64, "max": 2048}),
                "custom_height": ("INT", {"default": 512, "min": 64, "max": 2048}),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "filename_text")
    FUNCTION = "load_image"
    CATEGORY = "image"
    
    def IS_CHANGED(self, directory, mode, image_index, custom_width, custom_height):
        """处理状态变化，确保next模式能正确触发重新执行"""
        # 如果目录改变，重置索引
        if directory != self.last_directory:
            self.current_image_index = 0
            self.last_directory = directory
        
        # 如果模式改变，重置索引
        if mode != self.last_mode:
            self.current_image_index = 0
            self.last_mode = mode
        
        # 对于next模式，每次执行都应该触发重新计算
        if mode == "next":
            return True
        
        # 对于single模式，只有当参数改变时才重新计算
        return None
    
    def get_image_files(self, directory: str) -> List[str]:
        """从目录中获取所有图片文件"""
        if not os.path.exists(directory):
            return []
        
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp', '.gif']
        image_files = []
        
        for file in os.listdir(directory):
            if any(file.lower().endswith(ext) for ext in image_extensions):
                image_files.append(os.path.join(directory, file))
        
        # 使用自然数字排序，正确处理包含数字的文件名
        def natural_sort_key(filepath):
            import re
            # 将数字部分转换为整数进行比较
            filename = os.path.basename(filepath)
            return [int(text) if text.isdigit() else text.lower()
                   for text in re.split('([0-9]+)', filename)]
        
        image_files.sort(key=natural_sort_key)
        return image_files
    
    def load_image_file(self, image_path: str, target_width: int, target_height: int) -> Optional[np.ndarray]:
        """加载单个图片文件"""
        try:
            # 使用PIL加载图片
            pil_image = Image.open(image_path)
            
            # 转换为RGB模式（处理RGBA、灰度等格式）
            if pil_image.mode != 'RGB':
                pil_image = pil_image.convert('RGB')
            
            # 调整尺寸
            if pil_image.size != (target_width, target_height):
                pil_image = pil_image.resize((target_width, target_height), Image.Resampling.LANCZOS)
            
            # 转换为numpy数组
            image_array = np.array(pil_image)
            
            return image_array
        except Exception as e:
            print(f"加载图片失败 {image_path}: {e}")
            return None
    
    def load_image(self, directory: str, mode: str, image_index: int, 
                   custom_width: int, custom_height: int):
        """主要的图片加载函数"""
        # 获取图片文件列表
        self.image_files = self.get_image_files(directory)
        
        if not self.image_files:
            # 返回空图片和错误信息
            empty_image = np.zeros((custom_height, custom_width, 3), dtype=np.uint8)
            return (torch.from_numpy(empty_image).unsqueeze(0).float() / 255.0, "错误: 目录中没有找到图片文件")
        
        # 确定要加载的图片索引
        if mode == "single":
            current_index = min(image_index, len(self.image_files) - 1)
        else:  # next mode
            current_index = self.current_image_index % len(self.image_files)
            # 更新索引为下一张图片
            self.current_image_index = (self.current_image_index + 1) % len(self.image_files)
        
        # 加载图片
        image_path = self.image_files[current_index]
        image_array = self.load_image_file(image_path, custom_width, custom_height)
        
        if image_array is None:
            # 返回空图片和错误信息
            empty_image = np.zeros((custom_height, custom_width, 3), dtype=np.uint8)
            return (torch.from_numpy(empty_image).unsqueeze(0).float() / 255.0, f"错误: 无法加载图片 {image_path}")
        
        # 转换为tensor格式
        image_tensor = torch.from_numpy(image_array).unsqueeze(0).float() / 255.0
        
        # 获取文件名
        filename = os.path.basename(image_path)
        
        return (image_tensor, filename)