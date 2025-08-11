from .video_loader import VideoLoaderNode, ImageLoaderNode
from .save_text_node import SaveTextNode

NODE_CLASS_MAPPINGS = {
    "VideoLoader": VideoLoaderNode,
    "ImageLoader": ImageLoaderNode,
    "SaveText": SaveTextNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VideoLoader": "Video Loader",
    "ImageLoader": "Image Loader",
    "SaveText": "保存文本"
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']