# G2PW兼容接口包装类
# 提供与原始g2pw模块兼容的接口

import os
from typing import List
from .onnx_api import G2PWOnnxConverter


class G2PWPinyin:
    """G2PW拼音转换器兼容类"""
    
    def __init__(
        self,
        model_dir: str = "G2PWModel/",
        model_source: str = None,
        v_to_u: bool = False,
        neutral_tone_with_five: bool = True,
        style: str = "pinyin"
    ):
        """
        初始化G2PW拼音转换器
        
        Args:
            model_dir: 模型目录路径
            model_source: 模型源
            v_to_u: 是否将v转换为u
            neutral_tone_with_five: 是否使用五声调表示轻声
            style: 拼音风格，支持"pinyin"或"bopomofo"
        """
        # 确保路径格式正确
        if not os.path.isabs(model_dir):
            current_dir = os.path.dirname(os.path.abspath(__file__))
            model_dir = os.path.join(current_dir, model_dir)
        
        # 初始化ONNX转换器
        self.converter = G2PWOnnxConverter(
            model_dir=model_dir,
            style=style,
            model_source=model_source,
            enable_non_tradional_chinese=False
        )
        
        self.v_to_u = v_to_u
        self.neutral_tone_with_five = neutral_tone_with_five
        self.style = style
    
    def lazy_pinyin(self, text: str, neutral_tone_with_five: bool = True, style: str = "TONE3") -> List[str]:
        """
        将文本转换为拼音列表
        
        Args:
            text: 输入文本
            neutral_tone_with_five: 是否使用五声调表示轻声
            style: 拼音风格
            
        Returns:
            拼音列表
        """
        # 使用ONNX转换器进行拼音转换
        results = self.converter([text])
        
        if not results:
            return []
        
        # 提取拼音结果
        pinyin_list = []
        for char_pinyin in results[0]:
            if char_pinyin is None:
                # 对于非中文字符，返回空字符串
                pinyin_list.append("")
            else:
                pinyin_list.append(char_pinyin)
        
        return pinyin_list


def correct_pronunciation(word: str, pinyins: List[str]) -> List[str]:
    """
    多音字发音校正函数
    
    Args:
        word: 单词
        pinyins: 原始拼音列表
        
    Returns:
        校正后的拼音列表
    """
    # 这里可以实现多音字校正逻辑
    # 目前直接返回原始拼音
    return pinyins