import os
import sys
import threading
import time

from tqdm import tqdm

now_dir = os.getcwd()
sys.path.append(now_dir)

import re
import torch
from text.LangSegmenter import LangSegmenter
from text import chinese
from typing import Dict, List, Tuple
from text.cleaner import clean_text
from text import cleaned_text_to_sequence
from transformers import AutoModelForMaskedLM, AutoTokenizer
from TTS_infer_pack.text_segmentation_method import split_big_text, splits, get_method as get_seg_method

from tools.i18n.i18n import I18nAuto, scan_language_list

language=os.environ.get("language","Auto")
language=sys.argv[-1] if sys.argv[-1] in scan_language_list() else language
i18n = I18nAuto(language=language)
punctuation = set(['!', '?', '…', ',', '.', '-'," "])

def get_first(text:str) -> str:
    pattern = "[" + "".join(re.escape(sep) for sep in splits) + "]"
    text = re.split(pattern, text)[0].strip()
    return text

def merge_short_text_in_array(texts:str, threshold:int) -> list:
    if (len(texts)) < 2:
        return texts
    result = []
    text = ""
    for ele in texts:
        text += ele
        if len(text) >= threshold:
            result.append(text)
            text = ""
    if (len(text) > 0):
        if len(result) == 0:
            result.append(text)
        else:
            result[len(result) - 1] += text
    return result

class TextPreprocessor:
    def __init__(self, bert_model:AutoModelForMaskedLM, 
                 tokenizer:AutoTokenizer, device:torch.device):
        self.bert_model = bert_model
        self.tokenizer = tokenizer
        self.device = device
        

        

        

        
    def preprocess(self, text:str, lang:str, text_split_method:str, version:str="v2")->List[Dict]:
        print(i18n("############ 切分文本 ############"))
        text = self.replace_consecutive_punctuation(text)
        texts = self.pre_seg_text(text, lang, text_split_method)
        result = []
        print(i18n("############ 提取文本Bert特征 ############"))
        print(texts)
        try:
            # 确保音频输出目录存在
            output_dir = './音频输出'
            os.makedirs(output_dir, exist_ok=True)
            with open('./音频输出/tts-out.txt', 'w', encoding='utf-8') as f:
                f.write(str(texts))
        except Exception as e:
            print(f"保存文本切分结果时出错: {e}")
            pass
        for text in tqdm(texts):
            phones, bert_features, norm_text = self.segment_and_extract_feature_for_text(text, lang, version)
            if phones is None or norm_text=="":
                continue
            res={
                "phones": phones,
                "bert_features": bert_features,
                "norm_text": norm_text,
            }
            result.append(res)
        return result

    def pre_seg_text(self, text:str, lang:str, text_split_method:str):
        text = text.strip("\n")
        if len(text) == 0:
            return []
        if (text[0] not in splits and len(get_first(text)) < 4): 
            text = "。" + text if lang != "en" else "." + text
        print(i18n("实际输入的目标文本:"))
        print(text)
        
        seg_method = get_seg_method(text_split_method)
        text = seg_method(text)
        
        while "\n\n" in text:
            text = text.replace("\n\n", "\n")

        _texts = text.split("\n")
        _texts = self.filter_text(_texts)
        _texts = merge_short_text_in_array(_texts, 5)
        texts = []

        for text in _texts:
            # 解决输入目标文本的空行导致报错的问题
            if len(text.strip()) == 0:
                continue
            if not re.sub("\W+", "", text):
                # 检测一下，如果是纯符号，就跳过。
                continue
            if text[-1] not in splits:
                text += "。" if lang != "en" else "."

            # 解决句子过长导致Bert报错的问题
            if len(text) > 50:
                texts.extend(split_big_text(text))
            else:
                texts.append(text)

        print(i18n("实际输入的目标文本(切句后):"))
        print(texts)
        return texts
    
    def segment_and_extract_feature_for_text(self, text:str, language:str, version:str="v1")->Tuple[list, torch.Tensor, str]:
        return self.get_phones_and_bert(text, language, version)
        
    def get_phones_and_bert(self, text:str, language:str, version:str, final:bool=False):
        if language in {"en", "all_zh", "all_ja", "all_ko", "all_yue"}:
            language = language.replace("all_","")
            if language == "en":
                formattext = " ".join(tmp["text"] for tmp in LangSegmenter.getTexts(text))
            else:
                # 因无法区别中日韩文汉字,以用户输入为准
                formattext = text
            while "  " in formattext:
                formattext = formattext.replace("  ", " ")
            if language == "zh":
                if re.search(r'[A-Za-z]', formattext):
                    formattext = re.sub(r'[a-z]', lambda x: x.group(0).upper(), formattext)
                    formattext = chinese.mix_text_normalize(formattext)
                    return self.get_phones_and_bert(formattext,"zh",version)
                else:
                    phones, word2ph, norm_text = self.clean_text_inf(formattext, language, version)
                    bert = self.get_bert_feature(norm_text, word2ph).to(self.device)
            elif language == "yue" and re.search(r'[A-Za-z]', formattext):
                    formattext = re.sub(r'[a-z]', lambda x: x.group(0).upper(), formattext)
                    formattext = chinese.mix_text_normalize(formattext)
                    return self.get_phones_and_bert(formattext,"yue",version)
            else:
                phones, word2ph, norm_text = self.clean_text_inf(formattext, language, version)
                bert = torch.zeros(
                    (1024, len(phones)),
                    dtype=torch.float32,
                ).to(self.device)
        elif language in {"zh", "ja", "ko", "yue", "auto", "auto_yue"}:
            textlist=[]
            langlist=[]
            if language == "auto":
                for tmp in LangSegmenter.getTexts(text):
                    langlist.append(tmp["lang"])
                    textlist.append(tmp["text"])
            elif language == "auto_yue":
                for tmp in LangSegmenter.getTexts(text):
                    if tmp["lang"] == "zh":
                        tmp["lang"] = "yue"
                    langlist.append(tmp["lang"])
                    textlist.append(tmp["text"])
            else:
                for tmp in LangSegmenter.getTexts(text):
                    if tmp["lang"] == "en":
                        langlist.append(tmp["lang"])
                    else:
                        # 因无法区别中日韩文汉字,以用户输入为准
                        langlist.append(language)
                    textlist.append(tmp["text"])
            # print(textlist)
            # print(langlist)
            phones_list = []
            bert_list = []
            norm_text_list = []
            for i in range(len(textlist)):
                lang = langlist[i]
                phones, word2ph, norm_text = self.clean_text_inf(textlist[i], lang, version)
                bert = self.get_bert_inf(phones, word2ph, norm_text, lang)
                phones_list.append(phones)
                norm_text_list.append(norm_text)
                bert_list.append(bert)
            bert = torch.cat(bert_list, dim=1)
            phones = sum(phones_list, [])
            norm_text = ''.join(norm_text_list)

        if not final and len(phones) < 6:
            return self.get_phones_and_bert("." + text,language,version,final=True)

        return phones, bert, norm_text


    def get_bert_feature(self, text:str, word2ph:list)->torch.Tensor:
        with torch.inference_mode():
            # 清理之前的缓存
            self._cleanup_bert_cache()
            
            inputs = self.tokenizer(text, return_tensors="pt")
            for i in inputs:
                inputs[i] = inputs[i].to(self.device)
            # 确保输入张量与模型在同一设备上
            if self.bert_model.device != self.device:
                self.bert_model = self.bert_model.to(self.device)
            
            # 使用torch.no_grad()上下文管理器避免计算图构建
            res = self.bert_model(**inputs, output_hidden_states=True)
            
            # 提取需要的隐藏状态
            needed_hidden_states = res.hidden_states[-3:-2]
            feature = torch.cat(needed_hidden_states, -1)[0].cpu()[1:-1]
            
            # 彻底清理BERT模型相关对象
            self._cleanup_bert_objects(res, inputs)
                
        assert len(word2ph) == len(text)
        phone_level_feature = []
        for i in range(len(word2ph)):
            repeat_feature = feature[i].repeat(word2ph[i], 1)
            phone_level_feature.append(repeat_feature)
        phone_level_feature = torch.cat(phone_level_feature, dim=0)
        # 确保返回的特征张量在正确的设备上
        return phone_level_feature.T.to(self.device)
    
    def _cleanup_bert_cache(self):
        """清理BERT模型相关的缓存"""
        import gc
        
        # 清理PyTorch的缓存
        if hasattr(torch, 'cuda') and torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # 清理Python内部缓存
        gc.collect()
    
    def _cleanup_bert_objects(self, res, inputs):
        """彻底清理BERT模型相关的对象"""
        import gc
        
        # 清理模型输出对象
        if hasattr(res, 'hidden_states'):
            res.hidden_states = None
        if hasattr(res, 'attentions'):
            res.attentions = None
        if hasattr(res, 'last_hidden_state'):
            res.last_hidden_state = None
        
        # 清理输入张量
        for key in list(inputs.keys()):
            if hasattr(inputs[key], 'detach'):
                inputs[key].detach()
            inputs[key] = None
        
        # 清理BERT模型内部缓存（如果存在）
        if hasattr(self.bert_model, 'clear_cache'):
            self.bert_model.clear_cache()
        
        # 强制垃圾回收
        for _ in range(3):  # 多次清理确保彻底
            gc.collect()
            if hasattr(torch, 'cuda') and torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
    
    
    
    def clean_text_inf(self, text:str, language:str, version:str="v2"):
        phones, word2ph, norm_text = clean_text(text, language, version)
        phones = cleaned_text_to_sequence(phones, version)
        return phones, word2ph, norm_text

    def get_bert_inf(self, phones:list, word2ph:list, norm_text:str, language:str):
        language=language.replace("all_","")
        if language == "zh":
            feature = self.get_bert_feature(norm_text, word2ph).to(self.device)
        else:
            feature = torch.zeros(
                (1024, len(phones)),
                dtype=torch.float32,
            ).to(self.device)

        return feature

    def cleanup(self):
        """公共接口：清理BERT相关缓存以释放内存"""
        try:
            # 统一清理BERT缓存
            self._cleanup_bert_cache()

            # 尝试清理BERT模型内部缓存
            if hasattr(self.bert_model, 'clear_cache'):
                try:
                    self.bert_model.clear_cache()
                except Exception as e:
                    print(f"[警告] bert_model.clear_cache 调用失败: {e}")

            # 多轮垃圾回收与CUDA缓存释放，确保彻底释放显存
            import gc
            for _ in range(2):
                gc.collect()
                if hasattr(torch, 'cuda') and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
        except Exception as e:
            print(f"[警告] TextPreprocessor.cleanup 出错: {e}")


    def filter_text(self,texts):
        _text=[]
        if all(text in [None, " ", "\n",""] for text in texts):
            raise ValueError(i18n("请输入有效文本"))
        for text in texts:
            if text in  [None, " ", ""]:
                pass
            else:
                _text.append(text)
        return _text
    

    def replace_consecutive_punctuation(self,text):
        punctuations = ''.join(re.escape(p) for p in punctuation)
        pattern = f'([{punctuations}])([{punctuations}])+'
        result = re.sub(pattern, r'\1', text)
        return result