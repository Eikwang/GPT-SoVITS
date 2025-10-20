"""
# WebAPI文档

` python api_v2.py -a 127.0.0.1 -p 9880 -c GPT_SoVITS/configs/tts_infer.yaml `

## 执行参数:
    `-a` - `绑定地址, 默认"127.0.0.1"`
    `-p` - `绑定端口, 默认9880`
    `-c` - `TTS配置文件路径, 默认"GPT_SoVITS/configs/tts_infer.yaml"`

## 调用:

### 推理

endpoint: `/tts`
GET:
```
http://127.0.0.1:9880/tts?text=先帝创业未半而中道崩殂，今天下三分，益州疲弊，此诚危急存亡之秋也。&text_lang=zh&ref_audio_path=archive_jingyuan_1.wav&prompt_lang=zh&prompt_text=我是「罗浮」云骑将军景元。不必拘谨，「将军」只是一时的身份，你称呼我景元便可&text_split_method=cut5&batch_size=1&media_type=wav&streaming_mode=true
```

POST:
```json
{
    "text": "",                   # str.(required) text to be synthesized
    "text_lang": "",              # str.(required) language of the text to be synthesized
    "ref_audio_path": "",         # str.(required) reference audio path
    "aux_ref_audio_paths": [],     # list.(optional) auxiliary reference audio paths for multi-speaker tone fusion
    "prompt_text": "",            # str.(optional) prompt text for the reference audio
    "prompt_lang": "",            # str.(required) language of the prompt text for the reference audio
    "top_k": 5,                    # int. top k sampling
    "top_p": 1,                    # float. top p sampling
    "temperature": 1,              # float. temperature for sampling
    "text_split_method": "cut5",  # str. text split method, see text_segmentation_method.py for details.
    "batch_size": 1,               # int. batch size for inference
    "batch_threshold": 0.75,       # float. threshold for batch splitting.
    "split_bucket": true,          # bool. whether to split the batch into multiple buckets.
    "speed_factor": 1.0,           # float. control the speed of the synthesized audio.
    "streaming_mode": false,       # bool. whether to return a streaming response.
    "seed": -1,                    # int. random seed for reproducibility.
    "parallel_infer": true,        # bool. whether to use parallel inference.
    "repetition_penalty": 1.35,    # float. repetition penalty for T2S model.
    "sample_steps": 32,            # int. number of sampling steps for VITS model V3.
    "super_sampling": false        # bool. whether to use super-sampling for audio when using VITS model V3.
}
```

RESP:
成功: 直接返回 wav 音频流， http code 200
失败: 返回包含错误信息的 json, http code 400

### 命令控制

endpoint: `/control`

command:
"restart": 重新运行
"exit": 结束运行

GET:
```
http://127.0.0.1:9880/control?command=restart
```
POST:
```json
{
    "command": "restart"
}
```

RESP: 无


### 切换GPT模型

endpoint: `/set_gpt_weights`

GET:
```
http://127.0.0.1:9880/set_gpt_weights?weights_path=GPT_SoVITS/pretrained_models/s1bert25hz-2kh-longer-epoch=68e-step=50232.ckpt
```
RESP:
成功: 返回"success", http code 200
失败: 返回包含错误信息的 json, http code 400


### 切换Sovits模型

endpoint: `/set_sovits_weights`

GET:
```
http://127.0.0.1:9880/set_sovits_weights?weights_path=GPT_SoVITS/pretrained_models/s2G488k.pth
```

RESP:
成功: 返回"success", http code 200
失败: 返回包含错误信息的 json, http code 400

"""

import os
import sys
import traceback
from typing import Generator

# 优先使用项目内的site-packages
project_site_packages = os.path.join(os.getcwd(), 'py312', 'Lib', 'site-packages')
if project_site_packages not in sys.path:
    sys.path.insert(0, project_site_packages)

# 在最开始强制导入ctypes.util以修复librosa加载问题
import ctypes.util

# 确保正确设置环境变量
try:
    # 抑制ONNX Runtime警告
    os.environ['ORT_DISABLE_ALL_LOGS'] = '1'
    # 设置bert_path环境变量
    os.environ['bert_path'] = 'GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large'
except:
    pass

import torch
import torchaudio
import gc
import threading
import time
import warnings
import signal
import asyncio

# 抑制常见警告
warnings.filterwarnings("ignore", category=UserWarning, module="torch")
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*onnxruntime.*CUDA.*")

now_dir = os.getcwd()
sys.path.append(now_dir)
sys.path.append("%s/GPT_SoVITS" % (now_dir))

# 修复异步环境问题
try:
    # 确保正确设置Windows事件循环策略
    if hasattr(asyncio, 'WindowsSelectorEventLoopPolicy') and os.name == 'nt':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        asyncio.set_event_loop(asyncio.new_event_loop())
except Exception as e:
    print(f"[警告] 无法设置异步事件循环策略: {e}")


# 兼容性函数


# 增强内存管理函数

# 定期内存监控函数

import argparse
import subprocess
import wave
import numpy as np
import soundfile as sf
from fastapi import FastAPI, Response
from fastapi.responses import StreamingResponse, JSONResponse
import uvicorn
from io import BytesIO
from tools.i18n.i18n import I18nAuto
from GPT_SoVITS.TTS_infer_pack.TTS import TTS, TTS_Config
from GPT_SoVITS.TTS_infer_pack.text_segmentation_method import get_method_names as get_cut_method_names

# 修复pydantic导入问题
# 导入pydantic
from pydantic import BaseModel

# 确保在Windows上使用正确的异步循环策略
import asyncio
if hasattr(asyncio, 'WindowsSelectorEventLoopPolicy') and os.name == 'nt':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# print(sys.path)
i18n = I18nAuto()
cut_method_names = get_cut_method_names()

parser = argparse.ArgumentParser(description="GPT-SoVITS api")
parser.add_argument("-c", "--tts_config", type=str, default="GPT_SoVITS/configs/tts_infer.yaml", help="tts_infer路径")
parser.add_argument("-a", "--bind_addr", type=str, default="127.0.0.1", help="default: 127.0.0.1")
parser.add_argument("-p", "--port", type=int, default="9880", help="default: 9880")
args = parser.parse_args()
config_path = args.tts_config
# device = args.device
port = args.port
host = args.bind_addr
argv = sys.argv

if config_path in [None, ""]:
    config_path = "GPT_SoVITS/configs/tts_infer.yaml"

# 延迟初始化TTS配置和管道
print("准备TTS配置...")
tts_config = TTS_Config(config_path)
print(tts_config)
# 启动时立即创建TTS实例
tts_pipeline = TTS(tts_config)

# 检测CUDA并自动切换到GPU+半精度以提升推理性能
try:
    import torch as _torch_init
    if _torch_init.cuda.is_available():
        print("检测到CUDA可用，切换到GPU并启用半精度以加速推理")
        try:
            tts_pipeline.set_device(_torch_init.device("cuda"), save=True)
        except Exception as e:
            print(f"[警告] 切换设备到CUDA失败: {e}")
        try:
            tts_pipeline.enable_half_precision(True, save=True)
        except Exception as e:
            print(f"[警告] 启用半精度失败: {e}")
    else:
        print("CUDA不可用，保持CPU模式")
except Exception as e:
    print(f"[警告] 设备初始化检查失败: {e}")


APP = FastAPI()



class TTS_Request(BaseModel):
    text: str = None
    text_lang: str = None
    ref_audio_path: str = None
    aux_ref_audio_paths: list = None
    prompt_lang: str = None
    prompt_text: str = ""
    top_k: int = 5
    top_p: float = 1
    temperature: float = 1
    text_split_method: str = "cut5"
    batch_size: int = 1
    batch_threshold: float = 0.75
    split_bucket: bool = True
    speed_factor: float = 1.0
    fragment_interval: float = 0.3
    seed: int = -1
    media_type: str = "wav"
    streaming_mode: bool = False
    parallel_infer: bool = True
    repetition_penalty: float = 1.35
    sample_steps: int = 32
    super_sampling: bool = False


### modify from https://github.com/RVC-Boss/GPT-SoVITS/pull/894/files
def pack_ogg(io_buffer: BytesIO, data: np.ndarray, rate: int):
    with sf.SoundFile(io_buffer, mode="w", samplerate=rate, channels=1, format="ogg") as audio_file:
        audio_file.write(data)
    return io_buffer


def pack_raw(io_buffer: BytesIO, data: np.ndarray, rate: int):
    io_buffer.write(data.tobytes())
    return io_buffer


def pack_wav(io_buffer: BytesIO, data: np.ndarray, rate: int):
    # 统一输出为16位PCM，避免浮点WAV在部分客户端不兼容
    io_buffer = BytesIO()
    if data.dtype != np.int16:
        if np.issubdtype(data.dtype, np.floating):
            data = (np.clip(data, -1.0, 1.0) * 32767.0).astype(np.int16)
        else:
            data = data.astype(np.int16, copy=False)
    sf.write(io_buffer, data, rate, format="WAV", subtype="PCM_16")
    return io_buffer


def pack_aac(io_buffer: BytesIO, data: np.ndarray, rate: int):
    # 统一为int16，避免ffmpeg参数与数据不匹配导致缓冲膨胀或阻塞
    if data.dtype != np.int16:
        if np.issubdtype(data.dtype, np.floating):
            data = (np.clip(data, -1.0, 1.0) * 32767.0).astype(np.int16)
        else:
            data = data.astype(np.int16, copy=False)
    process = subprocess.Popen(
        [
            "ffmpeg",
            "-f",
            "s16le",  # 输入16位有符号小端整数PCM
            "-ar",
            str(rate),  # 设置采样率
            "-ac",
            "1",  # 单声道
            "-i",
            "pipe:0",  # 从管道读取输入
            "-c:a",
            "aac",  # 音频编码器为AAC
            "-b:a",
            "192k",  # 比特率
            "-vn",  # 不包含视频
            "-f",
            "adts",  # 输出AAC数据流格式
            "pipe:1",  # 将输出写入管道
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    out, _ = process.communicate(input=data.tobytes())
    # 显式关闭句柄以帮助Windows上尽快释放资源
    try:
        if process.stdin:
            process.stdin.close()
        if process.stdout:
            process.stdout.close()
        if process.stderr:
            process.stderr.close()
    except Exception:
        pass
    io_buffer.write(out)
    return io_buffer


def pack_audio(io_buffer: BytesIO, data: np.ndarray, rate: int, media_type: str):
    if media_type == "ogg":
        io_buffer = pack_ogg(io_buffer, data, rate)
    elif media_type == "aac":
        io_buffer = pack_aac(io_buffer, data, rate)
    elif media_type == "wav":
        io_buffer = pack_wav(io_buffer, data, rate)
    else:
        io_buffer = pack_raw(io_buffer, data, rate)
    io_buffer.seek(0)
    return io_buffer


# from https://huggingface.co/spaces/coqui/voice-chat-with-mistral/blob/main/app.py
def wave_header_chunk(frame_input=b"", channels=1, sample_width=2, sample_rate=32000):
    # This will create a wave header then append the frame input
    # It should be first on a streaming wav file
    # Other frames better should not have it (else you will hear some artifacts each chunk start)
    wav_buf = BytesIO()
    with wave.open(wav_buf, "wb") as vfout:
        vfout.setnchannels(channels)
        vfout.setsampwidth(sample_width)
        vfout.setframerate(sample_rate)
        vfout.writeframes(frame_input)

    wav_buf.seek(0)
    return wav_buf.read()

# 更稳健的WAV流式头：手工构造RIFF/WAVE并将data块长度设为大值，适配持续流
def wave_header_chunk_streaming(channels=1, sample_width=2, sample_rate=16000):
    import struct
    bits_per_sample = sample_width * 8
    byte_rate = sample_rate * channels * sample_width
    block_align = channels * sample_width
    # 使用较大的占位长度以便部分解码器持续读取
    data_size = 0x7FFFFFFF
    # RIFF块大小 = 4 ("WAVE") + (8 + fmt_chunk) + (8 + data_size)
    riff_size = 4 + (8 + 16) + (8 + data_size)
    header = b"RIFF" + struct.pack('<I', riff_size) + b"WAVE"
    # fmt chunk
    header += b"fmt " + struct.pack('<I', 16)  # PCM fmt chunk size
    header += struct.pack('<HHIIHH', 1, channels, sample_rate, byte_rate, block_align, bits_per_sample)
    # data chunk with large size placeholder
    header += b"data" + struct.pack('<I', data_size)
    return header


def handle_control(command: str):
    if command == "restart":
        os.execl(sys.executable, sys.executable, *argv)
    elif command == "exit":
        os.kill(os.getpid(), signal.SIGTERM)
        exit(0)


def check_params(req: dict):
    text: str = req.get("text", "")
    text_lang: str = req.get("text_lang", "")
    ref_audio_path: str = req.get("ref_audio_path", "")
    streaming_mode: bool = req.get("streaming_mode", False)
    media_type: str = req.get("media_type", "wav")
    prompt_lang: str = req.get("prompt_lang", "")
    text_split_method: str = req.get("text_split_method", "cut5")

    if ref_audio_path in [None, ""]:
        return JSONResponse(status_code=400, content={"message": "ref_audio_path is required"})
    if text in [None, ""]:
        return JSONResponse(status_code=400, content={"message": "text is required"})
    if text_lang in [None, ""]:
        return JSONResponse(status_code=400, content={"message": "text_lang is required"})
    elif text_lang.lower() not in tts_config.languages:
        return JSONResponse(
            status_code=400,
            content={"message": f"text_lang: {text_lang} is not supported in version {tts_config.version}"},
        )
    if prompt_lang in [None, ""]:
        return JSONResponse(status_code=400, content={"message": "prompt_lang is required"})
    elif prompt_lang.lower() not in tts_config.languages:
        return JSONResponse(
            status_code=400,
            content={"message": f"prompt_lang: {prompt_lang} is not supported in version {tts_config.version}"},
        )
    if media_type not in ["wav", "raw", "ogg", "aac"]:
        return JSONResponse(status_code=400, content={"message": f"media_type: {media_type} is not supported"})
    elif media_type == "ogg" and not streaming_mode:
        return JSONResponse(status_code=400, content={"message": "ogg format is not supported in non-streaming mode"})

    if text_split_method not in cut_method_names:
        return JSONResponse(
            status_code=400, content={"message": f"text_split_method:{text_split_method} is not supported"}
        )

    return None


async def tts_handle(req: dict):
    """
    Text to speech handler.
    """
    global tts_pipeline
    
    streaming_mode = req.get("streaming_mode", False)
    return_fragment = req.get("return_fragment", False)
    media_type = req.get("media_type", "wav")
    # 统一初始化响应以避免异常路径返回None被序列化为JSON null
    response = None

    check_res = check_params(req)
    if check_res is not None:
        return check_res

    if streaming_mode or return_fragment:
        req["return_fragment"] = True

    try:

        # 简化TTS推理执行，移除复杂的异步包装和内存管理
        try:
            print("[TTS推理] 开始执行TTS推理...")
            tts_generator = tts_pipeline.run(req)
            print("[TTS推理] TTS推理完成")
                    
        except Exception as e:
            print(f"[TTS错误] TTS推理失败: {e}")
            traceback.print_exc()
            
            # 基础内存清理
            try:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
                
            return JSONResponse(
                content={"error": f"TTS inference failed: {str(e)}"},
                status_code=500
            )

        if streaming_mode:
            
            def streaming_generator():
                nonlocal media_type, tts_generator  # 声明使用外层函数的media_type与生成器变量
                resampler_16k = None
                current_media_type = media_type
                # 局部导入以确保在不同线程/作用域下可用
                import torch
                import torchaudio
                import numpy as np
                try:
                    # 统一处理AAC为RAW，避免流式编码阻塞
                    if current_media_type == "aac":
                        current_media_type = "raw"
                    # 若为WAV，先行发送流式WAV头，使用16k单声道2字节采样宽度
                    if current_media_type == "wav":
                        try:
                            yield wave_header_chunk_streaming(channels=1, sample_width=2, sample_rate=16000)
                        except Exception:
                            # 若头部构造失败，保持兼容并继续输出原始PCM
                            pass
                        current_media_type = "raw"

                    for sr, chunk in tts_generator:
                        # 统一重采样到16k
                        if sr != 16000:
                            # 复用单个重采样器，避免每块都创建转换器
                            if resampler_16k is None:
                                resampler_16k = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
                            chunk_tensor = torch.from_numpy(chunk).unsqueeze(0)
                            chunk_tensor = chunk_tensor.to(torch.float32) / 32768.0
                            chunk = resampler_16k(chunk_tensor).squeeze(0).numpy()
                            sr = 16000
                        # 确保片段为int16，避免下游编码器阻塞及临时大数组
                        if chunk.dtype != np.int16:
                            # 假定输入范围为[-1,1]或float，统一转换
                            if np.issubdtype(chunk.dtype, np.floating):
                                chunk = (np.clip(chunk, -1.0, 1.0) * 32767.0).astype(np.int16)
                            else:
                                # 其他整数类型按int16截断
                                chunk = chunk.astype(np.int16, copy=False)
                        yield pack_audio(BytesIO(), chunk, sr, current_media_type).getvalue()
                finally:
                    # 确保生成器结束时释放引用，帮助GC
                    try:
                        del resampler_16k
                    except Exception:
                        pass
                    try:
                        del chunk_tensor
                    except Exception:
                        pass
                    try:
                        del chunk
                    except Exception:
                        pass
                    try:
                        del sr
                    except Exception:
                        pass
                    # 主动关闭与删除生成器，避免悬挂引用
                    try:
                        if hasattr(tts_generator, 'close'):
                            tts_generator.close()
                    except Exception:
                        pass
                    try:
                        del tts_generator
                    except Exception:
                        pass
                    # 记录统一内存与缓存状态
                    try:
                        import psutil, os, gc
                        process = psutil.Process(os.getpid())
                        mem = process.memory_info()
                        mem_full = None
                        try:
                            mem_full = process.memory_full_info()
                        except Exception:
                            pass
                        gpu_alloc_mb = 0.0
                        gpu_reserved_mb = 0.0
                        try:
                            import torch
                            if torch.cuda.is_available():
                                gpu_alloc_mb = torch.cuda.memory_allocated() / 1024 / 1024
                                gpu_reserved_mb = torch.cuda.memory_reserved() / 1024 / 1024
                        except Exception:
                            pass
                        handles = None
                        fds = None
                        threads = None
                        try:
                            handles = process.num_handles()
                        except Exception:
                            pass
                        try:
                            fds = process.num_fds()
                        except Exception:
                            pass
                        try:
                            threads = process.num_threads()
                        except Exception:
                            pass
                        gc_counts = None
                        try:
                            gc_counts = gc.get_count()
                        except Exception:
                            pass
                        stats = {
                            'rss_mb': mem.rss / 1024 / 1024,
                            'vms_mb': mem.vms / 1024 / 1024,
                            'uss_mb': (getattr(mem_full, 'uss', None) or 0) / 1024 / 1024 if mem_full else None,
                            'pss_mb': (getattr(mem_full, 'pss', None) or 0) / 1024 / 1024 if mem_full else None,
                            'gpu_allocated_mb': gpu_alloc_mb,
                            'gpu_reserved_mb': gpu_reserved_mb,
                            'handles': handles,
                            'fds': fds,
                            'threads': threads,
                            'gc_counts': gc_counts,
                        }
                        # 缓存规模统计
                        try:
                            from GPT_SoVITS.TTS_infer_pack.TTS import resample_transform_dict
                        except Exception:
                            resample_transform_dict = None
                        resample_cache_size = len(resample_transform_dict) if isinstance(resample_transform_dict, dict) else None
                        prompt_cache_size = None
                        try:
                            if 'tts_pipeline' in globals() and hasattr(tts_pipeline, 'prompt_cache') and isinstance(tts_pipeline.prompt_cache, dict):
                                # 估算 prompt_cache 存储的条目数量
                                prompt_cache_size = len([k for k in tts_pipeline.prompt_cache.keys()])
                        except Exception:
                            pass
                        delta = ""
                        try:
                            if 'pre_stats' in locals() and pre_stats:
                                # 计算线程与缓存规模的增量
                                pre_threads = pre_stats.get('threads') or 0
                                pre_resample_cache_size = None
                                pre_prompt_cache_size = None
                                try:
                                    from GPT_SoVITS.TTS_infer_pack.TTS import resample_transform_dict as _resample_dict_pre
                                    pre_resample_cache_size = len(_resample_dict_pre) if isinstance(_resample_dict_pre, dict) else None
                                except Exception:
                                    pass
                                try:
                                    if 'tts_pipeline' in globals() and hasattr(tts_pipeline, 'prompt_cache') and isinstance(tts_pipeline.prompt_cache, dict):
                                        pre_prompt_cache_size = len([k for k in tts_pipeline.prompt_cache.keys()])
                                except Exception:
                                    pass
                                delta_threads = None
                                if isinstance(stats.get('threads'), int) and isinstance(pre_threads, int):
                                    delta_threads = stats['threads'] - pre_threads
                                delta_resample = None
                                if isinstance(resample_cache_size, int) and isinstance(pre_resample_cache_size, int):
                                    delta_resample = resample_cache_size - pre_resample_cache_size
                                delta_prompt = None
                                if isinstance(prompt_cache_size, int) and isinstance(pre_prompt_cache_size, int):
                                    delta_prompt = prompt_cache_size - pre_prompt_cache_size
                                delta = (
                                    f" | ΔRSS:{(stats['rss_mb'] - pre_stats['rss_mb']):.1f}MB"
                                    f" ΔUSS:{(((stats.get('uss_mb') or 0) - (pre_stats.get('uss_mb') or 0))):.1f}MB"
                                    f" ΔGPU_alloc:{(stats['gpu_allocated_mb'] - pre_stats['gpu_allocated_mb']):.1f}MB"
                                    + (f" Δthreads:{delta_threads}" if delta_threads is not None else "")
                                    + (f" Δresample_cache:{delta_resample}" if delta_resample is not None else "")
                                    + (f" Δprompt_cache_keys:{delta_prompt}" if delta_prompt is not None else "")
                                )
                        except Exception:
                            pass
                        
                        print(f"[内存状态][流式结束] RSS:{stats['rss_mb']:.1f}MB VMS:{stats['vms_mb']:.1f}MB USS:{(stats['uss_mb'] or 0):.1f}MB PSS:{(stats['pss_mb'] or 0):.1f}MB | GPU alloc:{stats['gpu_allocated_mb']:.1f}MB reserved:{stats['gpu_reserved_mb']:.1f}MB | handles:{stats['handles']} fds:{stats['fds']} threads:{stats['threads']} gc:{stats['gc_counts']} | resample_cache:{resample_cache_size} prompt_cache_keys:{prompt_cache_size}{delta}")
                    except Exception:
                        pass

            # 为流式传输设置更兼容的Content-Type
            # wav采用标准mime，raw采用RFC格式audio/L16
            response_media_type = f"audio/{media_type}"
            try:
                if media_type == "wav":
                    response_media_type = "audio/wav"
                elif media_type == "raw":
                    response_media_type = "audio/L16; rate=16000; channels=1"
            except Exception:
                pass

            response = StreamingResponse(
                streaming_generator(),
                media_type=response_media_type,
            )
            # 流式路径由生成器finally恢复监控
            # 关键：流式路径需要立即返回响应，否则将返回None被序列化为JSON null
            return response

        else:
            # 处理非流式响应 - 简化版本
            try:
                print("[非流式] 开始处理音频数据...")
                audio_chunks = []
                sample_rate = None
                
                # 收集所有音频块
                for sr, chunk in tts_generator:
                    if sample_rate is None:
                        sample_rate = sr
                    audio_chunks.append(chunk)
                    
                if not audio_chunks:
                    return JSONResponse(
                        content={"error": "No audio data generated"},
                        status_code=500
                    )
                
                # 合并音频数据
                import numpy as np
                audio_data = np.concatenate(audio_chunks)
                
                # 简单的音频数据处理
                audio_bytes = pack_audio(BytesIO(), audio_data, sample_rate, media_type).getvalue()
                print(f"[非流式] 音频处理完成，数据大小: {len(audio_bytes)} bytes")

                return Response(audio_bytes, media_type=f"audio/{media_type}")
                
            except Exception as e:
                print(f"[非流式错误] 音频处理失败: {e}")
                traceback.print_exc()
                return JSONResponse(
                    content={"error": f"Audio processing failed: {str(e)}"},
                    status_code=500
                )

    except Exception as e:
        print(f"[TTS处理错误] {e}")
        traceback.print_exc()
        return JSONResponse(
            content={"error": f"TTS processing failed: {str(e)}"},
            status_code=500
        )


        # 在异常情况下，保证返回有效响应
        try:
            streaming_mode = req.get("streaming_mode", False)
        except Exception:
            streaming_mode = False
        if streaming_mode:
            # 流式异常回退：返回WAV头+短静音片段，避免客户端收到null
            def silent_streaming_generator():
                import numpy as np
                from io import BytesIO
                try:
                    yield wave_header_chunk_streaming(channels=1, sample_width=2, sample_rate=16000)
                except Exception:
                    pass
                # 200ms 静音
                silence = np.zeros(int(0.2 * 16000), dtype=np.int16)
                yield pack_audio(BytesIO(), silence, 16000, "raw").getvalue()
            try:
                return StreamingResponse(silent_streaming_generator(), media_type="audio/wav")
            except Exception:
                # 兜底：返回空WAV头
                try:
                    return Response(content=wave_header_chunk_streaming(channels=1, sample_width=2, sample_rate=16000), media_type="audio/wav")
                except Exception:
                    return Response(content=b"", media_type="audio/wav")
        else:
            # 非流式异常路径维持原行为
            return Response(content=f"TTS generation failed: {e}", media_type="application/json", status_code=400)


@APP.get("/")
async def health_check():
    """健康检查端点"""
    return JSONResponse(content={
        "status": "ok",
        "message": "GPT-SoVITS API is running",
        "version": "v2"
    })


@APP.get("/control")
async def control(command: str = None):
    if command is None:
        return JSONResponse(status_code=400, content={"message": "command is required"})
    handle_control(command)


@APP.get("/tts")
async def tts_get_endpoint(
    text: str = None,
    text_lang: str = None,
    ref_audio_path: str = None,
    aux_ref_audio_paths: list = None,
    prompt_lang: str = None,
    prompt_text: str = "",
    top_k: int = 5,
    top_p: float = 1,
    temperature: float = 1,
    text_split_method: str = "cut5",
    batch_size: int = 1,
    batch_threshold: float = 0.75,
    split_bucket: bool = True,
    speed_factor: float = 1.0,
    fragment_interval: float = 0.3,
    seed: int = -1,
    media_type: str = "wav",
    streaming_mode: bool = False,
    parallel_infer: bool = True,
    repetition_penalty: float = 1.35,
    sample_steps: int = 32,
    super_sampling: bool = False,
):
    req = {
        "text": text,
        "text_lang": text_lang.lower() if text_lang else "zh",
        "ref_audio_path": ref_audio_path,
        "aux_ref_audio_paths": aux_ref_audio_paths,
        "prompt_text": prompt_text,
        "prompt_lang": prompt_lang.lower() if prompt_lang else "zh",
        "top_k": top_k,
        "top_p": top_p,
        "temperature": temperature,
        "text_split_method": text_split_method,
        "batch_size": int(batch_size),
        "batch_threshold": float(batch_threshold),
        "speed_factor": float(speed_factor),
        "split_bucket": split_bucket,
        "fragment_interval": fragment_interval,
        "seed": seed,
        "media_type": media_type,
        "streaming_mode": streaming_mode,
        "parallel_infer": parallel_infer,
        "repetition_penalty": float(repetition_penalty),
        "sample_steps": int(sample_steps),
        "super_sampling": super_sampling,
    }
    return await tts_handle(req)


@APP.post("/tts")
async def tts_post_endpoint(request: TTS_Request):
    req = request.dict()
    return await tts_handle(req)


@APP.get("/set_refer_audio")
async def set_refer_aduio(refer_audio_path: str = None):
    try:
        tts_pipeline.set_ref_audio(refer_audio_path)
    except Exception as e:
        return JSONResponse(status_code=400, content={"message": "set refer audio failed", "Exception": str(e)})
    return JSONResponse(status_code=200, content={"message": "success"})


@APP.get("/set_gpt_weights")
async def set_gpt_weights(weights_path: str = None):
    try:
        if weights_path in ["", None]:
            return JSONResponse(status_code=400, content={"message": "gpt weight path is required"})
        
        # 清理旧模型内存
        
        tts_pipeline.init_t2s_weights(weights_path)
        
            
        
    except Exception as e:
        return JSONResponse(status_code=400, content={"message": "change gpt weight failed", "Exception": str(e)})

    return JSONResponse(status_code=200, content={"message": "success"})


@APP.get("/set_sovits_weights")
async def set_sovits_weights(weights_path: str = None):
    try:
        if weights_path in ["", None]:
            return JSONResponse(status_code=400, content={"message": "sovits weight path is required"})
        
        # 清理旧模型内存
        
        tts_pipeline.init_vits_weights(weights_path)
        
                
        
    except Exception as e:
        return JSONResponse(status_code=400, content={"message": "change sovits weight failed", "Exception": str(e)})
    return JSONResponse(status_code=200, content={"message": "success"})




if __name__ == "__main__":
    try:
        if host == "None":  # 在调用时使用 -a None 参数，可以让api监听双栈
            host = None
        
        print("启动API服务器...")
        
        # 启动uvicorn服务器
        import uvicorn
        
        # 使用简化的配置启动uvicorn
        uvicorn.run(
            app=APP, 
            host=host, 
            port=port, 
            workers=1,
            log_level="info"
        )
        
    except Exception:
        traceback.print_exc()
    finally:
        # 清理资源
        os.kill(os.getpid(), signal.SIGTERM)
        exit(0)


