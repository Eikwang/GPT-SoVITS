import argparse
import time
import requests
import tracemalloc
import json
import os
from datetime import datetime
import threading
import statistics
import psutil


def call_tts(host: str, port: int, mode: str, ref_audio_path: str, streaming: bool):
    url = f"http://{host}:{port}/tts"
    text = "测试一句话，观察内存变化。"
    payload = {
        "text": text,
        "text_lang": "zh",
        "ref_audio_path": ref_audio_path,
        "prompt_lang": "zh",
        "prompt_text": "参考文本",
        "batch_size": 1,
        "streaming_mode": streaming,
        "media_type": "wav",
        "parallel_infer": True,
        "repetition_penalty": 1.35,
        "sample_steps": 8,
        "top_k": 5,
        "top_p": 1.0,
        "temperature": 1.0,
        "text_split_method": "cut5",
        "speed_factor": 1.0,
        "fragment_interval": 0.3,
        "seed": -1,
        "super_sampling": False,
    }
    try:
        if mode.lower() == "get":
            resp = requests.get(url, params=payload, timeout=60)
        else:
            resp = requests.post(url, json=payload, timeout=60)
        resp.raise_for_status()
        return True, len(resp.content)
    except Exception as e:
        return False, str(e)


def get_memory_status(host: str, port: int):
    try:
        url = f"http://{host}:{port}/memory_status"
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        # 兼容不同返回结构
        if isinstance(data, dict):
            if "memory_info" in data:
                return True, data
            if "memory_stats" in data:
                return True, {"memory_info": data.get("memory_stats")}
        return True, data
    except Exception as e:
        return False, {"error": str(e)}


def manual_cleanup(host: str, port: int):
    try:
        url = f"http://{host}:{port}/cleanup_memory"
        resp = requests.post(url, timeout=20)
        resp.raise_for_status()
        return True, resp.json()
    except Exception as e:
        return False, {"error": str(e)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9880)
    parser.add_argument("--ref", required=True, help="参考音频绝对路径")
    parser.add_argument("--duration", type=int, default=120, help="压测时长(秒)")
    parser.add_argument("--mode", choices=["get", "post"], default="post")
    parser.add_argument("--streaming", action="store_true")
    parser.add_argument("--interval", type=float, default=1.0, help="请求间隔(秒)")
    parser.add_argument("--concurrency", type=int, default=1, help="并发请求数")
    parser.add_argument("--cpu_sample_interval", type=float, default=1.0, help="CPU采样间隔(秒)")
    parser.add_argument("--log", default=os.path.join("logs", f"memory_leak_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"))
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.log), exist_ok=True)
    with open(args.log, "w", encoding="utf-8") as lf:
        def log(msg):
            print(msg)
            lf.write(msg + "\n")
            lf.flush()

        log(f"Starting memory leak test. host={args.host} port={args.port} duration={args.duration}s mode={args.mode} streaming={args.streaming}")
        log(f"ref_audio_path={args.ref}")

        ok, ms = get_memory_status(args.host, args.port)
        log(f"Initial /memory_status ok={ok} stats={json.dumps(ms, ensure_ascii=False)}")

        # 初始化CPU采样
        process = psutil.Process(os.getpid())
        cpu_system_series = []
        cpu_process_series = []
        stop_event = threading.Event()

        def cpu_sampler():
            # 预热避免第一个0.0
            process.cpu_percent(interval=None)
            psutil.cpu_percent(interval=None)
            while not stop_event.is_set():
                cpu_system_series.append(psutil.cpu_percent(interval=None))
                cpu_process_series.append(process.cpu_percent(interval=None))
                time.sleep(args.cpu_sample_interval)

        sampler_thread = threading.Thread(target=cpu_sampler, daemon=True)
        sampler_thread.start()

        # 初始化延迟与内存序列采集
        latencies = []
        tracemalloc.start(25)
        start_snapshot = tracemalloc.take_snapshot()
        start_time = time.time()
        last_status = start_time
        success_count = 0
        error_count = 0
        mem_series = []  # [(t, rss_mb, uss_mb, gpu_allocated_mb, gpu_reserved_mb)]

        # 并发工作线程
        workers = []

        def worker_loop(idx: int):
            nonlocal success_count, error_count
            while time.time() - start_time < args.duration and not stop_event.is_set():
                t0 = time.time()
                ok, detail = call_tts(args.host, args.port, args.mode, args.ref, args.streaming)
                t1 = time.time()
                latencies.append(t1 - t0)
                if ok:
                    success_count += 1
                else:
                    error_count += 1
                    log(f"[ERROR] t{idx} failed: {detail}")
                time.sleep(args.interval)

        for i in range(max(1, args.concurrency)):
            th = threading.Thread(target=worker_loop, args=(i,), daemon=True)
            workers.append(th)
            th.start()

        # 主线程负责每5秒采样一次内存状态
        while time.time() - start_time < args.duration:
            now = time.time()
            if now - last_status >= 5.0:
                ok, ms_now = get_memory_status(args.host, args.port)
                try:
                    mi = ms_now.get("memory_info", {}) if isinstance(ms_now, dict) else {}
                    mem_series.append((now, mi.get("rss_mb"), mi.get("uss_mb"), mi.get("gpu_allocated_mb"), mi.get("gpu_reserved_mb")))
                except Exception:
                    mem_series.append((now, None, None, None, None))
                log(f"[MEM] ok={ok} stats={json.dumps(ms_now, ensure_ascii=False)}")
                last_status = now
            time.sleep(0.2)

        # 停止采样与工作线程
        stop_event.set()
        for th in workers:
            th.join(timeout=5)
        sampler_thread.join(timeout=5)

        end_snapshot = tracemalloc.take_snapshot()
        stats = end_snapshot.compare_to(start_snapshot, "lineno")

        ok, cleanup_res = manual_cleanup(args.host, args.port)
        log(f"Manual cleanup ok={ok} result={json.dumps(cleanup_res, ensure_ascii=False)}")

        ok, ms_fin = get_memory_status(args.host, args.port)
        log(f"Final /memory_status ok={ok} stats={json.dumps(ms_fin, ensure_ascii=False)}")

        # 统计延迟指标
        latencies_sorted = sorted(latencies)
        avg_latency = statistics.mean(latencies_sorted) if latencies_sorted else 0.0
        p50 = latencies_sorted[int(0.50 * len(latencies_sorted))] if latencies_sorted else 0.0
        p90 = latencies_sorted[int(0.90 * len(latencies_sorted))] if latencies_sorted else 0.0
        p99 = latencies_sorted[int(0.99 * len(latencies_sorted))] if latencies_sorted else 0.0

        # 统计CPU指标
        cpu_sys_avg = statistics.mean(cpu_system_series) if cpu_system_series else 0.0
        cpu_sys_peak = max(cpu_system_series) if cpu_system_series else 0.0
        cpu_proc_avg = statistics.mean(cpu_process_series) if cpu_process_series else 0.0
        cpu_proc_peak = max(cpu_process_series) if cpu_process_series else 0.0

        log(f"Requests success={success_count} errors={error_count}")
        log(f"Latency avg={avg_latency:.3f}s p50={p50:.3f}s p90={p90:.3f}s p99={p99:.3f}s")
        log(f"CPU system avg={cpu_sys_avg:.1f}% peak={cpu_sys_peak:.1f}% | process avg={cpu_proc_avg:.1f}% peak={cpu_proc_peak:.1f}%")
        log("Top 10 tracemalloc diffs:")
        for stat in stats[:10]:
            log(str(stat))

        # 输出内存时间序列摘要
        log("Memory series (timestamp, rss_mb, uss_mb, gpu_allocated_mb, gpu_reserved_mb):")
        for item in mem_series:
            log(str(item))

        tracemalloc.stop()


if __name__ == "__main__":
    main()
