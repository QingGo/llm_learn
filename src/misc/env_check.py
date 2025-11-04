#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TensorBoard + PyTorch add_graph 诊断脚本
可独立运行，用于排查 add_graph 静默退出 / CUDA 不可用 / OOM 等问题。
生成报告文件：tensorboard_diagnosis_report.txt
"""

import torch
from torch.utils.tensorboard import SummaryWriter
import os, gc, subprocess, traceback, shutil
from datetime import datetime
from transformer.model import Transformer

REPORT_PATH = "tensorboard_diagnosis_report.txt"

def log(msg, file=None, end="\n"):
    print(msg, end=end)
    if file:
        file.write(msg + "\n")

def section(title, file):
    sep = "=" * 80
    log(f"\n{sep}\n【{title}】\n{sep}", file)

def run_cmd(cmd):
    try:
        out = subprocess.check_output(cmd, shell=True, stderr=subprocess.STDOUT, timeout=10)
        return out.decode().strip()
    except Exception as e:
        return f"执行失败: {e}"

def check_env_info(f):
    section("PyTorch & 环境信息", f)
    log(f"时间: {datetime.now()}", f)
    log(f"PyTorch 版本: {torch.__version__}", f)
    log(f"CUDA 是否可用: {torch.cuda.is_available()}", f)
    log(f"torch.version.cuda: {torch.version.cuda}", f)
    log(f"torch.backends.cudnn.version(): {torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else 'N/A'}", f)

def check_gpu_status(f):
    section("GPU 状态检测", f)
    if not torch.cuda.is_available():
        log("⚠️ CUDA 不可用（当前为 CPU 环境）", f)
        log(run_cmd("nvidia-smi"), f)
        return "cpu"

    n = torch.cuda.device_count()
    log(f"✅ 检测到 {n} 个 GPU", f)
    for i in range(n):
        prop = torch.cuda.get_device_properties(i)
        log(f"GPU[{i}]: {prop.name}, 显存 {prop.total_memory / 1024**3:.1f} GB", f)
    log(run_cmd("nvidia-smi | head -n 20"), f)
    return "cuda"

def check_dependencies(f):
    section("关键依赖检查", f)
    for pkg in ["tensorboard", "protobuf", "onnx"]:
        try:
            __import__(pkg)
            version = run_cmd(f"pip show {pkg} | grep Version") or "未知版本"
            log(f"✅ {pkg} 已安装 ({version})", f)
        except ImportError:
            log(f"❌ {pkg} 未安装", f)

def check_disk_space(f):
    section("磁盘与权限", f)
    cwd = os.getcwd()
    total, used, free = shutil.disk_usage(cwd)
    log(f"当前目录: {cwd}", f)
    log(f"磁盘总量: {total / 1024**3:.1f} GB, 剩余: {free / 1024**3:.1f} GB", f)
    test_file = "tensorboard_write_test.txt"
    try:
        with open(test_file, "w") as wf:
            wf.write("test")
        os.remove(test_file)
        log("✅ 写入权限正常", f)
    except Exception as e:
        log(f"❌ 写入权限异常: {e}", f)

def check_oom_logs(f):
    section("系统 OOM Kill 检查", f)
    log(run_cmd("dmesg | grep -i 'killed process' | tail -n 5"), f)

def simple_add_graph_test(f, device):
    section("add_graph 功能测试", f)    
    # model = SimpleModel().to(device)
    # seq_len = 32
    # dummy_input = torch.randint(0, 10000, (4, seq_len), dtype=torch.long, device=device)
    model = Transformer(
        vocab_size=10000,
        d_model=64,
        seq_len=32,
        n_heads=4,
        d_hidden=256,
        stack=2
    )
    
    writer = SummaryWriter(log_dir="runs/diagnostic_test")
    device = torch.device("cpu")
    try:
        with torch.no_grad():
            writer.add_graph(model, (
                torch.zeros((16, model.seq_len), dtype=torch.long, device=device),
                torch.ones((16, model.seq_len), dtype=torch.long, device=device),
                torch.ones((16, model.seq_len), dtype=torch.bool, device=device),
                torch.ones((16, model.seq_len), dtype=torch.bool, device=device),
                ))
        log("✅ add_graph 成功生成图文件。", f)
    except RuntimeError as e:
        err = str(e).lower()
        if "out of memory" in err:
            log("❌ CUDA OOM（显存不足）", f)
        else:
            log("❌ RuntimeError:", f)
            log(traceback.format_exc(), f)
    except Exception:
        log("❌ 发生未知异常：", f)
        log(traceback.format_exc(), f)
    finally:
        writer.close()
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()
        log("🧹 清理完成。", f)

def generate_report():
    with open(REPORT_PATH, "w") as f:
        check_env_info(f)
        device = check_gpu_status(f)
        check_dependencies(f)
        check_disk_space(f)
        check_oom_logs(f)
        simple_add_graph_test(f, device)

        section("结论与建议", f)
        if not torch.cuda.is_available():
            log("⚠️ 检测到 CUDA 不可用，建议检查：", f)
            log("  1️⃣ 是否安装了 GPU 版 PyTorch", f)
            log("  2️⃣ `nvidia-smi` 输出是否正常", f)
            log("  3️⃣ 环境变量 CUDA_VISIBLE_DEVICES 是否被禁用", f)
        else:
            log("✅ CUDA 正常可用，add_graph 测试通过", f)

        log("\n报告已生成: " + os.path.abspath(REPORT_PATH), f)

    print(f"\n✅ 完成！请查看报告文件: {REPORT_PATH}")

if __name__ == "__main__":
    generate_report()
