import json, os
import matplotlib.pyplot as plt
import numpy as np

BASE = "./bench_results"
# 固定顺序
config_order = [
    "sglang_single", "sglang_dp", "sglang_tp",
    "vllm_single",  "vllm_dp",  "vllm_tp"
]

# 颜色：SGLang 蓝色系，vLLM 橙色系
colors = ['#1f77b4', '#4ba3d5', '#7fc8f8', '#ff7f0e', '#ffa940', '#ffcf70']

def load_data(base_dir):
    data = {}
    for cfg in config_order:
        filepath = os.path.join(base_dir, f"{cfg}.json")
        if not os.path.exists(filepath):
            print(f"Warning: {filepath} not found, skipping.")
            continue
        with open(filepath) as f:
            d = json.load(f)
        reqs = d['requests']
        # 提取指标列表
        ttft = [r['ttft'] for r in reqs]
        total_lat = [r['total_latency'] for r in reqs]
        # 处理 TPOT：如果JSON里有 avg_tpot 直接取，否则近似计算
        if 'avg_tpot' in reqs[0]:
            tpot = [r['avg_tpot'] for r in reqs]
            tpot_p95 = [r['p95_tpot'] for r in reqs]
        else:
            # 近似计算 TPOT = (total - ttft) / (tokens - 1)
            tpot = []
            tpot_p95 = []
            for r in reqs:
                if r['output_tokens'] > 1:
                    tp = (r['total_latency'] - r['ttft']) / (r['output_tokens'] - 1)
                else:
                    tp = 0
                tpot.append(tp)
                tpot_p95.append(tp)  # 近似时P95用平均代替，不够准确，建议用精确版
            print(f"Note: {cfg} used approximate TPOT (no avg_tpot field).")
        data[cfg] = {
            'ttft': ttft,
            'total_lat': total_lat,
            'tpot': tpot,
            'tpot_p95': tpot_p95,
            'throughput': d['throughput'],
            'wall_time': d['wall_time_seconds']
        }
    return data

data = load_data(BASE)

# 只取实际存在的配置
labels = [c for c in config_order if c in data]
if not labels:
    raise RuntimeError("No data loaded!")

# 提取绘图序列
ttft_lists = [data[c]['ttft'] for c in labels]
tpot_lists = [data[c]['tpot'] for c in labels]      # 用平均TPOT
tpot_p95_lists = [data[c]['tpot_p95'] for c in labels]
tps = [data[c]['throughput'] for c in labels]

# ---- 图1：TTFT 箱线图 ----
plt.figure(figsize=(10,6))
bplot = plt.boxplot(ttft_lists, labels=labels, patch_artist=True, showmeans=True)
for patch, color in zip(bplot['boxes'], colors[:len(labels)]):
    patch.set_facecolor(color)
plt.ylabel("TTFT (seconds)")
plt.title("Time-to-First-Token Comparison (Qwen3.5-4B, stream)")
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig("ttft_boxplot.png", dpi=150)

# ---- 图2：TPOT 箱线图（平均 TPOT） ----
plt.figure(figsize=(10,6))
bplot = plt.boxplot(tpot_lists, labels=labels, patch_artist=True, showmeans=True)
for patch, color in zip(bplot['boxes'], colors[:len(labels)]):
    patch.set_facecolor(color)
plt.ylabel("Average TPOT (seconds)")
plt.title("Per-Output-Token Latency Comparison (avg TPOT)")
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig("tpot_boxplot.png", dpi=150)

# ---- 图3：吞吐量柱状图 ----
plt.figure(figsize=(10,6))
bars = plt.bar(labels, tps, color=colors[:len(labels)])
plt.ylabel("Throughput (tokens/s)")
plt.title("Overall Throughput Comparison")
for bar, val in zip(bars, tps):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, f"{val:.1f}",
             ha='center', va='bottom', fontsize=9)
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig("throughput_bar.png", dpi=150)

# ---- 图4：延迟百分位对比（TTFT 与 TPOT 的 P50/P95/P99） ----
# 计算各个百分位数
import itertools
metrics = ['ttft', 'tpot']
percentiles = [50, 95, 99]
pdata = {}
for m in metrics:
    for p in percentiles:
        pdata[f"{m}_p{p}"] = [np.percentile(data[c][m], p) for c in labels]

# 绘制分组柱状图
fig, axes = plt.subplots(1, 2, figsize=(14,6), sharey=False)
for ax, metric in zip(axes, metrics):
    x = np.arange(len(labels))
    width = 0.25
    for i, p in enumerate(percentiles):
        vals = pdata[f"{metric}_p{p}"]
        ax.bar(x + i*width, vals, width, label=f'P{p}', color=f'C{i}')
    ax.set_title(f'{metric.upper()} Percentiles')
    ax.set_xticks(x + width)
    ax.set_xticklabels(labels, rotation=15, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig("latency_percentiles.png", dpi=150)

plt.show()