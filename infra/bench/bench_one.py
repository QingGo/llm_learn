import time, json, requests, argparse
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

print_lock = Lock()

def generate_random_text(seed, length_words):
    rng = np.random.RandomState(seed)
    vocab = ["the", "quick", "brown", "fox", "jumps", "over", "lazy", "dog",
             "model", "inference", "benchmark", "server", "parallel", "tensor",
             "sglang", "vllm", "throughput", "latency", "response", "token"]
    return " ".join(rng.choice(vocab, size=length_words))

def build_prompt(seed, n_words):
    text = generate_random_text(seed, n_words)
    return [{"role": "user", "content": text}]

def send_stream_request(url, model, seed, input_len, max_tokens, index):
    payload = {
        "model": model,
        "messages": build_prompt(seed, input_len),
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": True
    }
    t_start = time.time()
    token_times = []       # 每个 token 到达的绝对时间
    total_latency = None
    try:
        with requests.post(url, json=payload, stream=True, timeout=300) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines(decode_unicode=True):
                if not line:
                    continue
                if line.startswith("data: "):
                    data_str = line[6:]
                    if data_str == "[DONE]":
                        total_latency = time.time() - t_start
                        break
                    try:
                        chunk = json.loads(data_str)
                    except json.JSONDecodeError:
                        continue
                    delta = chunk.get("choices", [{}])[0].get("delta", {})
                    content = delta.get("content", "")
                    if content:
                        token_times.append(time.time())
        # 没有[DONE]的情况
        if total_latency is None:
            total_latency = time.time() - t_start
        output_tokens = len(token_times)
        if output_tokens == 0:
            # 无输出，可能错误
            return None
        # 计算 TTFT
        ttft = token_times[0] - t_start
        # 计算 TPOT 列表（相邻token间隔）
        intervals = np.diff(token_times)  # 长度 = output_tokens - 1
        avg_tpot = np.mean(intervals) if len(intervals) > 0 else 0.0
        p95_tpot = np.percentile(intervals, 95) if len(intervals) > 0 else 0.0
        with print_lock:
            print(f"[request {index:3d}] done, ttft={ttft:.3f}s, total={total_latency:.3f}s, "
                  f"tokens={output_tokens}, avg_tpot={avg_tpot*1000:.1f}ms")
        return {
            "ttft": ttft,
            "total_latency": total_latency,
            "output_tokens": output_tokens,
            "avg_tpot": avg_tpot,
            "p95_tpot": p95_tpot
        }
    except Exception as e:
        with print_lock:
            print(f"[request {index:3d}] failed: {e}")
        return None

def run_benchmark(args):
    base_seed = args.seed
    num = args.num_requests
    wall_start = time.time()
    results = []
    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = {
            executor.submit(send_stream_request, args.url, args.model,
                            base_seed + i, args.input_len, args.output_len, i): i
            for i in range(num)
        }
        for future in as_completed(futures):
            res = future.result()
            if res is not None:
                results.append(res)
    wall_end = time.time()
    wall_duration = wall_end - wall_start
    total_tokens = sum(r['output_tokens'] for r in results)
    throughput = total_tokens / wall_duration if wall_duration > 0 else 0.0
    return results, throughput, wall_duration

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--output-file", required=True)
    parser.add_argument("--num-requests", type=int, default=100)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--input-len", type=int, default=512)
    parser.add_argument("--output-len", type=int, default=128)
    parser.add_argument("--model", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    results, throughput, wall_time = run_benchmark(args)
    with open(args.output_file, "w") as f:
        json.dump({
            "config": vars(args),
            "throughput": throughput,
            "wall_time_seconds": wall_time,
            "requests": results
        }, f, indent=2)
    print(f"Results saved to {args.output_file}")
    print(f"Concurrency: {args.concurrency}, Throughput: {throughput:.2f} tokens/s, Wall time: {wall_time:.2f}s")