from vllm import LLM, SamplingParams

# 1. 初始化 LLM（模型将自动从 HuggingFace 下载到缓存目录）
llm = LLM(model="facebook/opt-125m")

# 2. 设置采样参数（等同于 generate 时的参数）
sampling_params = SamplingParams(
    temperature=0.8,
    top_p=0.95,
    max_tokens=50,
)

# 3. 准备提示词（prompts），可以一次传入多个，实现批量推理
prompts = [
    "What is the capital of France?",
    "Write a haiku about artificial intelligence:",
]

# 4. 生成
outputs = llm.generate(prompts, sampling_params)

# 5. 输出结果
for i, output in enumerate(outputs):
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt}")
    print(f"Generated: {generated_text}")
    print("-" * 50)