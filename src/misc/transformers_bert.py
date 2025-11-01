from transformers import BertConfig, BertModel
import torch

# 初始化BERT配置与模型
config = BertConfig(hidden_size=768, num_attention_heads=6, attention_probs_dropout_prob=0.1, attn_implementation="eager")
model = BertModel(config)
model.eval()

# 构造输入（batch_size=2, seq_len=5）
input_ids = torch.randint(0, config.vocab_size, (2, 5))

# 使用模型的公开接口获取注意力与各层隐藏态
with torch.no_grad():
    outputs = model(input_ids, output_attentions=True, output_hidden_states=True)

attentions = outputs.attentions  # list/tuple: 每层的注意力权重 [batch, heads, seq_len, seq_len]
hidden_states_list = outputs.hidden_states  # list/tuple: [embeddings, layer1, layer2, ...]

attention_probs = attentions[0]
context_layer = hidden_states_list[1]

# 通过内部注意力子模块计算 Q/K/V，并打印多头形状（以第 1 层为例）
attn = model.encoder.layer[0].attention.self
hs0 = hidden_states_list[0]
with torch.no_grad():
    q = attn.query(hs0)
    k = attn.key(hs0)
    v = attn.value(hs0)

num_heads = attn.num_attention_heads
head_size = attn.attention_head_size
batch_size, seq_len, _ = hs0.shape

# 将 [batch, seq_len, num_heads, head_size] 置换为 [batch, num_heads, seq_len, head_size]
# 方便下一步计算多头注意力权重
q_multi = q.view(batch_size, seq_len, num_heads, head_size).permute(0, 2, 1, 3)
k_multi = k.view(batch_size, seq_len, num_heads, head_size).permute(0, 2, 1, 3)
v_multi = v.view(batch_size, seq_len, num_heads, head_size).permute(0, 2, 1, 3)

print("Q维度（拆分多头后）:", q_multi.shape)
print("K维度（拆分多头后）:", k_multi.shape)
print("V维度（拆分多头后）:", v_multi.shape)
print("注意力权重维度:", attention_probs.shape)
print("注意力输出维度:", context_layer.shape)
'''
Q维度（拆分多头后）: torch.Size([2, 6, 5, 128])
K维度（拆分多头后）: torch.Size([2, 6, 5, 128])
V维度（拆分多头后）: torch.Size([2, 6, 5, 128])
注意力权重维度: torch.Size([2, 6, 5, 5])
注意力输出维度: torch.Size([2, 5, 768])
'''