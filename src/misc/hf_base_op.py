from transformers import pipeline

'''
HF_ENDPOINT=https://hf-mirror.com uv run src/misc/hf_base_op.py
默认下载位置 ~/.cache/huggingface/hub
'''

'''
Pipeline API
No model was supplied, defaulted to distilbert/distilbert-base-uncased-finetuned-sst-2-english and revision 714eb0f (https://huggingface.co/distilbert/distilbert-base-uncased-finetuned-sst-2-english).
Using a pipeline without specifying a model name and revision in production is not recommended.
model.safetensors: 100%|███████████████████████████████████████████| 268M/268M [00:55<00:00, 4.86MB/s]
tokenizer_config.json: 100%|████████████████████████████████████████| 48.0/48.0 [00:00<00:00, 234kB/s]
vocab.txt: 232kB [00:00, 9.47MB/s]
Device set to use cpu
[{'label': 'POSITIVE', 'score': 0.9995681643486023}]
config.json: 665B [00:00, 999kB/s]
model.safetensors: 100%|███████████████████████████████████████████| 548M/548M [01:22<00:00, 6.65MB/s]
generation_config.json: 100%|█████████████████████████████████████████| 124/124 [00:00<00:00, 870kB/s]
tokenizer_config.json: 100%|████████████████████████████████████████| 26.0/26.0 [00:00<00:00, 198kB/s]
vocab.json: 1.04MB [00:01, 1.01MB/s]
merges.txt: 456kB [00:01, 405kB/s] 
tokenizer.json: 1.36MB [00:01, 1.00MB/s]
Device set to use cpu
Truncation was not explicitly activated but `max_length` is provided a specific value, please use `truncation=True` to explicitly truncate examples to max length. Defaulting to 'longest_first' truncation strategy. If you encode pairs of sequences (GLUE-style) with the tokenizer you can select this strategy more precisely by providing a specific strategy to `truncation`.
Setting `pad_token_id` to `eos_token_id`:50256 for open-end generation.
Both `max_new_tokens` (=256) and `max_length`(=30) seem to have been set. `max_new_tokens` will take precedence. Please refer to the documentation for more information. (https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)
[{'generated_text': 'Transformers library is a library to implement and manipulate the transformers in the Java code.\n\nThis library provides a simple, easy to use, and powerful way to implement and manipulate the transformers in the Java code. It can be used to implement either a Java-based or a Java-based class.\n\nIn this tutorial we will demonstrate using the Java transformers, and how to implement transformers in the Java code. In this tutorial we will use the Java transformers to generate the transformers in the class.\n\nTo start, we will create an instance of the transformers:\n\npublic class Transformers implements JavaTransforms { public String transformers = new String[5]; public Transformers(int x, int y, int width) { return x == y; } public Transformers(int x, int y, int width) { return x == y; } }\n\nIn this example we will create a class that takes a Transformers class and a Transformers class as parameters. All the transformers we will create will be available as transformers in the class.\n\nThe transformers we will implement will have an initializer with a new class name, and will have a method that will be called when the transformers are constructed.'}]
No model was supplied, defaulted to distilbert/distilbert-base-cased-distilled-squad and revision 564e9b5 (https://hf-mirror.com/distilbert/distilbert-base-cased-distilled-squad).
Using a pipeline without specifying a model name and revision in production is not recommended.
config.json: 473B [00:00, 720kB/s]
model.safetensors: 100%|███████████████████████████████████████████| 261M/261M [00:32<00:00, 8.00MB/s]
tokenizer_config.json: 49.0B [00:00, 1.30kB/s]
vocab.txt: 213kB [00:04, 51.7kB/s] 
tokenizer.json: 436kB [00:03, 117kB/s]  
Device set to use cpu
{'score': 0.9507156211911933, 'start': 40, 'end': 76, 'answer': 'Pipeline, Tokenizer, PreTrainedModel'}
'''
# 1. 情感分析（默认模型：distilbert-base-uncased-finetuned-sst-2-english）
sentiment_analyzer = pipeline("sentiment-analysis")
print(sentiment_analyzer("I love learning Transformers!"))  # 输出：[{'label': 'POSITIVE', 'score': 0.9998}]

# 2. 文本生成（GPT-2）
text_generator = pipeline("text-generation", model="gpt2")
print(text_generator("Transformers library is", max_length=30, num_return_sequences=1))

# 3. 问答（抽取式，需上下文+问题）
question_answerer = pipeline("question-answering")
context = "Transformers has three core components: Pipeline, Tokenizer, PreTrainedModel"
question = "What are the core components of Transformers?"
print(question_answerer(question=question, context=context))  # 输出答案及置信度

'''
Tokenizer
tokenizer_config.json: 100%|█████████████████████████████████████████████████████████████████| 48.0/48.0 [00:00<00:00, 354kB/s]
vocab.txt: 232kB [00:00, 387kB/s]  
tokenizer.json: 466kB [00:00, 1.01MB/s]
config.json: 570B [00:00, 2.74MB/s]
编码结果： [101, 7592, 19081, 999, 102]
解码结果： hello transformers!
批量编码形状： torch.Size([3, 10])
'''


from transformers import BertTokenizer

# 1. 加载预训练Tokenizer（需与模型匹配）
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# 2. 核心方法：encode（文本转ID）
text = "Hello Transformers!"
encoded = tokenizer.encode(text, add_special_tokens=True)  # add_special_tokens：添加[CLS]和[SEP]
print("编码结果：", encoded)  # 输出：[101, 7592, 19081, 999, 102]

# 3. decode（ID转回文本）
decoded = tokenizer.decode(encoded, skip_special_tokens=True)
print("解码结果：", decoded)  # 输出："hello transformers!"

# 4. 批量处理（padding/truncation）
texts = ["I love NLP", "Transformers is powerful", "This is a long text that needs truncation"]
batch_encoded = tokenizer(
    texts,
    padding=True,  # 短文本补0
    truncation=True,  # 长文本截断
    max_length=10,  # 统一长度为10
    return_tensors="pt"  # 返回PyTorch张量（tf为TensorFlow张量）
)
print("批量编码形状：", batch_encoded["input_ids"].shape)  # 输出：torch.Size([3, 10])

'''
PreTrainedModel
model.safetensors: 100%|████████████████████████████████████████████████████████████████████| 440M/440M [01:36<00:00, 4.56MB/s]
Some weights of BertForSequenceClassification were not initialized from the model checkpoint at bert-base-uncased and are newly initialized: ['classifier.bias', 'classifier.weight']
You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
BertForSequenceClassification(
  (bert): BertModel(
    (embeddings): BertEmbeddings(
      (word_embeddings): Embedding(30522, 768, padding_idx=0)
      (position_embeddings): Embedding(512, 768)
      (token_type_embeddings): Embedding(2, 768)
      (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
      (dropout): Dropout(p=0.1, inplace=False)
    )
    (encoder): BertEncoder(
      (layer): ModuleList(
        (0-11): 12 x BertLayer(
          (attention): BertAttention(
            (self): BertSdpaSelfAttention(
              (query): Linear(in_features=768, out_features=768, bias=True)
              (key): Linear(in_features=768, out_features=768, bias=True)
              (value): Linear(in_features=768, out_features=768, bias=True)
              (dropout): Dropout(p=0.1, inplace=False)
            )
            (output): BertSelfOutput(
              (dense): Linear(in_features=768, out_features=768, bias=True)
              (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
              (dropout): Dropout(p=0.1, inplace=False)
            )
          )
          (intermediate): BertIntermediate(
            (dense): Linear(in_features=768, out_features=3072, bias=True)
            (intermediate_act_fn): GELUActivation()
          )
          (output): BertOutput(
            (dense): Linear(in_features=3072, out_features=768, bias=True)
            (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
            (dropout): Dropout(p=0.1, inplace=False)
          )
        )
      )
    )
    (pooler): BertPooler(
      (dense): Linear(in_features=768, out_features=768, bias=True)
      (activation): Tanh()
    )
  )
  (dropout): Dropout(p=0.1, inplace=False)
  (classifier): Linear(in_features=768, out_features=2, bias=True)
)
Epoch 1/3, Loss: 0.7087
Epoch 2/3, Loss: 0.6487
Epoch 3/3, Loss: 0.7399
预测类别： 1
'''


from transformers import BertForSequenceClassification, BertTokenizer, get_linear_schedule_with_warmup
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim

# 1. 加载模型（指定任务：文本分类，num_labels=2表示二分类）
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
print(model)

# 2. 准备输入（Tokenizer处理后的数据）
text = "Transformers is easy to learn"
inputs = tokenizer(text, return_tensors="pt")  # 输出：input_ids、attention_mask

# 3. 模型微调，冻结 classifier 以外的层
for param in model.parameters():
    param.requires_grad = False  # 冻结所有层（除classifier）
for param in model.classifier.parameters():
    param.requires_grad = True  # 解冻 classifier 层

# 准备示例数据
texts = ["I love this product!", "This is terrible.", "It's okay.", "Amazing experience!"]
labels = [1, 0, 0, 1]  # 1表示正面，0表示负面
encoded_data = tokenizer(
    texts,
    padding=True,
    truncation=True,
    max_length=128,
    return_tensors="pt"
)

dataset = TensorDataset(
    encoded_data["input_ids"],
    encoded_data["attention_mask"],
    torch.tensor(labels)
)

dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
# 只将需要训练的参数传递给优化器
optimizer = optim.AdamW(
    [p for p in model.parameters() if p.requires_grad],  # 只优化可训练参数
    lr=2e-5
)

epochs = 3
total_steps = len(dataloader) * epochs
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=0,
    num_training_steps=total_steps
)

# 训练循环
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)

for epoch in range(epochs):
    model.train()
    total_loss = 0
    
    for batch in dataloader:
        batch_input_ids = batch[0].to(device)
        batch_attention_mask = batch[1].to(device)
        batch_labels = batch[2].to(device)
        
        model.zero_grad()
        
        outputs = model(
            input_ids=batch_input_ids,
            attention_mask=batch_attention_mask,
            labels=batch_labels
        )
        
        loss = outputs.loss
        total_loss += loss.item()
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
    
    avg_train_loss = total_loss / len(dataloader)
    print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_train_loss:.4f}")

# 4. 模型推理（禁用梯度计算，提升速度）
with torch.no_grad():
    outputs = model(**inputs)  # 解包inputs作为参数
    logits = outputs.logits  # 模型输出（未归一化的概率）
    predictions = torch.argmax(logits, dim=1)  # 取概率最大的类别

print("预测类别：", predictions.item())  # 输出：0或1（对应二分类标签）

# 5. 保存与加载
model.save_pretrained("./my_bert_model")
tokenizer.save_pretrained("./my_bert_model")
# 重新加载
model = BertForSequenceClassification.from_pretrained("./my_bert_model")

'''
验证编码结果
编码ID列表： [101, 1045, 2293, 17953, 2361, 1998, 19081, 999, 102]
编码字典keys： KeysView({'input_ids': tensor([[  101,  1045,  2293, 17953,  2361,  1998, 19081,   999,   102]]), 'token_type_ids': tensor([[0, 0, 0, 0, 0, 0, 0, 0, 0]]), 'attention_mask': tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1]])})
input_ids形状： torch.Size([1, 9])
保留特殊标记解码： [CLS] i love nlp and transformers! [SEP]
去除特殊标记解码： i love nlp and transformers!
中文编码ID： tensor([[  101,  8228, 11285, 11789,  8180,  4696,  4638,  2523,  1962,  4500,
          8013,   102]])
中文解码： tokenizer 真 的 很 好 用 ！
'''

from transformers import BertTokenizer

# 1. 加载与模型匹配的Tokenizer（必须一致！）
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")  # 英文小写模型
chinese_tokenizer = BertTokenizer.from_pretrained("uer/roberta-base-finetuned-dianping-chinese")  # 中文模型

# 2. 单文本编码（两种常用方法）
# 方法1：encode（简洁，返回列表）
text = "I love NLP and Transformers!"
encoded_ids = tokenizer.encode(
    text,
    add_special_tokens=True  # 开启特殊标记（默认True）
)
print("编码ID列表：", encoded_ids)  # 输出：[101, 1045, 2293, 17953, 2361, 1998, 19081, 999, 102]

# 方法2：__call__（推荐，返回字典，含input_ids、attention_mask等）
encoded_dict = tokenizer(
    text,
    add_special_tokens=True,
    return_tensors="pt"  # 返回PyTorch张量（tf=TensorFlow，np=NumPy）
)
print("编码字典keys：", encoded_dict.keys())  # 输出：dict_keys(['input_ids', 'attention_mask', 'token_type_ids'])
print("input_ids形状：", encoded_dict["input_ids"].shape)  # 输出：torch.Size([1, 9])

# 3. 解码操作
decoded_text = tokenizer.decode(
    encoded_ids,
    skip_special_tokens=False  # 是否忽略特殊标记（False=保留，True=去除）
)
print("保留特殊标记解码：", decoded_text)  # 输出："[CLS] i love nlp and transformers! [SEP]"
print("去除特殊标记解码：", tokenizer.decode(encoded_ids, skip_special_tokens=True))  # 输出："i love nlp and transformers!"

# 4. 中文文本实操（对比英文）
chinese_text = "Tokenizer真的很好用！"
chinese_encoded = chinese_tokenizer(chinese_text, return_tensors="pt")
print("中文编码ID：", chinese_encoded["input_ids"])
print("中文解码：", chinese_tokenizer.decode(chinese_encoded["input_ids"][0], skip_special_tokens=True)) # tokenizer 真 的 很 好 用 ！

'''
特殊 token
无特殊标记ID： [2023, 2003, 1037, 3231, 1012]
[CLS] ID： 101
[SEP] ID： 102
[PAD] ID： 0
手动添加特殊Token： [101, 2026, 7661, 3793, 102]
'''
from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
text = "This is a test."

# 1. 禁止添加[CLS]和[SEP]
encoded_no_special = tokenizer(text, add_special_tokens=False)
print("无特殊标记ID：", encoded_no_special["input_ids"])  # 无101（[CLS]）和102（[SEP]） [2023, 2003, 1037, 3231, 1012]

# 2. 查看特殊Token的ID
print("[CLS] ID：", tokenizer.cls_token_id)  # 输出：101
print("[SEP] ID：", tokenizer.sep_token_id)  # 输出：102
print("[PAD] ID：", tokenizer.pad_token_id)  # 输出：0

# 3. 手动添加特殊Token（场景：自定义任务）
custom_text = tokenizer.cls_token + " My custom text " + tokenizer.sep_token
encoded_custom = tokenizer(custom_text, add_special_tokens=False)
print("手动添加特殊Token：", encoded_custom["input_ids"])

'''
截断
批量1长度（最长文本长度）： 23
批量2形状： torch.Size([3, 15])
第3个文本截断后： this is a very long text that will be truncated because its length
第1个文本的attention_mask： tensor([1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
'''
from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
texts = [
    "I love NLP.",  # 短文本
    "Transformers library is powerful and easy to learn.",  # 中长文本
    "This is a very long text that will be truncated because its length exceeds the max_length we set."  # 超长文本
]

# 1. 基础批量处理（设置padding=True以确保所有序列长度相同）
batch1 = tokenizer(texts, padding=True, return_tensors="pt")
print("批量1长度（最长文本长度）：", batch1["input_ids"].shape[1])  # 输出：超长文本的原始长度

# 2. 限制max_length+开启truncation
batch2 = tokenizer(
    texts,
    padding="max_length",  # 填充到max_length
    truncation=True,       # 截断超长文本
    max_length=15,         # 统一长度为15
    return_tensors="pt"
)
print("批量2形状：", batch2["input_ids"].shape)  # 输出：torch.Size([3, 15])
print("第3个文本截断后：", tokenizer.decode(batch2["input_ids"][2], skip_special_tokens=True))

# 3. 查看attention_mask（1=真实Token，0=PAD）
print("第1个文本的attention_mask：", batch2["attention_mask"][0])  # 前6位为1，后9位为0

'''
GPT-2
GPT2LMHeadModel(
  (transformer): GPT2Model(
    (wte): Embedding(50257, 768)
    (wpe): Embedding(1024, 768)
    (drop): Dropout(p=0.1, inplace=False)
    (h): ModuleList(
      (0-11): 12 x GPT2Block(
        (ln_1): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
        (attn): GPT2Attention(
          (c_attn): Conv1D(nf=2304, nx=768)
          (c_proj): Conv1D(nf=768, nx=768)
          (attn_dropout): Dropout(p=0.1, inplace=False)
          (resid_dropout): Dropout(p=0.1, inplace=False)
        )
        (ln_2): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
        (mlp): GPT2MLP(
          (c_fc): Conv1D(nf=3072, nx=768)
          (c_proj): Conv1D(nf=768, nx=3072)
          (act): NewGELUActivation()
          (dropout): Dropout(p=0.1, inplace=False)
        )
      )
    )
    (ln_f): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
  )
  (lm_head): Linear(in_features=768, out_features=50257, bias=False)
)
模型结构： gpt2
词汇表大小： 50257
'''

from transformers import GPT2Tokenizer, GPT2LMHeadModel

# 1. 加载预训练模型和Tokenizer（模型名可替换为"gpt2-medium"等更大模型）
model_name = "gpt2"  # 基础版（轻量，适合demo）
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)
print(model)

# 2. 关键配置：设置padding标记（GPT-2默认无PAD token，批量生成时需补充）
tokenizer.pad_token = tokenizer.eos_token  # 用结束标记（EOS）作为填充标记

# 3. 验证加载成功
print("模型结构：", model.config.model_type)  # 输出：gpt2
print("词汇表大小：", tokenizer.vocab_size)  # 输出：50257（GPT-2默认词汇表）

'''
英文文本生成
提示词： In the future, AI will help humans
生成结果： In the future, AI will help humans to solve problems that are not solved by humans.

"We are going to have to be able to solve problems that are not solved by humans," said Dr. Michael S. Karp, a professor
'''
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch

# 1. 重新加载模型和Tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

# 2. 预处理提示词
prompt = "In the future, AI will help humans"
inputs = tokenizer(
    prompt,
    return_tensors="pt",  # 返回PyTorch张量
    padding=False,  # 单文本无需填充
    truncation=False  # 提示词未超长（GPT-2最大支持1024长度）
)

# 3. 模型生成（核心参数）
model.eval()  # 切换为推理模式（禁用训练时的dropout）
with torch.no_grad():  # 禁用梯度计算，提升速度
    outputs = model.generate(
        **inputs,  # 解包input_ids和attention_mask
        max_length=50,  # 生成总长度（提示词长度+生成文本长度）
        num_return_sequences=1,  # 生成1个结果
        pad_token_id=tokenizer.pad_token_id,  # 填充标记ID（必需）
        eos_token_id=tokenizer.eos_token_id  # 结束标记ID（生成到EOS时停止）
    )

# 4. 解码生成结果
generated_text = tokenizer.decode(
    outputs[0],
    skip_special_tokens=True  # 忽略特殊标记（此处仅EOS，可省略）
)

print("提示词：", prompt)
print("生成结果：", generated_text)

'''
中文文本生成
pytorch_model.bin: 100%|████████████████████████████████████████████████████████████████████| 421M/421M [01:49<00:00, 3.86MB/s]
GPT2LMHeadModel(
  (transformer): GPT2Model(
    (wte): Embedding(21128, 768)
    (wpe): Embedding(1024, 768)
    (drop): Dropout(p=0.1, inplace=False)
    (h): ModuleList(
      (0-11): 12 x GPT2Block(
        (ln_1): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
        (attn): GPT2Attention(
          (c_attn): Conv1D(nf=2304, nx=768)
          (c_proj): Conv1D(nf=768, nx=768)
          (attn_dropout): Dropout(p=0.1, inplace=False)
          (resid_dropout): Dropout(p=0.1, inplace=False)
        )
        (ln_2): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
        (mlp): GPT2MLP(
          (c_fc): Conv1D(nf=3072, nx=768)
          (c_proj): Conv1D(nf=768, nx=3072)
          (act): NewGELUActivation()
          (dropout): Dropout(p=0.1, inplace=False)
        )
      )
    )
    (ln_f): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
  )
  (lm_head): Linear(in_features=768, out_features=21128, bias=False)
)
Setting `pad_token_id` to `eos_token_id`:50256 for open-end generation.
中文生成结果： 未 来 ， 人 工 智 能 将 帮 助 人 类 未 来 ， 人 工 智 能 将 帮 助 人 类scorethestorethestorethestorethest
'''
# 中文Demo：加载中文GPT-2模型，
# 这个中文GPT-2模型在预训练阶段就使用了BERT的分词策略（WordPiece分词），而不是GPT-2原生的Byte Pair Encoding (BPE)分词。
# 这是为了更好地处理中文文本，
tokenizer = BertTokenizer.from_pretrained("uer/gpt2-chinese-cluecorpussmall")
model = GPT2LMHeadModel.from_pretrained("uer/gpt2-chinese-cluecorpussmall")
# 确保设置pad_token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    print(f"已设置pad_token为: {tokenizer.pad_token}")
print(model)

# 中文提示词生成
prompt = "未来，人工智能将帮助人类"
inputs = tokenizer(prompt, return_tensors="pt")
with torch.no_grad():
    outputs = model.generate(**inputs, max_length=50, num_return_sequences=1)
print("中文生成结果：", tokenizer.decode(outputs[0], skip_special_tokens=True))

'''
生成参数调整
Setting `pad_token_id` to `eos_token_id`:50256 for open-end generation.

=== temperature=0.1 ===
The best way to learn AI is to learn from it.

The best way to learn AI is to learn from it.

The best way to learn AI is to learn from it.
Setting `pad_token_id` to `eos_token_id`:50256 for open-end generation.

=== temperature=1.0 ===
The best way to learn AI is by actually using them in your training. The simplest way I can do is to teach the human brain how to do what a neural network does, i.e.
Setting `pad_token_id` to `eos_token_id`:50256 for open-end generation.

=== temperature=1.8 ===
The best way to learn AI is on the inside—both outside and into everyday life.

Image via pixabay
Setting `pad_token_id` to `eos_token_id`:50256 for open-end generation.
长文本生成结果： How to improve reading efficiency, especially by increasing the number of words in your novel.
We don't know what that means for you and I think we all need to start considering how our minds work around distractions if they're going away from us when it comes time deciding which books are suitable or unsuitable at different stages after release so as notto get bored with them! The more things change throughout
Setting `pad_token_id` to `eos_token_id`:50256 for open-end generation.

=== 提示词1：Machine learning is a branch of AI ===
结果1： Machine learning is a branch of AI in AI that uses real-time software to collect information from information at certain points in time. The system is a great place to start when you're trying to build a real-time predictive algorithm or perform a statistical analysis in scientific research.

I wanted to
结果2： Machine learning is a branch of AI that is based on AI that works on the underlying concepts and algorithms of programming. The goal is to provide a generalizable representation of how humans are able to develop and use these concepts and algorithms in real-world situations. We believe that this is a major breakthrough and

=== 提示词2：Data is the fuel of AI development ===
结果1： Data is the fuel of AI development on the ground, but these are important factors in the decision going forward. What is important is that it is considered that we can achieve those goals.

How does this impact AI, exactly?

AI is like a game of chess, you are constantly
结果2： Data is the fuel of AI development, and most of which is now in the hands of an AI program. The next iteration will see a version of the engine that can produce higher quality videos, and make use of AI algorithms to accelerate and adapt to the times.

"In the short term
Setting `pad_token_id` to `eos_token_id`:50256 for open-end generation.
优化后生成结果： Explain quantum computing to a high school student in 3 sentences
If you want something like that, here are some questions: What is the probability of solving these problems? Is there an optimal solution for all three puzzles if one person solves them and two people do not solve it or does they end up with different answers. And how many students can be involved when each puzzle has its own answer (the more time spent doing homework)? If we go further down this list, maybe even asking about math might
'''

# 重新加载模型和分词器
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
# 不同temperature对比
prompts = "The best way to learn AI is"
temperatures = [0.1, 1.0, 1.8]

input = tokenizer(prompts, return_tensors="pt")
for temp in temperatures:
    with torch.no_grad():
        outputs = model.generate(
            **input,
            max_length=40,
            do_sample=True,  # 必须启用采样，temperature参数才会生效
            temperature=temp,
            num_return_sequences=1
        )
    print(f"\n=== temperature={temp} ===")
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))

# 长文本生成+抑制重复
prompt = "How to improve reading efficiency"
with torch.no_grad():
    outputs = model.generate(
        **tokenizer(prompt, return_tensors="pt"),
        max_length=80,  # 生成更长文本
        do_sample=True,  # 必须启用采样，temperature参数才会生效
        temperature=0.9,
        repetition_penalty=1.2,  # 抑制重复
        top_k=50,
        top_p=0.9,
    )
print("长文本生成结果：", tokenizer.decode(outputs[0], skip_special_tokens=True))

# 批量处理多个提示词
prompts = [
    "Machine learning is a branch of AI",
    "Data is the fuel of AI development"
]

# 编码批量提示词（需开启padding）
inputs = tokenizer(
    prompts,
    return_tensors="pt",
    padding=True,  # 填充到最长提示词长度
    truncation=True,
    max_length=30
    # pad_token参数不被tokenizer支持，已在tokenizer初始化时设置
)

# 批量生成2个结果
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_length=60,
        num_return_sequences=2,  # 每个提示词生成2个结果
        do_sample=True,  # 启用采样，支持生成多个序列
        temperature=0.8,
    )

# 输出所有结果
for i, prompt in enumerate(prompts):
    print(f"\n=== 提示词{i+1}：{prompt} ===")
    for j in range(2):
        print(f"结果{j+1}：", tokenizer.decode(outputs[i*2 + j], skip_special_tokens=True))


prompt = "Explain quantum computing to a high school student in 3 sentences"
with torch.no_grad():
    outputs = model.generate(
        **tokenizer(prompt, return_tensors="pt"),
        max_length=100,
        min_length=50,  # 最小生成长度（含提示词）
        temperature=0.8,
        top_k=40,
        top_p=0.85,
        repetition_penalty=1.2,
        do_sample=True,  # 开启采样（默认True，关闭则为贪心搜索）
    )
print("优化后生成结果：", tokenizer.decode(outputs[0], skip_special_tokens=True))