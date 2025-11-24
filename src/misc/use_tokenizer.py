import tiktoken
import tokenizers

tokenizer = tiktoken.get_encoding("cl100k_base")
tokenizer2 = tokenizers.Tokenizer.from_pretrained("gpt2")


def test_token(text: str) -> int:
    """
    测试文本的token化过程，包括使用tiktoken和tokenizers库进行编码和解码。

    Args:
        text: 待测试的文本字符串

    Returns:
        int: 文本的token数量

    处理步骤:
    1. 使用tiktoken库对文本进行编码，得到token列表
    2. 打印每个token的原始值和对应的解码文本
    3. 使用tokenizers库对文本进行编码，得到token列表
    4. 打印每个token的原始值和对应的解码文本
    """
    # 使用tiktoken库对文本进行编码，得到token列表
    print("tiktoken tokenization:")
    tokens = tokenizer.encode(text)
    print(tokens)
    for token in tokens:
        print(f"Token: {token}, Decoded: {tokenizer.decode([token])}.")

    # 使用tokenizers库对文本进行编码，得到token列表
    print("tokenizers tokenization:")
    tokens2 = tokenizer2.encode(text)
    print(tokens2)
    for token in tokens2.ids:
        print(f"Token: {token}, Decoded: {tokenizer2.decode([token])}.")


for text in ["75785793 + 6746479"]:
    print(f"Test text: {text}")
    test_token(text)
"""
Test text: 75785793 + 6746479
tiktoken tokenization:
[23776, 20907, 6365, 489, 220, 25513, 22644, 24]
Token: 23776, Decoded: 757.
Token: 20907, Decoded: 857.
Token: 6365, Decoded: 93.
Token: 489, Decoded:  +.
Token: 220, Decoded:  .
Token: 25513, Decoded: 674.
Token: 22644, Decoded: 647.
Token: 24, Decoded: 9.
tokenizers tokenization:
Encoding(num_tokens=9, attributes=[ids, type_ids, tokens, offsets, attention_mask, special_tokens_mask, overflowing])
Token: 2425, Decoded: 75.
Token: 3695, Decoded: 78.
Token: 3553, Decoded: 57.
Token: 6052, Decoded: 93.
Token: 1343, Decoded:  +.
Token: 718, Decoded:  6.
Token: 4524, Decoded: 74.
Token: 2414, Decoded: 64.
Token: 3720, Decoded: 79.
"""
