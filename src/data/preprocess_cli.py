import argparse

from .clean_cc_bc import preprocess_and_save


def main():
    parser = argparse.ArgumentParser(description="离线预处理：清洗/分块/分词并保存到磁盘")
    parser.add_argument("--output", type=str, required=True, help="保存到的目标路径（datasets.save_to_disk）")
    parser.add_argument("--max-seq-length", type=int, default=1024, help="最大序列长度")
    parser.add_argument("--min-seq-length", type=int, default=5, help="最小序列长度")
    parser.add_argument("--book-threshold", type=int, default=2000, help="长文本分割阈值")
    parser.add_argument("--pad-token-id", type=int, default=50256, help="填充token id")
    parser.add_argument("--tokenizer-path", type=str, default="./data/tokenizer.json", help="分词器保存/加载路径")
    parser.add_argument("--vocab-size", type=int, default=30000, help="新训练分词器的词表大小")
    parser.add_argument("--num-proc", type=int, default=8, help="预处理并行进程数")
    args = parser.parse_args()
    preprocess_and_save(
        output_path=args.output,
        max_seq_length=args.max_seq_length,
        min_seq_length=args.min_seq_length,
        book_threshold=args.book_threshold,
        pad_token_id=args.pad_token_id,
        tokenizer_path=args.tokenizer_path,
        vocab_size=args.vocab_size,
        num_proc=args.num_proc,
    )


if __name__ == "__main__":
    main()