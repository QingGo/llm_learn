

.PHONY: install
install:
	uv pip install -e .
	bash -c "source $(PWD)/.venv/bin/activate && \
	transformer-train --install-completion && \
	transformer-interactive --install-completion"

# 默认参数（可通过 make VAR=value 覆盖）
PROCESSED_PATH ?= ./data/processed_blocks
TOKENIZER_PATH ?= ./data/tokenizer.json
SEQ_LEN ?= 1024
# 小批量本地测试
BATCH_SIZE ?= 4
MIN_SEQ_LEN ?= 5
BOOK_THRESHOLD ?= 2000
PAD_TOKEN_ID ?= 50256
NUM_PROC ?= 8
VOCAB_SIZE ?= 30000

.PHONY: gpt-offline
gpt-offline:
	# 1) 离线预处理：清洗/分块/分词并保存
	uv run python -m src.data.preprocess_cli \
		--output $(PROCESSED_PATH) \
		--max-seq-length $(SEQ_LEN) \
		--min-seq-length $(MIN_SEQ_LEN) \
		--book-threshold $(BOOK_THRESHOLD) \
		--pad-token-id $(PAD_TOKEN_ID) \
		--tokenizer-path $(TOKENIZER_PATH) \
		--vocab-size $(VOCAB_SIZE) \
		--num-proc $(NUM_PROC)
	# 2) 使用离线数据进行训练
	uv run python -m src.gpt.cli \
		--processed-path $(PROCESSED_PATH) \
		--tokenizer-path $(TOKENIZER_PATH) \
		--batch-size $(BATCH_SIZE) \
		--seq-len $(SEQ_LEN)