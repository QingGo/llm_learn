

.PHONY: install
install:
	uv pip install -e .
	bash -c "source $(PWD)/.venv/bin/activate && \
	transformer-train --install-completion && \
	transformer-interactive --install-completion"