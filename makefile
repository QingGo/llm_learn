.PHONY: install
install:
	uv pip install -e .
	transformer-train --install-completion