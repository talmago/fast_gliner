.PHONY: clean style build dev publish help

PY_BINDINGS_DIR=py_src
MATURIN_TOML=Cargo.toml

clean:
	cargo clean
	rm -rf target/
	rm -rf $(PY_BINDINGS_DIR)/build/ $(PY_BINDINGS_DIR)/dist/
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete

style:
	poetry run ruff check $(PY_BINDINGS_DIR) --fix
	poetry run ruff format $(PY_BINDINGS_DIR)

dev:
	poetry env activate
	poetry install --no-root
	poetry run maturin develop --release --strip -m $(MATURIN_TOML)

build:
	poetry run maturin build --release --strip -m $(MATURIN_TOML)

release:
	poetry run maturin publish -m $(MATURIN_TOML)

help:
	@echo ""
	@echo "Makefile commands:"
	@echo "  clean      - Remove build artifacts"
	@echo "  style      - Run code formatting and linting"
	@echo "  dev        - Build and install locally in editable mode"
	@echo "  build      - Build a wheel for distribution"
	@echo "  release    - Publish to PyPI (requires auth)"
	@echo "  help       - Show this help message"
