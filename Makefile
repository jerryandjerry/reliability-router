.PHONY: install test lint demo eval serve dashboard clean

install:
	uv sync --all-extras

test:
	uv run --all-extras pytest

lint:
	uv run --all-extras ruff check src tests

demo:
	uv run reliability-router route --query "What is 2 + 2?"
	uv run reliability-router answer --query "Cite sources for the current CEO of Example Corp"

eval:
	uv run reliability-router evaluate

serve:
	uv run reliability-router serve --host 0.0.0.0 --port 8000

dashboard:
	uv run --all-extras streamlit run dashboard/app.py

clean:
	rm -rf .pytest_cache .ruff_cache .mypy_cache htmlcov .coverage .venv src/*.egg-info src/reliability_router.egg-info
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
