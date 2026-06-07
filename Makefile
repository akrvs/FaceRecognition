.PHONY: install dev lint format test cov serve docker evaluate

install:
	pip install ".[recognition]"

dev:
	pip install -e ".[recognition,dev]"

lint:
	ruff check src tests
	mypy src

format:
	ruff format src tests
	ruff check --fix src tests

test:
	pytest

cov:
	pytest --cov=visage --cov-report=term-missing

serve:
	visage serve

docker:
	docker build -t visage:latest .

evaluate:
	visage evaluate $(LFW_DIR) --pairs 2000
