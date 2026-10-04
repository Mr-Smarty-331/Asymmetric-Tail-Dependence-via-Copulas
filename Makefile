.PHONY: install test test-all test-fast smoke coverage lint run-dashboard clean

PYTHON ?= python3
PYTEST ?= pytest

install:
	$(PYTHON) -m pip install -r requirements.txt

test:
	$(PYTEST) -m "not network and not llm and not slow"

test-fast:
	$(PYTEST) -m "not network and not llm and not slow"

test-all:
	$(PYTEST)

smoke:
	$(PYTHON) -m pytest tests/test_smoke.py -v

coverage:
	$(PYTEST) --cov=src --cov-report=term-missing tests/

run-dashboard:
	streamlit run src/app.py

clean:
	rm -rf __pycache__ .pytest_cache .coverage htmlcov tests/__pycache__ src/__pycache__
