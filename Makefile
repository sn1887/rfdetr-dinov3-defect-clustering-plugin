PYTHON ?= python
PYTHONPATH := python-lib

.PHONY: validate test test-unit zip clean

validate:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/validate_plugin.py

test:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m pytest

test-unit:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m pytest tests/unit

zip: validate
	$(PYTHON) scripts/build_plugin_zip.py --output dist/rfdetr-dinov3-defect-clustering-plugin.zip

clean:
	rm -rf .pytest_cache .ruff_cache .mypy_cache htmlcov .coverage dist build
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
