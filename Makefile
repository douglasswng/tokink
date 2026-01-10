format:
	ruff format

check:
	ruff check --fix

format-check:
	make format
	make check

test:
	pytest

pre-commit:
	make format-check
	pre-commit run --all-files
