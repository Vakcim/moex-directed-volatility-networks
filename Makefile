.PHONY: compile test lint check

compile:
	python -m compileall -q v10_directed tests

test:
	PYTHONPATH=v10_directed:tests python -m unittest discover -s tests -v

lint:
	ruff check v10_directed tests

check: compile test
