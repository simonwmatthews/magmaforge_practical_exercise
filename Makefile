.PHONY: tests devinstall

tests:

ifdef args
	pytest -c tests/pytest.ini $(args)
else
	pytest -c tests/pytest.ini
endif

devinstall:
	pip install -e .