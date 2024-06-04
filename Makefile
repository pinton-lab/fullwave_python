PYTHON_VERSION=3.11.5
WHICH_PYTHON_COMMAND=$$(which python3)
res=${WHICH_PYTHON_COMMAND}
PROJECT_NAME=fullwave_simulation

install-pyenv:
	brew update
	brew install pyenv
install-precommit:
	brew update
	brew install pre-commit
install-poetry:
	curl -sSL https://install.python-poetry.org | python3 -
	poetry --version
install-python:
	pyenv install $(PYTHON_VERSION)
install:
	pyenv local $(PYTHON_VERSION)
	poetry config virtualenvs.in-project true
	poetry env use -- $(WHICH_PYTHON_COMMAND)
	export PYTHON_KEYRING_BACKEND=keyring.backends.null.Keyring && poetry install
	poetry run pre-commit install
test:
	pytest tests/ --cov=$(PROJECT_NAME) --cov-report=term-missing
