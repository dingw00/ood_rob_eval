# define the name of the virtual environment directory
VENV := .venv
CONFIG := config.yaml
SHELL := /bin/bash

# Install virtualenv, create a new virtual environment and install dependencies
build: requirements.txt
	pip install virtualenv
	virtualenv $(VENV)
	./$(VENV)/bin/python -m pip install --upgrade pip
	./$(VENV)/bin/pip install -r requirements.txt

# OoD robustness test scripts
ood_test: build $(CONFIG)
	./$(VENV)/bin/python ood_test.py --cfg $(CONFIG)

ood_eval: build $(CONFIG)
	./$(VENV)/bin/python ood_eval.py --cfg $(CONFIG)
	
randomized_smoothing_test: build $(CONFIG)
	./$(VENV)/bin/python randomized_smoothing_test.py --cfg $(CONFIG)

config: build
	./$(VENV)/bin/python run_gui.py
run: ood_test
eval: $(CONFIG)
	sed -i 's/eval_severity: True/eval_severity: False/g' ./$(CONFIG)
	make ood_eval
eval_severity:  # TODO
	sed -i 's/eval_severity: False/eval_severity: True/g' ./$(CONFIG)
	sed -i 's/severity: all/severity: 1/g' ./$(CONFIG)
	make ood_eval
	sed -i 's/severity: 1/severity: 2/g' ./$(CONFIG)
	make ood_eval
	sed -i 's/severity: 2/severity: 3/g' ./$(CONFIG)
	make ood_eval
	sed -i 's/severity: 3/severity: 4/g' ./$(CONFIG)
	make ood_eval
	sed -i 's/severity: 4/severity: 5/g' ./$(CONFIG)
	make ood_eval
	sed -i 's/severity: 5/severity: all/g' ./$(CONFIG)
clean:
	rm -rf $(VENV)
	find . -type d -name '__pycache__' -exec rm -rf {} +
# git clean -Xdf

.PHONY: all run clean
