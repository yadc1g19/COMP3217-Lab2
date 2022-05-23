PYTHON = python

.PHONY = help setup run

.DEFAULT_GOAL = run

help:
	@echo To run the project type 'make'-
	
run:
	mkdir graph
	${PYTHON} cw_main.py
	
setup: 
	pip install -r requirements.txt