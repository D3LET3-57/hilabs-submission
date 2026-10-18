.PHONY: install predict predict-kev engine test demo reports clean

install:
	pip install -e ".[dense,gpu]"

# Reproduce the submitted predictions (single command, no server required)
predict:
	python predict.py --input data/test_input.csv --output predictions/test_predictions.csv

# Live two-stage run (requires KEV_ENDPOINT / a running Kev server)
predict-kev:
	python predict.py --input data/test_input.csv --output predictions/test_predictions_live.csv \
		--mode kev --nperm 8

# Deterministic engine baseline (offline; needs retrieval artifacts)
engine:
	python predict.py --input data/test_input.csv --output predictions/test_predictions_engine.csv --mode engine

test:
	pytest -q

demo:
	python demo/inspect.py --nperm 8

reports:
	cd reports && tectonic -X compile report_project.tex && tectonic -X compile report_architecture.tex && tectonic -X compile deck_final.tex

clean:
	rm -f predictions/*_live.csv
