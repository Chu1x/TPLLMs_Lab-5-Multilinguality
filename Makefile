LANGUAGES := fr es en ar zh
OUTPUT_DIR := outputs/lab5
CONTROLLED_OUTPUT_DIR := outputs/lab5_controlled
CONTROLLED_ARGS := \
	--languages $(LANGUAGES) \
	--data-dir data \
	--output-dir $(CONTROLLED_OUTPUT_DIR) \
	--max-train-samples 1000 \
	--max-dev-samples 300 \
	--max-test-samples 300 \
	--max-length 256 \
	--epochs 2 \
	--batch-size 4 \
	--resume

.PHONY: install data warmup stats test smoke pilot controlled controlled-fr controlled-es controlled-en controlled-ar controlled-zh full snippets controlled-snippets plots controlled-plots report-plots-png check-report report-pdf

install:
	python3 -m pip install -r requirements.txt

data:
	python3 scripts/download_ud_data.py --include-sequoia-warmup

warmup:
	python3 scripts/inspect_ud_warmup.py data/sequoia/test.conllu

stats:
	python3 scripts/run_lab5_experiments.py \
		--languages $(LANGUAGES) \
		--data-dir data \
		--output-dir $(OUTPUT_DIR) \
		--epochs 3 \
		--batch-size 4 \
		--use-cpu \
		--resume \
		--stats-only

test:
	python3 -m pytest test_lab5_multilinguality.py

smoke:
	python3 scripts/run_lab5_experiments.py \
		--languages $(LANGUAGES) \
		--train-languages fr \
		--data-dir data \
		--output-dir outputs/lab5_smoke \
		--max-train-samples 8 \
		--max-dev-samples 8 \
		--max-test-samples 8 \
		--epochs 1 \
		--batch-size 2

pilot:
	python3 scripts/run_lab5_experiments.py \
		--languages $(LANGUAGES) \
		--data-dir data \
		--output-dir outputs/lab5_pilot \
		--max-train-samples 200 \
		--max-dev-samples 100 \
		--max-test-samples 100 \
		--epochs 1 \
		--batch-size 8 \
		--resume

controlled:
	python3 scripts/run_lab5_experiments.py $(CONTROLLED_ARGS)

controlled-fr:
	python3 scripts/run_lab5_experiments.py $(CONTROLLED_ARGS) --train-languages fr

controlled-es:
	python3 scripts/run_lab5_experiments.py $(CONTROLLED_ARGS) --train-languages es

controlled-en:
	python3 scripts/run_lab5_experiments.py $(CONTROLLED_ARGS) --train-languages en

controlled-ar:
	python3 scripts/run_lab5_experiments.py $(CONTROLLED_ARGS) --train-languages ar --use-cpu

controlled-zh:
	python3 scripts/run_lab5_experiments.py $(CONTROLLED_ARGS) --train-languages zh

full:
	python3 scripts/run_lab5_experiments.py \
		--languages $(LANGUAGES) \
		--data-dir data \
		--output-dir $(OUTPUT_DIR) \
		--epochs 3 \
		--batch-size 4 \
		--max-length 256 \
		--use-cpu \
		--resume

snippets:
	python3 scripts/report_snippets.py --output-dir $(OUTPUT_DIR)

controlled-snippets:
	python3 scripts/report_snippets.py --output-dir $(CONTROLLED_OUTPUT_DIR)

plots:
	python3 scripts/plot_lab5_outputs.py \
		--output-dir $(OUTPUT_DIR) \
		--warmup-dir $(OUTPUT_DIR)/warmup \
		--plots-dir $(OUTPUT_DIR)/plots

controlled-plots:
	python3 scripts/plot_lab5_outputs.py \
		--output-dir $(CONTROLLED_OUTPUT_DIR) \
		--warmup-dir $(OUTPUT_DIR)/warmup \
		--plots-dir $(CONTROLLED_OUTPUT_DIR)/plots

report-plots-png:
	MPLCONFIGDIR=outputs/matplotlib-cache python3 scripts/plot_lab5_png_outputs.py \
		--output-dir $(OUTPUT_DIR) \
		--warmup-dir $(OUTPUT_DIR)/warmup \
		--plots-dir $(OUTPUT_DIR)/plots
	MPLCONFIGDIR=outputs/matplotlib-cache python3 scripts/plot_lab5_png_outputs.py \
		--output-dir $(CONTROLLED_OUTPUT_DIR) \
		--warmup-dir $(OUTPUT_DIR)/warmup \
		--plots-dir $(CONTROLLED_OUTPUT_DIR)/plots

check-report:
	python3 scripts/check_report.py lab5_report.md lab5_answers.md

report-pdf: report-plots-png
	pandoc lab5_report.md \
		--from gfm \
		--pdf-engine=xelatex \
		--resource-path=. \
		-V geometry:margin=1in \
		-V colorlinks=true \
		-o lab5_report.pdf
