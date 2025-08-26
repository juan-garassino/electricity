# Run full pipeline
run_full:
	python electricity/main.py --input_csv complete_dataset.csv --horizon 1 --epochs 100

# Quick experiment (no CLI args)
run_quick:
	python electricity/main.py

# Keep leaky columns for prediction experiments
run_keep_leaky:
	python electricity/main.py --input_csv data.csv --keep_leaky --epochs 50

# Skip preprocessing if already done
run_skip_preprocessing:
	python electricity/main.py --input_csv data.csv --skip_preprocessing