# Reproduction Entrypoints

All scripts are plain Python entrypoints intended to be run from the repository root.

- `prepare_data.py` — prints the expected dataset layout and creates local data directories (`data/cifar10/`, ...).
- `prepare_checkpoints.py` — prints expected checkpoint locations and reports which files are missing.
- `run_main_experiments.py` — orchestrates the retained main results: proxy attacks (exp1), topology tampering (exp2), quantization stability (exp3), and system overhead (exp4).
- `run_ablations.py` — orchestrates the retained key ablations: measurement binding, freshness, activation chains, chain state, and signature mode (exp6 family).
- `generate_figures.py` — runs the retained figure builder and summary-table generator; outputs go to `results/`.

Typical full run:

```bash
python scripts/prepare_data.py
python scripts/prepare_checkpoints.py
python scripts/run_main_experiments.py
python scripts/run_ablations.py
python scripts/generate_figures.py
```

Note: some experiments are practical only with GPU acceleration, and checkpoint
availability determines which model variants can be reproduced exactly.
