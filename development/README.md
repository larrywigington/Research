# Development Space for SCS and PDLP

This directory contains implementations and experiments for both the Splitting Conic Solver (SCS) and the Primal-Dual Linear Programming (PDLP) solver.

## Structure
- **scs_implementation/**: Implementation, tests, and benchmarks for SCS.
- **pdlp_implementation/**: Implementation, tests, and benchmarks for PDLP.
- **utils/**: Shared utility functions for both solvers.
- **notebooks/**: Jupyter notebooks for interactive development.
- **requirements.txt**: Dependencies for the project.
- **Makefile**: Automation for setup, testing, and benchmarking.

## Setup
Run:
```sh
pip install -r requirements.txt
```

## Compilation
Run:
```sh
make test-scs
make test-pdlp
make benchmark
```
