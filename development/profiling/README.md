# Profiling Solvers

## Overview
This directory contains scripts for profiling different linear programming solvers, including:
- **SciPy linprog (Baseline)**
- **SCS Solver (GPU-Accelerated)**
- **PDLP Solver (GPU-Accelerated)**

Profiling helps measure:
- **Execution time**: How long each solver takes to solve a given problem.
- **Memory usage**: The peak memory consumption during execution.
- **Iteration counts** (if applicable): How many iterations each solver needs to converge.

## Running the Profiling Script
To execute the profiling, run the following command:
```bash
python profiling/profile_solvers.py
```

This will output performance metrics for each solver.

## Profiling Metrics
For each solver, the script will print:
```
--- Profiling <Solver Name> ---
Execution Time: X.XXXXXX sec
Memory Usage: X.XX MB
```

## Expanding Profiling Capabilities
### 1. GPU Profiling
- Enable memory tracking for CUDA kernels.
- Measure execution time per kernel.

### 2. Large-Scale Benchmarks
- Run solvers on larger linear programming instances.
- Compare performance at different problem scales.

### 3. Logging Results
- Store profiling outputs in a log file for future comparison.

## Next Steps
If you’d like to expand profiling, modify `profile_solvers.py` or add new benchmarking functions!
