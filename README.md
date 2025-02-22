# PhD Research, Writing, and Development Plan

## Overview
**Topic:** GPU-Accelerated PDHG for Decentralized Swarm Navigation  
**Goal:** Develop GPU-accelerated decentralized algorithms for efficient and effective decision-making in autonomous swarms, focusing on trajectory planning using a GPU-based Primal-Dual Hybrid Gradient (PDHG) solver.  
**Target Defense:** Summer 2027  

Each chapter of the dissertation will be structured as a **stand-alone, publishable journal article**. The research will begin with a literature review on **GPU-accelerated linear programming solvers** and culminate in the development of **decentralized swarm navigation algorithms**.

---
## Research Plan
### **Phase 1: Literature Review (Spring 2024 – Fall 2024)**
- Study GPU-accelerated linear programming solvers.
- Investigate first-order vs. second-order solvers and preconditioning techniques.
- Review PDHG for large-scale optimization and its stability challenges.
- Explore swarm robotics and decentralized decision-making strategies.

**Deliverable:** Survey paper on GPU-accelerated first-order methods for decentralized decision-making.

### **Phase 2: GPU-Accelerated PDHG for Noisy LPs (Spring 2025 – Fall 2025)**
- Implement CUDA-optimized PDHG solver.
- Explore adaptive step-size selection and preconditioning techniques.
- Benchmark against CPU-based solvers (CPLEX/HiGHS).

**Deliverable:** Journal article on GPU-accelerated PDHG and its application to noisy LPs.

### **Phase 3: Decentralized PDHG for Multi-Agent Navigation (Spring 2026 – Fall 2026)**
- Implement distributed PDHG for multi-agent motion planning.
- Develop decentralized collision avoidance and swarm coordination strategies.

**Deliverable:** Journal article on decentralized PDHG for real-time swarm trajectory planning.

### **Phase 4: Real-World Implementation (Spring 2027 – Summer 2027)**
- Build FPV and autonomous drones with Jetson Nano.
- Integrate PDHG solver for real-time execution.
- Conduct field experiments and evaluate swarm performance.

**Deliverable:** Journal article on real-world implementation and performance evaluation.

---
## Dissertation Writing Plan
| Chapter | Tentative Title |
|---------|----------------|
| 1 | Introduction to GPU-Accelerated Optimization for Swarm Decision-Making |
| 2 | Review of GPU-Accelerated Linear Programming Solvers |
| 3 | GPU-Accelerated PDHG for Noisy Linear Programs |
| 4 | Decentralized PDHG for Multi-Agent Navigation |
| 5 | Real-World Implementation of GPU-Accelerated Swarm Navigation |
| 6 | Conclusion and Future Directions |

---
## Development and Implementation Plan
| Phase | Task | Timeline |
|-------|------|----------|
| **Simulation Development** | Implement CUDA PDHG solver | Spring 2025 |
|  | Benchmark vs. CPU solvers | Fall 2025 |
| **Swarm Algorithm Development** | Develop distributed PDHG | Spring 2026 |
|  | Implement local communication | Fall 2026 |
| **Hardware Integration** | Assemble FPV drones | Spring 2027 |
|  | Deploy PDHG on Jetson Nano | Summer 2027 |
| **Testing and Evaluation** | Simulated swarm tests | Spring 2027 |
|  | Field experiments | Summer 2027 |

---
## Project Directory Structure
```
📦 phd-research
├── 📂 dissertation          # LaTeX files for the dissertation
│   ├── 📂 chapters         # Individual chapter files
│   │   ├── 01_introduction.tex
│   │   ├── 02_literature_review.tex
│   │   ├── 03_pdhg_solver.tex
│   │   ├── 04_decentralized_pdhg.tex
│   │   ├── 05_realworld_tests.tex
│   │   ├── 06_conclusion.tex
│   ├── 📂 appendices       # Additional appendices
│   │   ├── appendix_1.tex
│   ├── 📂 additional_resources  # Supplemental files
│   │   ├── classified_chapter.tex
│   │   ├── classified_meta.bib
│   │   ├── classified_thesis.tex
│   │   ├── exec_sum_with_refs.tex
│   │   ├── supplementals.tex
│   │   ├── _README.txt
│   ├── 📂 figures          # Figures, plots, and diagrams
│   │   ├── dragon.jpg
│   │   ├── npslogo.pdf
│   │   ├── bash-long.sh
│   │   ├── Python-short.py
│   ├── 📂 tables           # Any tables used in LaTeX
│   ├── 📂 frontmatter      # Front matter and setup files
│   │   ├── acronyms.tex
│   │   ├── front-matter.tex
│   │   ├── SETUP.txt
│   ├── main.tex            # Main dissertation file
│   ├── references.bib      # BibTeX references
│   ├── nps-ieee.bst        # Bibliography style file
│   ├── nps-informs.bst     # Bibliography style file
│   ├── npsreport2018.cls   # LaTeX class file
│   ├── nps_sf298.sty       # Formatting style file
│
├── 📂 papers                # Collection of relevant PDFs
│
├── 📂 development           # Code development workspace
│   ├── 📂 notebooks         # Jupyter notebooks for solver testing
│   │   ├── pdlp_testing.ipynb
│   │   ├── scs_testing.ipynb
│   ├── 📂 scs_implementation  # GPU-accelerated SCS solver
│   │   ├── 📂 benchmarks     # Benchmarking scripts
│   │   ├── 📂 experiments    # Experimental setups
│   │   ├── 📂 src           # Solver implementation
│   │   │   ├── gpu_kernels.py    # Custom CUDA kernels
│   │   │   ├── scs_solver.py     # Main SCS solver
│   │   │   ├── utils.py          # Utility functions
│   │   ├── 📂 tests         # Functional tests
│   │   │   ├── test_baseline.py  # Baseline solver tests
│   │   │   ├── test_scs_solver.py  # SCS solver tests
│   │   ├── README.md
│   │   ├── requirements.txt
│   │   ├── setup.py
│   ├── 📂 pdlp_implementation # GPU-accelerated PDLP solver
│   │   ├── 📂 benchmarks     # Benchmarking scripts
│   │   ├── 📂 experiments    # Experimental setups
│   │   ├── 📂 src           # Solver implementation
│   │   │   ├── gpu_kernels.py    # Custom CUDA kernels
│   │   │   ├── pdlp_solver.py    # Main PDLP solver
│   │   │   ├── utils.py          # Utility functions
│   │   ├── 📂 tests         # Functional tests
│   │   │   ├── test_baseline.py  # Baseline solver tests
│   │   │   ├── test_pdlp_solver.py  # PDLP solver tests
│   │   ├── README.md
│   │   ├── requirements.txt
│   │   ├── setup.py
│   ├── 📂 test_data         # Transportation problem generator
│   │   ├── 📂 data          # Generated datasets
│   │   ├── generate_transportation.py  # Problem generator
│   │   ├── utils.py              # Utility functions
│   ├── 📂 profiling        # Performance benchmarking
│   │   ├── profile_solvers.py  # Main profiling script
│   │   ├── README.md           # Overview of profiling methods
│   ├── 📂 utils             # Shared utility functions
│   │   ├── matrix_operations.py
│   │   ├── plotting.py
│   │   ├── profiling.py
│   │   ├── README.md
│   ├── create.py            # Script to initialize project structure
│   ├── environment.yml       # Conda environment configuration
│   ├── Makefile              # Automation for setup, testing, and benchmarking
│   ├── README.md             # Project overview
│   ├── requirements.txt       # Python dependencies
│
├── 📂 tools                 # Utility scripts
│   ├── compile.sh          # Compile LaTeX + BibTeX
│   ├── fetch_bibtex.py     # Fetch BibTeX from DOI
│
├── 📂 docs                  # Notes, meeting logs, planning
│
├── .gitignore               # Ignore logs, compiled files
├── README.md                # Project overview
└── LICENSE                  # Open-source license (if applicable)
```


---
## Target Conferences and Journals
- **Optimization & Algorithms:** SIAM Journal on Optimization, Mathematical Programming, JMLR.
- **Robotics & Swarm Systems:** IEEE Transactions on Robotics, Autonomous Robots Journal, Swarm Intelligence Journal.
- **GPU Computing:** Journal of Parallel and Distributed Computing, ACM Transactions on Graphics.

---
## Timeline Summary
| Phase | Research Focus | Publication | Development |
|-------|---------------|----------------------|----------------------|
| **Spring 2024 – Fall 2024** | Literature Review | Survey Paper | - |
| **Spring 2025 – Fall 2025** | GPU-Accelerated PDHG | Journal Paper 1 | CUDA Solver |
| **Spring 2026 – Fall 2026** | Decentralized Swarm Navigation | Journal Paper 2 | Multi-Agent PDHG |
| **Spring 2027 – Summer 2027** | Real-World Drone Implementation | Journal Paper 3 | Hardware Integration |
| **Summer 2027** | Defense | Final Dissertation Submission | - |

---
## Next Steps
- [ ] Set up a literature review database.
- [ ] Develop an early CUDA prototype for the PDHG solver.
- [ ] Establish an experimental setup for testing multi-agent PDHG.

---
This repository will track all progress, source code, and experimental results related to the research. Let me know if you need any modifications! 🚀
