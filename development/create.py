import os

def create_directory_structure():
    directories = [
        "scs_implementation/src",
        "scs_implementation/tests",
        "scs_implementation/benchmarks",
        "scs_implementation/experiments",
        "pdlp_implementation/src",
        "pdlp_implementation/tests",
        "pdlp_implementation/benchmarks",
        "pdlp_implementation/experiments",
        "utils",
        "notebooks"
    ]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

def create_files():
    files = {
        "scs_implementation/README.md": "# SCS Implementation\n\nImplementation details for Splitting Conic Solver (SCS).\n",
        "scs_implementation/requirements.txt": "numpy\nscipy\nmatplotlib\ncvxpy\ncupy\npytest\n",
        "scs_implementation/setup.py": "from setuptools import setup, find_packages\n\nsetup(name='scs_implementation', version='0.1', packages=find_packages())\n",
        "pdlp_implementation/README.md": "# PDLP Implementation\n\nImplementation details for Primal-Dual Linear Programming (PDLP).\n",
        "pdlp_implementation/requirements.txt": "numpy\nscipy\nmatplotlib\ncvxpy\ncupy\npytest\n",
        "pdlp_implementation/setup.py": "from setuptools import setup, find_packages\n\nsetup(name='pdlp_implementation', version='0.1', packages=find_packages())\n",
        "utils/matrix_operations.py": "# Common matrix operations for solvers\n",
        "utils/profiling.py": "# Performance profiling tools\n",
        "utils/plotting.py": "# Visualization tools\n",
        "utils/README.md": "# Utilities\n\nCommon utilities used in both SCS and PDLP implementations.\n",
        "notebooks/scs_testing.ipynb": "# Jupyter Notebook for testing SCS\n",
        "notebooks/pdlp_testing.ipynb": "# Jupyter Notebook for testing PDLP\n",
        "environment.yml": "name: phd-research\nchannels:\n  - conda-forge\ndependencies:\n  - python=3.10\n  - numpy\n  - scipy\n  - matplotlib\n  - cvxpy\n  - cupy\n  - pytest\n  - jupyter\n",
        "requirements.txt": "numpy\nscipy\nmatplotlib\ncvxpy\ncupy\npytest\njupyter\n",
        "Makefile": "setup:\n\tpip install -r requirements.txt\n\n"
                    "test-scs:\n\tpytest scs_implementation/tests/\n\n"
                    "test-pdlp:\n\tpytest pdlp_implementation/tests/\n\n"
                    "benchmark:\n\tpython benchmarks/run_all.py\n\n"
                    "clean:\n\trm -rf __pycache__\n",
        "README.md": "# Development Space for SCS and PDLP\n\nThis directory contains implementations and experiments for both the Splitting Conic Solver (SCS) and the Primal-Dual Linear Programming (PDLP) solver.\n\n## Structure\n- **scs_implementation/**: Implementation, tests, and benchmarks for SCS.\n- **pdlp_implementation/**: Implementation, tests, and benchmarks for PDLP.\n- **utils/**: Shared utility functions for both solvers.\n- **notebooks/**: Jupyter notebooks for interactive development.\n- **requirements.txt**: Dependencies for the project.\n- **Makefile**: Automation for setup, testing, and benchmarking.\n\n## Setup\nRun:\n```sh\npip install -r requirements.txt\n```\n\n## Compilation\nRun:\n```sh\nmake test-scs\nmake test-pdlp\nmake benchmark\n```\n"
    }
    for file, content in files.items():
        if os.path.dirname(file):
            os.makedirs(os.path.dirname(file), exist_ok=True)
        with open(file, 'w') as f:
            f.write(content)

def main():
    create_directory_structure()
    create_files()
    print("Development directory structure and files created successfully!")

if __name__ == "__main__":
    main()
