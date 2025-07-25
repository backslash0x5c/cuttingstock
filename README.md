This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.
- English ver. -> CLAUDE.md
- Japanese.ver -> CLAUDE_ja.md

## Project Overview

This is a cutting stock optimization project for rebar (reinforcing bar) cutting. The system solves the classic cutting stock problem to minimize material waste when cutting rebar pieces to required lengths from available stock rods.

## Quick Start

### Prerequisites
Install required dependencies:
```bash
pip install pandas pulp streamlit tqdm
```

### Basic Usage
1. **Run command-line optimization**:
   ```bash
   python cutting_optimizer_ver1.py
   ```
   This uses the default D19 diameter with data from `task.csv`.

2. **Run web interface** (most user-friendly):
   ```bash
   streamlit run streamlit_cutting_app.py
   ```
   Open browser to http://localhost:8501 for interactive interface.

3. **Change rebar diameter**: 
   Edit the `diameter` variable in the script (line 139 in cutting_optimizer_ver1.py):
   ```python
   diameter = 'D10'  # Options: D10, D13, D16, D19, D22
   ```

### Expected Output
The optimizer will display:
- Cutting patterns: `rod_length = [cut1, cut2, ...] [waste]`
- Processing time, total loss, material usage, and yield rate
- Validation that all requirements are met

## Core Architecture

### Algorithm Versions
- **cutting_optimizer_ver1.py**: Main optimization engine using backtracking and linear programming
- **cutting_optimizer_ver2.py**: Incomplete column generation & branch-price implementation
- **subset_sum_solver_ver1.py**: Backtracking algorithm for finding combinations
- **subset_sum_solver_ver2.py**: Optimized version of subset sum solver
- **cutting_comb_ver1-4.py**: Various cutting combination generation approaches

### Key Components

1. **Data Input System** (`read_csv.py:46-72`)
   - Reads rebar specifications from `base_pattern.csv` and cutting requirements from `task.csv`
   - Returns available rod lengths and required cut specifications for each diameter

2. **Combination Generation** (`cutting_optimizer_ver1.py:16-96`)
   - Uses backtracking to find all valid cutting combinations
   - Filters combinations that fit within available rod lengths
   - Sorts by material efficiency (lowest waste first)

3. **Linear Programming Optimization** (`cutting_optimizer_ver1.py:98-136`)
   - Uses PuLP library to solve the cutting stock problem as integer linear program
   - Minimizes total material loss while meeting all cutting requirements

4. **Web Interface** (`streamlit_cutting_app.py`)
   - Provides interactive web UI for inputting requirements and viewing results
   - Supports both CSV upload and manual input
   - Shows cutting patterns, material utilization, and downloadable results

## Common Development Tasks

### Running the Optimization
```bash
python cutting_optimizer_ver1.py
```

### Running the Streamlit Web App
```bash
streamlit run streamlit_cutting_app.py
```

### Testing Different Diameters
Modify the diameter variable in the main scripts (D10, D13, D16, D19, D22 are supported).

## Dependencies

The project requires:
- `pandas`: CSV data processing
- `pulp`: Linear programming optimization
- `streamlit`: Web interface
- `tqdm`: Progress bars (cutting_comb_ver4.py)

Install with:
```bash
pip install pandas pulp streamlit tqdm
```

## Data Format

### base_pattern.csv
Defines available rod lengths for each rebar diameter:
```csv
,D10,D13,D16,D19,D22
"4,000",◯,◯,◯,◯,◯
"4,500",◯,◯,◯,◯,◯
```

### task.csv  
Specifies required cutting quantities:
```csv
,鉄筋経,D10,D13,D16,D19,D22
1,4495,1,,,2,
2,3585,2,,,10,
```

## Key Algorithms

### Backtracking Combination Generation (`subset_sum_solver_ver2.py:8-45`)
Generates all unique combinations of required cuts that fit within available rod lengths using recursive backtracking with pruning.

### Linear Programming Formulation (`cutting_optimizer_ver1.py:98-136`)
- **Variables**: Number of times each cutting pattern is used
- **Objective**: Minimize total material waste
- **Constraints**: Meet exact demand for each required length

### Penalty-Based Heuristic (`cutting_comb_ver4.py:71-216`)
Alternative approach using penalty parameters to balance waste minimization vs. over-production when exact solutions aren't feasible.

## Testing and Validation

Each optimizer validates results by verifying that the total cuts produced exactly match the required quantities. Results include:
- Total material length used
- Material waste (loss)
- Yield rate percentage
- Detailed cutting patterns

## Performance Considerations

- Combination generation is exponential in problem size
- Use `cutting_optimizer_ver1.py` for production - it's the most stable implementation
- `cutting_comb_ver4.py` includes parameter tuning for difficult instances
- Processing time varies significantly with problem complexity
