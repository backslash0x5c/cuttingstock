# Cutting Stock Optimization System

## Project Overview

This is a rebar cutting stock optimization system that solves the cutting stock problem for steel reinforcement bars. The system calculates optimal cutting patterns to minimize material waste when cutting rebars of various lengths from standard stock lengths.

## Core Architecture

### Main Components

1. **Cutting Optimizers** (`cutting_optimizer_ver2.py`): Solve the cutting stock problem using integer linear programming
2. **Data Readers** (`read_xlsx.py`): Parse Excel files containing cutting requirements
3. **Web Interface** (`cutting_optimizer_streamlit.py`): Streamlit-based user interface

### Key Algorithms

- **Depth-First Search (DFS)**: Used to generate all valid cutting combinations within material constraints
- **Integer Linear Programming (ILP)**: Optimizes cutting patterns to minimize waste using PuLP library
- **Column Generation**: Experimental approach in `cutting_optimizer_dual.py` (incomplete)

## Standard Rebar Specifications

The system supports five rebar diameters with predefined available stock lengths:

```python
BASE_PATTERNS = {
    'D10': [4000, 4500, 5500, 6000],
    'D13': [3000, 4000, 4500, 5500, 6000, 7500],
    'D16': [4000, 4500, 5500, 6000, 7000],
    'D19': [3500, 4000, 4500, 5500, 6000],
    'D22': [4000, 4500, 5500, 6000]
}
```

## Common Development Tasks

### Required libraries
- `streamlit`: Web interface
- `pulp`: Integer linear programming solver
- `openpyxl`: Excel file processing
- `pandas`: Data processing

### Running the Streamlit Application
```bash
streamlit run cutting_optimizer_streamlit.py
```

### Running Command-Line Optimizers
```bash
python cutting_optimizer_ver2.py
```

### Testing with Different Diameters
Modify the `diameter` variable in the main functions (typically 'D10', 'D13', 'D16', 'D19', 'D22').

## Data Input Formats

### Excel Files
The system reads cutting requirements from Excel files with the following structure:
- Sheet contains "鉄筋径" (rebar diameter) headers
- Length values in mm in the first data column
- Quantity values in corresponding diameter columns

## Algorithm Flow

1. **Data Input**: Read cutting requirements from Excel
2. **Combination Generation**: Use DFS to find all valid cutting patterns within stock length constraints
3. **Optimization**: Solve ILP to minimize total waste while meeting demand
4. **Output**: Display cutting instructions with waste calculations

## Performance Considerations

- Combination generation can be computationally expensive for large problems
- Default optimization timeout is 60 seconds
- Results include processing time, yield rate, and total material usage statistics

## File Dependencies

- `cutting_optimizer_streamlit.py` → main web interface
- `cutting_optimizer_ver2.py` → command-line optimizer with Excel integration
- `read_xlsx.py` → Excel file parser with multi-sheet support
- Excel files: `required_cuts.xlsx` → input data

## Testing

The system includes sample result files in the `result/` directory for validation and comparison of optimization results across different rebar diameters.