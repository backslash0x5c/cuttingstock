## Project Overview

This is a rebar cutting stock optimization system that solves the cutting stock problem for steel reinforcement bars using depth-first search algorithms and integer linear programming.

## Commands

### Running the Application
```bash
# Run the Streamlit web interface
streamlit run cutting_optimizer_streamlit.py

# Run command-line optimizer
python cutting_optimizer_ver2.py
```

### Dependencies
The system requires these Python packages:
- `streamlit`: Web interface
- `pulp`: Integer linear programming solver
- `openpyxl`: Excel file processing
- `pandas`: Data processing

### Testing with Different Diameters
Modify the `diameter` variable in the main functions (typically 'D10', 'D13', 'D16', 'D19', 'D22').

## Architecture

### Core Components

1. **Optimizers**:
   - `cutting_optimizer_streamlit.py`: Main web interface with integrated optimization
   - `cutting_optimizer_ver2.py`: Command-line version with Excel integration

2. **Data Processing**:
   - `read_xlsx.py`: Excel parser for cutting requirements with multi-sheet support
   - `result_sheet.py`: Results output to Excel format

### Key Algorithms

- **Depth-First Search (DFS)**: Generates all valid cutting combinations within material constraints using recursive exploration
- **Integer Linear Programming (ILP)**: Optimizes cutting patterns using PuLP library to minimize waste

### Rebar Specifications

The system supports five standard rebar diameters with predefined stock lengths:

```python
BASE_PATTERNS = {
    'D10': [4000, 4500, 5500, 6000],
    'D13': [3000, 4000, 4500, 5500, 6000, 7500],
    'D16': [4000, 4500, 5500, 6000, 7000],
    'D19': [3500, 4000, 4500, 5500, 6000],
    'D22': [4000, 4500, 5500, 6000]
}
```

### Data Flow

1. **Input**: Excel files with cutting requirements (`required_cuts.xlsx`)
2. **Processing**: Extract rebar diameter data using `find_cell_position()` and `get_diameter_column_index()`
3. **Combination Generation**: Use DFS to find valid cutting patterns within stock constraints
4. **Optimization**: Solve ILP problem with 120-second timeout to minimize waste
5. **Output**: Display cutting instructions with yield rates and material usage statistics

### File Structure

- **Input files**: Excel files with "鉄筋径" (rebar diameter) headers
- **Result files**: Located in `result/` directory with processing results and comparisons

### Performance Considerations

- Combination generation is computationally expensive for large problems
- Default optimization timeout: 120 seconds
- Results include processing time and yield rate calculations
- DFS uses pruning to avoid exploring invalid branches when current_sum > max_sum