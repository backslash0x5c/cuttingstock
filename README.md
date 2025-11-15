# Rebar Cutting Stock Optimization Application

A Streamlit-based web application that optimizes rebar cutting patterns to minimize waste. It uses depth-first search (DFS) algorithms and integer linear programming (ILP) to generate optimal cutting plans.

## Key Features

### 🎯 Optimization Features
- **Multi-material optimization**: Optimization using all available rod lengths
- **Single-material optimization**: Optimization using only specific rod lengths
- **Reuse scrap materials**: Prioritize using existing scrap to reduce new material usage

### 📊 Input Methods
1. **XLSX File Upload**: Automatically load data from Excel cutting specification sheets
2. **Manual Input**: Directly input required cut lengths and quantities
3. **Reuse Scrap CSV Input**: Import existing scrap data via CSV file (optional)

### 📈 Results Display
- Yield rate calculation (with/without reuse)
- Detailed cutting pattern display
- Reusable scrap list (above threshold)
- Execution history storage and display
- Processing time and combination count statistics

### 💾 Download Features
- **Reusable Scrap List CSV**: Combines new scraps with unused reuse scraps
- **Optimization Results Excel**: Includes cutting patterns, summary tables, and sheets

## Usage

### Starting the Application

```bash
streamlit run cutting_optimizer_streamlit_v2.py
```

### Basic Workflow

1. **Select Rebar Diameter**: Choose from D10, D13, D16, D19, D22
2. **Input Reuse Scraps (Optional)**: Upload CSV file
   ```csv
   Length (mm),Quantity
   405,3
   1000,1
   1915,2
   ```
3. **Input Cutting Instructions**: Specify required cuts via XLSX file or manual input
4. **Set Scrap Threshold**: Set minimum length for reusable scraps (default: 400mm)
5. **Execute Optimization**: Click button to start calculation
6. **Review and Download Results**: Check results in tabs for each material and download as needed

## Dependencies

```bash
pip install streamlit pandas pulp openpyxl
```

- `streamlit`: Web interface
- `pandas`: Data processing and CSV/Excel operations
- `pulp`: Integer linear programming solver
- `openpyxl`: Excel file read/write

## Supported Rebar Diameters and Available Rod Lengths

```python
BASE_PATTERNS = {
    'D10': [4000, 4500, 5500, 6000],
    'D13': [3000, 4000, 4500, 5500, 6000, 7500],
    'D16': [4000, 4500, 5500, 6000, 7000],
    'D19': [3500, 4000, 4500, 5500, 6000],
    'D22': [4000, 4500, 5500, 6000]
}
```

## Algorithms

### 1. Depth-First Search (DFS)
- Generates all possible cutting combinations within available material lengths
- Prunes invalid combinations (current_sum > max_sum)
- Reuses calculated combinations for efficiency

### 2. Integer Linear Programming (ILP)
- Solves optimization problem using PuLP library
- Objective function: Minimize total loss (scrap)
- Constraints:
  - Demand constraints: Satisfy required quantities for each cut length
  - Reuse scrap constraints: Do not exceed available quantities for each reuse material

### 3. Reuse Scrap Processing
- Generates cutting patterns from input reuse scraps
- Prioritizes usage during optimization (with quantity constraints)
- Combines unused scraps with new scraps for download

## Data Flow

```
[Input]
  ├─ Cutting Instructions (XLSX/Manual)
  └─ Reuse Scraps (CSV)
      ↓
[Processing]
  ├─ Combination Generation (DFS)
  ├─ Optimization Calculation (ILP)
  └─ Result Aggregation
      ↓
[Output]
  ├─ Cutting Pattern Display
  ├─ Yield Rate Calculation
  ├─ Reusable Scrap List (CSV)
  └─ Optimization Results (Excel)
```

## Reuse Scrap Circulation

### Input Example
```csv
Length (mm),Quantity
405,3
1000,1
1915,2
```

### Optimization Results
- 405mm: 1 piece used
- 1915mm: 2 pieces used
- New scraps: 610mm×5, 1020mm×2, 830mm×3

### Download CSV (Combined)
```csv
Length (mm),Quantity
1020,2
1000,1
830,3
610,5
405,2
```

## Performance Settings

- **Optimization Time Limit**: 10-3600 seconds (default: 120 seconds)
- **Combination Generation**: May take time for large-scale problems
- **Result Storage**: Saved in session state, updates display without recalculation when threshold changes

## File Structure

```
cuttingstock/
├── cutting_optimizer_streamlit_v2.py  # Main application
├── README.md                          # This file (English)
├── README_ja.md                       # Japanese README
└── test_scrap.csv                     # Test scrap data
```

## Output Details

### Cutting Pattern Table
- ID, usage count, loss, base material length, cutting pattern

### Output Summary Table
- Total quantities for each cut length
- Formula-based sum row

### Cutting Instruction Summary Table (XLSX upload)
- Cutting instruction data per sheet
- Sum row and difference calculations

### Summary Sheet
- Diameter, yield rate, total material length, processing time
- Difference display between output results and cutting instructions

## Notes

- CSV files should be in 2-column format: "Length (mm),Quantity"
- Scraps of the same length are automatically aggregated
- Reuse scrap usage is optional; normal optimization runs without upload
- For large-scale problems, recommend setting longer time limits

## License

This project is a tool for rebar cutting optimization.
