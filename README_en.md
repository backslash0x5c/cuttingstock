# Rebar Cutting Stock Optimization Application

A Streamlit-based web application that optimizes rebar cutting patterns to minimize waste. It uses depth-first search (DFS) algorithms and integer linear programming (ILP) to generate optimal cutting plans.

## Key Features

### 🎯 Optimization Features
- **Multi-diameter optimization**: Simultaneously optimize for all rebar diameters (D10, D13, D16, D19, D22)
- **Multi-material optimization**: Optimization using all available rod lengths for each diameter
- **Flexible material selection**: Choose specific rod lengths for each diameter via checkboxes
- **Reuse scrap materials**: Prioritize using existing scrap to reduce new material usage
- **Sheet-based assignment**: Assign cutting patterns to construction sheets with proper execution order

### 📊 Input Methods
1. **XLSX File Upload**: Automatically load data from Excel cutting specification sheets (supports all diameters)
2. **Reuse Scrap CSV Input**: Import existing scrap data via CSV file with diameter information (optional)

### 📈 Results Display
- **Consolidated sheet-based view**: Shows cutting order by construction sheet across all diameters
- **Inventory operations tracking**: Records material withdrawal and storage operations for each sheet
- **Color-coded display**: Different background colors for each sheet for easy identification
- Yield rate calculation (with/without reuse) for each diameter
- Detailed cutting pattern display per diameter
- Reusable scrap list (above threshold) consolidated across all diameters
- Processing time and combination count statistics per diameter

### 💾 Download Features
- **Sheet-based Cutting Order Excel**: Complete cutting sequence organized by construction sheet
- **Consolidated Reusable Scrap List CSV**: Combines all diameter scraps with unused reuse scraps
- **Per-diameter Optimization Results Excel**: Includes cutting patterns, summary tables, and sheets for each diameter

## Usage

### Starting the Application

```bash
streamlit run cutting_optimizer_streamlit_v4.py
```

### Basic Workflow

1. **Select Material Lengths**: Use checkboxes to select available rod lengths for each diameter (D10-D22)
2. **Set Time Limit**: Configure optimization time limit (10-3600 seconds, default: 120s)
3. **Upload XLSX File**: Upload cutting specification sheet containing all diameter data
4. **Input Reuse Scraps (Optional)**: Upload CSV file with 3 columns (diameter, length, quantity)
   ```csv
   Diameter,Length (mm),Quantity
   D13,405,3
   D13,1000,1
   D16,1915,2
   ```
5. **Set Scrap Threshold**: Set minimum length for reusable scraps (default: 400mm)
6. **Execute Optimization**: Click button to start multi-diameter optimization
7. **Review Sheet-based Results**: Check consolidated cutting order by construction sheet
8. **Review Diameter Details**: Check detailed results in tabs for each diameter
9. **Download Results**: Download sheet-based cutting order and consolidated scrap list

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
- Generates cutting patterns from input reuse scraps per diameter
- Prioritizes usage during optimization (with quantity constraints)
- Combines unused scraps with new scraps for download

### 4. Sheet-based Pattern Assignment
- Assigns cutting patterns to construction sheets in execution order
- Tracks inventory operations (withdrawal/storage) between sheets
- Minimizes intermediate inventory by matching patterns to sheet demands
- Maintains original sheet order from XLSX file

## Data Flow

```
[Input]
  ├─ Cutting Instructions (XLSX - all diameters)
  ├─ Reuse Scraps (CSV - with diameter column)
  └─ Material Selection (checkbox per diameter)
      ↓
[Processing - Per Diameter]
  ├─ Combination Generation (DFS)
  ├─ Optimization Calculation (ILP)
  └─ Pattern-to-Sheet Assignment
      ↓
[Consolidation]
  ├─ Merge results across diameters
  ├─ Sort by sheet order and diameter
  └─ Track inventory operations
      ↓
[Output]
  ├─ Sheet-based Cutting Order (Excel)
  ├─ Diameter-specific Results (tabs)
  ├─ Yield Rate Calculation (per diameter)
  ├─ Consolidated Reusable Scrap List (CSV)
  └─ Per-diameter Optimization Results (Excel)
```

## Multi-Diameter Reuse Scrap Circulation

### Input Example (CSV with diameter)
```csv
Diameter,Length (mm),Quantity
D13,405,3
D13,1000,1
D16,1915,2
```

### Optimization Results (per diameter)
- **D13**: 405mm×1 used, 1000mm×1 unused
- **D16**: 1915mm×2 used
- **New scraps**: D13: 610mm×5, D16: 1020mm×2, 830mm×3

### Download CSV (Consolidated across all diameters)
```csv
Diameter,Length (mm),Quantity
D16,1020,2
D13,1000,1
D16,830,3
D13,610,5
D13,405,2
```

## Performance Settings

- **Optimization Time Limit**: 10-3600 seconds (default: 120 seconds)
- **Combination Generation**: May take time for large-scale problems
- **Result Storage**: Saved in session state, updates display without recalculation when threshold changes

## File Structure

```
cuttingstock/
├── cutting_optimizer_streamlit_v4.py  # Main application (multi-diameter support)
├── cutting_optimizer_streamlit_v2.py  # Previous version (single diameter)
├── README.md                          # This file (English)
├── README_ja.md                       # Japanese README
└── test_scrap.csv                     # Test scrap data
```

## Output Details

### Sheet-based Cutting Order Table (Consolidated)
- Cutting sequence number, sheet name, diameter, operation type
- Number of rods, base material length, cutting pattern, loss
- Color-coded by sheet for easy identification
- Includes inventory withdrawal and storage operations

### Per-Diameter Cutting Pattern Table
- ID, cutting order, sheet name, operation type
- Base material length, cutting pattern, loss
- Color-coded by sheet and highlighted for reuse materials

### Output Summary Table (per diameter)
- Total quantities for each cut length
- Formula-based sum row

### Cutting Instruction Summary Table (XLSX upload, per diameter)
- Cutting instruction data per sheet
- Sum row and difference calculations

### Summary Sheet (per diameter)
- Diameter, yield rate (with/without reuse), total material length
- Processing time, scrap threshold
- Difference display between output results and cutting instructions

## Notes

- **Reuse scrap CSV files** should be in 3-column format: "Diameter,Length (mm),Quantity"
- **Material selection** is required for each diameter you want to optimize
- **Sheet order** in the consolidated view follows the original XLSX file order
- Scraps of the same diameter and length are automatically aggregated
- Reuse scrap usage is optional; normal optimization runs without upload
- For large-scale problems (multiple diameters, many cut types), recommend setting longer time limits
- The app automatically validates that selected materials can accommodate the cutting requirements

## License

This project is a tool for rebar cutting optimization.
