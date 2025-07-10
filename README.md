# Data Analysis Scripts for JSONL Files

This repository contains scripts to efficiently analyze large JSONL (JSON Lines) files containing Sanskrit and Tibetan language data.

## Files Created

1. **`analyze_data.py`** - Main analysis script with comprehensive features
2. **`run_analysis.py`** - Simple runner script for quick analysis
3. **`analysis_report.json`** - Detailed analysis report (generated after running)

## Analysis Results Summary

Based on the analysis of your data files:

### Sanskrit Data
- **Total Files**: 7 files
- **Total Records**: 132,943 records
- **Structure**: Each record contains `text` and `metadata` fields
- **Average Text Length**: ~11 characters
- **Data Format**: Short Sanskrit phrases/words

### Tibetan Data  
- **Total Files**: 93 files
- **Total Records**: 184,225 records
- **Structure**: Each record contains `text` and `metadata` fields
- **Average Text Length**: ~159,733 characters
- **Data Format**: Long Tibetan texts (likely complete texts/documents)

## Usage

### Quick Analysis
```bash
python run_analysis.py
```

### Detailed Analysis with Options
```bash
# Basic analysis
python analyze_data.py

# With larger sample size
python analyze_data.py --sample-size 200

# Save detailed report
python analyze_data.py --save-report

# Analyze specific directories
python analyze_data.py --directories sanskrit tibetan

# Get help
python analyze_data.py --help
```

## Script Features

The analysis script provides:

1. **Efficient Processing**: Handles large files without loading everything into memory
2. **File Statistics**: Count of records per file and total records
3. **Structure Analysis**: Identifies all JSON fields and their types
4. **Content Analysis**: Analyzes text fields and calculates statistics
5. **Sample Data**: Shows sample records from each dataset
6. **Detailed Reports**: Optional JSON report with comprehensive analysis

## Data Structure

Both Sanskrit and Tibetan files follow this structure:

```json
{
  "text": "actual text content",
  "metadata": {
    "author": "author name or null",
    "title": "title or null", 
    "file_lang": "language code",
    "format": "file format",
    "dataset": "dataset name",
    "file_name": "original filename",
    "file_path": "original file path"
  }
}
```

## Key Insights

1. **Sanskrit files** contain shorter text snippets (words/phrases)
2. **Tibetan files** contain much longer texts (complete documents/books)
3. Both datasets have rich metadata for tracking sources
4. Total dataset size: **317,168 records** across **100 files**

## Requirements

- Python 3.6+
- Standard library modules only (no external dependencies)

## Output Files

- `analysis_report.json`: Detailed analysis in JSON format
- Console output with summary statistics and sample data 