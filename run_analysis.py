#!/usr/bin/env python3
"""
Simple runner script for data analysis
"""

import subprocess
import sys
import os

def main():
    print("Starting data analysis...")
    print("This will analyze your Sanskrit and Tibetan JSONL files efficiently.")
    print("-" * 50)
    
    # Check if the main analysis script exists
    if not os.path.exists("analyze_data.py"):
        print("Error: analyze_data.py not found!")
        sys.exit(1)
    
    # Run the analysis with save report enabled
    try:
        result = subprocess.run([
            sys.executable, "analyze_data.py", 
            "--save-report", 
            "--sample-size", "50"
        ], check=True, capture_output=True, text=True)
        
        print(result.stdout)
        if result.stderr:
            print("Warnings/Errors:")
            print(result.stderr)
            
    except subprocess.CalledProcessError as e:
        print(f"Error running analysis: {e}")
        if e.stdout:
            print("Output:", e.stdout)
        if e.stderr:
            print("Error details:", e.stderr)
        sys.exit(1)
    
    print("\nAnalysis complete!")
    print("Check 'analysis_report.json' for detailed results.")

if __name__ == "__main__":
    main() 