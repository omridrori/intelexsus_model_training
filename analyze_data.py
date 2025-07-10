#!/usr/bin/env python3
"""
Data Analysis Script for Large JSONL Files
Efficiently analyzes Sanskrit and Tibetan language data files
"""

import json
import os
import sys
from pathlib import Path
from collections import defaultdict, Counter
import argparse
from typing import Dict, List, Any, Iterator, Optional
import time

class JSONLAnalyzer:
    def __init__(self, max_sample_lines=100):
        self.max_sample_lines = max_sample_lines
        self.results = {}
        
    def safe_read_jsonl_sample(self, filepath: Path, max_lines: Optional[int] = None) -> Iterator[Dict]:
        """Safely read a sample of lines from a JSONL file"""
        max_lines = max_lines or self.max_sample_lines
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                for i, line in enumerate(f):
                    if i >= max_lines:
                        break
                    line = line.strip()
                    if line:
                        try:
                            yield json.loads(line)
                        except json.JSONDecodeError as e:
                            print(f"Warning: Invalid JSON in {filepath} at line {i+1}: {e}")
                            continue
        except Exception as e:
            print(f"Error reading {filepath}: {e}")
            return
    
    def count_lines(self, filepath: Path) -> int:
        """Efficiently count lines in a file"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return sum(1 for line in f if line.strip())
        except Exception as e:
            print(f"Error counting lines in {filepath}: {e}")
            return 0
    
    def analyze_structure(self, sample_data: List[Dict]) -> Dict:
        """Analyze the structure of JSON objects"""
        if not sample_data:
            return {}
        
        # Collect all keys
        all_keys = set()
        key_types = defaultdict(Counter)
        key_presence = defaultdict(int)
        
        for item in sample_data:
            if isinstance(item, dict):
                all_keys.update(item.keys())
                for key, value in item.items():
                    key_presence[key] += 1
                    key_types[key][type(value).__name__] += 1
        
        # Calculate key statistics
        total_items = len(sample_data)
        key_stats = {}
        for key in all_keys:
            key_stats[key] = {
                'presence_rate': key_presence[key] / total_items,
                'types': dict(key_types[key])
            }
        
        return {
            'total_keys': len(all_keys),
            'all_keys': sorted(all_keys),
            'key_statistics': key_stats
        }
    
    def analyze_content(self, sample_data: List[Dict]) -> Dict:
        """Analyze content characteristics"""
        if not sample_data:
            return {}
        
        text_fields = []
        numeric_fields = []
        
        # Identify text and numeric fields
        for item in sample_data:
            if isinstance(item, dict):
                for key, value in item.items():
                    if isinstance(value, str) and len(value) > 10:
                        text_fields.append(key)
                    elif isinstance(value, (int, float)):
                        numeric_fields.append(key)
        
        # Count field occurrences
        text_field_counts = Counter(text_fields)
        numeric_field_counts = Counter(numeric_fields)
        
        # Sample text lengths for text fields
        text_lengths = defaultdict(list)
        for item in sample_data:
            if isinstance(item, dict):
                for key, value in item.items():
                    if isinstance(value, str) and key in text_field_counts:
                        text_lengths[key].append(len(value))
        
        # Calculate average text lengths
        avg_text_lengths = {}
        for key, lengths in text_lengths.items():
            if lengths:
                avg_text_lengths[key] = sum(lengths) / len(lengths)
        
        return {
            'text_fields': dict(text_field_counts),
            'numeric_fields': dict(numeric_field_counts),
            'average_text_lengths': avg_text_lengths
        }
    
    def analyze_file(self, filepath: Path) -> Dict:
        """Analyze a single JSONL file"""
        print(f"Analyzing {filepath}...")
        
        # Count total lines
        total_lines = self.count_lines(filepath)
        
        # Read sample data
        sample_data = list(self.safe_read_jsonl_sample(filepath))
        
        # Analyze structure and content
        structure_analysis = self.analyze_structure(sample_data)
        content_analysis = self.analyze_content(sample_data)
        
        return {
            'file_path': str(filepath),
            'total_records': total_lines,
            'sample_size': len(sample_data),
            'structure': structure_analysis,
            'content': content_analysis
        }
    
    def analyze_directory(self, directory: Path) -> Dict:
        """Analyze all JSONL files in a directory"""
        if not directory.exists():
            return {}
        
        results = {}
        jsonl_files = list(directory.glob("*.jsonl"))
        
        if not jsonl_files:
            return {}
        
        print(f"\nAnalyzing {len(jsonl_files)} files in {directory}...")
        
        total_records = 0
        for filepath in sorted(jsonl_files):
            file_analysis = self.analyze_file(filepath)
            results[filepath.name] = file_analysis
            total_records += file_analysis['total_records']
        
        return {
            'directory': str(directory),
            'total_files': len(jsonl_files),
            'total_records': total_records,
            'files': results
        }
    
    def generate_summary(self, analysis_results: Dict) -> str:
        """Generate a human-readable summary"""
        summary = []
        summary.append("=" * 60)
        summary.append("DATA ANALYSIS SUMMARY")
        summary.append("=" * 60)
        
        for lang, data in analysis_results.items():
            if not data:
                continue
                
            summary.append(f"\n{lang.upper()} DATA:")
            summary.append(f"  Directory: {data['directory']}")
            summary.append(f"  Total Files: {data['total_files']}")
            summary.append(f"  Total Records: {data['total_records']:,}")
            
            if data['files']:
                # Get a sample file analysis for structure overview
                sample_file = next(iter(data['files'].values()))
                structure = sample_file.get('structure', {})
                
                if structure:
                    summary.append(f"  Common Fields: {', '.join(structure.get('all_keys', []))}")
                
                content = sample_file.get('content', {})
                if content.get('text_fields'):
                    summary.append(f"  Text Fields: {', '.join(content['text_fields'].keys())}")
                if content.get('average_text_lengths'):
                    summary.append("  Average Text Lengths:")
                    for field, avg_len in content['average_text_lengths'].items():
                        summary.append(f"    {field}: {avg_len:.1f} characters")
        
        summary.append("\n" + "=" * 60)
        return "\n".join(summary)
    
    def save_detailed_report(self, analysis_results: Dict, output_file: str = "analysis_report.json"):
        """Save detailed analysis to JSON file"""
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(analysis_results, f, indent=2, ensure_ascii=False)
        print(f"Detailed analysis saved to {output_file}")

def main():
    parser = argparse.ArgumentParser(description='Analyze JSONL data files')
    parser.add_argument('--sample-size', type=int, default=100, 
                        help='Number of lines to sample from each file (default: 100)')
    parser.add_argument('--save-report', action='store_true',
                        help='Save detailed report to JSON file')
    parser.add_argument('--directories', nargs='+', default=['sanskrit', 'tibetan'],
                        help='Directories to analyze (default: sanskrit tibetan)')
    
    args = parser.parse_args()
    
    # Initialize analyzer
    analyzer = JSONLAnalyzer(max_sample_lines=args.sample_size)
    
    # Analyze each directory
    results = {}
    for dir_name in args.directories:
        directory = Path(dir_name)
        if directory.exists():
            results[dir_name] = analyzer.analyze_directory(directory)
        else:
            print(f"Warning: Directory {dir_name} not found")
    
    # Generate and display summary
    summary = analyzer.generate_summary(results)
    print(summary)
    
    # Save detailed report if requested
    if args.save_report:
        analyzer.save_detailed_report(results)
    
    # Show some sample data
    print("\nSAMPLE DATA PREVIEW:")
    print("-" * 40)
    for lang, data in results.items():
        if data and data['files']:
            sample_file = next(iter(data['files'].keys()))
            filepath = Path(lang) / sample_file
            
            print(f"\nSample from {filepath}:")
            sample_lines = list(analyzer.safe_read_jsonl_sample(filepath, 3))
            for i, line in enumerate(sample_lines, 1):
                print(f"  Record {i}: {json.dumps(line, ensure_ascii=False)[:200]}...")
                if len(json.dumps(line, ensure_ascii=False)) > 200:
                    print("    (truncated)")

if __name__ == "__main__":
    main() 