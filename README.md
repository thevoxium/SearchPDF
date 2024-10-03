# Enhanced PDF Search Tool

This tool allows you to search through multiple PDF files in a directory using TF-IDF scoring.

## Features

- Extracts text from multiple PDF files
- Computes TF-IDF scores for efficient searching
- Stores processed data for faster subsequent searches
- Supports multiple queries in a single session
- Provides an option to update stored data

## Requirements

- Python 3.6+
- pdfplumber
- argparse

Install required packages:

```
pip install pdfplumber argparse
```

## Usage

1. First run or update data:
   ```
   python search.py /path/to/pdf/folder --update
   ```

2. Subsequent runs (using stored data):
   ```
   python search.py /path/to/pdf/folder
   ```

3. Enter search queries when prompted. Press Enter without a query to exit.

## Output

For each query, the tool displays:
- Matching documents
- Relevance scores
- Page numbers where matches were found

## Note

The tool creates a `pdf_search_data.pkl` file in the specified folder to store processed data.