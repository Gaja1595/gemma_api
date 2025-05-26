# Bunchtest Tester

A simple GUI application to test the `/bunchtest` API endpoint with multiple resume files and one job description file.

## Overview

The Bunchtest Tester is a desktop application that allows you to:

1. Select multiple resume files (PDF or DOCX)
2. Select a job description file (PDF or DOCX)
3. Call the `/bunchtest` API endpoint
4. View the ranked results in the UI
5. Save the results to a JSON file

This tool is designed to help you test and evaluate how well multiple candidates match a specific job description, making it easier to identify the best candidates for a position.

## Features

- **Multiple Resume Selection**: Upload and process multiple resume files at once
- **Ranked Results**: View candidates ranked by their match score
- **Detailed Analysis**: See detailed scores and rationales for each candidate
- **Split View Interface**: Browse the candidate list on the left and view details on the right
- **Export Functionality**: Save results to JSON for further analysis

## Requirements

- Python 3.6 or higher
- Tkinter (usually included with Python)
- Requests library (`pip install requests`)

## Installation

1. Ensure you have Python installed on your system
2. Install the required dependencies:
   ```
   pip install requests
   ```
3. Download the `bunchtest_tester.py` file

## Usage

1. Run the application:
   ```
   python bunchtest_tester.py
   ```

2. Add multiple resume files by clicking "Add Files..."
3. Select a job description file by clicking "Browse..."
4. Verify the API URL (default is `http://localhost:8000`)
5. Click "Process Files" to call the bunchtest API
6. View the results in the UI
7. Save the results to a JSON file if desired

## API Configuration

By default, the application connects to `http://localhost:8000`. You can change this in the UI if your API is hosted elsewhere.

## Results Explanation

The results are displayed in a split view:

1. **Left Panel**: Shows a ranked list of candidates with their scores and fit categories
2. **Right Panel**: Shows detailed information about the selected candidate in three tabs:
   - **Summary**: Shows the overall score, fit category, summary text, and detailed scores with progress bars
   - **JSON**: Shows the raw JSON response for the selected candidate
   - **Rationale**: Shows the detailed rationale for each evaluation criterion

## Saving Results

You can save the complete results (including all candidates) to a JSON file by clicking the "Save Results" button. The file will include:

- Results for all candidates
- Job description title
- Number of resumes processed
- Timestamp

## Troubleshooting

If you encounter issues:

1. **API Connection Errors**: Verify the API URL and ensure the server is running
2. **File Format Errors**: Ensure you're using supported file formats (PDF or DOCX)
3. **Empty Results**: Check that your resume and job description files contain valid content

## License

This tool is provided as-is for testing purposes.

## Related Tools

- **Intervet Tester**: For testing single resume-job description matches
