# Intervet Tester

A simple GUI application to test the intervet2 API with resume and job description files.

## Features

- Select resume and job description files (PDF or DOCX)
- Call the intervet2 API
- View results in a user-friendly interface
- Save results to JSON files for later analysis
- Reuse with different resume-JD combinations

## Screenshots

![Intervet Tester UI](https://i.imgur.com/placeholder.png)

## Requirements

- Python 3.7 or higher
- Tkinter (usually included with Python)
- Requests library

## Installation

1. Ensure you have Python installed
2. Install the required dependencies:

```bash
pip install requests
```

## Usage

1. Run the application:

```bash
python intervet_tester.py
```

2. Select a resume file (PDF or DOCX)
3. Select a job description file (PDF or DOCX)
4. Click "Process Files" to call the intervet2 API
5. View the results in the UI
6. Save the results to a JSON file if desired

## API Configuration

By default, the application connects to `http://localhost:8000`. You can change this in the UI if your API is hosted elsewhere.

## Results Explanation

The results are displayed in three tabs:

1. **Summary**: Shows the overall score, fit category, summary text, and detailed scores with progress bars
2. **JSON**: Shows the raw JSON response from the API
3. **Rationale**: Shows the detailed rationale for each evaluation criterion

## Saving Results

Click the "Save Results" button to save the results to a JSON file. The default filename includes the resume filename, JD filename, and timestamp.

## Using Results with SOTA Models

The saved JSON files can be used for comparison with state-of-the-art models:

1. Save the results from the Intervet Tester
2. Open the JSON file
3. Copy the content
4. Paste it into your SOTA model's input along with your evaluation prompt

## Troubleshooting

### API Connection Issues

If you encounter connection issues with the API:
1. Ensure the API is running and accessible
2. Check the API URL in the UI
3. Verify network connectivity

### File Format Issues

If you encounter issues with file formats:
1. Ensure all resume and JD files are in PDF or DOCX format
2. Check that the files are not corrupted
3. Try converting the files to a different format
