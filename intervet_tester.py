"""
Intervet Tester

A simple GUI application to test the intervet2 API with resume and job description files.
The application allows you to:
1. Select a resume file (PDF or DOCX)
2. Select a job description file (PDF or DOCX)
3. Call the intervet2 API
4. View the results in the UI
5. Save the results to a JSON file

Usage:
    python intervet_tester.py
"""

import os
import sys
import json
import time
import requests
import tkinter as tk
from tkinter import ttk, filedialog, scrolledtext, messagebox
from datetime import datetime

# API endpoint configuration
API_BASE_URL = "http://localhost:8000"  # Change this to your API URL

class IntervetTesterApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Intervet Tester")
        self.root.geometry("1000x800")
        self.root.minsize(800, 600)

        # Variables
        self.resume_path = tk.StringVar()
        self.jd_path = tk.StringVar()
        self.api_url = tk.StringVar(value=API_BASE_URL)
        self.status = tk.StringVar(value="Ready")
        self.result = None

        # Create UI
        self.create_ui()

    def create_ui(self):
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Input section
        input_frame = ttk.LabelFrame(main_frame, text="Input Files", padding="10")
        input_frame.pack(fill=tk.X, pady=5)

        # Resume file selection
        ttk.Label(input_frame, text="Resume File:").grid(row=0, column=0, sticky=tk.W, pady=5)
        ttk.Entry(input_frame, textvariable=self.resume_path, width=50).grid(row=0, column=1, padx=5, pady=5)
        ttk.Button(input_frame, text="Browse...", command=self.browse_resume).grid(row=0, column=2, padx=5, pady=5)

        # JD file selection
        ttk.Label(input_frame, text="Job Description File:").grid(row=1, column=0, sticky=tk.W, pady=5)
        ttk.Entry(input_frame, textvariable=self.jd_path, width=50).grid(row=1, column=1, padx=5, pady=5)
        ttk.Button(input_frame, text="Browse...", command=self.browse_jd).grid(row=1, column=2, padx=5, pady=5)

        # API URL
        ttk.Label(input_frame, text="API URL:").grid(row=2, column=0, sticky=tk.W, pady=5)
        ttk.Entry(input_frame, textvariable=self.api_url, width=50).grid(row=2, column=1, padx=5, pady=5)

        # Action buttons
        action_frame = ttk.Frame(main_frame)
        action_frame.pack(fill=tk.X, pady=10)

        ttk.Button(action_frame, text="Process Files", command=self.process_files).pack(side=tk.LEFT, padx=5)
        ttk.Button(action_frame, text="Save Results", command=self.save_results).pack(side=tk.LEFT, padx=5)
        ttk.Button(action_frame, text="Clear", command=self.clear_results).pack(side=tk.LEFT, padx=5)

        # Status bar
        status_frame = ttk.Frame(main_frame)
        status_frame.pack(fill=tk.X, side=tk.BOTTOM, pady=5)

        ttk.Label(status_frame, text="Status:").pack(side=tk.LEFT)
        ttk.Label(status_frame, textvariable=self.status).pack(side=tk.LEFT, padx=5)

        # Results section
        results_frame = ttk.LabelFrame(main_frame, text="Results", padding="10")
        results_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        # Create notebook for different result views
        self.notebook = ttk.Notebook(results_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        # Summary tab
        summary_frame = ttk.Frame(self.notebook)
        self.notebook.add(summary_frame, text="Summary")

        # Summary content
        summary_content_frame = ttk.Frame(summary_frame)
        summary_content_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Score and category
        self.score_var = tk.StringVar(value="Score: N/A")
        self.category_var = tk.StringVar(value="Category: N/A")

        score_label = ttk.Label(summary_content_frame, textvariable=self.score_var, font=("Arial", 14, "bold"))
        score_label.pack(anchor=tk.W, pady=5)

        category_label = ttk.Label(summary_content_frame, textvariable=self.category_var, font=("Arial", 12))
        category_label.pack(anchor=tk.W, pady=5)

        # Summary text
        ttk.Label(summary_content_frame, text="Summary:", font=("Arial", 10, "bold")).pack(anchor=tk.W, pady=5)
        self.summary_text = scrolledtext.ScrolledText(summary_content_frame, wrap=tk.WORD, height=5)
        self.summary_text.pack(fill=tk.X, expand=False, pady=5)

        # Detailed scores
        ttk.Label(summary_content_frame, text="Detailed Scores:", font=("Arial", 10, "bold")).pack(anchor=tk.W, pady=5)

        self.scores_frame = ttk.Frame(summary_content_frame)
        self.scores_frame.pack(fill=tk.X, expand=False, pady=5)

        # Create progress bars for each score category
        self.score_bars = {}
        self.score_labels = {}

        score_categories = [
            ("skills_match_direct", "Skills Match (Direct)"),
            ("skills_match_subjective", "Skills Match (Subjective)"),
            ("experience_match", "Experience Match"),
            ("reliability", "Reliability"),
            ("location_match", "Location Match"),
            ("academic_match", "Academic Match"),
            ("alma_mater", "Alma Mater"),
            ("certifications", "Certifications")
        ]

        for i, (key, label) in enumerate(score_categories):
            ttk.Label(self.scores_frame, text=label).grid(row=i, column=0, sticky=tk.W, pady=2)

            self.score_bars[key] = ttk.Progressbar(self.scores_frame, length=300, mode="determinate")
            self.score_bars[key].grid(row=i, column=1, padx=5, pady=2)

            self.score_labels[key] = ttk.Label(self.scores_frame, text="0")
            self.score_labels[key].grid(row=i, column=2, padx=5, pady=2)

        # JSON tab
        json_frame = ttk.Frame(self.notebook)
        self.notebook.add(json_frame, text="JSON")

        # JSON content
        json_content_frame = ttk.Frame(json_frame)
        json_content_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Add a copy button
        copy_button = ttk.Button(json_content_frame, text="Copy JSON to Clipboard", command=self.copy_json_to_clipboard)
        copy_button.pack(anchor=tk.NE, pady=5)

        # JSON text area
        self.json_text = scrolledtext.ScrolledText(json_content_frame, wrap=tk.WORD)
        self.json_text.pack(fill=tk.BOTH, expand=True, pady=5)

        # Rationale tab
        rationale_frame = ttk.Frame(self.notebook)
        self.notebook.add(rationale_frame, text="Rationale")

        # Rationale content
        rationale_content_frame = ttk.Frame(rationale_frame)
        rationale_content_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Add a copy button
        copy_button = ttk.Button(rationale_content_frame, text="Copy Rationale to Clipboard", command=self.copy_rationale_to_clipboard)
        copy_button.pack(anchor=tk.NE, pady=5)

        # Rationale text area
        self.rationale_text = scrolledtext.ScrolledText(rationale_content_frame, wrap=tk.WORD)
        self.rationale_text.pack(fill=tk.BOTH, expand=True, pady=5)

    def browse_resume(self):
        file_path = filedialog.askopenfilename(
            title="Select Resume File",
            filetypes=[("Document Files", "*.pdf;*.docx"), ("All Files", "*.*")]
        )
        if file_path:
            self.resume_path.set(file_path)

    def browse_jd(self):
        file_path = filedialog.askopenfilename(
            title="Select Job Description File",
            filetypes=[("Document Files", "*.pdf;*.docx"), ("All Files", "*.*")]
        )
        if file_path:
            self.jd_path.set(file_path)

    def process_files(self):
        resume_file = self.resume_path.get()
        jd_file = self.jd_path.get()
        api_url = self.api_url.get()

        if not resume_file or not os.path.exists(resume_file):
            messagebox.showerror("Error", "Please select a valid resume file.")
            return

        if not jd_file or not os.path.exists(jd_file):
            messagebox.showerror("Error", "Please select a valid job description file.")
            return

        if not api_url:
            messagebox.showerror("Error", "Please enter a valid API URL.")
            return

        # Update status
        self.status.set("Processing files...")
        self.root.update_idletasks()

        try:
            # Call the API
            endpoint = f"{api_url}/intervet2"

            # Prepare the files for upload
            files = {
                'resume_file': (os.path.basename(resume_file), open(resume_file, 'rb'), 'application/octet-stream'),
                'jd_file': (os.path.basename(jd_file), open(jd_file, 'rb'), 'application/octet-stream')
            }

            # Make the request
            start_time = time.time()
            response = requests.post(endpoint, files=files)
            elapsed_time = time.time() - start_time

            # Close file handles
            for file_obj in files.values():
                file_obj[1].close()

            if response.status_code == 200:
                # Parse the response
                self.result = response.json()

                # Update the UI
                self.update_results_ui()

                # Update status
                self.status.set(f"Processing completed in {elapsed_time:.2f} seconds")
            else:
                messagebox.showerror("API Error", f"Error {response.status_code}: {response.text}")
                self.status.set(f"Error: API returned status code {response.status_code}")

        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {str(e)}")
            self.status.set(f"Error: {str(e)}")

    def update_results_ui(self):
        if not self.result:
            return

        # Update summary tab
        self.score_var.set(f"Score: {self.result.get('total_score', 'N/A')}/100")
        self.category_var.set(f"Category: {self.result.get('fit_category', 'N/A')}")

        # Update summary text
        self.summary_text.delete(1.0, tk.END)
        self.summary_text.insert(tk.END, self.result.get('summary', 'No summary available.'))

        # Update score bars
        detailed_scores = self.result.get('detailed_scores', {})
        for key, bar in self.score_bars.items():
            score = detailed_scores.get(key, 0)

            # Determine the maximum value for this category
            max_score = 25 if key == "skills_match_direct" else 20 if key == "experience_match" else 15 if key == "skills_match_subjective" else 10

            # Update the progress bar
            bar["maximum"] = max_score
            bar["value"] = score

            # Update the label
            self.score_labels[key].config(text=f"{score}/{max_score}")

        # Update JSON tab
        self.json_text.delete(1.0, tk.END)
        self.json_text.insert(tk.END, json.dumps(self.result, indent=2))

        # Update rationale tab
        self.rationale_text.delete(1.0, tk.END)

        # Check for both 'rationale' and 'detailed_rationale' keys since the API might use either
        rationale = self.result.get('detailed_rationale', self.result.get('rationale', {}))
        for key, value in rationale.items():
            self.rationale_text.insert(tk.END, f"{key.replace('_', ' ').title()}:\n")
            self.rationale_text.insert(tk.END, f"{value}\n\n")

        # Switch to summary tab
        self.notebook.select(0)

    def save_results(self):
        if not self.result:
            messagebox.showerror("Error", "No results to save.")
            return

        # Get resume and JD filenames
        resume_filename = os.path.basename(self.resume_path.get())
        jd_filename = os.path.basename(self.jd_path.get())

        # Create a default filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        default_filename = f"intervet_result_{resume_filename.split('.')[0]}_{jd_filename.split('.')[0]}_{timestamp}.json"

        # Ask for save location
        file_path = filedialog.asksaveasfilename(
            title="Save Results",
            defaultextension=".json",
            initialfile=default_filename,
            filetypes=[("JSON Files", "*.json"), ("All Files", "*.*")]
        )

        if not file_path:
            return

        try:
            with open(file_path, 'w') as f:
                json.dump(self.result, f, indent=2)

            self.status.set(f"Results saved to {file_path}")
            messagebox.showinfo("Success", f"Results saved to {file_path}")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to save results: {str(e)}")
            self.status.set(f"Error: {str(e)}")

    def copy_json_to_clipboard(self):
        """Copy the JSON content to the clipboard."""
        if not self.result:
            messagebox.showerror("Error", "No results to copy.")
            return

        # Convert the result to a formatted JSON string
        json_str = json.dumps(self.result, indent=2)

        # Copy to clipboard
        self.root.clipboard_clear()
        self.root.clipboard_append(json_str)

        # Update status
        self.status.set("JSON copied to clipboard")
        messagebox.showinfo("Success", "JSON content copied to clipboard")

    def copy_rationale_to_clipboard(self):
        """Copy the rationale content to the clipboard."""
        if not self.result:
            messagebox.showerror("Error", "No results to copy.")
            return

        # Get the rationale text
        rationale_text = self.rationale_text.get(1.0, tk.END)

        if not rationale_text.strip():
            messagebox.showerror("Error", "No rationale content to copy.")
            return

        # Copy to clipboard
        self.root.clipboard_clear()
        self.root.clipboard_append(rationale_text)

        # Update status
        self.status.set("Rationale copied to clipboard")
        messagebox.showinfo("Success", "Rationale content copied to clipboard")

    def clear_results(self):
        # Clear result data
        self.result = None

        # Reset UI elements
        self.score_var.set("Score: N/A")
        self.category_var.set("Category: N/A")
        self.summary_text.delete(1.0, tk.END)
        self.json_text.delete(1.0, tk.END)
        self.rationale_text.delete(1.0, tk.END)

        # Reset score bars
        for key, bar in self.score_bars.items():
            bar["value"] = 0
            self.score_labels[key].config(text="0")

        # Reset status
        self.status.set("Ready")

def main():
    root = tk.Tk()
    IntervetTesterApp(root)  # Create the app instance
    root.mainloop()

if __name__ == "__main__":
    main()
