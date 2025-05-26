"""
Bunchtest Tester

A simple GUI application to test the bunchtest API with multiple resume files and one job description file.
The application allows you to:
1. Select multiple resume files (PDF or DOCX)
2. Select a job description file (PDF or DOCX)
3. Call the bunchtest API
4. View the ranked results in the UI
5. Save the results to a JSON file

Usage:
    python bunchtest_tester.py
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

class BunchtestTesterApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Bunchtest Tester")
        self.root.geometry("1200x800")
        self.root.minsize(800, 600)

        # Variables
        self.resume_files = []  # List to store multiple resume files
        self.jd_path = tk.StringVar()
        self.api_url = tk.StringVar(value=API_BASE_URL)
        self.status = tk.StringVar(value="Ready")
        self.result = None
        self.selected_candidate_index = 0  # Index of the currently selected candidate

        # Create UI
        self.create_ui()

    def create_ui(self):
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Input section
        input_frame = ttk.LabelFrame(main_frame, text="Input Files", padding="10")
        input_frame.pack(fill=tk.X, pady=5)

        # Resume files selection
        ttk.Label(input_frame, text="Resume Files:").grid(row=0, column=0, sticky=tk.W, pady=5)

        # Frame for resume files list and buttons
        resume_files_frame = ttk.Frame(input_frame)
        resume_files_frame.grid(row=0, column=1, padx=5, pady=5, sticky=tk.W)

        # Listbox to display selected resume files
        self.resume_listbox = tk.Listbox(resume_files_frame, width=50, height=5)
        self.resume_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Scrollbar for resume listbox
        resume_scrollbar = ttk.Scrollbar(resume_files_frame, orient=tk.VERTICAL, command=self.resume_listbox.yview)
        resume_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.resume_listbox.config(yscrollcommand=resume_scrollbar.set)

        # Buttons for resume files
        resume_buttons_frame = ttk.Frame(input_frame)
        resume_buttons_frame.grid(row=0, column=2, padx=5, pady=5)

        ttk.Button(resume_buttons_frame, text="Add Files...", command=self.add_resume_files).pack(pady=2)
        ttk.Button(resume_buttons_frame, text="Remove Selected", command=self.remove_selected_resume).pack(pady=2)
        ttk.Button(resume_buttons_frame, text="Clear All", command=self.clear_resume_files).pack(pady=2)

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

        # Results section with split view
        results_frame = ttk.LabelFrame(main_frame, text="Results", padding="10")
        results_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        # Create a PanedWindow for split view
        paned_window = ttk.PanedWindow(results_frame, orient=tk.HORIZONTAL)
        paned_window.pack(fill=tk.BOTH, expand=True)

        # Left panel - Candidate list
        left_panel = ttk.Frame(paned_window)
        paned_window.add(left_panel, weight=1)

        ttk.Label(left_panel, text="Ranked Candidates:", font=("Arial", 10, "bold")).pack(anchor=tk.W, pady=5)

        # Treeview for candidate list
        self.candidates_tree = ttk.Treeview(left_panel, columns=("score", "category"), show="headings")
        self.candidates_tree.heading("score", text="Score")
        self.candidates_tree.heading("category", text="Category")
        self.candidates_tree.column("score", width=50)
        self.candidates_tree.column("category", width=120)
        self.candidates_tree.pack(fill=tk.BOTH, expand=True, pady=5)

        # Bind selection event
        self.candidates_tree.bind("<<TreeviewSelect>>", self.on_candidate_selected)

        # Right panel - Candidate details
        right_panel = ttk.Frame(paned_window)
        paned_window.add(right_panel, weight=3)

        # Create notebook for different result views
        self.notebook = ttk.Notebook(right_panel)
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
        self.candidate_var = tk.StringVar(value="Candidate: N/A")

        candidate_label = ttk.Label(summary_content_frame, textvariable=self.candidate_var, font=("Arial", 14, "bold"))
        candidate_label.pack(anchor=tk.W, pady=5)

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

    def add_resume_files(self):
        file_paths = filedialog.askopenfilenames(
            title="Select Resume Files",
            filetypes=[("Document Files", "*.pdf;*.docx"), ("All Files", "*.*")]
        )
        if file_paths:
            for file_path in file_paths:
                if file_path not in self.resume_files:
                    self.resume_files.append(file_path)
                    self.resume_listbox.insert(tk.END, os.path.basename(file_path))

    def remove_selected_resume(self):
        selected_indices = self.resume_listbox.curselection()
        if not selected_indices:
            return

        # Remove in reverse order to avoid index shifting
        for index in sorted(selected_indices, reverse=True):
            del self.resume_files[index]
            self.resume_listbox.delete(index)

    def clear_resume_files(self):
        self.resume_files = []
        self.resume_listbox.delete(0, tk.END)

    def browse_jd(self):
        file_path = filedialog.askopenfilename(
            title="Select Job Description File",
            filetypes=[("Document Files", "*.pdf;*.docx"), ("All Files", "*.*")]
        )
        if file_path:
            self.jd_path.set(file_path)

    def process_files(self):
        if not self.resume_files:
            messagebox.showerror("Error", "Please select at least one resume file.")
            return

        jd_file = self.jd_path.get()
        api_url = self.api_url.get()

        if not jd_file or not os.path.exists(jd_file):
            messagebox.showerror("Error", "Please select a valid job description file.")
            return

        if not api_url:
            messagebox.showerror("Error", "Please enter a valid API URL.")
            return

        # Update status
        self.status.set(f"Processing {len(self.resume_files)} resume files...")
        self.root.update_idletasks()

        try:
            # Call the API
            endpoint = f"{api_url}/bunchtest"

            # Prepare the files for upload
            files = []

            # Add JD file
            files.append(
                ('jd_file', (os.path.basename(jd_file), open(jd_file, 'rb'), 'application/octet-stream'))
            )

            # Add multiple resume files - use the same key name for all files
            for resume_file in self.resume_files:
                files.append(
                    ('resume_files', (os.path.basename(resume_file), open(resume_file, 'rb'), 'application/octet-stream'))
                )

            # Make the request
            start_time = time.time()
            response = requests.post(endpoint, files=files)
            elapsed_time = time.time() - start_time

            # Close file handles
            for file_tuple in files:
                # Each file_tuple is ('key', (filename, file_object, content_type))
                file_tuple[1][1].close()

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

        # Clear the candidates tree
        for item in self.candidates_tree.get_children():
            self.candidates_tree.delete(item)

        # Populate the candidates tree
        results = self.result.get('results', [])
        for i, candidate in enumerate(results):
            score = candidate.get('total_score', 0)
            category = candidate.get('fit_category', 'Unknown')
            filename = candidate.get('resume_filename', f'Candidate {i+1}')

            # Insert into tree
            item_id = self.candidates_tree.insert('', 'end', text=filename, values=(score, category))

            # If this is the first candidate, select it
            if i == 0:
                self.candidates_tree.selection_set(item_id)
                self.display_candidate_details(candidate, filename)

    def on_candidate_selected(self, event):
        selected_items = self.candidates_tree.selection()
        if not selected_items:
            return

        # Get the selected item
        item_id = selected_items[0]

        # Get the index of the selected item
        index = self.candidates_tree.index(item_id)

        # Get the candidate data
        candidate = self.result.get('results', [])[index]
        filename = candidate.get('resume_filename', f'Candidate {index+1}')

        # Display the candidate details
        self.display_candidate_details(candidate, filename)

    def display_candidate_details(self, candidate, filename):
        # Update candidate name
        self.candidate_var.set(f"Candidate: {filename}")

        # Update score and category
        self.score_var.set(f"Score: {candidate.get('total_score', 'N/A')}/100")
        self.category_var.set(f"Category: {candidate.get('fit_category', 'N/A')}")

        # Update summary text
        self.summary_text.delete(1.0, tk.END)
        self.summary_text.insert(tk.END, candidate.get('summary', 'No summary available.'))

        # Update score bars
        detailed_scores = candidate.get('detailed_scores', {})
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
        self.json_text.insert(tk.END, json.dumps(candidate, indent=2))

        # Update rationale tab
        self.rationale_text.delete(1.0, tk.END)

        # Check for both 'rationale' and 'detailed_rationale' keys since the API might use either
        rationale = candidate.get('detailed_rationale', candidate.get('rationale', {}))
        for key, value in rationale.items():
            self.rationale_text.insert(tk.END, f"{key.replace('_', ' ').title()}:\n")
            self.rationale_text.insert(tk.END, f"{value}\n\n")

    def save_results(self):
        if not self.result:
            messagebox.showerror("Error", "No results to save.")
            return

        # Create a default filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        default_filename = f"bunchtest_result_{timestamp}.json"

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
        selected_items = self.candidates_tree.selection()
        if not selected_items:
            messagebox.showerror("Error", "No candidate selected.")
            return

        # Get the selected item
        item_id = selected_items[0]

        # Get the index of the selected item
        index = self.candidates_tree.index(item_id)

        # Get the candidate data
        candidate = self.result.get('results', [])[index]

        # Convert the result to a formatted JSON string
        json_str = json.dumps(candidate, indent=2)

        # Copy to clipboard
        self.root.clipboard_clear()
        self.root.clipboard_append(json_str)

        # Update status
        self.status.set("JSON copied to clipboard")
        messagebox.showinfo("Success", "JSON content copied to clipboard")

    def copy_rationale_to_clipboard(self):
        """Copy the rationale content to the clipboard."""
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

        # Clear the candidates tree
        for item in self.candidates_tree.get_children():
            self.candidates_tree.delete(item)

        # Reset UI elements
        self.candidate_var.set("Candidate: N/A")
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
    BunchtestTesterApp(root)  # Create the app instance
    root.mainloop()

if __name__ == "__main__":
    main()
