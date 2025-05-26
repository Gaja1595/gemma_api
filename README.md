# Gemma 3:4B API

The Gemma 3:4B API provides a comprehensive suite of tools for interacting with the Gemma 3:4B large language model and performing various HR-related tasks. This API is designed to streamline and automate processes such as advanced text generation, document parsing, and candidate evaluation.

## Core Capabilities

The API offers the following key functionalities:

*   **Text Generation**: Generate contextually relevant text based on prompts, with or without accompanying image data.
*   **Resume Parsing**: Extract structured information from resumes in PDF or DOCX format. Employs a robust waterfall text extraction mechanism, including image-based OCR for scanned documents.
*   **Job Description (JD) Parsing**: Parse job descriptions from PDF or DOCX files, extracting key details into a structured format. Also utilizes the waterfall text extraction mechanism.
*   **Interview Question Generation**: Create tailored interview questions based on:
    *   A job description and a candidate's resume.
    *   A job description only.
*   **Candidate-Job Fit Evaluation**: Assess the suitability of a candidate for a specific role by comprehensively comparing their resume against the job description.
*   **VAPI Call Summary Extraction**: Extract key information (e.g., salary expectations, notice period) from VAPI (Voice API) call summaries or transcripts.
*   **Batch Resume Evaluation**: Evaluate multiple resumes against a single job description in a batch process, with results sorted by match score.

## API Endpoints

This section details the available API endpoints, their functionalities, expected request formats, and responses.

### Root

*   **Method**: `GET`
*   **Path**: `/`
*   **Description**: Provides a welcome message and a list of all available API endpoints with their descriptions.
*   **Request Body**: None.
*   **Response**: JSON object containing a welcome message and a list of endpoint details.

### Generate Text

*   **Method**: `POST`
*   **Path**: `/generate`
*   **Description**: Generates a text-based response from the Gemma model based on a given prompt and optional conversation history.
*   **Request Body**: `GenerateRequest` model.
    *   `prompt` (string, required): The prompt to send to the model.
    *   `history` (string, optional): The conversation history. Defaults to an empty string.
    ```json
    {
        "prompt": "Tell me about the future of AI.",
        "history": "User: What is AI?\nAssistant: AI is artificial intelligence."
    }
    ```
*   **Response**: JSON object with the generated text: `{"response": "Generated text from the model."}`.

### Generate with Image

*   **Method**: `POST`
*   **Path**: `/generate_with_image`
*   **Description**: Generates a response based on an uploaded image and a text prompt, leveraging multimodal capabilities.
*   **Request Body**: Form data.
    *   `file` (UploadFile, required): Image file (supported formats: JPG, PNG, GIF, BMP).
    *   `prompt` (string, required): Text prompt related to the image.
    *   `history` (string, optional): Optional conversation history.
*   **Response**: JSON object with the generated text: `{"response": "Generated text based on the image and prompt."}`.

### Parse Resume

*   **Method**: `POST`
*   **Path**: `/resume`
*   **Description**: Parses a resume file (PDF or DOCX) and extracts structured information. Uses a waterfall mechanism for text extraction (standard parsing then image-based fallback if needed).
*   **Request Body**:
    *   `file` (UploadFile, required): Resume file (PDF or DOCX).
*   **Response**: `ResumeResponse` JSON object containing comprehensively parsed resume data, including name, contact information, education, skills (differentiated into `direct_skills` and `subjective_skills` with context), experience, projects, certifications, etc., along with a `confidence_score` and `confidence_details`.

### Parse Job Description (JD)

*   **Method**: `POST`
*   **Path**: `/jd_parser`
*   **Description**: Parses a job description file (PDF or DOCX) and extracts structured information. Uses a waterfall mechanism for text extraction.
*   **Request Body**:
    *   `file` (UploadFile, required): Job description file (PDF or DOCX).
*   **Response**: `JobDescriptionResponse` JSON object containing parsed job description data, including job title, company, location, skills, experience, education, responsibilities, etc., along with a `confidence_score`.

### Generate Interview Questions (JD & Resume)

*   **Method**: `POST`
*   **Path**: `/jd`
*   **Description**: Generates tailored interview questions based on a job description file and resume data.
*   **Request Body**: Form data.
    *   `file` (UploadFile, required): Job description file (PDF or DOCX).
    *   `request_data_json` (string, required): JSON string containing `resume_json` (parsed resume data, typically from the `/resume` endpoint) and question scales (0-10 for technical, past experience, case study, situation handling, personality test questions). See `JDQuestionRequest` model for structure.
    ```json
    {
        "resume_json": {
            "name": "John Doe",
            "direct_skills": {"Python": "Expert", "Java": "Intermediate"},
            "experience": [{"role": "Software Engineer", "duration": "2 years"}]
        },
        "technical_questions": 7,
        "past_experience_questions": 5,
        "case_study_questions": 3,
        "situation_handling_questions": 4,
        "personality_test_questions": 2
    }
    ```
*   **Response**: JSON object with categorized interview questions (technical, past experience, case study, situation handling, personality test). The number of questions per category is proportional to the provided scale (0-10 maps to 0-5 questions).

### Generate Interview Questions (JD Only)

*   **Method**: `POST`
*   **Path**: `/jd_only`
*   **Description**: Generates interview questions based solely on an uploaded job description file.
*   **Request Body**:
    *   `file` (UploadFile, required): Job description file (PDF or DOCX).
*   **Response**: JSON object with 5 questions for each category (technical, past experience, case study, situation handling, personality test).

### Evaluate Candidate-Job Fit (JSON Inputs)

*   **Method**: `POST`
*   **Path**: `/intervet`
*   **Description**: Evaluates how well a candidate's resume matches a job description, using JSON inputs for both resume and JD data.
*   **Request Body**: `IntervetRequest` model.
    *   `resume_json` (object, required): Parsed resume data (typically from the `/resume` endpoint).
    *   `jd_json` (object, required): Parsed job description data (typically from the `/jd_parser` endpoint).
    ```json
    {
        "resume_json": {
            "name": "Jane Doe",
            "direct_skills": {"Python": "Proficient", "Machine Learning": "Experienced"},
            "experience": [{"company_name": "Tech Corp", "role": "Data Scientist", "duration": "3 years"}]
        },
        "jd_json": {
            "job_title": "Senior Data Scientist",
            "required_skills": ["Python", "Machine Learning", "Deep Learning"],
            "required_experience": "5+ years"
        }
    }
    ```
*   **Response**: JSON object containing an overall `total_score` (0-100), `fit_category` (e.g., "Excellent Match"), `summary` of the evaluation, detailed `scores` for various criteria (skills, experience, reliability, location, education, certifications), and `rationale` for each score.

### Evaluate Candidate-Job Fit (File Uploads)

*   **Method**: `POST`
*   **Path**: `/intervet2`
*   **Description**: Evaluates how well a candidate's resume matches a job description, using direct file uploads for both resume and JD.
*   **Request Body**: Form data.
    *   `resume_file` (UploadFile, required): Resume file (PDF or DOCX).
    *   `jd_file` (UploadFile, required): Job description file (PDF or DOCX).
*   **Response**: JSON object with evaluation results, similar in structure to the `/intervet` endpoint response (including `total_score`, `fit_category`, `summary`, `scores`, and `rationale`).

### Extract from Call Summary (Interfix)

*   **Method**: `POST`
*   **Path**: `/interfix`
*   **Description**: Extracts key information from a VAPI (Voice API) call summary or transcript.
*   **Request Body**: `InterfixRequest` model.
    *   `summary` (string, required): The call summary or transcript text.
    ```json
    {
        "summary": "The candidate mentioned a notice period of 1 month and expects a salary around 80,000. They are looking for a new role to grow their skills. Available for an interview next week, preferably in the afternoon."
    }
    ```
*   **Response**: `InterfixResponse` JSON object with extracted fields: `offer_in_hand` (float, optional), `notice_period` (string, optional), `expected_salary` (float, optional), `reason_to_switch` (string, optional), `preferred_time_for_interview` (string, optional), and `preferred_date_for_interview` (string, optional).

### Batch Evaluate Resumes

*   **Method**: `POST`
*   **Path**: `/bunchtest`
*   **Description**: Evaluates multiple resume files (PDF or DOCX) against a single job description file (PDF or DOCX) in a batch process.
*   **Request Body**: Form data.
    *   `resume_files` (List[UploadFile], required): List of resume files.
    *   `jd_file` (UploadFile, required): A single job description file.
*   **Response**: JSON object containing a list of `results` (each result is similar to the `/intervet2` response, including `resume_filename`), `jd_title`, `resume_count`, and a `timestamp`. Results are sorted by `total_score` in descending order.

## Key Features and Functionalities

This section provides a more detailed explanation of the core features offered by the Gemma 3:4B API.

### Text Generation

The API provides powerful text generation capabilities leveraging the Gemma model:

*   **`/generate` Endpoint**: This endpoint uses the Gemma model to generate text-based responses. It takes a user-provided `prompt` and an optional `history` of the conversation. The model then continues the conversation or answers the prompt based on this input.
*   **`/generate_with_image` Endpoint**: This endpoint extends the text generation capability to multimodal inputs. It accepts an image file (e.g., JPG, PNG) along with a `prompt` and optional `history`. The Gemma model considers both the visual information from the image and the textual context to generate a relevant response.

### Resume Parsing (`/resume` endpoint)

*   **Overall Functionality**: The `/resume` endpoint is designed to process resume files (PDF or DOCX) and convert them into structured JSON data. This facilitates the automated extraction of key candidate information for HR systems and analysis.
*   **Key Information Extracted**: The service extracts a comprehensive set of fields, including but not limited to:
    *   `name`, `email`, `phone`, `location`, `summary`
    *   `education`: (List of entries with `degree`, `institution`, `year`, etc.)
    *   `direct_skills`: (Dictionary of skills explicitly listed, e.g., in a "Skills" section, with context)
    *   `subjective_skills`: (Dictionary of skills inferred from the text of experience or projects, with context)
    *   `experience`: (List of entries with `company_name`, `role`, `duration`, `key_responsibilities`, etc.)
    *   `projects`: (List of entries with `name`, `description`, `technologies_used`, etc.)
    *   `certifications`, `languages`, `social_media`, `publications`, `achievements`, `volunteer_experience`, `domain_of_interest`
    *   `confidence_score` and `confidence_details`.
*   **Waterfall Text Extraction**: To ensure high accuracy in text extraction from various resume formats, the API employs a waterfall mechanism. It first attempts to extract text using standard PDF/DOCX parsing libraries. If the extracted text is insufficient (which can happen with image-based PDFs or scanned documents), it automatically falls back to an image-based extraction method. This fallback treats the document as an image and uses the LLM's vision capabilities to "read" and extract the text content.
*   **Confidence Score**: The `ResumeResponse` includes a `confidence_score` (a float between 0.0 and 1.0) and `confidence_details` (a dictionary with scores per field). These scores represent the model's assessment of the accuracy and completeness of the extracted information. A higher score indicates greater confidence in the parsed data. This is determined by analyzing factors like the clarity of the resume sections, consistency in formatting, and how well the extracted data aligns with expected patterns for each field.

### Job Description Parsing (`/jd_parser` endpoint)

*   **Overall Functionality**: The `/jd_parser` endpoint processes job description files (PDF or DOCX) and transforms their content into a structured JSON format. This is useful for standardizing job postings and enabling programmatic analysis of requirements.
*   **Key Information Extracted**: Key fields include:
    *   `job_title`, `company_name`, `location`, `job_type`, `work_mode`, `summary`
    *   `responsibilities`: (List of strings)
    *   `required_skills`: (List of strings)
    *   `preferred_skills`: (List of strings)
    *   `required_experience`: (e.g., "3+ years")
    *   `education_requirements`: (List of strings)
    *   `education_details`: (Object with `degree_level`, `field_of_study`, etc.)
    *   `salary_range`, `benefits`
    *   `confidence_score`.
*   **Waterfall Text Extraction**: Similar to resume parsing, this endpoint uses a waterfall approach for text extraction. It starts with standard methods for PDF/DOCX files and can fall back to image-based extraction if the initial attempt does not yield sufficient text, ensuring robust parsing across different JD formats.
*   **Confidence Score**: The `JobDescriptionResponse` includes a `confidence_score` (a float). This score reflects the model's confidence in how accurately it has parsed the job description into the structured schema.

### Interview Question Generation

The API provides two endpoints for generating interview questions:

*   **`/jd` Endpoint (Job Description & Resume)**: This endpoint creates tailored interview questions by analyzing both an uploaded job description file (PDF/DOCX) and provided resume data (as a JSON payload). The request allows users to specify `technical_questions`, `past_experience_questions`, `case_study_questions`, `situation_handling_questions`, and `personality_test_questions` on a scale of 0-10, which then determines the number of questions generated for each category (up to 5 per category).
*   **`/jd_only` Endpoint (Job Description Only)**: When resume data is not available, this endpoint generates interview questions based solely on the content of an uploaded job description file (PDF/DOCX). It typically generates a fixed set of questions (e.g., 5) for each category.
*   **Typical Question Categories**:
    *   Technical Questions
    *   Past Experience Questions
    *   Case Study Questions
    *   Situation Handling Questions
    *   Personality Test Questions

### Candidate-Job Fit Evaluation (`/intervet` and `/intervet2` endpoints)

These endpoints assess how well a candidate's profile aligns with a job's requirements:

*   **`/intervet` Endpoint**: This endpoint requires the resume data and job description data to be provided in JSON format (these are typically the outputs of the `/resume` and `/jd_parser` endpoints, respectively).
*   **`/intervet2` Endpoint**: This endpoint offers more convenience by allowing direct file uploads for both the resume (PDF/DOCX) and the job description (PDF/DOCX). The API handles the parsing of these files internally before conducting the evaluation.
*   **Main Evaluation Criteria**: The candidate-job fit is determined by evaluating several factors, including:
    *   Skills Match (both `direct_skills` explicitly listed and `subjective_skills` inferred from experience/projects)
    *   Years of Experience
    *   Reliability (assessed from job tenure and frequency of job changes)
    *   Location Match
    *   Education Match (comparing academic qualifications against requirements)
    *   Relevant Certifications
*   **Output**: The evaluation response includes:
    *   An `total_score` (typically 0-100).
    *   A `fit_category` (e.g., "Excellent Match", "Strong Match", "Good Match", "Moderate Match", "Weak Match").
    *   A textual `summary` of the evaluation.
    *   Detailed `scores` and `rationale` for each criterion, explaining the basis of the assessment and highlighting specific strengths or gaps.

### VAPI Call Summary Extraction (`/interfix` endpoint)

*   **Purpose**: The `/interfix` endpoint is designed to extract structured information from unstructured text summaries or transcripts generated by VAPI (Voice API) call agents, often used in initial HR screening calls.
*   **Key Information Extracted**: It aims to pull out key details useful for recruiters, such as `offer_in_hand` (candidate's current offer/salary), `notice_period`, `expected_salary`, `reason_to_switch`, `preferred_time_for_interview`, and `preferred_date_for_interview`.

### Batch Resume Evaluation (`/bunchtest` endpoint)

*   **Purpose**: This endpoint facilitates the efficient screening of multiple candidates for a single job opening. Users can upload multiple resume files (PDF or DOCX) along with one job description file (PDF or DOCX).
*   **Functionality**: The API processes each resume individually, performing a comprehensive candidate-job fit evaluation (identical to the `/intervet2` endpoint) against the common job description.
*   **Output**: The results are returned as a list, where each item represents the detailed evaluation for one resume. Crucially, these results are **sorted by the `total_score` in descending order**, enabling recruiters to quickly identify the top candidates from the batch. The response also includes the job title from the JD, the number of resumes processed, and a timestamp.

## Data Models/Schemas

The API utilizes Pydantic models to define the structure of request and response bodies, ensuring data consistency and providing clear specifications. Below is a brief overview of key data models:

*   **`GenerateRequest`**: Used for the `/generate` endpoint. It includes `prompt` (string, required) for the user's input and `history` (string, optional) for the preceding conversation context.

*   **`ResumeResponse`**: This is the detailed JSON structure returned by the `/resume` endpoint. It encapsulates all parsed information from a resume. Key components include:
    *   Basic candidate details (`name`, `email`, `phone`, `location`, `summary`).
    *   Nested structures for detailed sections like `education` (list of `EducationEntry`), `experience` (list of `ExperienceEntry`), and `projects` (list of `ProjectEntry`).
    *   `direct_skills` and `subjective_skills` (dictionaries mapping skills to their context/rationale).
    *   Lists for `certifications` (list of `CertificationEntry`), `languages` (list of `LanguageEntry`), `social_media`, `publications`, `achievements`, `volunteer_experience`, and `domain_of_interest`.
    *   `confidence_score` (float) and `confidence_details` (dict) for parsing accuracy.
    *   Models like `EducationEntry`, `ExperienceEntry`, `ProjectEntry`, and `CertificationEntry` define the schema for individual items within these lists, capturing granular details (e.g., `EducationEntry` includes `degree`, `institution`, `year`).

*   **`JobDescriptionResponse`**: This model defines the JSON structure for the output of the `/jd_parser` endpoint. It includes:
    *   Core job details (`job_title`, `company_name`, `location`, `job_type`, `work_mode`, `summary`).
    *   Lists for `responsibilities`, `required_skills`, `preferred_skills`, `education_requirements`.
    *   Nested structures like `education_details` (object with `degree_level`, `field_of_study`), `benefits` (list of `BenefitEntry`), and `requirements` (list of `RequirementEntry`).
    *   `confidence_score` (float) for parsing accuracy.

*   **`JDQuestionRequest`**: The request model for the `/jd` endpoint (generating interview questions with JD and resume). It requires:
    *   `resume_json` (object): The parsed resume data, typically from the `/resume` endpoint.
    *   Scales (integer, 0-10) for various question categories: `technical_questions`, `past_experience_questions`, `case_study_questions`, `situation_handling_questions`, `personality_test_questions`.

*   **`IntervetRequest`**: Used for the `/intervet` endpoint (candidate-job fit evaluation with JSON inputs). It contains:
    *   `resume_json` (object): Parsed resume data.
    *   `jd_json` (object): Parsed job description data.

*   **`InterfixRequest`**: The request model for the `/interfix` endpoint (VAPI call summary extraction). It includes:
    *   `summary` (string): The text summary or transcript of the call.

*   **`InterfixResponse`**: The response model for the `/interfix` endpoint. It structures the extracted call information, such as `offer_in_hand` (float, optional), `notice_period` (string, optional), `expected_salary` (float, optional), `reason_to_switch` (string, optional), `preferred_time_for_interview` (string, optional), and `preferred_date_for_interview` (string, optional).

These models ensure that data exchanged with the API is well-defined and validated, promoting reliable integrations.
