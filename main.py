import json
import os
import re
import tempfile
import logging
import time
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from pydantic import BaseModel, Field  # <-- Required to define request schema
import PyPDF2
import docx  # For DOCX file processing
from typing import Dict, List, Optional, Literal, Any, Callable, Tuple

# Import metrics logging
from metrics_logger import RequestMetrics, log_model_metrics, log_file_processing_metrics, shutdown_metrics_logger

# Configure logging with a more user-friendly format
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.StreamHandler()
    ]
)

# Create a custom filter to simplify some log messages
class UserFriendlyFilter(logging.Filter):
    def filter(self, record):
        # Simplify common log messages
        if "Sending prompt to" in record.msg:
            record.msg = "🤖 Processing with Gemma 3:4b model..."
        elif "Received response from" in record.msg:
            record.msg = "✅ Received model response in {:.2f}s".format(
                float(record.msg.split("in ")[1].split("s")[0])
            )
        elif "Extracting text from PDF" in record.msg or "Extracting text from DOCX" in record.msg:
            record.msg = "📄 Extracting text from document..."
        elif "Successfully extracted" in record.msg and "characters from" in record.msg:
            record.msg = "✅ Successfully extracted document text"
        elif "Parsing resume with" in record.msg:
            record.msg = "🔍 Analyzing resume content..."
        elif "Parsing job description with" in record.msg:
            record.msg = "🔍 Analyzing job description content..."
        elif "Starting resume parsing" in record.msg:
            record.msg = "🚀 Starting resume analysis..."
        elif "Starting job description parsing" in record.msg:
            record.msg = "🚀 Starting job description analysis..."
        elif "Normalizing" in record.msg:
            record.msg = "📊 Organizing extracted data..."

        return True

# Get logger and add the filter
logger = logging.getLogger("gemma-api")
logger.addFilter(UserFriendlyFilter())

# Fix SSL certificate issue for ollama client
os.environ.pop('SSL_CERT_FILE', None)

# Now import ollama after fixing the SSL issue
import ollama

MODEL_NAME = "gemma3:4b"

# Define detailed resume schema models
class EducationEntry(BaseModel):
    degree: str
    institution: str
    year: str
    gpa: Optional[str] = None
    location: Optional[str] = None
    major: Optional[str] = None
    minor: Optional[str] = None
    achievements: Optional[List[str]] = None
    courses: Optional[List[str]] = None

class ExperienceEntry(BaseModel):
    company_name: str
    role: str
    duration: str
    key_responsibilities: str
    location: Optional[str] = None
    achievements: Optional[List[str]] = None
    technologies_used: Optional[List[str]] = None
    team_size: Optional[str] = None
    industry: Optional[str] = None

class ProjectEntry(BaseModel):
    name: str
    description: Optional[str] = None
    duration: Optional[str] = None
    technologies_used: Optional[List[str]] = None
    url: Optional[str] = None
    role: Optional[str] = None
    achievements: Optional[List[str]] = None

class CertificationEntry(BaseModel):
    name: str
    issuer: Optional[str] = None
    date: Optional[str] = None
    expiry: Optional[str] = None
    url: Optional[str] = None

class SocialMediaEntry(BaseModel):
    platform: str
    url: str
    username: Optional[str] = None

class LanguageEntry(BaseModel):
    name: str
    proficiency: Optional[str] = None

class PublicationEntry(BaseModel):
    title: str
    publisher: Optional[str] = None
    date: Optional[str] = None
    url: Optional[str] = None
    authors: Optional[List[str]] = None
    description: Optional[str] = None

class AchievementEntry(BaseModel):
    title: str
    date: Optional[str] = None
    issuer: Optional[str] = None
    description: Optional[str] = None

class VolunteerEntry(BaseModel):
    organization: str
    role: str
    duration: Optional[str] = None
    description: Optional[str] = None

class ResumeResponse(BaseModel):
    name: str
    email: Optional[str] = None
    phone: Optional[str] = None
    summary: Optional[str] = None
    location: Optional[str] = None
    education: List[EducationEntry] = []
    skills: List[str] = []  # Original skills list
    direct_skills: Dict[str, str] = {}  # Skills explicitly mentioned in resume
    subjective_skills: Dict[str, str] = {}  # Skills inferred from experience/projects
    experience: List[ExperienceEntry] = []
    projects: List[ProjectEntry] = []
    certifications: List[CertificationEntry] = []
    languages: List[LanguageEntry] = []
    social_media: List[SocialMediaEntry] = []
    publications: List[PublicationEntry] = []
    achievements: List[AchievementEntry] = []
    volunteer_experience: List[VolunteerEntry] = []
    domain_of_interest: List[str] = []
    references: List[Dict[str, str]] = []
    confidence_score: float = 0.0  # Overall confidence in the parsing
    confidence_details: Dict[str, float] = {}  # Detailed breakdown of confidence scores by field

class BenefitEntry(BaseModel):
    title: str
    description: Optional[str] = None

class RequirementEntry(BaseModel):
    title: str
    description: Optional[str] = None
    is_mandatory: Optional[bool] = True

class JobDescriptionResponse(BaseModel):
    job_title: str
    company_name: Optional[str] = None
    location: Optional[str] = None
    job_type: Optional[str] = None  # Full-time, Part-time, Contract, etc.
    work_mode: Optional[str] = None  # Remote, Hybrid, On-site
    department: Optional[str] = None
    summary: Optional[str] = None
    responsibilities: List[str] = []
    required_skills: List[str] = []
    preferred_skills: List[str] = []
    required_experience: Optional[str] = None  # e.g., "3+ years"
    education_requirements: List[str] = []
    salary_range: Optional[str] = None
    benefits: List[BenefitEntry] = []
    requirements: List[RequirementEntry] = []
    application_deadline: Optional[str] = None
    posting_date: Optional[str] = None
    contact_information: Optional[Dict[str, str]] = None
    company_description: Optional[str] = None
    industry: Optional[str] = None
    career_level: Optional[str] = None  # Entry, Mid, Senior
    confidence_score: float = 0.0  # Overall confidence in the parsing

    # Additional fields
    class Config:
        extra = "allow"  # Allow additional fields beyond the predefined schema
        schema_extra = {
            "example": {
                "name": "John Doe",
                "email": "john.doe@example.com",
                "phone": "123-456-7890",
                "summary": "Experienced software engineer with 5+ years in machine learning and AI.",
                "location": "San Francisco, CA",
                "education": [
                    {
                        "degree": "B.Tech Computer Science",
                        "institution": "Example University",
                        "year": "2018-2022",
                        "gpa": "3.8/4.0",
                        "location": "San Francisco, CA",
                        "major": "Computer Science",
                        "courses": ["Machine Learning", "Data Structures", "Algorithms"]
                    }
                ],
                "skills": {
                    "Python": "Developed multiple ML models using Python and scikit-learn",
                    "Machine Learning": "Implemented recommendation systems using collaborative filtering",
                    "NLP": "Built text classification models for sentiment analysis"
                },
                "experience": [
                    {
                        "company_name": "Example Corp",
                        "role": "ML Engineer",
                        "duration": "2022-Present",
                        "key_responsibilities": "Developed ML models for recommendation systems",
                        "location": "San Francisco, CA",
                        "technologies_used": ["Python", "TensorFlow", "PyTorch"]
                    }
                ],
                "projects": [
                    {
                        "name": "Resume Parser",
                        "description": "Built an AI-powered resume parser using NLP techniques",
                        "technologies_used": ["Python", "spaCy", "FastAPI"]
                    }
                ],
                "certifications": [
                    {
                        "name": "AWS Certified ML Specialist",
                        "issuer": "Amazon Web Services",
                        "date": "2023"
                    }
                ],
                "languages": [
                    {"name": "English", "proficiency": "Native"},
                    {"name": "Spanish", "proficiency": "Intermediate"}
                ],
                "social_media": [
                    {"platform": "LinkedIn", "url": "https://linkedin.com/in/johndoe"},
                    {"platform": "GitHub", "url": "https://github.com/johndoe"}
                ],
                "domain_of_interest": ["AI", "ML", "NLP"],
                "confidence_score": 0.92
            }
        }

app = FastAPI(
    title="Gemma 3:4B API",
    description="API for interacting with Gemma 3:4B model and parsing resumes",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create a metrics middleware
class MetricsMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Extract the endpoint path
        path = request.url.path

        # Create a metrics tracker for this request
        metrics = RequestMetrics(endpoint=path)

        # Store metrics in request state for access in route handlers
        request.state.metrics = metrics

        try:
            # Call the next middleware or route handler
            response = await call_next(request)

            # Mark request as complete with success status
            metrics.mark_complete(status_code=response.status_code)

            return response

        except Exception as e:
            # Mark request as complete with error
            metrics.mark_complete(status_code=500, error=str(e))
            raise

# Add the metrics middleware
app.add_middleware(MetricsMiddleware)

# Register lifespan context for startup/shutdown events
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(_: FastAPI):  # Use _ to indicate unused variable
    # Startup: nothing to do
    yield
    # Shutdown: close metrics logger
    shutdown_metrics_logger()

# Update app with lifespan
app.router.lifespan_context = lifespan

class GenerateRequest(BaseModel):
    prompt: str
    history: str = ""  # optional default value

class JDQuestionRequest(BaseModel):
    resume_json: Dict[str, Any] = Field(..., description="Resume data in JSON format, typically obtained from the /resume endpoint")
    technical_questions: int = Field(..., ge=0, le=10, description="Scale 0-10 for technical questions")
    past_experience_questions: int = Field(..., ge=0, le=10, description="Scale 0-10 for past experience questions")
    case_study_questions: int = Field(..., ge=0, le=10, description="Scale 0-10 for case study questions")
    situation_handling_questions: int = Field(..., ge=0, le=10, description="Scale 0-10 for situation handling questions")
    personality_test_questions: int = Field(..., ge=0, le=10, description="Scale 0-10 for personality test questions")

class InterfixRequest(BaseModel):
    summary: str = Field(..., description="Summary or transcript of VAPI response (AI call agent)")

    class Config:
        schema_extra = {
            "example": {
                "summary": "The call was an automated HR screening for an AI-powered full stack developer position. The candidate indicated they have a 2-month notice period, expect a salary of 1 lakh rupees monthly, and are seeking to change jobs primarily to relocate to the company's location. They expressed flexibility for both in-office and remote work arrangements."
            }
        }

class InterfixResponse(BaseModel):
    offer_in_hand: Optional[float] = None
    notice_period: Optional[str] = None
    expected_salary: Optional[float] = None
    reason_to_switch: Optional[str] = None
    preferred_time_for_interview: Optional[str] = None
    preferred_date_for_interview: Optional[str] = None

class IntervetRequest(BaseModel):
    resume_json: Dict[str, Any] = Field(..., description="Resume data in JSON format, typically obtained from the /resume endpoint")
    jd_json: Dict[str, Any] = Field(..., description="Job description data in JSON format, typically obtained from the /jd_parser endpoint")

    class Config:
        schema_extra = {
            "example": {
                "resume_json": {
                    "name": "John Doe",
                    "email": "john.doe@example.com",
                    "phone": "+1234567890",
                    "education": [
                        {"degree": "B.Tech Computer Science", "institution": "Example University", "year": "2018-2022"}
                    ],
                    "skills": ["Python", "Machine Learning", "FastAPI", "SQL"],
                    "experience": [
                        {"company_name": "Tech Corp", "role": "Software Engineer", "duration": "2022-Present",
                         "key_responsibilities": "Developed APIs and ML models"}
                    ],
                    "projects": ["Resume Parser", "Recommendation System"],
                    "domain_of_interest": ["AI", "Web Development"]
                },
                "jd_json": {
                    "job_title": "Full Stack Developer",
                    "required_skills": ["Python", "JavaScript", "SQL"],
                    "required_experience": "2+ years",
                    "education_requirements": ["Bachelor's degree in Computer Science or related field"]
                }
            }
        }

def get_response(prompt: str, timeout_seconds: int = 60, max_tokens: int = 1000, image_path: str = None, request_metrics: RequestMetrics = None) -> str:
    try:
        logger.info(f"Sending prompt to {MODEL_NAME} with {timeout_seconds}s timeout")

        # Initialize response
        response = ""

        # Set up the stream with timeout
        from concurrent.futures import ThreadPoolExecutor

        start_time = time.time()

        # Create a function to process the stream with timeout tracking
        def process_stream():
            nonlocal response
            try:
                # Prepare options
                options = {"num_predict": max_tokens}

                # If image path is provided, use it in the prompt
                if image_path and os.path.exists(image_path):
                    logger.info(f"Including image from path: {image_path}")

                    # For Gemma 3, we can use the format that includes image paths directly
                    # The format may vary depending on the Ollama version and model
                    try:
                        # First attempt: Use the images parameter (newer Ollama versions)
                        result = ollama.generate(
                            model=MODEL_NAME,
                            prompt=prompt,
                            images=[image_path],
                            stream=False,
                            options=options
                        )
                        return result
                    except Exception as img_error:
                        logger.warning(f"Error using images parameter: {img_error}. Trying alternative method...")

                        # Second attempt: Use a special format in the prompt (older Ollama versions)
                        # This is a fallback method that might work with some Ollama configurations
                        with open(image_path, 'rb') as img_file:
                            import base64
                            img_data = base64.b64encode(img_file.read()).decode('utf-8')

                            # Create a special prompt format that includes the image data
                            # This format might work with some Ollama configurations
                            special_prompt = f"[img]{img_data}[/img]\n{prompt}"

                            result = ollama.generate(
                                model=MODEL_NAME,
                                prompt=special_prompt,
                                stream=False,
                                options=options
                            )
                            return result
                else:
                    # Use non-streaming mode with text-only prompt
                    result = ollama.generate(
                        model=MODEL_NAME,
                        prompt=prompt,
                        stream=False,
                        options=options
                    )
                    return result
            except Exception as e:
                logger.error(f"Error in model generation: {e}")
                raise

        # Use ThreadPoolExecutor to run with a strict timeout
        with ThreadPoolExecutor() as executor:
            future = executor.submit(process_stream)
            try:
                result = future.result(timeout=timeout_seconds)
                processing_time = time.time() - start_time
                logger.info(f"Received response from {MODEL_NAME} in {processing_time:.2f}s")

                # Extract the response text
                response_text = result["response"]

                # Log metrics if a request_metrics object was provided
                if request_metrics:
                    # Mark first byte received
                    request_metrics.mark_first_byte()

                    # Log model metrics
                    log_model_metrics(
                        request_metrics=request_metrics,
                        model_name=MODEL_NAME,
                        prompt_length=len(prompt),
                        response_length=len(response_text),
                        processing_time=processing_time,
                        error=None
                    )

                    # Add token usage if available in the response
                    if "eval_count" in result:
                        request_metrics.add_metric("eval_count", result["eval_count"])
                    if "prompt_eval_count" in result:
                        request_metrics.add_metric("prompt_eval_count", result["prompt_eval_count"])
                    if "total_duration" in result:
                        request_metrics.add_metric("model_total_duration", result["total_duration"])

                return response_text

            except TimeoutError:
                logger.warning(f"Response generation timed out after {timeout_seconds}s")

                # Log error metrics if a request_metrics object was provided
                if request_metrics:
                    log_model_metrics(
                        request_metrics=request_metrics,
                        model_name=MODEL_NAME,
                        prompt_length=len(prompt),
                        response_length=0,
                        processing_time=timeout_seconds,
                        error=f"Timeout after {timeout_seconds}s"
                    )

                raise HTTPException(
                    status_code=504,
                    detail=f"Response generation timed out after {timeout_seconds} seconds. Try reducing the complexity of your request."
                )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating response: {e}")

        # Log error metrics if a request_metrics object was provided
        if request_metrics:
            log_model_metrics(
                request_metrics=request_metrics,
                model_name=MODEL_NAME,
                prompt_length=len(prompt),
                response_length=0,
                processing_time=time.time() - start_time if 'start_time' in locals() else 0,
                error=str(e)
            )

        raise HTTPException(status_code=500, detail=f"Error generating response: {str(e)}")

def extract_text_from_pdf(file_path: str) -> str:
    """Extract text from a PDF file."""
    try:
        logger.info(f"Extracting text from PDF: {file_path}")
        text = ""
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            total_pages = len(pdf_reader.pages)
            logger.info(f"PDF has {total_pages} pages")

            for page_num in range(total_pages):
                page = pdf_reader.pages[page_num]
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
                else:
                    logger.warning(f"No text extracted from page {page_num + 1}")

        if not text.strip():
            logger.error("No text could be extracted from the PDF")
            raise HTTPException(status_code=400, detail="Could not extract text from the PDF. The file might be scanned or protected.")

        logger.info(f"Successfully extracted {len(text)} characters from PDF")
        return text
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error extracting text from PDF: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing the PDF file: {str(e)}")


def extract_text_from_docx(file_path: str) -> str:
    """Extract text from a DOCX file."""
    try:
        logger.info(f"Extracting text from DOCX: {file_path}")
        doc = docx.Document(file_path)

        # Extract text from paragraphs
        paragraphs_text = [paragraph.text for paragraph in doc.paragraphs if paragraph.text.strip()]
        logger.info(f"Extracted {len(paragraphs_text)} paragraphs from DOCX")

        # Extract text from tables
        tables_text = []
        for table in doc.tables:
            for row in table.rows:
                row_text = [cell.text for cell in row.cells if cell.text.strip()]
                if row_text:
                    tables_text.append(' | '.join(row_text))

        if tables_text:
            logger.info(f"Extracted text from {len(doc.tables)} tables in DOCX")

        # Combine all text
        all_text = '\n'.join(paragraphs_text + tables_text)

        if not all_text.strip():
            logger.error("No text could be extracted from the DOCX file")
            raise HTTPException(status_code=400, detail="Could not extract text from the DOCX file. The file might be empty or corrupted.")

        logger.info(f"Successfully extracted {len(all_text)} characters from DOCX")
        return all_text
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error extracting text from DOCX: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing the DOCX file: {str(e)}")


def extract_text_from_image(file_path: str) -> str:
    """Extract text from a file by treating it as an image and using the LLM's vision capabilities."""
    logger.info(f"Attempting to extract text from file as image: {file_path}")

    try:
        # Create a prompt that asks the model to extract all text from the image
        image_prompt = """
        This document contains important text. Please extract ALL text content from this image.
        Format your response as plain text only, preserving the structure as much as possible.
        Include ALL text visible in the document, including:
        - Headers and titles
        - Bullet points and numbered lists
        - Dates and contact information
        - Skills, requirements, and qualifications
        - Any other textual information present

        Do not add any commentary, just extract the text content exactly as it appears.
        Preserve formatting like bullet points, section headers, and paragraph breaks as much as possible.
        """

        # Get response with a longer timeout for image processing
        response = get_response(
            prompt=image_prompt,
            timeout_seconds=120,
            max_tokens=2000,
            image_path=file_path
        )

        # Clean up the response to remove any potential commentary
        text = response.strip()

        # Check if we got a meaningful response
        if len(text) < 50:  # Arbitrary threshold for meaningful content
            logger.warning(f"Image-based text extraction returned very little text ({len(text)} chars)")
            raise ValueError("Image-based text extraction returned insufficient content")

        logger.info(f"Successfully extracted {len(text)} characters using image-based extraction")
        return text

    except Exception as e:
        logger.error(f"Error in image-based text extraction: {e}")
        raise ValueError(f"Failed to extract text using image-based approach: {str(e)}")

def extract_text_from_file(file_path: str, file_type: Literal["pdf", "docx"], request_metrics: RequestMetrics = None) -> str:
    """Extract text from a file based on its type with waterfall fallback mechanism.

    This function implements a waterfall mechanism for text extraction:
    1. First, it tries to extract text using the standard method based on file type
    2. If that fails, it falls back to treating the file as an image and using the LLM's vision capabilities
    """
    start_time = time.time()
    file_size = os.path.getsize(file_path) if os.path.exists(file_path) else 0

    try:
        # First attempt: Use standard extraction based on file type
        if file_type == "pdf":
            text = extract_text_from_pdf(file_path)
            extraction_method = "pdf_text"
        elif file_type == "docx":
            text = extract_text_from_docx(file_path)
            extraction_method = "docx_text"
        else:
            raise ValueError(f"Unsupported file type: {file_type}")

        # Log file processing metrics if request_metrics is provided
        if request_metrics:
            processing_time = time.time() - start_time
            log_file_processing_metrics(
                request_metrics=request_metrics,
                file_type=file_type,
                file_size=file_size,
                extraction_method=extraction_method,
                text_length=len(text),
                processing_time=processing_time
            )

        return text

    except Exception as primary_error:
        # If standard extraction fails, log the error and try image-based extraction
        logger.warning(f"Standard text extraction failed: {primary_error}. Attempting image-based extraction as fallback.")

        try:
            # Second attempt: Use image-based extraction as fallback
            text = extract_text_from_image(file_path)

            # Log file processing metrics if request_metrics is provided
            if request_metrics:
                processing_time = time.time() - start_time
                log_file_processing_metrics(
                    request_metrics=request_metrics,
                    file_type=file_type,
                    file_size=file_size,
                    extraction_method="image_based",
                    text_length=len(text),
                    processing_time=processing_time
                )

            return text

        except Exception as fallback_error:
            # If both methods fail, raise a comprehensive error
            logger.error(f"Both standard and image-based text extraction failed. Primary error: {primary_error}, Fallback error: {fallback_error}")

            # Log failure metrics if request_metrics is provided
            if request_metrics:
                processing_time = time.time() - start_time
                log_file_processing_metrics(
                    request_metrics=request_metrics,
                    file_type=file_type,
                    file_size=file_size,
                    extraction_method="failed",
                    text_length=0,
                    processing_time=processing_time
                )
                request_metrics.add_metric("extraction_error", f"{primary_error} | {fallback_error}")

            raise HTTPException(
                status_code=400,
                detail=f"Could not extract text from the file using any available method. The file might be corrupted, protected, or in an unsupported format."
            )

def calculate_resume_confidence_score(parsed_data: Dict) -> float:
    """
    Calculate a confidence score for the parsed resume data.

    This function evaluates how confident the model is in the correctness of the extracted information.
    It analyzes the structure, consistency, and clarity of the parsed data to estimate
    the likelihood that the information was correctly extracted.

    The scoring system works as follows:
    1. Each field has a weight based on how reliably it can be extracted
    2. Fields are scored based on structural correctness and consistency
    3. The final score represents the model's confidence in the extraction accuracy
    4. A higher score means higher confidence in the correctness of the extracted data

    Returns:
        float: overall_confidence_score
    """
    # Define weights for different fields based on extraction reliability
    field_weights = {
        "name": 0.10,  # Names are usually clear and easy to extract
        "email": 0.10,  # Emails have a standard format and are easy to identify
        "phone": 0.08,  # Phone numbers have patterns but can vary in format
        "summary": 0.05,  # Summaries can be ambiguous
        "location": 0.07,  # Locations are usually clear
        "education": 0.12,  # Education sections are usually well-structured
        "skills": 0.15,  # Skills can be scattered or in different formats
        "experience": 0.15,  # Experience is complex and can be ambiguous
        "projects": 0.08,  # Projects can be mixed with experience
        "certifications": 0.05,  # Certifications are usually clear
        "languages": 0.05  # Languages are usually in a standard format
    }

    # Calculate confidence based on structural correctness and consistency
    score = 0.0
    total_weight = 0.0
    detailed_scores = {}

    for field, weight in field_weights.items():
        total_weight += weight
        field_score = 0.0

        if field not in parsed_data:
            detailed_scores[field] = 0.0
            continue

        field_value = parsed_data[field]

        # Check if field has a value
        if field_value is None:
            detailed_scores[field] = 0.0
            continue

        # For string fields - check if they match expected patterns
        if isinstance(field_value, str):
            if field_value.strip():
                if field == "email":
                    # Check if it looks like an email (contains @ and .)
                    if "@" in field_value and "." in field_value.split("@")[1]:
                        field_score = 0.95  # High confidence for well-formed emails
                    else:
                        field_score = 0.3   # Low confidence for malformed emails
                elif field == "phone":
                    # Check if it looks like a phone number (contains digits and common separators)
                    if re.search(r'[\d\(\)\+\-\s\.]{7,}', field_value) and sum(c.isdigit() for c in field_value) >= 7:
                        field_score = 0.9   # High confidence for well-formed phone numbers
                    else:
                        field_score = 0.4   # Lower confidence for unusual formats
                else:
                    # For other string fields, confidence is based on length and content
                    # Very short or very long values might be less reliable
                    length = len(field_value.strip())
                    if 2 <= length <= 100:
                        field_score = 0.85  # Reasonable length, higher confidence
                    else:
                        field_score = 0.6   # Unusual length, lower confidence
            else:
                field_score = 0.0

        # For list fields - check structure and consistency
        elif isinstance(field_value, list):
            if len(field_value) > 0:
                if field == "education":
                    # Check structure of education entries
                    structure_scores = []
                    for entry in field_value:
                        if isinstance(entry, dict):
                            # Check for expected keys in education entries
                            has_degree = bool(entry.get("degree", "").strip())
                            has_institution = bool(entry.get("institution", "").strip())
                            has_year = bool(entry.get("year", "").strip())

                            # Calculate structure score for this entry
                            entry_score = (has_degree + has_institution + has_year) / 3
                            structure_scores.append(entry_score)

                    # Average structure score across all entries
                    if structure_scores:
                        field_score = sum(structure_scores) / len(structure_scores)
                        # Adjust based on number of entries (more entries = more complex extraction)
                        field_score = field_score * (0.7 + 0.3 / (1 + 0.1 * len(field_value)))
                    else:
                        field_score = 0.5  # Moderate confidence for unusual structure

                elif field == "experience":
                    # Check structure of experience entries
                    structure_scores = []
                    for entry in field_value:
                        if isinstance(entry, dict):
                            # Check for expected keys in experience entries
                            has_company = bool(entry.get("company_name", "").strip())
                            has_role = bool(entry.get("role", "").strip())
                            has_duration = bool(entry.get("duration", "").strip())
                            has_responsibilities = bool(entry.get("key_responsibilities", "").strip())

                            # Calculate structure score for this entry
                            entry_score = (has_company + has_role + has_duration + has_responsibilities) / 4
                            structure_scores.append(entry_score)

                    # Average structure score across all entries
                    if structure_scores:
                        field_score = sum(structure_scores) / len(structure_scores)
                        # Adjust based on number of entries (more entries = more complex extraction)
                        field_score = field_score * (0.7 + 0.3 / (1 + 0.1 * len(field_value)))
                    else:
                        field_score = 0.5  # Moderate confidence for unusual structure

                elif field == "skills":
                    # For skills list, check if entries look like skills (typically 1-4 words)
                    valid_skills = 0
                    for skill in field_value:
                        if isinstance(skill, str):
                            words = len(skill.split())
                            if 1 <= words <= 4 and len(skill) <= 50:
                                valid_skills += 1

                    # Calculate confidence based on proportion of valid-looking skills
                    if field_value:
                        field_score = valid_skills / len(field_value)
                        # Adjust based on number of skills (very few or very many skills might be less reliable)
                        if len(field_value) < 3 or len(field_value) > 30:
                            field_score *= 0.8
                    else:
                        field_score = 0.0

                else:
                    # For other list fields, base confidence on consistency of structure
                    if all(isinstance(item, str) for item in field_value):
                        field_score = 0.8  # High confidence for consistent string lists
                    elif all(isinstance(item, dict) for item in field_value):
                        field_score = 0.75  # Good confidence for consistent dict lists
                    else:
                        field_score = 0.5  # Lower confidence for mixed-type lists
            else:
                field_score = 0.7  # Empty list might be correct (no data) or incorrect (missed data)

        # For dict fields (skills when in dictionary format)
        elif isinstance(field_value, dict):
            if len(field_value) > 0:
                if field == "skills":
                    # For skills dict, check if keys look like skills and values provide context
                    valid_entries = 0
                    for skill, context in field_value.items():
                        if isinstance(skill, str) and isinstance(context, str):
                            skill_words = len(skill.split())
                            if 1 <= skill_words <= 4 and len(skill) <= 50 and context.strip():
                                valid_entries += 1

                    # Calculate confidence based on proportion of valid-looking skill entries
                    if field_value:
                        field_score = valid_entries / len(field_value)
                    else:
                        field_score = 0.0
                else:
                    # For other dict fields, moderate confidence
                    field_score = 0.7
            else:
                field_score = 0.5  # Empty dict might be correct or incorrect

        # Add to total score
        score += weight * field_score
        detailed_scores[field] = round(field_score * 100, 1)  # Store as percentage

    # Normalize score
    if total_weight > 0:
        normalized_score = score / total_weight
    else:
        normalized_score = 0.0

    return round(normalized_score, 2)  # Return only the overall score


def calculate_jd_confidence_score(parsed_data: Dict) -> float:
    """
    Calculate a confidence score for the parsed job description data.

    This function evaluates how confident the model is in the correctness of the extracted information.
    It analyzes the structure, consistency, and clarity of the parsed data to estimate
    the likelihood that the information was correctly extracted.

    The scoring system works as follows:
    1. Each field has a weight based on how reliably it can be extracted
    2. Fields are scored based on structural correctness and consistency
    3. The final score represents the model's confidence in the extraction accuracy
    4. A higher score means higher confidence in the correctness of the extracted data

    Returns:
        float: overall_confidence_score
    """
    # Define weights for different fields based on extraction reliability
    field_weights = {
        "job_title": 0.12,  # Job titles are usually clear and prominent
        "company_name": 0.10,  # Company names are usually clear
        "location": 0.08,  # Locations are usually clear
        "job_type": 0.05,  # Job types can be ambiguous
        "work_mode": 0.05,  # Work modes can be ambiguous
        "summary": 0.08,  # Summaries can be mixed with other content
        "responsibilities": 0.15,  # Responsibilities are complex and can be scattered
        "required_skills": 0.15,  # Skills can be mixed with responsibilities
        "preferred_skills": 0.10,  # Preferred skills can be ambiguous
        "required_experience": 0.07,  # Experience requirements can be embedded in text
        "education_requirements": 0.05  # Education requirements are usually clear
    }

    # Calculate confidence based on structural correctness and consistency
    score = 0.0
    total_weight = 0.0
    detailed_scores = {}

    for field, weight in field_weights.items():
        total_weight += weight
        field_score = 0.0

        if field not in parsed_data:
            detailed_scores[field] = 0.0
            continue

        field_value = parsed_data[field]

        # Check if field has a value
        if field_value is None:
            detailed_scores[field] = 0.7  # Null might be correct (no data in JD)
            continue

        # For string fields - check if they match expected patterns
        if isinstance(field_value, str):
            if field_value.strip():
                if field == "job_title":
                    # Job titles are usually concise
                    words = len(field_value.split())
                    if 1 <= words <= 10:
                        field_score = 0.9  # High confidence for reasonable job titles
                    else:
                        field_score = 0.6  # Lower confidence for unusually long titles

                elif field == "required_experience":
                    # Check if it looks like an experience requirement (contains years/months and numbers)
                    if re.search(r'\d+\s*(?:year|yr|month|mo|yrs|years|months)', field_value.lower()):
                        field_score = 0.9  # High confidence for well-formed experience requirements
                    else:
                        field_score = 0.6  # Lower confidence for unusual formats

                else:
                    # For other string fields, confidence is based on length and content
                    length = len(field_value.strip())
                    if field in ["company_name", "location", "job_type", "work_mode"] and length <= 50:
                        field_score = 0.85  # Reasonable length for these fields
                    elif field == "summary" and 20 <= length <= 500:
                        field_score = 0.8   # Reasonable length for summary
                    else:
                        field_score = 0.7   # Default confidence
            else:
                field_score = 0.0

        # For list fields - check structure and consistency
        elif isinstance(field_value, list):
            if len(field_value) > 0:
                if field in ["responsibilities", "required_skills", "preferred_skills"]:
                    # Check if entries look like responsibilities or skills (complete phrases/sentences)
                    valid_entries = 0
                    for entry in field_value:
                        if isinstance(entry, str):
                            # Responsibilities and skills should be reasonably sized text
                            if 3 <= len(entry.strip()) <= 200:
                                valid_entries += 1

                    # Calculate confidence based on proportion of valid-looking entries
                    if field_value:
                        field_score = valid_entries / len(field_value)

                        # Adjust based on number of entries
                        if field == "responsibilities" and not (3 <= len(field_value) <= 15):
                            field_score *= 0.9  # Unusual number of responsibilities
                        elif field == "required_skills" and not (2 <= len(field_value) <= 20):
                            field_score *= 0.9  # Unusual number of required skills
                    else:
                        field_score = 0.0

                elif field == "education_requirements":
                    # Education requirements are usually short phrases
                    valid_entries = 0
                    for entry in field_value:
                        if isinstance(entry, str):
                            words = len(entry.split())
                            if 2 <= words <= 15:
                                valid_entries += 1

                    # Calculate confidence based on proportion of valid-looking entries
                    if field_value:
                        field_score = valid_entries / len(field_value)
                    else:
                        field_score = 0.0

                else:
                    # For other list fields, base confidence on consistency
                    if all(isinstance(item, str) for item in field_value):
                        field_score = 0.8  # High confidence for consistent string lists
                    elif all(isinstance(item, dict) for item in field_value):
                        field_score = 0.75  # Good confidence for consistent dict lists
                    else:
                        field_score = 0.5  # Lower confidence for mixed-type lists
            else:
                field_score = 0.7  # Empty list might be correct (no data) or incorrect (missed data)

        # For dict fields or complex objects
        elif isinstance(field_value, dict) or (isinstance(field_value, list) and all(isinstance(item, dict) for item in field_value)):
            # Complex structures like benefits or requirements
            field_score = 0.75  # Moderate confidence for complex structures

        # Add to total score
        score += weight * field_score
        detailed_scores[field] = round(field_score * 100, 1)  # Store as percentage

    # Normalize score
    if total_weight > 0:
        normalized_score = score / total_weight
    else:
        normalized_score = 0.0

    return round(normalized_score, 2)  # Return only the overall score

def convert_skills_to_dict(parsed_data: Dict) -> Dict:
    """
    Convert skills from list to dictionary with context and separate into direct and subjective skills.

    - Direct skills: Skills explicitly mentioned in the resume's skills section
    - Subjective skills: Skills inferred from projects or experience sections
    """
    if "skills" not in parsed_data or not isinstance(parsed_data["skills"], list):
        return parsed_data

    # Get the original skills list
    skills_list = parsed_data["skills"]

    # Initialize direct and subjective skills dictionaries
    direct_skills = {}
    subjective_skills = {}

    # Extract context from experience
    experience_context = {}
    if "experience" in parsed_data and isinstance(parsed_data["experience"], list):
        for exp in parsed_data["experience"]:
            if isinstance(exp, dict):
                # Extract text from experience entries
                exp_text = ""
                for _, value in exp.items():  # Use _ to indicate unused variable
                    if isinstance(value, str):
                        exp_text += value + " "

                # Check which skills are mentioned in this experience
                for skill in skills_list:
                    if isinstance(skill, str) and skill.lower() in exp_text.lower():
                        # Use the role and company as context
                        role = exp.get("role", "")
                        company = exp.get("company_name", "")
                        if role and company:
                            experience_context[skill] = f"Used as {role} at {company}"

    # Extract context from projects
    project_context = {}
    if "projects" in parsed_data and isinstance(parsed_data["projects"], list):
        for proj in parsed_data["projects"]:
            if isinstance(proj, dict) and "description" in proj:
                proj_desc = proj.get("description", "")
                if isinstance(proj_desc, str):
                    for skill in skills_list:
                        if isinstance(skill, str) and skill.lower() in proj_desc.lower():
                            project_context[skill] = f"Used in project: {proj.get('name', 'Unknown project')}"

    # Categorize skills as direct or subjective
    for skill in skills_list:
        if isinstance(skill, str):
            # If the skill is mentioned in experience or projects, it's subjective
            if skill in experience_context or skill in project_context:
                if skill in experience_context:
                    subjective_skills[skill] = experience_context[skill]
                else:
                    subjective_skills[skill] = project_context[skill]
            else:
                # Otherwise, it's a direct skill
                direct_skills[skill] = "Mentioned in resume"

    # Replace the skills list with the dictionaries
    parsed_data["direct_skills"] = direct_skills
    parsed_data["subjective_skills"] = subjective_skills

    # Keep the original skills list for backward compatibility
    parsed_data["skills"] = skills_list

    return parsed_data

def parse_resume_with_gemma(resume_text: str) -> Dict:
    """Parse resume text using the Gemma model."""
    logger.info("Parsing resume with Gemma model")

    prompt = f"""
    You are an expert resume parser. Your task is to extract ALL structured information from the resume text below.

    Follow these guidelines:
    1. Extract ALL information that is explicitly mentioned in the resume text.
    2. Format your response as a valid JSON object with EXACTLY the following structure:

    {{
        "name": "Full Name",
        "email": "email@example.com" or null,
        "phone": "+1234567890" or null,
        "education": [
            {{
                "degree": "Full Degree Name (Including Specialization)",
                "institution": "Institution Name",
                "year": "Year or Date Range"
            }}
        ],
        "direct_skills": {{
            "Skill 1": "Mentioned in skills section",
            "Skill 2": "Listed in technical proficiencies",
            ...
        }},
        "subjective_skills": {{
            "Skill 3": "Used in Project X as described in projects section",
            "Skill 4": "Applied during role at Company Y as mentioned in experience",
            ...
        }},
        "experience": [
            {{
                "company_name": "Company Name with Location if mentioned",
                "role": "Job Title",
                "duration": "Date Range",
                "key_responsibilities": "Detailed description of responsibilities and achievements"
            }}
        ],
        "projects": [
            {{
                "name": "Project Name",
                "description": "Detailed project description including technologies used"
            }}
        ],
        "certifications": [],
        "domain_of_interest": ["Interest 1", "Interest 2", ...],
        "languages_known": ["Language 1", "Language 2", ...],
        "achievements": [],
        "publications": [],
        "volunteer_experience": [],
        "references": [],
        "summary": null,
        "personal_projects": [],
        "social_media": ["platform1.com/username", "platform2.com/username"]
    }}

    3. For arrays, if no information is available, use an empty array []
    4. For string fields, if no information is available, use null
    5. Do not make up or infer information that is not explicitly stated in the resume
    6. Ensure the JSON is properly formatted and valid
    7. IMPORTANT: For skills extraction:
       - "direct_skills" should include skills explicitly mentioned in a skills or technical proficiencies section
       - "subjective_skills" should include skills inferred from project descriptions or work experience
       - Both should be dictionaries where the key is the skill and the value is a brief rationale or context
    8. IMPORTANT: For experience entries, include all details in the key_responsibilities field as a single string with line breaks (\n)
    9. IMPORTANT: For projects, include all details in the description field as a single string with line breaks (\n)
    10. IMPORTANT: Make sure all JSON is valid - check for missing commas, extra commas, proper quotes, and proper nesting of objects and arrays
    
    Resume text:
    {resume_text}

    Respond ONLY with the JSON object, nothing else. Do not include explanations, markdown formatting, or code blocks.
    """

    try:
        # Use a 120 second timeout for resume parsing
        response = get_response(prompt, timeout_seconds=120)

        # Log the raw response for debugging
        logger.info(f"Raw response length: {len(response)}")
        if len(response) < 500:
            logger.info(f"Raw response content: {response}")
        else:
            logger.info(f"Raw response preview: {response[:500]}...")

        # Try to extract JSON from the response
        json_str = response.strip()

        # Remove any markdown formatting if present
        if "```json" in json_str:
            json_str = json_str.split("```json")[1].split("```")[0].strip()
        elif "```" in json_str:
            json_str = json_str.split("```")[1].split("```")[0].strip()

        # Try to find the JSON object if there's additional text
        if json_str.find('{') >= 0 and json_str.rfind('}') >= 0:
            start = json_str.find('{')
            end = json_str.rfind('}') + 1
            json_str = json_str[start:end]

        # Log the extracted JSON string for debugging
        logger.info(f"Extracted JSON string length: {len(json_str)}")
        if len(json_str) < 500:
            logger.info(f"Extracted JSON string: {json_str}")
        else:
            logger.info(f"Extracted JSON string preview: {json_str[:500]}...")

        # Try to parse the JSON string into a Python dictionary
        try:
            parsed_data = json.loads(json_str)
            return parsed_data
        except json.JSONDecodeError as json_err:
            # Log detailed error information
            error_line = json_str.splitlines()[json_err.lineno-1] if json_err.lineno <= len(json_str.splitlines()) else "Line not available"
            logger.error(f"JSON decode error at line {json_err.lineno}, column {json_err.colno}, char {json_err.pos}: {json_err.msg}")
            logger.error(f"Error line content: {error_line}")

            # Attempt to fix common JSON errors
            logger.info("Attempting to fix JSON formatting issues...")

            # Fix 1: Try to repair common JSON syntax errors
            fixed_json = json_str

            # Fix trailing commas in arrays and objects
            fixed_json = fixed_json.replace(",]", "]").replace(",}", "}")

            # Fix missing commas between array elements or object properties
            fixed_json = fixed_json.replace("}{", "},{").replace("][", "],[").replace("\"\"", "\",\"")

            # Fix unescaped quotes in strings
            # This regex finds unescaped quotes within string values
            fixed_json = re.sub(r'(?<!\\)"(?=.*?[^\\]":)', '\\"', fixed_json)

            # Fix 2: If still failing, try a more aggressive approach - use a minimal valid structure
            try:
                parsed_data = json.loads(fixed_json)
                logger.info("Successfully fixed JSON formatting issues")
                return parsed_data
            except json.JSONDecodeError:
                logger.warning("Could not fix JSON formatting issues with simple repairs")

                # Create a minimal valid structure as fallback
                fallback_data = {
                    "name": "Unknown",
                    "email": None,
                    "phone": None,
                    "education": [],
                    "direct_skills": {},
                    "subjective_skills": {},
                    "experience": [],
                    "projects": [],
                    "certifications": [],
                    "domain_of_interest": [],
                    "languages_known": [],
                    "social_media": [],
                    "achievements": [],
                    "publications": [],
                    "volunteer_experience": [],
                    "references": [],
                    "summary": None,
                    "error": "Failed to parse resume data due to JSON formatting issues",
                    "details": str(json_err)
                }

                # Try to extract some basic information using regex patterns
                # Name pattern - look for a name-like pattern at the beginning of the JSON
                name_match = re.search(r'"name"\s*:\s*"([^"]+)"', json_str)
                if name_match:
                    fallback_data["name"] = name_match.group(1)

                # Email pattern
                email_match = re.search(r'"email"\s*:\s*"([^"]+@[^"]+)"', json_str)
                if email_match:
                    fallback_data["email"] = email_match.group(1)

                # Phone pattern
                phone_match = re.search(r'"phone"\s*:\s*"([^"]+)"', json_str)
                if phone_match:
                    fallback_data["phone"] = phone_match.group(1)

                # Skills pattern - try to extract skills
                direct_skills_match = re.search(r'"direct_skills"\s*:\s*{(.*?)}', json_str, re.DOTALL)
                if direct_skills_match:
                    skills_str = direct_skills_match.group(1)
                    skills = re.findall(r'"([^"]+)"\s*:\s*"([^"]+)"', skills_str)
                    fallback_data["direct_skills"] = dict(skills)

                subjective_skills_match = re.search(r'"subjective_skills"\s*:\s*{(.*?)}', json_str, re.DOTALL)
                if subjective_skills_match:
                    skills_str = subjective_skills_match.group(1)
                    skills = re.findall(r'"([^"]+)"\s*:\s*"([^"]+)"', skills_str)
                    fallback_data["subjective_skills"] = dict(skills)

                logger.info(f"Created fallback data structure with partial information: {fallback_data['name']}")
                return fallback_data

    except Exception as e:
        logger.error(f"Error in Gemma parsing: {e}")
        # Return a minimal valid structure instead of raising an exception
        return {
            "name": "Unknown",
            "email": None,
            "phone": None,
            "education": [],
            "direct_skills": {},
            "subjective_skills": {},
            "experience": [],
            "projects": [],
            "certifications": [],
            "domain_of_interest": [],
            "languages_known": [],
            "social_media": [],
            "achievements": [],
            "publications": [],
            "volunteer_experience": [],
            "references": [],
            "summary": None,
            "error": "Failed to parse resume data",
            "details": str(e)
        }


def parse_jd_with_gemma(jd_text: str) -> Dict:
    """Parse job description text using the Gemma model."""
    logger.info("Parsing job description with Gemma model")

    prompt = f"""
    You are an expert job description parser. Your task is to extract ALL structured information from the job description text below.

    Follow these guidelines:
    1. Extract ALL information that is explicitly mentioned in the job description text.
    2. Format your response as a valid JSON object with EXACTLY the following structure:

    {{
        "job_title": "Full Job Title",
        "company_name": "Company Name" or null,
        "location": "Job Location" or null,
        "job_type": "Full-time/Part-time/Contract/etc." or null,
        "work_mode": "Remote/Hybrid/On-site" or null,
        "department": "Department Name" or null,
        "summary": "Brief job summary or overview" or null,
        "responsibilities": [
            "Responsibility 1",
            "Responsibility 2",
            ...
        ],
        "required_skills": [
            "Required Skill 1",
            "Required Skill 2",
            ...
        ],
        "preferred_skills": [
            "Preferred Skill 1",
            "Preferred Skill 2",
            ...
        ],
        "required_experience": "Experience requirement (e.g., '3+ years')" or null,
        "education_requirements": [
            "Education Requirement 1",
            "Education Requirement 2",
            ...
        ],
        "education_details": {{
            "degree_level": "Bachelor's/Master's/PhD/etc." or null,
            "field_of_study": "Computer Science/Engineering/etc." or null,
            "is_required": true or false,
            "alternatives": "Alternative education paths if mentioned" or null
        }},
        "salary_range": "Salary information if mentioned" or null,
        "benefits": [
            {{
                "title": "Benefit Title",
                "description": "Benefit Description" or null
            }},
            ...
        ],
        "requirements": [
            {{
                "title": "Requirement Title",
                "description": "Requirement Description" or null,
                "is_mandatory": true or false
            }},
            ...
        ],
        "application_deadline": "Application deadline if mentioned" or null,
        "posting_date": "Job posting date if mentioned" or null,
        "industry": "Industry type if mentioned" or null,
        "career_level": "Entry/Mid/Senior level if mentioned" or null
    }}

    3. For arrays, if no information is available, use an empty array []
    4. For string fields, if no information is available, use null
    5. Do not make up or infer information that is not explicitly stated in the job description
    6. Ensure the JSON is properly formatted and valid
    7. IMPORTANT: Distinguish between required skills and preferred/nice-to-have skills
    8. IMPORTANT: For responsibilities and skills, list each item separately in the array
    9. IMPORTANT: If years of experience are mentioned for specific skills, include that in the skill description
    10. IMPORTANT: Make sure all JSON is valid - check for missing commas, extra commas, proper quotes, and proper nesting of objects and arrays
    11. IMPORTANT: Be thorough in extracting ALL skills mentioned in the job description, even if they are embedded in paragraphs
    12. IMPORTANT: For education requirements, be comprehensive and include both degree levels (Bachelor's, Master's, etc.) and fields of study (Computer Science, Engineering, etc.)
    13. IMPORTANT: Pay special attention to abbreviations like CSE, IT, AIDA, etc. and include them in the appropriate fields

    Job Description text:
    {jd_text}

    Respond ONLY with the JSON object, nothing else. Do not include explanations, markdown formatting, or code blocks.
    """

    try:
        # Use a 120 second timeout for JD parsing
        response = get_response(prompt, timeout_seconds=120)

        # Log the raw response for debugging
        logger.info(f"Raw JD response length: {len(response)}")
        if len(response) < 500:
            logger.info(f"Raw JD response content: {response}")
        else:
            logger.info(f"Raw JD response preview: {response[:500]}...")

        # Try to extract JSON from the response
        json_str = response.strip()

        # Remove any markdown formatting if present
        if "```json" in json_str:
            json_str = json_str.split("```json")[1].split("```")[0].strip()
        elif "```" in json_str:
            json_str = json_str.split("```")[1].split("```")[0].strip()

        # Try to find the JSON object if there's additional text
        if json_str.find('{') >= 0 and json_str.rfind('}') >= 0:
            start = json_str.find('{')
            end = json_str.rfind('}') + 1
            json_str = json_str[start:end]

        # Log the extracted JSON string for debugging
        logger.info(f"Extracted JD JSON string length: {len(json_str)}")
        if len(json_str) < 500:
            logger.info(f"Extracted JD JSON string: {json_str}")
        else:
            logger.info(f"Extracted JD JSON string preview: {json_str[:500]}...")

        # Try to parse the JSON string into a Python dictionary
        try:
            parsed_data = json.loads(json_str)
            return parsed_data
        except json.JSONDecodeError as json_err:
            # Log detailed error information
            error_line = json_str.splitlines()[json_err.lineno-1] if json_err.lineno <= len(json_str.splitlines()) else "Line not available"
            logger.error(f"JD JSON decode error at line {json_err.lineno}, column {json_err.colno}, char {json_err.pos}: {json_err.msg}")
            logger.error(f"Error line content: {error_line}")

            # Attempt to fix common JSON errors
            logger.info("Attempting to fix JD JSON formatting issues...")

            # Fix 1: Try to repair common JSON syntax errors
            fixed_json = json_str

            # Fix trailing commas in arrays and objects
            fixed_json = fixed_json.replace(",]", "]").replace(",}", "}")

            # Fix missing commas between array elements or object properties
            fixed_json = fixed_json.replace("}{", "},{").replace("][", "],[").replace("\"\"", "\",\"")

            # Fix unescaped quotes in strings
            # This regex finds unescaped quotes within string values
            fixed_json = re.sub(r'(?<!\\)"(?=.*?[^\\]":)', '\\"', fixed_json)

            # Fix 2: If still failing, try a more aggressive approach - use a minimal valid structure
            try:
                parsed_data = json.loads(fixed_json)
                logger.info("Successfully fixed JD JSON formatting issues")
                return parsed_data
            except json.JSONDecodeError:
                logger.warning("Could not fix JD JSON formatting issues with simple repairs")

                # Create a minimal valid structure as fallback
                fallback_data = {
                    "job_title": "Unknown",
                    "company_name": None,
                    "location": None,
                    "job_type": None,
                    "work_mode": None,
                    "department": None,
                    "summary": None,
                    "responsibilities": [],
                    "required_skills": [],
                    "preferred_skills": [],
                    "required_experience": None,
                    "education_requirements": [],
                    "education_details": {
                        "degree_level": None,
                        "field_of_study": None,
                        "is_required": True,
                        "alternatives": None
                    },
                    "salary_range": None,
                    "benefits": [],
                    "requirements": [],
                    "application_deadline": None,
                    "posting_date": None,
                    "industry": None,
                    "career_level": None,
                    "error": "Failed to parse job description data due to JSON formatting issues",
                    "details": str(json_err)
                }

                # Try to extract some basic information using regex patterns
                # Job title pattern
                title_match = re.search(r'"job_title"\s*:\s*"([^"]+)"', json_str)
                if title_match:
                    fallback_data["job_title"] = title_match.group(1)

                # Company name pattern
                company_match = re.search(r'"company_name"\s*:\s*"([^"]+)"', json_str)
                if company_match:
                    fallback_data["company_name"] = company_match.group(1)

                # Required skills pattern - try to extract skills array
                skills_match = re.search(r'"required_skills"\s*:\s*\[(.*?)\]', json_str, re.DOTALL)
                if skills_match:
                    skills_str = skills_match.group(1)
                    # Extract quoted strings from the skills array
                    skills = re.findall(r'"([^"]+)"', skills_str)
                    if skills:
                        fallback_data["required_skills"] = skills

                # Preferred skills pattern - try to extract preferred skills array
                pref_skills_match = re.search(r'"preferred_skills"\s*:\s*\[(.*?)\]', json_str, re.DOTALL)
                if pref_skills_match:
                    pref_skills_str = pref_skills_match.group(1)
                    # Extract quoted strings from the preferred skills array
                    pref_skills = re.findall(r'"([^"]+)"', pref_skills_str)
                    if pref_skills:
                        fallback_data["preferred_skills"] = pref_skills

                # Education requirements pattern - try to extract education requirements array
                edu_match = re.search(r'"education_requirements"\s*:\s*\[(.*?)\]', json_str, re.DOTALL)
                if edu_match:
                    edu_str = edu_match.group(1)
                    # Extract quoted strings from the education requirements array
                    education_reqs = re.findall(r'"([^"]+)"', edu_str)
                    if education_reqs:
                        fallback_data["education_requirements"] = education_reqs

                # Education details pattern - try to extract education details object
                degree_level_match = re.search(r'"degree_level"\s*:\s*"([^"]+)"', json_str)
                if degree_level_match:
                    fallback_data["education_details"]["degree_level"] = degree_level_match.group(1)

                field_of_study_match = re.search(r'"field_of_study"\s*:\s*"([^"]+)"', json_str)
                if field_of_study_match:
                    fallback_data["education_details"]["field_of_study"] = field_of_study_match.group(1)

                # Required experience pattern
                exp_match = re.search(r'"required_experience"\s*:\s*"([^"]+)"', json_str)
                if exp_match:
                    fallback_data["required_experience"] = exp_match.group(1)

                # Responsibilities pattern - try to extract responsibilities array
                resp_match = re.search(r'"responsibilities"\s*:\s*\[(.*?)\]', json_str, re.DOTALL)
                if resp_match:
                    resp_str = resp_match.group(1)
                    # Extract quoted strings from the responsibilities array
                    responsibilities = re.findall(r'"([^"]+)"', resp_str)
                    if responsibilities:
                        fallback_data["responsibilities"] = responsibilities

                logger.info(f"Created fallback JD data structure with partial information: {fallback_data['job_title']}")
                return fallback_data

    except Exception as e:
        logger.error(f"Error in JD parsing: {e}")
        # Return a minimal valid structure instead of raising an exception
        return {
            "job_title": "Unknown",
            "company_name": None,
            "location": None,
            "job_type": None,
            "work_mode": None,
            "department": None,
            "summary": None,
            "responsibilities": [],
            "required_skills": [],
            "preferred_skills": [],
            "required_experience": None,
            "education_requirements": [],
            "education_details": {
                "degree_level": None,
                "field_of_study": None,
                "is_required": True,
                "alternatives": None
            },
            "salary_range": None,
            "benefits": [],
            "requirements": [],
            "application_deadline": None,
            "posting_date": None,
            "industry": None,
            "career_level": None,
            "error": "Failed to parse job description data",
            "details": str(e)
        }



def normalize_resume_data(parsed_data: Dict, convert_skills_to_dict_format: bool = False) -> Dict:
    """Normalize parsed resume data to a consistent schema."""
    logger.info("Normalizing resume data to consistent schema")

    # Define the fixed schema fields
    normalized_data = {
        "name": "",
        "email": None,
        "phone": None,
        "education": [],
        "direct_skills": {},
        "subjective_skills": {},
        "experience": [],
        "projects": [],
        "certifications": [],
        "domain_of_interest": [],
        "languages_known": [],
        "social_media": [],
        "achievements": [],
        "publications": [],
        "volunteer_experience": [],
        "references": [],
        "summary": None,
        "confidence_score": 0.0,
        "confidence_details": {}
    }

    # Copy data from parsed_data to normalized_data for fields that exist in both
    for field in normalized_data.keys():
        if field in parsed_data and field not in ["confidence_score", "confidence_details"]:
            # Special handling for skills if converting to dictionary
            if field == "skills" and convert_skills_to_dict_format:
                if isinstance(parsed_data[field], list):
                    # Process skills to separate direct and subjective skills
                    processed_data = convert_skills_to_dict({
                        "skills": parsed_data[field],
                        "experience": parsed_data.get("experience", []),
                        "projects": parsed_data.get("projects", [])
                    })

                    # Keep original skills list if not converting to dict format
                    if not convert_skills_to_dict_format:
                        normalized_data["skills"] = parsed_data[field]
                    else:
                        # For backward compatibility, also provide the combined skills dict
                        combined_skills = {}
                        combined_skills.update(processed_data.get("direct_skills", {}))
                        combined_skills.update(processed_data.get("subjective_skills", {}))
                        normalized_data["skills"] = combined_skills

                    # Set direct and subjective skills
                    normalized_data["direct_skills"] = processed_data.get("direct_skills", {})
                    normalized_data["subjective_skills"] = processed_data.get("subjective_skills", {})
                elif isinstance(parsed_data[field], dict):
                    normalized_data["skills"] = parsed_data[field]
            else:
                normalized_data[field] = parsed_data[field]

    # Handle direct_skills and subjective_skills if they exist in parsed_data
    if "direct_skills" in parsed_data:
        normalized_data["direct_skills"] = parsed_data["direct_skills"]
    if "subjective_skills" in parsed_data:
        normalized_data["subjective_skills"] = parsed_data["subjective_skills"]

    # Handle special case for languages vs languages_known
    if "languages" in parsed_data and "languages_known" not in parsed_data:
        normalized_data["languages_known"] = parsed_data["languages"]

    # Ensure name is a string
    if normalized_data["name"] is None:
        normalized_data["name"] = ""

    # Calculate confidence score
    confidence_score = calculate_resume_confidence_score(normalized_data)
    normalized_data["confidence_score"] = confidence_score

    return normalized_data

def normalize_jd_data(parsed_data: Dict) -> Dict:
    """Normalize parsed job description data to a consistent schema."""
    logger.info("Normalizing job description data to consistent schema")

    # Define the fixed schema fields
    normalized_data = {
        "job_title": "",
        "company_name": None,
        "location": None,
        "job_type": None,
        "work_mode": None,
        "department": None,
        "summary": None,
        "responsibilities": [],
        "required_skills": [],
        "preferred_skills": [],
        "required_experience": None,
        "education_requirements": [],
        "education_details": {
            "degree_level": None,
            "field_of_study": None,
            "is_required": True,
            "alternatives": None
        },
        "salary_range": None,
        "benefits": [],
        "requirements": [],
        "application_deadline": None,
        "posting_date": None,
        "contact_information": None,
        "company_description": None,
        "industry": None,
        "career_level": None,
        "confidence_score": 0.0,
        "confidence_details": {}
    }

    # Copy data from parsed_data to normalized_data for fields that exist in both
    for field in normalized_data.keys():
        if field in parsed_data and field not in ["confidence_score", "confidence_details"]:
            normalized_data[field] = parsed_data[field]

    # Ensure job_title is a string
    if normalized_data["job_title"] is None:
        normalized_data["job_title"] = ""

    # Calculate confidence score
    confidence_score = calculate_jd_confidence_score(normalized_data)
    normalized_data["confidence_score"] = confidence_score

    return normalized_data


def parse_resume(resume_text: str, convert_skills_to_dict_format: bool = False) -> Dict:
    """Use Gemma model to parse resume text."""
    logger.info("Starting resume parsing")

    try:
        # Use the primary LLM-based parsing method
        logger.info("Parsing resume with Gemma model")
        result = parse_resume_with_gemma(resume_text)

        # Check if we got a valid result
        if not result or not isinstance(result, dict):
            logger.error(f"Invalid result from parse_resume_with_gemma: {type(result)}")
            raise ValueError("Invalid result format from resume parser")

        # Check if the result contains an error field
        if "error" in result:
            logger.warning(f"Resume parsing returned an error: {result.get('error')}")
            # We'll still normalize it to ensure consistent schema

        # Normalize the result to ensure consistent schema
        normalized_result = normalize_resume_data(result, convert_skills_to_dict_format)

        # Add a message if there was an error in parsing
        if "error" in result:
            normalized_result["error"] = result["error"]
            if "details" in result:
                normalized_result["details"] = result["details"]

        return normalized_result

    except Exception as e:
        logger.error(f"Error in resume parsing: {e}")

        # Return a minimal structure if parsing fails
        result = {
            "error": "Failed to parse resume",
            "details": str(e),
            "name": "Unknown",
            "email": None,
            "phone": None,
            "direct_skills": {},
            "subjective_skills": {},
            "education": [],
            "experience": [],
        }

        # Normalize the result to ensure consistent schema
        normalized_result = normalize_resume_data(result, convert_skills_to_dict_format)

        return normalized_result


def extract_skills_from_text(text: str) -> List[str]:
    """Extract skills from text using pattern matching and common skill keywords."""
    logger.info("Attempting fallback skills extraction from text")

    # Common technical skills to look for
    common_skills = [
        "Python", "Java", "JavaScript", "C++", "C#", "Ruby", "PHP", "Swift", "Kotlin", "Go", "Rust",
        "HTML", "CSS", "SQL", "NoSQL", "MongoDB", "MySQL", "PostgreSQL", "Oracle", "Redis",
        "React", "Angular", "Vue", "Node.js", "Express", "Django", "Flask", "Spring", "ASP.NET",
        "AWS", "Azure", "GCP", "Docker", "Kubernetes", "Jenkins", "Git", "GitHub", "GitLab",
        "Machine Learning", "Deep Learning", "AI", "Data Science", "TensorFlow", "PyTorch", "Keras",
        "Big Data", "Hadoop", "Spark", "Kafka", "Airflow", "ETL", "Data Mining", "Data Analysis",
        "DevOps", "CI/CD", "Agile", "Scrum", "Kanban", "Jira", "Confluence", "REST API", "GraphQL",
        "Microservices", "Serverless", "Linux", "Unix", "Windows", "MacOS", "Android", "iOS",
        "Testing", "QA", "Selenium", "JUnit", "TestNG", "Cypress", "Jest", "Mocha", "Chai",
        "UI/UX", "Figma", "Sketch", "Adobe XD", "Photoshop", "Illustrator", "InDesign",
        "Project Management", "Product Management", "Scrum Master", "Product Owner",
        "Communication", "Leadership", "Teamwork", "Problem Solving", "Critical Thinking",
        "Time Management", "Adaptability", "Creativity", "Attention to Detail", "Analytical Skills"
    ]

    # Extract skills using pattern matching
    extracted_skills = []

    # Convert text to lowercase for case-insensitive matching
    text_lower = text.lower()

    # Look for skills in the text
    for skill in common_skills:
        if skill.lower() in text_lower:
            # Check if it's a standalone word or part of a phrase
            # This helps avoid false positives like "go" matching in "going"
            if len(skill) <= 2:  # For very short skills like "Go", "C#", etc.
                # Look for word boundaries or specific contexts
                if re.search(r'\b' + re.escape(skill.lower()) + r'\b', text_lower) or \
                   re.search(r'skills?.*' + re.escape(skill.lower()), text_lower) or \
                   re.search(r'technologies?.*' + re.escape(skill.lower()), text_lower) or \
                   re.search(r'experience.*' + re.escape(skill.lower()), text_lower):
                    extracted_skills.append(skill)
            else:
                extracted_skills.append(skill)

    # Look for skills mentioned in specific contexts
    skill_contexts = [
        r'proficient in\s+([A-Za-z0-9+#/\s]+)',
        r'experience with\s+([A-Za-z0-9+#/\s]+)',
        r'knowledge of\s+([A-Za-z0-9+#/\s]+)',
        r'familiar with\s+([A-Za-z0-9+#/\s]+)',
        r'skills:?\s*([A-Za-z0-9+#/\s,]+)',
        r'technologies:?\s*([A-Za-z0-9+#/\s,]+)',
        r'requirements:?\s*([A-Za-z0-9+#/\s,]+)'
    ]

    for pattern in skill_contexts:
        matches = re.finditer(pattern, text_lower)
        for match in matches:
            skill_text = match.group(1).strip()
            # Split by common separators
            for skill in re.split(r'[,;/|]|\sand\s', skill_text):
                skill = skill.strip()
                if skill and len(skill) > 2 and skill not in [s.lower() for s in extracted_skills]:
                    # Capitalize the first letter of each word
                    formatted_skill = ' '.join(word.capitalize() for word in skill.split())
                    extracted_skills.append(formatted_skill)

    # Remove duplicates while preserving order
    unique_skills = []
    for skill in extracted_skills:
        if skill not in unique_skills:
            unique_skills.append(skill)

    logger.info(f"Fallback extraction found {len(unique_skills)} skills")
    return unique_skills


def extract_education_from_text(text: str) -> List[str]:
    """Extract education requirements from text using pattern matching."""
    logger.info("Attempting fallback education extraction from text")

    # Common education patterns to look for
    education_patterns = [
        r"(?:bachelor|master|phd|doctorate|bs|ba|ms|ma|b\.s\.|b\.a\.|m\.s\.|m\.a\.|ph\.d\.)['']?s?\s+(?:degree|in)\s+([A-Za-z\s]+)",
        r"degree\s+(?:in|required)?\s+([A-Za-z\s]+)",
        r"(?:bachelor|master|phd|doctorate|bs|ba|ms|ma|b\.s\.|b\.a\.|m\.s\.|m\.a\.|ph\.d\.)['']?s?\s+(?:or equivalent)",
        r"(?:bachelor|master|phd|doctorate|bs|ba|ms|ma|b\.s\.|b\.a\.|m\.s\.|m\.a\.|ph\.d\.)['']?s?(?:\s+degree)?",
        r"education:?\s*([A-Za-z0-9+#/\s,]+)",
        r"qualifications:?\s*([A-Za-z0-9+#/\s,]+)"
    ]

    # Extract education requirements using pattern matching
    extracted_education = []

    # Convert text to lowercase for case-insensitive matching
    text_lower = text.lower()

    # Look for education requirements in the text
    for pattern in education_patterns:
        matches = re.finditer(pattern, text_lower)
        for match in matches:
            if len(match.groups()) > 0:
                edu_text = match.group(0).strip()
            else:
                edu_text = match.group(0).strip()

            # Clean up the text
            edu_text = re.sub(r'\s+', ' ', edu_text)

            if edu_text and edu_text not in [e.lower() for e in extracted_education]:
                # Capitalize the first letter of each word
                formatted_edu = ' '.join(word.capitalize() for word in edu_text.split())
                extracted_education.append(formatted_edu)

    # Check for specific degree mentions
    degree_types = [
        "Bachelor's degree", "Master's degree", "PhD", "Doctorate",
        "BS", "BA", "MS", "MA", "B.S.", "B.A.", "M.S.", "M.A.", "Ph.D."
    ]

    for degree in degree_types:
        if degree.lower() in text_lower and degree not in extracted_education:
            extracted_education.append(degree)

    # Check for specific field mentions
    fields = [
        "Computer Science", "Engineering", "Information Technology",
        "Business", "Mathematics", "Statistics", "Data Science"
    ]

    for field in fields:
        field_pattern = f"{field.lower()}"
        if field_pattern in text_lower:
            # Check if this field is already part of an extracted education requirement
            if not any(field.lower() in edu.lower() for edu in extracted_education):
                # Add as a standalone field
                extracted_education.append(field)

    # Remove duplicates while preserving order
    unique_education = []
    for edu in extracted_education:
        if edu not in unique_education:
            unique_education.append(edu)

    logger.info(f"Fallback extraction found {len(unique_education)} education requirements")
    return unique_education


def parse_jd(jd_text: str) -> Dict:
    """Use Gemma model to parse job description text."""
    logger.info("Starting job description parsing")

    try:
        # Use the primary LLM-based parsing method
        logger.info("Parsing job description with Gemma model")
        result = parse_jd_with_gemma(jd_text)

        # Check if we got a valid result
        if not result or not isinstance(result, dict):
            logger.error(f"Invalid result from parse_jd_with_gemma: {type(result)}")
            raise ValueError("Invalid result format from JD parser")

        # Check if the result contains an error field
        if "error" in result:
            logger.warning(f"JD parsing returned an error: {result.get('error')}")
            # We'll still normalize it to ensure consistent schema

        # Normalize the result to ensure consistent schema
        normalized_result = normalize_jd_data(result)

        # Fallback mechanism for skills extraction if no skills were found
        if not normalized_result["required_skills"]:
            logger.warning("No required skills found in JD, attempting fallback extraction")
            extracted_skills = extract_skills_from_text(jd_text)
            if extracted_skills:
                logger.info(f"Fallback extraction found {len(extracted_skills)} skills")
                normalized_result["required_skills"] = extracted_skills

        # Fallback mechanism for education requirements if none were found
        if not normalized_result["education_requirements"] and not normalized_result["education_details"]["degree_level"]:
            logger.warning("No education requirements found in JD, attempting fallback extraction")
            extracted_education = extract_education_from_text(jd_text)
            if extracted_education:
                logger.info(f"Fallback extraction found education requirements: {extracted_education}")
                normalized_result["education_requirements"] = extracted_education

                # Try to extract degree level and field
                for req in extracted_education:
                    if "bachelor" in req.lower() or "b." in req.lower() or "bs" in req.lower() or "ba" in req.lower():
                        normalized_result["education_details"]["degree_level"] = "Bachelor's degree"
                        # Try to extract field
                        if "comput" in req.lower() or "cs" in req.lower() or "cse" in req.lower() or "software" in req.lower():
                            normalized_result["education_details"]["field_of_study"] = "Computer Science"
                        elif "engineer" in req.lower():
                            normalized_result["education_details"]["field_of_study"] = "Engineering"
                        break
                    elif "master" in req.lower() or "m." in req.lower() or "ms" in req.lower() or "ma" in req.lower():
                        normalized_result["education_details"]["degree_level"] = "Master's degree"
                        break

        # Add a message if there was an error in parsing
        if "error" in result:
            normalized_result["error"] = result["error"]
            if "details" in result:
                normalized_result["details"] = result["details"]

        return normalized_result

    except Exception as e:
        logger.error(f"Error in job description parsing: {e}")

        # Return a minimal structure if parsing fails
        result = {
            "error": "Failed to parse job description",
            "details": str(e),
            "job_title": "Unknown",
            "required_skills": [],
            "preferred_skills": [],
            "responsibilities": [],
            "education_requirements": [],
            "education_details": {
                "degree_level": None,
                "field_of_study": None,
                "is_required": True,
                "alternatives": None
            }
        }

        # Normalize the result to ensure consistent schema
        normalized_result = normalize_jd_data(result)

        return normalized_result

@app.get("/")
async def root():
    return {
        "message": "Welcome to the Gemma3 API",
        "endpoints": [
            {
                "path": "/generate",
                "method": "POST",
                "description": "Generate a response from the Gemma model",
                "body": {"prompt": "string", "history": "string (optional)"}
            },
            {
                "path": "/generate_with_image",
                "method": "POST",
                "description": "Generate a response based on an image and text prompt",
                "body": "Upload an image file and provide a text prompt"
            },
            {
                "path": "/resume",
                "method": "POST",
                "description": "Parse a resume (PDF or DOCX) and extract ALL structured information with skills as a dictionary",
                "body": "Upload a PDF or DOCX file"
            },
            {
                "path": "/jd_parser",
                "method": "POST",
                "description": "Parse a job description (PDF or DOCX) and extract structured information in JSON format",
                "body": "Upload a PDF or DOCX file"
            },
            {
                "path": "/jd",
                "method": "POST",
                "description": "Generate interview questions based on a job description and resume data",
                "body": "Upload a JD file (PDF/DOCX) and provide question scales and resume data as JSON"
            },
            {
                "path": "/jd_only",
                "method": "POST",
                "description": "Generate interview questions based only on a job description (no resume data needed)",
                "body": "Upload a JD file (PDF/DOCX)"
            },
            {
                "path": "/intervet",
                "method": "POST",
                "description": "Evaluate how well a candidate's resume matches a job description",
                "body": {"resume_json": "Resume data from /resume endpoint", "jd_json": "JD data from /jd_parser endpoint"}
            },
            {
                "path": "/intervet2",
                "method": "POST",
                "description": "Evaluate how well a candidate's resume matches a job description using file uploads",
                "body": "Upload a resume file (PDF/DOCX) and a job description file (PDF/DOCX)"
            },
            {
                "path": "/interfix",
                "method": "POST",
                "description": "Extract key information from a VAPI call summary or transcript",
                "body": {"summary": "string (call summary or transcript)"}
            },
            {
                "path": "/bunchtest",
                "method": "POST",
                "description": "Batch evaluate multiple resumes against one job description",
                "body": "Upload multiple resume files and one job description file"
            }
        ]
    }

@app.post("/generate")
async def generate_endpoint(request: GenerateRequest):
    try:
        full_prompt = request.history + "\nUser: " + request.prompt + "\nAssistant: "
        response = get_response(full_prompt)
        return {"response": response}
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise HTTPException(status_code=500, detail="An unexpected error occurred.")

@app.post("/generate_with_image", summary="Generate a response based on text and image", description="Upload an image and provide a prompt to get a response related to the image")
async def generate_with_image_endpoint(
    file: UploadFile = File(..., description="Image file to analyze"),
    prompt: str = Form(..., description="Text prompt related to the image"),
    history: str = Form("", description="Optional conversation history")
):
    """Generate a response based on an image and text prompt.

    - **file**: Upload an image file (JPG, PNG, etc.)
    - **prompt**: Text prompt related to the image
    - **history**: Optional conversation history

    Returns a JSON object with the generated response.
    """
    try:
        # Check if the file is an image
        file_extension = os.path.splitext(file.filename.lower())[1]
        supported_extensions = [".jpg", ".jpeg", ".png", ".gif", ".bmp"]

        if file_extension not in supported_extensions:
            logger.warning(f"Unsupported file format: {file.filename}")
            raise HTTPException(status_code=400, detail=f"Only image files are supported: {', '.join(supported_extensions)}")

        logger.info(f"Processing image file: {file.filename}")

        # Create a temporary file to store the uploaded image
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as temp_file:
            # Write the uploaded file content to the temporary file
            content = await file.read()
            if not content:
                logger.error("Uploaded file is empty")
                raise HTTPException(status_code=400, detail="The uploaded file is empty.")

            temp_file.write(content)
            temp_file_path = temp_file.name
            logger.info(f"Saved uploaded image to temporary location: {temp_file_path}")

        try:
            # Create a multimodal prompt that includes the image
            image_prompt = f"I'm looking at this image. {prompt}"

            # Combine with history if provided
            if history:
                image_prompt = f"{history}\n\n{image_prompt}"

            logger.info(f"Sending image to Gemma with prompt: {image_prompt}")

            # Get response with a longer timeout for image processing
            # Pass the image path to the model
            response = get_response(
                prompt=image_prompt,
                timeout_seconds=120,
                max_tokens=1500,
                image_path=temp_file_path
            )

            return {"response": response}
        finally:
            # Clean up the temporary file
            if os.path.exists(temp_file_path):
                logger.info(f"Removing temporary file: {temp_file_path}")
                os.remove(temp_file_path)

    except HTTPException as e:
        # Re-raise HTTP exceptions
        raise e
    except Exception as e:
        logger.error(f"Unexpected error in image processing: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")

@app.post("/resume", response_model=Dict, summary="Parse a resume", description="Upload a PDF or DOCX resume and extract ALL structured information using the Gemma 3:4B model")
async def parse_resume_endpoint(
    request: Request,
    file: UploadFile = File(..., description="Resume file to parse (PDF or DOCX format)")
):
    """Parse a resume and extract ALL structured information.

    - **file**: Upload a PDF or DOCX file containing a resume

    Returns a comprehensive JSON object with ALL parsed resume information, including but not limited to:
    - Basic information (name, email, phone)
    - Education history
    - Work experience
    - Skills (as a dictionary with context about where each skill was used)
    - Projects
    - Certifications
    - Domain of interest
    - Languages
    - Achievements
    - Publications
    - And any other relevant information found in the resume

    Note: This endpoint uses a waterfall mechanism for text extraction:
    1. First attempts to extract text using standard PDF/DOCX parsing
    2. If that fails, falls back to treating the document as an image and using the LLM's vision capabilities

    The confidence score indicates how confident the model is in the correctness of the extracted information.
    A higher score means the model is more confident that the extracted data accurately represents what's in the
    resume. The score is calculated by analyzing the structure, consistency, and patterns in the
    extracted data.
    """
    # Get the metrics tracker from request state
    metrics = getattr(request.state, "metrics", None)
    if metrics:
        metrics.add_metric("endpoint", "resume")

    try:
        # Check if the file is a supported format (PDF or DOCX)
        file_extension = os.path.splitext(file.filename.lower())[1]

        if file_extension == '.pdf':
            file_type = "pdf"
            suffix = '.pdf'
        elif file_extension == '.docx':
            file_type = "docx"
            suffix = '.docx'
        else:
            logger.warning(f"Unsupported file format: {file.filename}")
            if metrics:
                metrics.add_metric("error", f"Unsupported file format: {file_extension}")
            raise HTTPException(status_code=400, detail="Only PDF and DOCX files are supported.")

        logger.info(f"Processing resume file: {file.filename} (type: {file_type})")
        if metrics:
            metrics.add_metric("file_name", file.filename)
            metrics.add_metric("file_type", file_type)

        # Create a temporary file to store the uploaded file
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            # Write the uploaded file content to the temporary file
            content = await file.read()
            if not content:
                logger.error("Uploaded file is empty")
                if metrics:
                    metrics.add_metric("error", "Empty file")
                raise HTTPException(status_code=400, detail="The uploaded file is empty.")

            temp_file.write(content)
            temp_file_path = temp_file.name
            logger.info(f"Saved uploaded file to temporary location: {temp_file_path}")

            if metrics:
                metrics.add_metric("file_size", len(content))

        try:
            # Extract text from the file using waterfall mechanism
            # This will first try standard extraction, then fall back to image-based extraction if needed
            start_extraction_time = time.time()
            resume_text = extract_text_from_file(temp_file_path, file_type, request_metrics=metrics)
            extraction_time = time.time() - start_extraction_time

            if metrics:
                metrics.add_metric("text_extraction_time", extraction_time)
                metrics.add_metric("extracted_text_length", len(resume_text))

            # Parse the resume text using the Gemma model
            start_parsing_time = time.time()

            # Parse the resume text using the Gemma model (always convert skills to dictionary format)
            parsed_data = parse_resume(resume_text, convert_skills_to_dict_format=True)

            parsing_time = time.time() - start_parsing_time
            if metrics:
                metrics.add_metric("parsing_time", parsing_time)
                metrics.add_metric("confidence_score", parsed_data.get("confidence_score", 0))

            # Remove any internal fields that might be present
            for field in list(parsed_data.keys()):
                if field.startswith("_"):
                    parsed_data.pop(field)

            # Log the number of fields extracted
            if metrics:
                metrics.add_metric("fields_extracted", len(parsed_data))
                metrics.add_metric("skills_count", len(parsed_data.get("skills", [])))
                metrics.add_metric("education_count", len(parsed_data.get("education", [])))
                metrics.add_metric("experience_count", len(parsed_data.get("experience", [])))
                metrics.add_metric("total_processing_time", extraction_time + parsing_time)

            return parsed_data
        finally:
            # Clean up the temporary file
            if os.path.exists(temp_file_path):
                logger.info(f"Removing temporary file: {temp_file_path}")
                os.remove(temp_file_path)

    except HTTPException as e:
        # Log the error in metrics
        if metrics:
            metrics.add_metric("error", str(e))
            metrics.add_metric("error_status", e.status_code)
        # Re-raise HTTP exceptions
        raise e
    except Exception as e:
        # Log the error in metrics
        if metrics:
            metrics.add_metric("error", str(e))
            metrics.add_metric("error_status", 500)
        logger.error(f"Unexpected error in resume parsing: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")

def generate_questions_for_category(jd_text: str, resume_data: Dict, category: str, num_questions: int) -> List[str]:
    """Generate interview questions for a specific category."""
    if num_questions <= 0:
        return []

    logger.info(f"Generating {num_questions} questions for category: {category}")

    # Map category to a more readable name
    category_name = {
        "technical_questions": "Technical",
        "past_experience_questions": "Past Experience",
        "case_study_questions": "Case Study",
        "situation_handling_questions": "Situation Handling",
        "personality_test_questions": "Personality Test"
    }.get(category, category)

    # Special handling for personality test questions
    if category == "personality_test_questions":
        # For personality questions, use a more general prompt focused on personality traits
        prompt = f"""
        You are an expert interview question generator. Your task is to create {num_questions} personality assessment questions.

        Candidate Resume Data (in JSON format):
        {json.dumps(resume_data, indent=2)}

        Generate exactly {num_questions} personality assessment questions that:
        - Focus on general personality traits like teamwork, leadership, adaptability, work ethic, etc.
        - Help understand the candidate's character, values, and working style
        - Are NOT specific to the job description but rather about the person's general traits
        - Reveal how the candidate might fit into different team environments
        - Assess soft skills and interpersonal abilities

        Examples of good personality questions:
        - "How do you handle criticism or feedback from colleagues or supervisors?"
        - "Describe a situation where you had to adapt to a significant change at work. How did you handle it?"
        - "What motivates you the most in your professional life?"
        - "How do you prioritize tasks when facing multiple deadlines?"
        - "Describe your ideal work environment and management style."

        IMPORTANT: Respond ONLY with a JSON array of strings, where each string is a question.
        Example: ["Question 1?", "Question 2?", "Question 3?"]

        DO NOT include any explanations, markdown formatting, or code blocks in your response.
        Your entire response should be ONLY the JSON array, nothing else.
        """
    else:
        # For all other categories, use the original prompt
        prompt = f"""
        You are an expert interview question generator. Your task is to create {num_questions} tailored {category_name} questions based on a job description and a candidate's resume.

        Job Description:
        {jd_text}

        Candidate Resume Data (in JSON format):
        {json.dumps(resume_data, indent=2)}

        Generate exactly {num_questions} {category_name} questions that are:
        - Relevant to the job description
        - Tailored to the candidate's background and experience
        - Specific and detailed (not generic)
        - Designed to assess the candidate's fit for the role

        IMPORTANT: Respond ONLY with a JSON array of strings, where each string is a question.
        Example: ["Question 1?", "Question 2?", "Question 3?"]

        DO NOT include any explanations, markdown formatting, or code blocks in your response.
        Your entire response should be ONLY the JSON array, nothing else.
        """

    try:
        # Use a shorter timeout for individual category generation
        response = get_response(prompt, timeout_seconds=45, max_tokens=500)

        # Try to extract JSON array from the response
        json_str = response.strip()

        # Remove any markdown formatting if present
        if "```json" in json_str:
            json_str = json_str.split("```json")[1].split("```")[0].strip()
        elif "```" in json_str:
            json_str = json_str.split("```")[1].split("```")[0].strip()

        # Try to find the JSON array if there's additional text
        if json_str.find('[') >= 0 and json_str.rfind(']') >= 0:
            start = json_str.find('[')
            end = json_str.rfind(']') + 1
            json_str = json_str[start:end]

        # Parse the JSON string into a Python list
        try:
            questions = json.loads(json_str)

            # Ensure we have a list of strings
            if isinstance(questions, list):
                # Convert all items to strings and filter out empty ones
                questions = [str(q) for q in questions if q]
                return questions[:num_questions]  # Limit to requested number
            else:
                logger.warning(f"Expected a list but got {type(questions)} for {category}")
                return []

        except json.JSONDecodeError:
            # If JSON parsing fails, try to extract questions using regex
            # Look for text that looks like questions (ends with ? or has question-like structure)
            question_pattern = re.compile(r'"([^"]+\?)"')
            matches = question_pattern.findall(json_str)

            if matches:
                return matches[:num_questions]  # Limit to requested number
            else:
                logger.warning(f"Could not extract questions for {category}")
                return []

    except Exception as e:
        logger.error(f"Error generating questions for {category}: {e}")
        return []

def generate_interview_questions(jd_text: str, resume_data: Dict, question_scales: Dict) -> Dict:
    """Generate interview questions based on job description and resume data."""
    logger.info("Starting interview question generation using category-by-category approach")

    # Define the categories and map scale values to actual question counts
    categories = [
        "technical_questions",
        "past_experience_questions",
        "case_study_questions",
        "situation_handling_questions",
        "personality_test_questions"
    ]

    # Initialize results dictionary
    questions_data = {}

    # Generate questions for each category in parallel
    from concurrent.futures import ThreadPoolExecutor

    def process_category(category):
        # Map scale (0-10) to number of questions (0-5)
        scale = question_scales.get(category, 0)
        num_questions = min(5, max(0, int(scale / 2)))

        if num_questions > 0:
            return category, generate_questions_for_category(jd_text, resume_data, category, num_questions)
        else:
            return category, []

    # Use ThreadPoolExecutor to process categories in parallel
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(process_category, category): category for category in categories}

        for future in futures:
            try:
                category, questions = future.result()
                questions_data[category] = questions
                logger.info(f"Generated {len(questions)} questions for {category}")
            except Exception as e:
                logger.error(f"Error processing category: {e}")
                # Initialize with empty list on error
                questions_data[futures[future]] = []

    # Ensure all categories are present
    for category in categories:
        if category not in questions_data:
            questions_data[category] = []

    logger.info("Successfully generated all interview questions")
    return questions_data

@app.post("/jd_parser", response_model=Dict, summary="Parse a job description", description="Upload a PDF or DOCX job description and extract ALL structured information using the Gemma 3:4B model")
async def parse_jd_endpoint(
    file: UploadFile = File(..., description="Job description file to parse (PDF or DOCX format)")
):
    """Parse a job description and extract ALL structured information.

    - **file**: Upload a PDF or DOCX file containing a job description

    Returns a comprehensive JSON object with parsed job description information, including but not limited to:
    - Job title
    - Company name
    - Location
    - Job type (Full-time, Part-time, etc.)
    - Work mode (Remote, Hybrid, On-site)
    - Required skills
    - Preferred skills
    - Required experience
    - Education requirements
    - Responsibilities
    - Benefits
    - And any other relevant information found in the job description

    Note: This endpoint only returns the parsed JD data in JSON format. It does not generate interview questions.
    Use the /jd or /jd_only endpoints if you need interview questions based on the job description.

    This endpoint uses a waterfall mechanism for text extraction:
    1. First attempts to extract text using standard PDF/DOCX parsing
    2. If that fails, falls back to treating the document as an image and using the LLM's vision capabilities

    The confidence score indicates how confident the model is in the correctness of the extracted information.
    A higher score means the model is more confident that the extracted data accurately represents what's in the
    job description. The score is calculated by analyzing the structure, consistency, and patterns in the
    extracted data.
    """
    try:
        # Check if the file is a supported format (PDF or DOCX)
        file_extension = os.path.splitext(file.filename.lower())[1]

        if file_extension == '.pdf':
            file_type = "pdf"
            suffix = '.pdf'
        elif file_extension == '.docx':
            file_type = "docx"
            suffix = '.docx'
        else:
            logger.warning(f"Unsupported file format: {file.filename}")
            raise HTTPException(status_code=400, detail="Only PDF and DOCX files are supported.")

        logger.info(f"Processing job description file: {file.filename} (type: {file_type})")

        # Create a temporary file to store the uploaded file
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            # Write the uploaded file content to the temporary file
            content = await file.read()
            if not content:
                logger.error("Uploaded file is empty")
                raise HTTPException(status_code=400, detail="The uploaded file is empty.")

            temp_file.write(content)
            temp_file_path = temp_file.name
            logger.info(f"Saved uploaded file to temporary location: {temp_file_path}")

        try:
            # Extract text from the file using waterfall mechanism
            # This will first try standard extraction, then fall back to image-based extraction if needed
            jd_text = extract_text_from_file(temp_file_path, file_type)

            # Parse the job description text using the Gemma model
            logger.info(f"Extracted {len(jd_text)} characters of text from job description file")

            # Log a preview of the extracted text for debugging
            text_preview = jd_text[:500] + "..." if len(jd_text) > 500 else jd_text
            logger.info(f"Text preview: {text_preview}")

            # Parse the job description
            start_parsing_time = time.time()
            parsed_data = parse_jd(jd_text)
            parsing_time = time.time() - start_parsing_time

            logger.info(f"Job description parsing completed in {parsing_time:.2f} seconds")

            # Log parsing results
            if "job_title" in parsed_data:
                logger.info(f"Parsed job title: {parsed_data['job_title']}")

            if "required_skills" in parsed_data:
                skills_count = len(parsed_data["required_skills"])
                logger.info(f"Extracted {skills_count} required skills")
                if skills_count > 0:
                    skills_preview = ", ".join(parsed_data["required_skills"][:5])
                    if skills_count > 5:
                        skills_preview += f", ... ({skills_count-5} more)"
                    logger.info(f"Skills preview: {skills_preview}")

            if "education_requirements" in parsed_data:
                edu_count = len(parsed_data["education_requirements"])
                logger.info(f"Extracted {edu_count} education requirements")
                if edu_count > 0:
                    logger.info(f"Education requirements: {parsed_data['education_requirements']}")

            if "education_details" in parsed_data and isinstance(parsed_data["education_details"], dict):
                logger.info(f"Education details: {parsed_data['education_details']}")

            # Remove any internal fields that might be present
            for field in list(parsed_data.keys()):
                if field.startswith("_"):
                    parsed_data.pop(field)

            return parsed_data
        finally:
            # Clean up the temporary file
            if os.path.exists(temp_file_path):
                logger.info(f"Removing temporary file: {temp_file_path}")
                os.remove(temp_file_path)

    except HTTPException as e:
        # Re-raise HTTP exceptions
        raise e
    except Exception as e:
        logger.error(f"Unexpected error in job description parsing: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")


@app.post("/jd", response_model=Dict, summary="Generate interview questions", description="Upload a job description and provide resume data to generate tailored interview questions")
async def generate_interview_questions_endpoint(
    file: UploadFile = File(..., description="Job description file to parse (PDF or DOCX format)"),
    request_data_json: str = Form(None, description="JSON string containing resume data and question scales")
):
    """Generate tailored interview questions based on a job description and resume data.

    - **file**: Upload a PDF or DOCX file containing a job description
    - **request_data_json**: JSON data containing resume information and question scales

    The JSON data should have the following structure:
    ```json
    {
        "resume_json": { ... resume data ... },
        "technical_questions": 5,  // Scale 0-10
        "past_experience_questions": 3,  // Scale 0-10
        "case_study_questions": 2,  // Scale 0-10
        "situation_handling_questions": 4,  // Scale 0-10
        "personality_test_questions": 2  // Scale 0-10
    }
    ```

    Returns a JSON object with categorized interview questions:
    - Technical questions
    - Past experience questions
    - Case study questions
    - Situation handling questions
    - Personality test questions

    The number of questions in each category is proportional to the scale values provided (0-10).
    For efficiency, the scale values are converted to a maximum of 5 questions per category.

    Note: This endpoint processes each question category in parallel for faster response times.
    """
    try:
        # Check if the file is a supported format (PDF or DOCX)
        file_extension = os.path.splitext(file.filename.lower())[1]

        if file_extension == '.pdf':
            file_type = "pdf"
            suffix = '.pdf'
        elif file_extension == '.docx':
            file_type = "docx"
            suffix = '.docx'
        else:
            logger.warning(f"Unsupported file format: {file.filename}")
            raise HTTPException(status_code=400, detail="Only PDF and DOCX files are supported.")

        # Parse the request data JSON
        try:
            # Check if we have any data
            if request_data_json is None:
                raise HTTPException(
                    status_code=400,
                    detail="Missing request data. Please provide the 'request_data_json' field."
                )

            # Parse the JSON data
            request_data_dict = json.loads(request_data_json)
            request_data_obj = JDQuestionRequest(**request_data_dict)

            logger.info("Successfully parsed request data")
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="Invalid JSON in request data.")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid request data: {str(e)}")

        logger.info(f"Processing job description file: {file.filename} (type: {file_type})")

        # Create a temporary file to store the uploaded file
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            # Write the uploaded file content to the temporary file
            content = await file.read()
            if not content:
                logger.error("Uploaded file is empty")
                raise HTTPException(status_code=400, detail="The uploaded file is empty.")

            temp_file.write(content)
            temp_file_path = temp_file.name
            logger.info(f"Saved uploaded file to temporary location: {temp_file_path}")

        try:
            # Extract text from the file based on its type
            jd_text = extract_text_from_file(temp_file_path, file_type)

            # Prepare question scales dictionary
            question_scales = {
                "technical_questions": request_data_obj.technical_questions,
                "past_experience_questions": request_data_obj.past_experience_questions,
                "case_study_questions": request_data_obj.case_study_questions,
                "situation_handling_questions": request_data_obj.situation_handling_questions,
                "personality_test_questions": request_data_obj.personality_test_questions
            }

            # Generate interview questions
            questions_data = generate_interview_questions(jd_text, request_data_obj.resume_json, question_scales)

            return questions_data
        finally:
            # Clean up the temporary file
            if os.path.exists(temp_file_path):
                logger.info(f"Removing temporary file: {temp_file_path}")
                os.remove(temp_file_path)

    except HTTPException as e:
        # Re-raise HTTP exceptions
        raise e
    except Exception as e:
        logger.error(f"Unexpected error in interview question generation: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")

def normalize_skill(skill: str) -> str:
    """
    Normalize a skill name by removing common variations and standardizing format.

    Args:
        skill: The skill name to normalize

    Returns:
        Normalized skill name
    """
    # Handle None or empty strings
    if not skill:
        return ""

    # Convert to lowercase
    normalized = str(skill).lower()

    # Remove common suffixes and prefixes
    normalized = re.sub(r'\b(programming|development|developer|engineer|engineering|specialist|expert|proficiency|basics of|basic|advanced|intermediate|experience with|experience in|knowledge of|skills in)\b', '', normalized)

    # Remove punctuation and extra whitespace
    normalized = re.sub(r'[^\w\s]', '', normalized)
    normalized = re.sub(r'\s+', ' ', normalized).strip()

    return normalized


def get_skill_variations(skill: str) -> List[str]:
    """
    Generate common variations of a skill name.

    Args:
        skill: The base skill name

    Returns:
        List of skill variations
    """
    # Handle None or empty strings
    if not skill:
        return []

    variations = [skill]
    normalized = normalize_skill(skill)

    # Skip empty normalized skills
    if not normalized:
        return variations

    # Add normalized version if different
    if normalized != str(skill).lower():
        variations.append(normalized)

    # Add common variations for programming languages
    if normalized in ["python", "java", "javascript", "c++", "c#", "ruby", "php", "go", "rust", "typescript"]:
        variations.extend([
            f"{normalized} programming",
            f"{normalized} development",
            f"{normalized} coding",
            f"{normalized} language"
        ])

    # Add variations for frameworks
    if normalized in ["react", "angular", "vue", "django", "flask", "spring", "laravel", "express"]:
        variations.extend([
            f"{normalized} framework",
            f"{normalized} development",
            f"{normalized}.js"
        ])

    # Add variations for databases
    if normalized in ["sql", "mysql", "postgresql", "mongodb", "oracle", "sqlite", "redis"]:
        variations.extend([
            f"{normalized} database",
            f"{normalized} db"
        ])

    # Add variations for cloud platforms
    if normalized in ["aws", "azure", "gcp", "google cloud"]:
        variations.extend([
            f"{normalized} platform",
            f"{normalized} services",
            f"{normalized} cloud"
        ])

    # Special cases
    if normalized == "ml":
        variations.extend(["machine learning"])
    elif normalized == "ai":
        variations.extend(["artificial intelligence"])
    elif normalized == "ui":
        variations.extend(["user interface", "ui design"])
    elif normalized == "ux":
        variations.extend(["user experience", "ux design"])

    # Remove duplicates while preserving order
    unique_variations = []
    for v in variations:
        if v and v not in unique_variations:
            unique_variations.append(v)

    return unique_variations


def is_skill_match(resume_skill: str, jd_skill: str) -> bool:
    """
    Check if a resume skill matches a job description skill using semantic matching.

    Args:
        resume_skill: Skill from resume
        jd_skill: Skill from job description

    Returns:
        True if skills match semantically, False otherwise
    """
    # Handle None or empty strings
    if not resume_skill or not jd_skill:
        return False

    # Normalize both skills
    resume_skill_norm = normalize_skill(resume_skill)
    jd_skill_norm = normalize_skill(jd_skill)

    # Skip if either normalized skill is empty
    if not resume_skill_norm or not jd_skill_norm:
        return False

    # Direct match after normalization
    if resume_skill_norm == jd_skill_norm:
        return True

    # Check if one is a substring of the other
    # Only consider substantial matches to avoid false positives
    # For example, "C" shouldn't match "C++" but "Python" should match "Python programming"
    if len(resume_skill_norm) > 2 and len(jd_skill_norm) > 2:
        if (resume_skill_norm in jd_skill_norm) or (jd_skill_norm in resume_skill_norm):
            return True

    # Get variations of both skills and check for overlaps
    resume_variations = get_skill_variations(resume_skill)
    jd_variations = get_skill_variations(jd_skill)

    # Check if any variation of resume skill matches any variation of jd skill
    for rv in resume_variations:
        if not rv:  # Skip empty variations
            continue
        for jv in jd_variations:
            if not jv:  # Skip empty variations
                continue
            # Direct match
            if rv.lower() == jv.lower():
                return True
            # Substring match for substantial strings
            if len(rv) > 2 and len(jv) > 2:
                if rv.lower() in jv.lower() or jv.lower() in rv.lower():
                    return True

    return False


def find_matching_skills(resume_skills: List[str], jd_skills: List[str]) -> Tuple[List[str], List[str]]:
    """
    Find matching skills between resume and job description using semantic matching.

    Args:
        resume_skills: List of skills from resume
        jd_skills: List of skills from job description

    Returns:
        Tuple of (matched_skills, unmatched_skills)
    """
    # Handle None values
    if resume_skills is None:
        resume_skills = []
    if jd_skills is None:
        jd_skills = []

    matched = []
    unmatched = []

    # Filter out empty skills
    valid_resume_skills = [s for s in resume_skills if s]
    valid_jd_skills = [s for s in jd_skills if s]

    for jd_skill in valid_jd_skills:
        found_match = False
        for resume_skill in valid_resume_skills:
            if is_skill_match(resume_skill, jd_skill):
                matched.append(jd_skill)
                found_match = True
                break

        if not found_match:
            unmatched.append(jd_skill)

    return matched, unmatched


def evaluate_direct_skills_match(resume_data: Dict, jd_data: Dict) -> Tuple[float, str, Dict]:
    """
    Evaluate direct skills match between resume and job description.

    Returns:
        Tuple containing:
        - score (float): Score out of 25
        - rationale (str): Explanation of the score
        - details (Dict): Additional details like matched and missing skills
    """
    logger.info("Evaluating direct skills match")

    # Get direct skills from resume
    resume_direct_skills_list = []

    # First check if we have the new direct_skills field
    if "direct_skills" in resume_data and isinstance(resume_data["direct_skills"], dict):
        resume_direct_skills_list = list(resume_data["direct_skills"].keys())
    # Fallback to the original skills field for backward compatibility
    elif "skills" in resume_data:
        if isinstance(resume_data["skills"], list):
            resume_direct_skills_list = resume_data["skills"]
        elif isinstance(resume_data["skills"], dict):
            resume_direct_skills_list = list(resume_data["skills"].keys())

    # Get required and preferred skills from JD
    jd_required_skills_list = []
    jd_preferred_skills_list = []

    if "required_skills" in jd_data and isinstance(jd_data["required_skills"], list):
        jd_required_skills_list = jd_data["required_skills"]

    if "preferred_skills" in jd_data and isinstance(jd_data["preferred_skills"], list):
        jd_preferred_skills_list = jd_data["preferred_skills"]

    # Calculate direct skills match using semantic matching
    if not jd_required_skills_list:
        return 0, "No required skills specified in the job description", {}

    # Log skills for debugging
    logger.info(f"Resume direct skills: {resume_direct_skills_list}")
    logger.info(f"JD required skills: {jd_required_skills_list}")
    if jd_preferred_skills_list:
        logger.info(f"JD preferred skills: {jd_preferred_skills_list}")

    # Find matching and non-matching required skills
    matched_required, missing_required = find_matching_skills(resume_direct_skills_list, jd_required_skills_list)
    required_match_percentage = len(matched_required) / len(jd_required_skills_list) if jd_required_skills_list else 0

    # Log matching results
    logger.info(f"Matched required skills: {matched_required}")
    logger.info(f"Missing required skills: {missing_required}")
    logger.info(f"Required skills match percentage: {required_match_percentage:.2f}")

    # Bonus for preferred skills
    preferred_bonus = 0
    matched_preferred = []
    if jd_preferred_skills_list:
        # Find matching preferred skills
        matched_preferred, _ = find_matching_skills(resume_direct_skills_list, jd_preferred_skills_list)
        preferred_match_percentage = len(matched_preferred) / len(jd_preferred_skills_list) if jd_preferred_skills_list else 0
        preferred_bonus = preferred_match_percentage * 0.2  # 20% bonus max for preferred skills

        # Log preferred skills matching
        logger.info(f"Matched preferred skills: {matched_preferred}")
        logger.info(f"Preferred skills match percentage: {preferred_match_percentage:.2f}")
        logger.info(f"Preferred skills bonus: {preferred_bonus:.2f}")

    # Calculate final direct skills score (out of 25)
    skills_score = min(25, (required_match_percentage * 20) + (preferred_bonus * 5))
    logger.info(f"Final direct skills score: {skills_score:.2f}/25")

    # Generate rationale
    rationale = f"Matched {len(matched_required)}/{len(jd_required_skills_list)} required skills"
    if jd_preferred_skills_list:
        rationale += f" and {len(matched_preferred)}/{len(jd_preferred_skills_list)} preferred skills"

    # List matched and missing skills
    if matched_required:
        rationale += f". Matched required skills: {', '.join(matched_required)}"

    if missing_required:
        rationale += f". Missing required skills: {', '.join(missing_required)}"

    details = {
        "matched_required": matched_required,
        "missing_required": missing_required,
        "matched_preferred": matched_preferred,
        "required_match_percentage": required_match_percentage,
        "preferred_match_percentage": preferred_match_percentage if jd_preferred_skills_list else 0
    }

    return skills_score, rationale, details


def evaluate_experience_match(resume_data: Dict, jd_data: Dict) -> Tuple[float, str, Dict]:
    """
    Evaluate experience match between resume and job description.

    Returns:
        Tuple containing:
        - score (float): Score out of 20
        - rationale (str): Explanation of the score
        - details (Dict): Additional details about the experience match
    """
    logger.info("Evaluating experience match")

    # Extract required years of experience from JD
    required_yoe = None
    if "required_experience" in jd_data and jd_data["required_experience"]:
        yoe_match = re.search(r'(\d+)[\+\-]?\s*(?:year|yr|yrs|years)', jd_data["required_experience"].lower())
        if yoe_match:
            required_yoe = int(yoe_match.group(1))

    # Calculate candidate's total years of experience
    candidate_yoe = 0
    if "experience" in resume_data and isinstance(resume_data["experience"], list):
        for exp in resume_data["experience"]:
            if isinstance(exp, dict) and "duration" in exp:
                duration = exp["duration"]
                if isinstance(duration, str):
                    # Try to extract years from duration strings like "2018-2022" or "3 years"
                    year_match = re.search(r'(\d{4})\s*-\s*(\d{4}|\bpresent\b)', duration.lower())
                    if year_match:
                        start_year = int(year_match.group(1))
                        end_year = 2023  # Default to current year if "present"
                        if year_match.group(2).isdigit():
                            end_year = int(year_match.group(2))
                        candidate_yoe += (end_year - start_year)
                    else:
                        # Try direct year specification
                        direct_years = re.search(r'(\d+)\s*(?:year|yr|yrs|years)', duration.lower())
                        if direct_years:
                            candidate_yoe += int(direct_years.group(1))

    # Calculate experience match score
    if required_yoe is not None and candidate_yoe > 0:
        # Check if within 20% of required experience
        if candidate_yoe >= required_yoe * 0.8:
            if candidate_yoe <= required_yoe * 1.2:
                # Perfect match - within 20% range
                experience_score = 20
                rationale = f"Candidate has {candidate_yoe} years of experience, which is within the ideal range for the required {required_yoe} years"
            else:
                # Over-experienced but still good
                over_percentage = min(100, ((candidate_yoe - required_yoe * 1.2) / required_yoe) * 100)
                experience_score = max(10, 20 - (over_percentage / 10))
                rationale = f"Candidate has {candidate_yoe} years of experience, which is {over_percentage:.0f}% more than the required {required_yoe} years"
        else:
            # Under-experienced
            under_percentage = ((required_yoe * 0.8) - candidate_yoe) / (required_yoe * 0.8) * 100
            experience_score = max(0, 20 - (under_percentage / 5))
            rationale = f"Candidate has {candidate_yoe} years of experience, which is {under_percentage:.0f}% less than the minimum recommended {required_yoe * 0.8:.1f} years"
    else:
        experience_score = 10  # Neutral score if we can't determine
        if required_yoe is None:
            rationale = "No specific years of experience requirement found in the job description"
        else:
            rationale = f"Could not determine candidate's years of experience to compare with required {required_yoe} years"

    details = {
        "candidate_yoe": candidate_yoe,
        "required_yoe": required_yoe
    }

    return experience_score, rationale, details


def evaluate_reliability(resume_data: Dict, _: Dict) -> Tuple[float, str, Dict]:
    """
    Evaluate candidate reliability based on experience to job ratio.

    Returns:
        Tuple containing:
        - score (float): Score out of 10
        - rationale (str): Explanation of the score
        - details (Dict): Additional details about reliability
    """
    logger.info("Evaluating reliability")

    # Calculate candidate's total years of experience
    candidate_yoe = 0
    if "experience" in resume_data and isinstance(resume_data["experience"], list):
        for exp in resume_data["experience"]:
            if isinstance(exp, dict) and "duration" in exp:
                duration = exp["duration"]
                if isinstance(duration, str):
                    # Try to extract years from duration strings like "2018-2022" or "3 years"
                    year_match = re.search(r'(\d{4})\s*-\s*(\d{4}|\bpresent\b)', duration.lower())
                    if year_match:
                        start_year = int(year_match.group(1))
                        end_year = 2023  # Default to current year if "present"
                        if year_match.group(2).isdigit():
                            end_year = int(year_match.group(2))
                        candidate_yoe += (end_year - start_year)
                    else:
                        # Try direct year specification
                        direct_years = re.search(r'(\d+)\s*(?:year|yr|yrs|years)', duration.lower())
                        if direct_years:
                            candidate_yoe += int(direct_years.group(1))

    # Count number of companies
    num_companies = len(resume_data.get("experience", [])) if isinstance(resume_data.get("experience"), list) else 0

    if candidate_yoe > 0 and num_companies > 0:
        avg_tenure = candidate_yoe / num_companies

        # Score based on average tenure
        if avg_tenure >= 3:
            # Excellent tenure - 3+ years per company
            reliability_score = 10
            rationale = f"Excellent stability with average tenure of {avg_tenure:.1f} years per company"
        elif avg_tenure >= 2:
            # Good tenure - 2-3 years per company
            reliability_score = 8
            rationale = f"Good stability with average tenure of {avg_tenure:.1f} years per company"
        elif avg_tenure >= 1.5:
            # Moderate tenure - 1.5-2 years per company
            reliability_score = 6
            rationale = f"Moderate stability with average tenure of {avg_tenure:.1f} years per company"
        elif avg_tenure >= 1:
            # Below average tenure - 1-1.5 years per company
            reliability_score = 4
            rationale = f"Below average stability with average tenure of {avg_tenure:.1f} years per company"
        else:
            # Poor tenure - less than 1 year per company
            reliability_score = 2
            rationale = f"Frequent job changes with average tenure of {avg_tenure:.1f} years per company"
    else:
        # Can't determine reliability
        reliability_score = 5
        rationale = "Could not determine reliability from experience history"

    details = {
        "candidate_yoe": candidate_yoe,
        "num_companies": num_companies,
        "avg_tenure": candidate_yoe / num_companies if num_companies > 0 else 0
    }

    return reliability_score, rationale, details


def evaluate_location_match(resume_data: Dict, jd_data: Dict) -> Tuple[float, str, Dict]:
    """
    Evaluate location match between resume and job description.

    Returns:
        Tuple containing:
        - score (float): Score out of 10
        - rationale (str): Explanation of the score
        - details (Dict): Additional details about location match
    """
    logger.info("Evaluating location match")

    # Extract locations
    jd_location = None
    if "location" in jd_data and jd_data["location"]:
        jd_location = jd_data["location"].lower()

    resume_location = None
    if "location" in resume_data and resume_data["location"]:
        resume_location = resume_data["location"].lower()

    # Extract locations from experience
    experience_locations = []
    if "experience" in resume_data and isinstance(resume_data["experience"], list):
        for exp in resume_data["experience"]:
            if isinstance(exp, dict) and "company_name" in exp:
                company_name = exp["company_name"]
                if isinstance(company_name, str):
                    # Try to extract location from company name (often in format "Company Name, Location")
                    loc_match = re.search(r',\s*([A-Za-z\s,]+)$', company_name)
                    if loc_match:
                        location = loc_match.group(1).strip().lower()
                        if location and location not in experience_locations:
                            experience_locations.append(location)

    # Calculate location match score
    if jd_location and resume_location:
        # Direct location match
        if jd_location in resume_location or resume_location in jd_location:
            location_score = 10
            rationale = f"Current location ({resume_location}) matches job location ({jd_location})"
        elif any(jd_location in loc or loc in jd_location for loc in experience_locations):
            # Match with previous work locations
            location_score = 7
            rationale = f"Previous work location matches job location ({jd_location})"
        else:
            # No match
            location_score = 0
            rationale = f"Current location ({resume_location}) does not match job location ({jd_location})"
    elif jd_location and experience_locations:
        # Check if any previous work location matches
        if any(jd_location in loc or loc in jd_location for loc in experience_locations):
            location_score = 7
            rationale = f"Previous work location matches job location ({jd_location})"
        else:
            location_score = 0
            rationale = f"No location match found with job location ({jd_location})"
    elif jd_location:
        # No candidate location information
        location_score = 0
        rationale = f"Could not determine candidate's location to compare with job location ({jd_location})"
    else:
        # No job location specified
        location_score = 5
        rationale = "No specific location requirement found in the job description"

    details = {
        "jd_location": jd_location,
        "resume_location": resume_location,
        "experience_locations": experience_locations
    }

    return location_score, rationale, details


def evaluate_subjective_skills_match(resume_data: Dict, jd_data: Dict) -> Tuple[float, str, Dict]:
    """
    Evaluate subjective skills match between resume and job description.

    Returns:
        Tuple containing:
        - score (float): Score out of 15
        - rationale (str): Explanation of the score
        - details (Dict): Additional details like matched subjective skills
    """
    logger.info("Evaluating subjective skills match")

    # Get direct skills from resume (for comparison to avoid duplicates)
    resume_direct_skills_list = []
    if "direct_skills" in resume_data and isinstance(resume_data["direct_skills"], dict):
        resume_direct_skills_list = list(resume_data["direct_skills"].keys())
    elif "skills" in resume_data:
        if isinstance(resume_data["skills"], list):
            resume_direct_skills_list = resume_data["skills"]
        elif isinstance(resume_data["skills"], dict):
            resume_direct_skills_list = list(resume_data["skills"].keys())

    # Get subjective skills from resume
    resume_subjective_skills_list = []

    # First check if we have the new subjective_skills field
    if "subjective_skills" in resume_data and isinstance(resume_data["subjective_skills"], dict):
        resume_subjective_skills_list = list(resume_data["subjective_skills"].keys())
        logger.info(f"Using {len(resume_subjective_skills_list)} skills from subjective_skills field")
    else:
        # Fallback to extracting from experience and projects text
        logger.info("No subjective_skills field found, extracting from experience and projects text")
        experience_text = ""
        projects_text = ""

        if "experience" in resume_data and isinstance(resume_data["experience"], list):
            for exp in resume_data["experience"]:
                if isinstance(exp, dict):
                    for _, value in exp.items():  # Use _ to indicate unused variable
                        if isinstance(value, str):
                            experience_text += value + " "

        if "projects" in resume_data and isinstance(resume_data["projects"], list):
            for project in resume_data["projects"]:
                if isinstance(project, dict):
                    for _, value in project.items():  # Use _ to indicate unused variable
                        if isinstance(value, str):
                            projects_text += value + " "
                elif isinstance(project, str):
                    projects_text += project + " "

        combined_text = (experience_text + " " + projects_text).lower()

        # Extract potential skills from experience and projects text
        # Look for skills mentioned in specific contexts
        skill_contexts = [
            r'proficient in\s+([A-Za-z0-9+#/\s]+)',
            r'experience with\s+([A-Za-z0-9+#/\s]+)',
            r'knowledge of\s+([A-Za-z0-9+#/\s]+)',
            r'familiar with\s+([A-Za-z0-9+#/\s]+)',
            r'skills:?\s*([A-Za-z0-9+#/\s,]+)',
            r'technologies:?\s*([A-Za-z0-9+#/\s,]+)',
            r'developed\s+([A-Za-z0-9+#/\s,]+)',
            r'implemented\s+([A-Za-z0-9+#/\s,]+)',
            r'using\s+([A-Za-z0-9+#/\s,]+)'
        ]

        extracted_skills = []
        for pattern in skill_contexts:
            matches = re.finditer(pattern, combined_text)
            for match in matches:
                skill_text = match.group(1).strip()
                # Split by common separators
                for skill in re.split(r'[,;/|]|\sand\s', skill_text):
                    skill = skill.strip()
                    if skill and len(skill) > 2:
                        extracted_skills.append(skill)

        resume_subjective_skills_list = extracted_skills

    # Get all skills from JD
    jd_required_skills_list = []
    jd_preferred_skills_list = []

    if "required_skills" in jd_data and isinstance(jd_data["required_skills"], list):
        jd_required_skills_list = jd_data["required_skills"]

    if "preferred_skills" in jd_data and isinstance(jd_data["preferred_skills"], list):
        jd_preferred_skills_list = jd_data["preferred_skills"]

    # Combine required and preferred skills from JD
    all_jd_skills_list = jd_required_skills_list + jd_preferred_skills_list

    if not all_jd_skills_list:
        return 0, "No skills specified in the job description", {}

    # Log subjective skills for debugging
    if resume_subjective_skills_list:
        logger.info(f"Found {len(resume_subjective_skills_list)} potential subjective skills")
        if resume_subjective_skills_list:
            logger.info(f"Subjective skills sample: {', '.join(resume_subjective_skills_list[:10])}" +
                       ("..." if len(resume_subjective_skills_list) > 10 else ""))

    # Check for skills matches using semantic matching
    subjective_matches = []

    # Check if any of the subjective skills match JD skills
    for jd_skill in all_jd_skills_list:
        # Skip empty skills
        if not jd_skill:
            continue

        # Skip if this skill is already matched in the direct skills
        if any(is_skill_match(direct_skill, jd_skill) for direct_skill in resume_direct_skills_list if direct_skill):
            continue

        # Check if any subjective skill matches this JD skill
        for subjective_skill in resume_subjective_skills_list:
            if not subjective_skill:
                continue

            if is_skill_match(subjective_skill, jd_skill):
                subjective_matches.append(jd_skill)
                break

    # Log subjective matches
    if subjective_matches:
        logger.info(f"Found {len(subjective_matches)} additional skills in experience/projects: {', '.join(subjective_matches)}")
    else:
        logger.info("No additional skills found in experience/projects")

    # Calculate subjective skills score
    if all_jd_skills_list:
        # Calculate percentage based on additional matches found
        subjective_match_percentage = len(subjective_matches) / len(all_jd_skills_list)
        subjective_score = min(15, subjective_match_percentage * 15)
        logger.info(f"Subjective skills match percentage: {subjective_match_percentage:.2f}")
        logger.info(f"Subjective skills score: {subjective_score:.2f}/15")

        # Generate rationale
        if subjective_matches:
            rationale = f"Found {len(subjective_matches)} additional skills in experience/projects: {', '.join(subjective_matches)}"
        else:
            rationale = "No additional skills found in experience/projects"
    else:
        subjective_score = 0
        rationale = "No skills specified in the job description"

    details = {
        "subjective_matches": subjective_matches,
        "subjective_match_percentage": len(subjective_matches) / len(all_jd_skills_list) if all_jd_skills_list else 0
    }

    return subjective_score, rationale, details


def evaluate_education_match(resume_data: Dict, jd_data: Dict) -> Tuple[float, str, Dict]:
    """
    Evaluate education match between resume and job description.

    Returns:
        Tuple containing:
        - score (float): Score out of 10
        - rationale (str): Explanation of the score
        - details (Dict): Additional details about education match
    """
    logger.info("Evaluating education match")

    # Extract education requirements from JD
    jd_education_reqs = []
    if "education_requirements" in jd_data and isinstance(jd_data["education_requirements"], list):
        jd_education_reqs = jd_data["education_requirements"]

    # Extract education details from JD
    jd_degree_level = None
    jd_field_of_study = None
    if "education_details" in jd_data and isinstance(jd_data["education_details"], dict):
        if "degree_level" in jd_data["education_details"]:
            jd_degree_level = jd_data["education_details"]["degree_level"]
        if "field_of_study" in jd_data["education_details"]:
            jd_field_of_study = jd_data["education_details"]["field_of_study"]

    # Extract candidate education
    candidate_education = []
    if "education" in resume_data and isinstance(resume_data["education"], list):
        candidate_education = resume_data["education"]

    # If no education requirements or candidate has no education, return neutral score
    if not jd_education_reqs and not jd_degree_level:
        return 5, "No specific education requirements found in the job description", {}

    if not candidate_education:
        return 0, "No education information found in the candidate's resume", {}

    # Check for matches
    matches = []

    # Helper function to check if a degree matches the required level
    def matches_degree_level(degree_text, required_level):
        if not degree_text or not required_level:
            return False

        degree_text = degree_text.lower()
        required_level = required_level.lower()

        # Check for direct match
        if required_level in degree_text:
            return True

        # Check for common abbreviations and variations
        if required_level in ["bachelor", "bachelor's", "bachelors", "bs", "ba", "b.s.", "b.a.", "undergraduate"]:
            return any(term in degree_text for term in [
                "bachelor", "bachelor's", "bachelors", "bs", "ba", "b.s.", "b.a.", "undergraduate",
                "btech", "b.tech", "b.e.", "be", "b tech", "b.sc", "bsc"
            ])

        if required_level in ["master", "master's", "masters", "ms", "ma", "m.s.", "m.a.", "graduate", "post graduate", "postgraduate"]:
            return any(term in degree_text for term in [
                "master", "master's", "masters", "ms", "ma", "m.s.", "m.a.", "graduate",
                "mtech", "m.tech", "m.e.", "me", "m tech", "m.sc", "msc", "post graduate", "postgraduate"
            ])

        if required_level in ["phd", "ph.d.", "doctorate", "doctoral"]:
            return any(term in degree_text for term in ["phd", "ph.d.", "ph.d", "doctorate", "doctoral"])

        return False

    # Helper function to check if a field of study matches
    def matches_field_of_study(degree_text, required_field):
        if not degree_text or not required_field:
            return False

        degree_text = degree_text.lower()
        required_field = required_field.lower()

        # Check for direct match
        if required_field in degree_text:
            return True

        # Use semantic matching for fields
        field_mappings = {
            "computer science": ["computer science", "cs", "cse", "computer engineering", "computing", "software engineering",
                               "information technology", "it", "information systems", "aida", "ai", "artificial intelligence",
                               "data analytics", "data science", "computer applications", "mca", "comp sci"],
            "engineering": ["engineering", "engineer", "technology", "tech", "eng"],
            "business": ["business", "management", "mba", "administration", "finance", "marketing", "economics", "commerce"],
            "science": ["science", "physics", "chemistry", "biology", "mathematics", "math", "statistics"],
            "arts": ["arts", "humanities", "liberal arts", "social sciences", "psychology", "sociology", "philosophy"]
        }

        # Check if required field is in any of the mapping categories
        for category, variations in field_mappings.items():
            if required_field in variations:
                # Check if degree text contains any variation from this category
                return any(variation in degree_text for variation in variations)

        return False

    # Check each education entry against requirements
    highest_match_level = 0  # 0=no match, 1=field match, 2=level match, 3=both match

    for edu in candidate_education:
        if not isinstance(edu, dict):
            continue

        degree_text = edu.get("degree", "")
        institution = edu.get("institution", "")

        if not degree_text:
            continue

        # Check against specific education requirements
        for req in jd_education_reqs:
            if req.lower() in degree_text.lower():
                matches.append(f"Education matches requirement: {req}")
                highest_match_level = max(highest_match_level, 3)

        # Check against degree level and field of study
        level_match = jd_degree_level and matches_degree_level(degree_text, jd_degree_level)
        field_match = jd_field_of_study and matches_field_of_study(degree_text, jd_field_of_study)

        if level_match and field_match:
            matches.append(f"Education ({degree_text} from {institution}) matches required {jd_degree_level} in {jd_field_of_study}")
            highest_match_level = max(highest_match_level, 3)
        elif level_match:
            matches.append(f"Education level ({degree_text}) matches required {jd_degree_level}")
            highest_match_level = max(highest_match_level, 2)
        elif field_match:
            matches.append(f"Field of study ({degree_text}) matches required {jd_field_of_study}")
            highest_match_level = max(highest_match_level, 1)

    # Calculate education match score
    if highest_match_level == 3:
        # Perfect match - both level and field
        education_score = 10
        rationale = matches[0] if matches else "Education matches both required level and field of study"
    elif highest_match_level == 2:
        # Level match but not field
        education_score = 7
        rationale = matches[0] if matches else "Education matches required level but not field of study"
    elif highest_match_level == 1:
        # Field match but not level
        education_score = 5
        rationale = matches[0] if matches else "Education matches required field of study but not level"
    else:
        # No match
        education_score = 0
        rationale = "Education does not match job requirements"

    details = {
        "jd_education_reqs": jd_education_reqs,
        "jd_degree_level": jd_degree_level,
        "jd_field_of_study": jd_field_of_study,
        "candidate_education": candidate_education,
        "matches": matches
    }

    return education_score, rationale, details


def evaluate_certifications(resume_data: Dict, jd_data: Dict) -> Tuple[float, str, Dict]:
    """
    Evaluate certifications match between resume and job description.

    Returns:
        Tuple containing:
        - score (float): Score out of 10
        - rationale (str): Explanation of the score
        - details (Dict): Additional details about certifications match
    """
    logger.info("Evaluating certifications match")

    # Extract certifications from resume
    candidate_certs = []
    if "certifications" in resume_data and isinstance(resume_data["certifications"], list):
        for cert in resume_data["certifications"]:
            if isinstance(cert, dict) and "name" in cert:
                candidate_certs.append(cert["name"])
            elif isinstance(cert, str):
                candidate_certs.append(cert)

    # Extract certification requirements from JD
    jd_cert_reqs = []

    # Look in requirements section
    if "requirements" in jd_data and isinstance(jd_data["requirements"], list):
        for req in jd_data["requirements"]:
            if isinstance(req, dict) and "title" in req:
                title = req["title"].lower()
                if "certification" in title or "certificate" in title:
                    if "description" in req and req["description"]:
                        jd_cert_reqs.append(req["description"])
                    else:
                        jd_cert_reqs.append(title)

    # Also check in responsibilities and preferred skills
    cert_keywords = ["certification", "certificate", "certified", "license", "accreditation"]

    if "responsibilities" in jd_data and isinstance(jd_data["responsibilities"], list):
        for resp in jd_data["responsibilities"]:
            if isinstance(resp, str) and any(keyword in resp.lower() for keyword in cert_keywords):
                jd_cert_reqs.append(resp)

    if "preferred_skills" in jd_data and isinstance(jd_data["preferred_skills"], list):
        for skill in jd_data["preferred_skills"]:
            if isinstance(skill, str) and any(keyword in skill.lower() for keyword in cert_keywords):
                jd_cert_reqs.append(skill)

    # If no certification requirements or candidate has no certifications
    if not jd_cert_reqs:
        return 5, "No specific certification requirements found in the job description", {}

    if not candidate_certs:
        return 0, "No certifications found in the candidate's resume", {}

    # Check for matches using semantic matching
    matches = []

    for jd_cert in jd_cert_reqs:
        for candidate_cert in candidate_certs:
            # Use the is_skill_match function for semantic matching
            if is_skill_match(jd_cert, candidate_cert):
                matches.append(f"Certification match: {candidate_cert}")
                break

    # Calculate certifications match score
    if matches:
        match_percentage = len(matches) / len(jd_cert_reqs)
        cert_score = min(10, match_percentage * 10)
        rationale = f"Matched {len(matches)}/{len(jd_cert_reqs)} certification requirements: {', '.join(matches)}"
    else:
        cert_score = 0
        rationale = "No matching certifications found"

    details = {
        "jd_cert_reqs": jd_cert_reqs,
        "candidate_certs": candidate_certs,
        "matches": matches
    }

    return cert_score, rationale, details


def calculate_candidate_job_fit(resume_data: Dict, jd_data: Dict) -> Dict:
    """
    Calculate how well a candidate's resume matches a job description.

    This function evaluates the fit between a candidate and a job based on multiple criteria:
    1. Skills matching (direct and from projects/experience)
    2. Years of experience
    3. Reliability (experience to job ratio)
    4. Location match
    5. Academic qualifications
    6. Alma mater prestige
    7. Relevant certifications

    Returns a dictionary with overall score and detailed rationale.

    This implementation uses a modular approach, calling the LLM separately for each
    evaluation criterion to potentially improve accuracy and handle LLM fatigue.
    """
    logger.info("Calculating candidate-job fit score using modular approach")

    # Initialize scores and rationale dictionaries
    scores = {}
    rationale = {}

    # 1. Evaluate direct skills match (25%)
    logger.info("Step 1: Evaluating direct skills match")
    try:
        direct_skills_score, direct_skills_rationale, _ = evaluate_direct_skills_match(resume_data, jd_data)
        scores["skills_match_direct"] = direct_skills_score
        rationale["skills_match_direct"] = direct_skills_rationale
    except Exception as e:
        logger.error(f"Error evaluating direct skills match: {e}")
        scores["skills_match_direct"] = 0
        rationale["skills_match_direct"] = f"Error evaluating direct skills match: {str(e)}"

    # 2. Evaluate subjective skills match (15%)
    logger.info("Step 2: Evaluating subjective skills match")
    try:
        subjective_skills_score, subjective_skills_rationale, _ = evaluate_subjective_skills_match(resume_data, jd_data)
        scores["skills_match_subjective"] = subjective_skills_score
        rationale["skills_match_subjective"] = subjective_skills_rationale
    except Exception as e:
        logger.error(f"Error evaluating subjective skills match: {e}")
        scores["skills_match_subjective"] = 0
        rationale["skills_match_subjective"] = f"Error evaluating subjective skills match: {str(e)}"

    # 3. Evaluate experience match (20%)
    logger.info("Step 3: Evaluating experience match")
    try:
        experience_score, experience_rationale, _ = evaluate_experience_match(resume_data, jd_data)
        scores["experience_match"] = experience_score
        rationale["experience_match"] = experience_rationale
    except Exception as e:
        logger.error(f"Error evaluating experience match: {e}")
        scores["experience_match"] = 0
        rationale["experience_match"] = f"Error evaluating experience match: {str(e)}"

    # 4. Evaluate reliability (10%)
    logger.info("Step 4: Evaluating reliability")
    try:
        reliability_score, reliability_rationale, _ = evaluate_reliability(resume_data, jd_data)
        scores["reliability"] = reliability_score
        rationale["reliability"] = reliability_rationale
    except Exception as e:
        logger.error(f"Error evaluating reliability: {e}")
        scores["reliability"] = 0
        rationale["reliability"] = f"Error evaluating reliability: {str(e)}"

    # 5. Evaluate location match (10%)
    logger.info("Step 5: Evaluating location match")
    try:
        location_score, location_rationale, _ = evaluate_location_match(resume_data, jd_data)
        scores["location_match"] = location_score
        rationale["location_match"] = location_rationale
    except Exception as e:
        logger.error(f"Error evaluating location match: {e}")
        scores["location_match"] = 0
        rationale["location_match"] = f"Error evaluating location match: {str(e)}"

    # 6. Evaluate education match (10%)
    logger.info("Step 6: Evaluating education match")
    try:
        education_score, education_rationale, _ = evaluate_education_match(resume_data, jd_data)
        scores["education_match"] = education_score
        rationale["education_match"] = education_rationale
    except Exception as e:
        logger.error(f"Error evaluating education match: {e}")
        scores["education_match"] = 0
        rationale["education_match"] = f"Error evaluating education match: {str(e)}"

    # 7. Evaluate certifications match (5%)
    logger.info("Step 7: Evaluating certifications match")
    try:
        certifications_score, certifications_rationale, _ = evaluate_certifications(resume_data, jd_data)
        # Adjust certification score to be out of 5 instead of 10
        certifications_score = certifications_score / 2
        scores["certifications_match"] = certifications_score
        rationale["certifications_match"] = certifications_rationale
    except Exception as e:
        logger.error(f"Error evaluating certifications match: {e}")
        scores["certifications_match"] = 0
        rationale["certifications_match"] = f"Error evaluating certifications match: {str(e)}"

    # Calculate total score (out of 100)
    total_score = sum(scores.values())

    # Determine fit category
    fit_category = ""
    if total_score >= 85:
        fit_category = "Excellent Match"
    elif total_score >= 70:
        fit_category = "Strong Match"
    elif total_score >= 55:
        fit_category = "Good Match"
    elif total_score >= 40:
        fit_category = "Moderate Match"
    else:
        fit_category = "Weak Match"

    # Create summary
    summary = f"The candidate is a {fit_category.lower()} for this position with a score of {total_score}/100. "

    # Add key strengths and weaknesses
    strengths = []
    weaknesses = []

    for category, score in scores.items():
        # Define max scores for each category
        if category == "skills_match_direct":
            max_score = 25
        elif category == "experience_match":
            max_score = 20
        elif category == "skills_match_subjective":
            max_score = 15
        elif category == "certifications_match":
            max_score = 5
        else:
            max_score = 10

        percentage = (score / max_score) * 100

        if percentage >= 80:
            strengths.append(category.replace("_", " ").title())
        elif percentage <= 30:
            weaknesses.append(category.replace("_", " ").title())

    if strengths:
        summary += f"Key strengths: {', '.join(strengths)}. "

    if weaknesses:
        summary += f"Areas for improvement: {', '.join(weaknesses)}."

    # Return the result
    return {
        "total_score": total_score,
        "fit_category": fit_category,
        "summary": summary,
        "scores": scores,
        "rationale": rationale
    }

def generate_jd_only_questions(jd_text: str) -> Dict:
    """Generate interview questions based only on job description."""
    logger.info("Starting JD-only interview question generation")

    # Define the categories
    categories = [
        "technical_questions",
        "past_experience_questions",
        "case_study_questions",
        "situation_handling_questions",
        "personality_test_questions"
    ]

    # Initialize results dictionary
    questions_data = {}

    # Generate 5 questions for each category in parallel
    from concurrent.futures import ThreadPoolExecutor

    def process_category_jd_only(category):
        # Map category to a more readable name
        category_name = {
            "technical_questions": "Technical",
            "past_experience_questions": "Past Experience",
            "case_study_questions": "Case Study",
            "situation_handling_questions": "Situation Handling",
            "personality_test_questions": "Personality Test"
        }.get(category, category)

        # Special handling for personality test questions
        if category == "personality_test_questions":
            # For personality questions, use a more general prompt focused on personality traits
            prompt = f"""
            You are an expert interview question generator. Your task is to create 5 personality assessment questions.

            Generate exactly 5 personality assessment questions that:
            - Focus on general personality traits like teamwork, leadership, adaptability, work ethic, etc.
            - Help understand the candidate's character, values, and working style
            - Are NOT specific to the job description but rather about the person's general traits
            - Reveal how the candidate might fit into different team environments
            - Assess soft skills and interpersonal abilities

            Examples of good personality questions:
            - "How do you handle criticism or feedback from colleagues or supervisors?"
            - "Describe a situation where you had to adapt to a significant change at work. How did you handle it?"
            - "What motivates you the most in your professional life?"
            - "How do you prioritize tasks when facing multiple deadlines?"
            - "Describe your ideal work environment and management style."

            IMPORTANT: Respond ONLY with a JSON array of strings, where each string is a question.
            Example: ["Question 1?", "Question 2?", "Question 3?", "Question 4?", "Question 5?"]

            DO NOT include any explanations, markdown formatting, or code blocks in your response.
            Your entire response should be ONLY the JSON array, nothing else.
            """
        else:
            # For all other categories, use the original prompt
            prompt = f"""
            You are an expert interview question generator. Your task is to create 5 tailored {category_name} questions based on a job description.

            Job Description:
            {jd_text}

            Generate exactly 5 {category_name} questions that are:
            - Relevant to the job description
            - Specific and detailed (not generic)
            - Designed to assess a candidate's fit for the role

            IMPORTANT: Respond ONLY with a JSON array of strings, where each string is a question.
            Example: ["Question 1?", "Question 2?", "Question 3?", "Question 4?", "Question 5?"]

            DO NOT include any explanations, markdown formatting, or code blocks in your response.
            Your entire response should be ONLY the JSON array, nothing else.
            """

        try:
            # Use a shorter timeout for individual category generation
            response = get_response(prompt, timeout_seconds=45, max_tokens=500)

            # Try to extract JSON array from the response
            json_str = response.strip()

            # Remove any markdown formatting if present
            if "```json" in json_str:
                json_str = json_str.split("```json")[1].split("```")[0].strip()
            elif "```" in json_str:
                json_str = json_str.split("```")[1].split("```")[0].strip()

            # Try to find the JSON array if there's additional text
            if json_str.find('[') >= 0 and json_str.rfind(']') >= 0:
                start = json_str.find('[')
                end = json_str.rfind(']') + 1
                json_str = json_str[start:end]

            # Parse the JSON string into a Python list
            try:
                questions = json.loads(json_str)

                # Ensure we have a list of strings
                if isinstance(questions, list):
                    # Convert all items to strings and filter out empty ones
                    questions = [str(q) for q in questions if q]
                    return category, questions[:5]  # Limit to 5 questions
                else:
                    logger.warning(f"Expected a list but got {type(questions)} for {category}")
                    return category, []

            except json.JSONDecodeError:
                # If JSON parsing fails, try to extract questions using regex
                # Look for text that looks like questions (ends with ? or has question-like structure)
                question_pattern = re.compile(r'"([^"]+\?)"')
                matches = question_pattern.findall(json_str)

                if matches:
                    return category, matches[:5]  # Limit to 5 questions
                else:
                    logger.warning(f"Could not extract questions for {category}")
                    return category, []

        except Exception as e:
            logger.error(f"Error generating questions for {category}: {e}")
            return category, []

    # Use ThreadPoolExecutor to process categories in parallel
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(process_category_jd_only, category): category for category in categories}

        for future in futures:
            try:
                category, questions = future.result()
                questions_data[category] = questions
                logger.info(f"Generated {len(questions)} questions for {category}")
            except Exception as e:
                logger.error(f"Error processing category: {e}")
                # Initialize with empty list on error
                questions_data[futures[future]] = []

    # Ensure all categories are present
    for category in categories:
        if category not in questions_data:
            questions_data[category] = []

    logger.info("Successfully generated all JD-only interview questions")
    return questions_data


def generate_jd_only_questions(jd_text: str) -> Dict:
    """Generate interview questions based only on job description."""
    logger.info("Generating interview questions based only on job description")

    # Parse the job description text to extract key information
    try:
        jd_data = parse_job_description(jd_text)
        logger.info("Successfully parsed job description")
    except Exception as e:
        logger.error(f"Error parsing job description: {str(e)}")
        jd_data = {}

    # Define question categories
    categories = [
        "technical_skills",
        "behavioral",
        "experience",
        "problem_solving",
        "personality_test"
    ]

    # Generate questions for each category
    questions_data = {}

    # Generate questions using the AI model
    prompt = generate_jd_only_questions_prompt(jd_data)

    try:
        # Generate questions using the AI model
        response = generate_with_gemma(prompt)
        logger.info("Successfully generated questions with AI model")

        # Parse the response to extract questions by category
        parsed_questions = parse_questions_from_response(response, categories)
        questions_data.update(parsed_questions)
    except Exception as e:
        logger.error(f"Error generating questions with AI model: {str(e)}")
        # Provide fallback questions if AI generation fails
        for category in categories:
            questions_data[category] = generate_fallback_questions(category, jd_data)

    # Ensure all categories are present in the output
    for category in categories:
        if category not in questions_data:
            questions_data[category] = []

    logger.info("Successfully generated JD-only interview questions")
    return questions_data


def generate_jd_only_questions_prompt(jd_data: Dict) -> str:
    """Generate a prompt for the AI model to generate interview questions based on job description."""
    prompt = """You are an expert interviewer with deep knowledge of technical and behavioral interview questions.

Based on the following job description, generate interview questions in the following categories:
1. Technical Skills Questions (5 questions)
2. Behavioral Questions (5 questions)
3. Experience Questions (5 questions)
4. Problem Solving Questions (5 questions)
5. Personality Test Questions (5 questions)

For each category, provide thoughtful, specific questions that will help assess the candidate's fit for this role.
The personality test questions should focus on general personality traits, not specific job requirements.

Job Description:
"""

    # Add job description details to the prompt
    if isinstance(jd_data, dict):
        # Add job title
        if "job_title" in jd_data and jd_data["job_title"]:
            prompt += f"\nJob Title: {jd_data['job_title']}\n"

        # Add company name
        if "company_name" in jd_data and jd_data["company_name"]:
            prompt += f"Company: {jd_data['company_name']}\n"

        # Add job description
        if "job_description" in jd_data and jd_data["job_description"]:
            prompt += f"\nDescription: {jd_data['job_description']}\n"

        # Add required skills
        if "required_skills" in jd_data and isinstance(jd_data["required_skills"], list) and jd_data["required_skills"]:
            prompt += f"\nRequired Skills: {', '.join(jd_data['required_skills'])}\n"

        # Add preferred skills
        if "preferred_skills" in jd_data and isinstance(jd_data["preferred_skills"], list) and jd_data["preferred_skills"]:
            prompt += f"\nPreferred Skills: {', '.join(jd_data['preferred_skills'])}\n"

        # Add required experience
        if "required_experience" in jd_data and jd_data["required_experience"]:
            prompt += f"\nRequired Experience: {jd_data['required_experience']}\n"

        # Add education requirements
        if "education_requirements" in jd_data and isinstance(jd_data["education_requirements"], list) and jd_data["education_requirements"]:
            prompt += f"\nEducation Requirements: {', '.join(jd_data['education_requirements'])}\n"

    # Add formatting instructions
    prompt += """
Please format your response as follows:

TECHNICAL_SKILLS:
1. [Question 1]
2. [Question 2]
...

BEHAVIORAL:
1. [Question 1]
2. [Question 2]
...

EXPERIENCE:
1. [Question 1]
2. [Question 2]
...

PROBLEM_SOLVING:
1. [Question 1]
2. [Question 2]
...

PERSONALITY_TEST:
1. [Question 1]
2. [Question 2]
...

Make sure each question is specific, relevant to the job description, and designed to assess the candidate's qualifications effectively.
"""

    return prompt


def parse_questions_from_response(response: str, categories: List[str]) -> Dict[str, List[str]]:
    """Parse the AI model's response to extract questions by category."""
    result = {category: [] for category in categories}

    # Define patterns to match each category section
    patterns = {
        "technical_skills": r"TECHNICAL_SKILLS:(?:\r?\n)((?:(?:\d+\.\s+.+)(?:\r?\n|$))+)",
        "behavioral": r"BEHAVIORAL:(?:\r?\n)((?:(?:\d+\.\s+.+)(?:\r?\n|$))+)",
        "experience": r"EXPERIENCE:(?:\r?\n)((?:(?:\d+\.\s+.+)(?:\r?\n|$))+)",
        "problem_solving": r"PROBLEM_SOLVING:(?:\r?\n)((?:(?:\d+\.\s+.+)(?:\r?\n|$))+)",
        "personality_test": r"PERSONALITY_TEST:(?:\r?\n)((?:(?:\d+\.\s+.+)(?:\r?\n|$))+)"
    }

    # Extract questions for each category
    for category, pattern in patterns.items():
        if category in categories:
            matches = re.search(pattern, response, re.IGNORECASE | re.MULTILINE)
            if matches:
                # Extract the questions block
                questions_block = matches.group(1)
                # Split into individual questions
                questions = re.findall(r"\d+\.\s+(.+)(?:\r?\n|$)", questions_block)
                result[category] = questions

    return result


def generate_fallback_questions(category: str, _: Dict = None) -> List[str]:
    """Generate fallback questions if AI model fails."""
    # Default questions by category
    fallback_questions = {
        "technical_skills": [
            "Can you describe your experience with the main technical skills required for this role?",
            "How do you stay updated with the latest developments in your technical field?",
            "Describe a technical challenge you faced and how you overcame it.",
            "What technical tools or software are you most proficient with?",
            "How would you rate your expertise in the required technical skills for this position?"
        ],
        "behavioral": [
            "Describe a situation where you had to work under pressure to meet a deadline.",
            "Tell me about a time when you had to resolve a conflict within your team.",
            "How do you prioritize tasks when you have multiple responsibilities?",
            "Describe a situation where you had to adapt to a significant change at work.",
            "Tell me about a time when you went above and beyond what was required."
        ],
        "experience": [
            "What aspects of your previous experience are most relevant to this role?",
            "Describe your most significant professional achievement and why it matters.",
            "What challenges in your previous roles have prepared you for this position?",
            "How has your career progression prepared you for this role?",
            "What unique perspective would you bring to this position based on your experience?"
        ],
        "problem_solving": [
            "Describe a complex problem you solved and your approach to solving it.",
            "How do you approach problems that don't have clear solutions?",
            "Tell me about a time when you had to make a decision with incomplete information.",
            "What analytical tools or methods do you use when solving problems?",
            "Describe a situation where your initial approach to a problem didn't work and how you adjusted."
        ],
        "personality_test": [
            "How would your colleagues describe your work style?",
            "What motivates you professionally?",
            "How do you handle stress or pressure in the workplace?",
            "What type of work environment brings out your best performance?",
            "What are your long-term career goals and how does this position fit into them?"
        ]
    }

    # Return the questions for the specified category
    return fallback_questions.get(category, [])


def generate_jd_only_questions(jd_text: str) -> Dict:
    """Generate interview questions based only on job description."""
    try:
        # Parse the job description text into structured data
        jd_data = parse_job_description(jd_text)

        # Generate a prompt for the AI model
        prompt = generate_jd_only_questions_prompt(jd_data)

        # Generate questions using the AI model
        response = generate_with_gemma(prompt)

        # Parse the response into structured data
        categories = ["technical_skills", "behavioral", "experience", "problem_solving", "personality_test"]
        questions_data = parse_questions_from_response(response, categories)

        # Fill in any missing categories with fallback questions
        for category in categories:
            if category not in questions_data or not questions_data[category]:
                questions_data[category] = generate_fallback_questions(category, jd_data)

        logger.info(f"Generated {sum(len(questions) for questions in questions_data.values())} interview questions from JD")
        return questions_data
    except Exception as e:
        logger.error(f"Error generating JD-only interview questions: {str(e)}")
        # Return empty questions data with all categories
        return {category: [] for category in ["technical_skills", "behavioral", "experience", "problem_solving", "personality_test"]}

def parse_job_description(jd_text: str) -> Dict:
    """Parse job description text into structured data."""
    try:
        # Generate a prompt for the AI model
        prompt = """You are an expert job description analyzer. Extract structured information from the following job description.

Job Description:
"""
        prompt += jd_text

        prompt += """

Extract and return the following information in JSON format:
1. job_title: The title of the job
2. company_name: The name of the company
3. job_description: The full description of the job
4. required_skills: A list of required skills for the job
5. preferred_skills: A list of preferred/nice-to-have skills for the job
6. required_experience: The required years of experience
7. education_requirements: A list of education requirements
8. location: The job location

Return ONLY the JSON object with these fields, nothing else.
"""

        # Generate structured data using the AI model
        response = generate_with_gemma(prompt)

        # Parse the JSON response
        # Find JSON object in the response (it might be surrounded by markdown code blocks or other text)
        json_match = re.search(r'```(?:json)?\s*({[\s\S]*?})\s*```|({[\s\S]*?})', response)
        if json_match:
            json_str = json_match.group(1) or json_match.group(2)
            try:
                jd_data = json.loads(json_str)

                # Ensure all expected fields are present
                expected_fields = ["job_title", "company_name", "job_description", "required_skills",
                                  "preferred_skills", "required_experience", "education_requirements", "location"]
                for field in expected_fields:
                    if field not in jd_data:
                        if field in ["required_skills", "preferred_skills", "education_requirements"]:
                            jd_data[field] = []
                        else:
                            jd_data[field] = ""

                return jd_data
            except json.JSONDecodeError:
                logger.error("Failed to parse JSON from AI response")

        # Fallback: Return a basic structure if parsing fails
        return {
            "job_title": "",
            "company_name": "",
            "job_description": jd_text,
            "required_skills": [],
            "preferred_skills": [],
            "required_experience": "",
            "education_requirements": [],
            "location": ""
        }
    except Exception as e:
        logger.error(f"Error parsing job description: {str(e)}")
        return {
            "job_title": "",
            "company_name": "",
            "job_description": jd_text,
            "required_skills": [],
            "preferred_skills": [],
            "required_experience": "",
            "education_requirements": [],
            "location": ""
        }


def generate_with_gemma(prompt: str) -> str:
    """Generate text using the Gemma model."""
    try:
        # Use the existing get_response function
        response = get_response(prompt, timeout_seconds=60, max_tokens=1000)
        return response
    except Exception as e:
        logger.error(f"Error generating with Gemma: {str(e)}")
        return ""



def generate_jd_only_questions(jd_text: str) -> Dict:
    """Generate interview questions based only on job description."""
    try:
        # Parse the job description text into structured data
        jd_data = parse_job_description(jd_text)

        # Generate a prompt for the AI model
        prompt = generate_jd_only_questions_prompt(jd_data)

        # Generate questions using the AI model
        response = generate_with_gemma(prompt)

        # Parse the response into structured data
        categories = ["technical_skills", "behavioral", "experience", "problem_solving", "personality_test"]
        questions_data = parse_questions_from_response(response, categories)

        # Fill in any missing categories with fallback questions
        for category in categories:
            if category not in questions_data or not questions_data[category]:
                questions_data[category] = generate_fallback_questions(category, jd_data)

        logger.info(f"Generated {sum(len(questions) for questions in questions_data.values())} interview questions from JD")
        return questions_data
    except Exception as e:
        logger.error(f"Error generating JD-only interview questions: {str(e)}")
        # Return empty questions data with all categories
        return {category: [] for category in ["technical_skills", "behavioral", "experience", "problem_solving", "personality_test"]}

def generate_jd_only_questions(jd_text: str) -> Dict:
    """Generate interview questions based only on job description."""
    logger.info("Starting JD-only interview question generation")

    # Define the categories
    categories = [
        "technical_questions",
        "past_experience_questions",
        "case_study_questions",
        "situation_handling_questions",
        "personality_test_questions"
    ]

    # Initialize results dictionary
    questions_data = {}

    # Generate 5 questions for each category in parallel
    from concurrent.futures import ThreadPoolExecutor

    def process_category_jd_only(category):
        # Map category to a more readable name
        category_name = {
            "technical_questions": "Technical",
            "past_experience_questions": "Past Experience",
            "case_study_questions": "Case Study",
            "situation_handling_questions": "Situation Handling",
            "personality_test_questions": "Personality Test"
        }.get(category, category)

        # Special handling for personality test questions
        if category == "personality_test_questions":
            # For personality questions, use a more general prompt focused on personality traits
            prompt = f"""
            You are an expert interview question generator. Your task is to create 5 personality assessment questions.

            Generate exactly 5 personality assessment questions that:
            - Focus on general personality traits like teamwork, leadership, adaptability, work ethic, etc.
            - Help understand the candidate's character, values, and working style
            - Are NOT specific to the job description but rather about the person's general traits
            - Reveal how the candidate might fit into different team environments
            - Assess soft skills and interpersonal abilities

            Examples of good personality questions:
            - "How do you handle criticism or feedback from colleagues or supervisors?"
            - "Describe a situation where you had to adapt to a significant change at work. How did you handle it?"
            - "What motivates you the most in your professional life?"
            - "How do you prioritize tasks when facing multiple deadlines?"
            - "Describe your ideal work environment and management style."

            IMPORTANT: Respond ONLY with a JSON array of strings, where each string is a question.
            Example: ["Question 1?", "Question 2?", "Question 3?", "Question 4?", "Question 5?"]

            DO NOT include any explanations, markdown formatting, or code blocks in your response.
            Your entire response should be ONLY the JSON array, nothing else.
            """
        else:
            # For all other categories, use the original prompt
            prompt = f"""
            You are an expert interview question generator. Your task is to create 5 tailored {category_name} questions based on a job description.

            Job Description:
            {jd_text}

            Generate exactly 5 {category_name} questions that are:
            - Relevant to the job description
            - Specific and detailed (not generic)
            - Designed to assess a candidate's fit for the role

            IMPORTANT: Respond ONLY with a JSON array of strings, where each string is a question.
            Example: ["Question 1?", "Question 2?", "Question 3?", "Question 4?", "Question 5?"]

            DO NOT include any explanations, markdown formatting, or code blocks in your response.
            Your entire response should be ONLY the JSON array, nothing else.
            """

        try:
            # Use a shorter timeout for individual category generation
            response = get_response(prompt, timeout_seconds=45, max_tokens=500)

            # Try to extract JSON array from the response
            json_str = response.strip()

            # Remove any markdown formatting if present
            if "```json" in json_str:
                json_str = json_str.split("```json")[1].split("```")[0].strip()
            elif "```" in json_str:
                json_str = json_str.split("```")[1].split("```")[0].strip()

            # Try to find the JSON array if there's additional text
            if json_str.find('[') >= 0 and json_str.rfind(']') >= 0:
                start = json_str.find('[')
                end = json_str.rfind(']') + 1
                json_str = json_str[start:end]

            # Parse the JSON string into a Python list
            try:
                questions = json.loads(json_str)

                # Ensure we have a list of strings
                if isinstance(questions, list):
                    # Convert all items to strings and filter out empty ones
                    questions = [str(q) for q in questions if q]
                    return category, questions[:5]  # Limit to 5 questions
                else:
                    logger.warning(f"Expected a list but got {type(questions)} for {category}")
                    return category, []

            except json.JSONDecodeError:
                # If JSON parsing fails, try to extract questions using regex
                # Look for text that looks like questions (ends with ? or has question-like structure)
                question_pattern = re.compile(r'"([^"]+\?)"')
                matches = question_pattern.findall(json_str)

                if matches:
                    return category, matches[:5]  # Limit to 5 questions
                else:
                    logger.warning(f"Could not extract questions for {category}")
                    return category, []

        except Exception as e:
            logger.error(f"Error generating questions for {category}: {e}")
            return category, []

    # Use ThreadPoolExecutor to process categories in parallel
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(process_category_jd_only, category): category for category in categories}

        for future in futures:
            try:
                category, questions = future.result()
                questions_data[category] = questions
                logger.info(f"Generated {len(questions)} questions for {category}")
            except Exception as e:
                logger.error(f"Error processing category: {e}")
                # Initialize with empty list on error
                questions_data[futures[future]] = []

    # Ensure all categories are present
    for category in categories:
        if category not in questions_data:
            questions_data[category] = []

    logger.info("Successfully generated all JD-only interview questions")
    return questions_data

@app.post("/jd_only", response_model=Dict, summary="Generate interview questions from job description only", description="Upload a job description and get tailored interview questions without needing resume data")
async def generate_jd_only_questions_endpoint(file: UploadFile = File(..., description="Job description file to parse (PDF or DOCX format)")):
    """Generate tailored interview questions based only on a job description.

    - **file**: Upload a PDF or DOCX file containing a job description

    Returns a JSON object with 5 questions for each of these categories:
    - Technical questions
    - Past experience questions
    - Case study questions
    - Situation handling questions
    - Personality test questions

    Note: This endpoint processes each question category in parallel for faster response times.
    """
    try:
        # Check if the file is a supported format (PDF or DOCX)
        file_extension = os.path.splitext(file.filename.lower())[1]

        if file_extension == '.pdf':
            file_type = "pdf"
            suffix = '.pdf'
        elif file_extension == '.docx':
            file_type = "docx"
            suffix = '.docx'
        else:
            logger.warning(f"Unsupported file format: {file.filename}")
            raise HTTPException(status_code=400, detail="Only PDF and DOCX files are supported.")

        logger.info(f"Processing job description file: {file.filename} (type: {file_type})")

        # Create a temporary file to store the uploaded file
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            # Write the uploaded file content to the temporary file
            content = await file.read()
            if not content:
                logger.error("Uploaded file is empty")
                raise HTTPException(status_code=400, detail="The uploaded file is empty.")

            temp_file.write(content)
            temp_file_path = temp_file.name
            logger.info(f"Saved uploaded file to temporary location: {temp_file_path}")

        try:
            # Extract text from the file based on its type
            jd_text = extract_text_from_file(temp_file_path, file_type)

            # Generate interview questions based only on the JD
            questions_data = generate_jd_only_questions(jd_text)

            return questions_data
        finally:
            # Clean up the temporary file
            if os.path.exists(temp_file_path):
                logger.info(f"Removing temporary file: {temp_file_path}")
                os.remove(temp_file_path)

    except HTTPException as e:
        # Re-raise HTTP exceptions
        raise e
    except Exception as e:
        logger.error(f"Unexpected error in JD-only question generation: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")

@app.post("/intervet", response_model=Dict, summary="Evaluate candidate-job fit", description="Evaluate how well a candidate's resume matches a job description")
async def evaluate_candidate_job_fit_endpoint(request: IntervetRequest):
    """Evaluate how well a candidate's resume matches a job description.

    - **resume_json**: Resume data in JSON format (typically obtained from the /resume endpoint)
    - **jd_json**: Job description data in JSON format (typically obtained from the /jd_parser endpoint)

    Returns a comprehensive evaluation with:
    - Overall match score (0-100)
    - Fit category (Excellent/Strong/Good/Moderate/Weak Match)
    - Summary of the match
    - Detailed scores for each evaluation criterion
    - Detailed rationale for each score

    The evaluation is based on multiple criteria:
    1. Skills matching (direct and from projects/experience)
    2. Years of experience
    3. Reliability (experience to job ratio)
    4. Location match
    5. Academic qualifications
    6. Alma mater prestige
    7. Relevant certifications
    """
    try:
        logger.info("Starting candidate-job fit evaluation")

        # Ensure we have valid JSON data
        if not request.resume_json:
            raise HTTPException(status_code=400, detail="Missing or invalid resume_json data")

        if not request.jd_json:
            raise HTTPException(status_code=400, detail="Missing or invalid jd_json data")

        # Calculate the match score
        result = calculate_candidate_job_fit(request.resume_json, request.jd_json)

        return result
    except HTTPException as e:
        # Re-raise HTTP exceptions
        raise e
    except Exception as e:
        logger.error(f"Error in candidate-job fit evaluation: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An error occurred during evaluation: {str(e)}")

@app.post("/bunchtest", response_model=Dict, summary="Batch evaluate multiple resumes against one job description", description="Upload multiple resume files and one job description file for batch evaluation")
async def batch_evaluate_resumes_endpoint(
    request: Request,
    resume_files: List[UploadFile] = File(..., description="Multiple resume files to parse (PDF or DOCX format)"),
    jd_file: UploadFile = File(..., description="Job description file to parse (PDF or DOCX format)")
):
    """Evaluate how well multiple resumes match a job description, taking all as file uploads.

    - **resume_files**: Upload multiple PDF or DOCX files containing resumes
    - **jd_file**: Upload a PDF or DOCX file containing a job description

    Returns a dictionary with:
    - A list of evaluation results, one for each resume
    - Each evaluation contains the same comprehensive information as the /intervet2 endpoint
    - Results are sorted by overall match score in descending order

    This endpoint processes each resume individually through the intervet pipeline and compares it with the same job description.
    """
    try:
        logger.info(f"Starting batch evaluation of {len(resume_files)} resumes against one job description")

        # Get metrics tracker from request state
        metrics = getattr(request.state, "metrics", None)
        if metrics:
            metrics.add_metric("endpoint_type", "bunchtest")
            metrics.add_metric("resume_count", len(resume_files))

        # Process JD file first (only once)
        jd_temp_file = None
        try:
            # Save JD file to temp file
            jd_content = await jd_file.read()
            jd_file_ext = os.path.splitext(jd_file.filename)[1].lower()

            # Determine file type
            if jd_file_ext == '.pdf':
                jd_file_type = "pdf"
            elif jd_file_ext in ['.doc', '.docx']:
                jd_file_type = "docx"
            else:
                raise HTTPException(status_code=400, detail="Job description file must be PDF or DOCX format")

            # Create temp file
            jd_temp_file = tempfile.mktemp(suffix=jd_file_ext)
            with open(jd_temp_file, "wb") as f:
                f.write(jd_content)

            # Extract text from JD file
            logger.info(f"Extracting text from job description file: {jd_file.filename}")
            jd_text = extract_text_from_file(jd_temp_file, jd_file_type)

            # Parse JD text
            logger.info(f"Parsing job description text ({len(jd_text)} characters)")
            jd_data = parse_jd(jd_text)

            # Process each resume file
            results = []
            for i, resume_file in enumerate(resume_files):
                logger.info(f"Processing resume {i+1}/{len(resume_files)}: {resume_file.filename}")

                resume_temp_file = None
                try:
                    # Save resume file to temp file
                    resume_content = await resume_file.read()
                    resume_file_ext = os.path.splitext(resume_file.filename)[1].lower()

                    # Determine file type
                    if resume_file_ext == '.pdf':
                        resume_file_type = "pdf"
                    elif resume_file_ext in ['.doc', '.docx']:
                        resume_file_type = "docx"
                    else:
                        logger.warning(f"Skipping file {resume_file.filename} - not a PDF or DOCX file")
                        continue

                    # Create temp file
                    resume_temp_file = tempfile.mktemp(suffix=resume_file_ext)
                    with open(resume_temp_file, "wb") as f:
                        f.write(resume_content)

                    # Extract text from resume file
                    logger.info(f"Extracting text from resume file: {resume_file.filename}")
                    resume_text = extract_text_from_file(resume_temp_file, resume_file_type)

                    # Parse resume text
                    logger.info(f"Parsing resume text ({len(resume_text)} characters)")
                    resume_data = parse_resume(resume_text, convert_skills_to_dict_format=True)

                    # Calculate match score
                    logger.info(f"Evaluating resume {i+1} against job description")
                    result = calculate_candidate_job_fit(resume_data, jd_data)

                    # Add resume filename to result
                    result["resume_filename"] = resume_file.filename

                    # Add result to list
                    results.append(result)

                finally:
                    # Clean up resume temp file
                    if resume_temp_file and os.path.exists(resume_temp_file):
                        os.remove(resume_temp_file)

            # Sort results by total score in descending order
            results.sort(key=lambda x: x.get("total_score", 0), reverse=True)

            # Return results
            return {
                "results": results,
                "jd_title": jd_data.get("job_title", "Unknown Job"),
                "resume_count": len(results),
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }

        finally:
            # Clean up JD temp file
            if jd_temp_file and os.path.exists(jd_temp_file):
                os.remove(jd_temp_file)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in batch evaluation: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An error occurred during batch evaluation: {str(e)}")

@app.post("/interfix", response_model=InterfixResponse, summary="Extract key information from call summary", description="Extract structured data from a VAPI call summary or transcript")
async def interfix_endpoint(request: InterfixRequest):
    """Extract key information from a VAPI call summary or transcript.

    - **summary**: The summary or transcript text from an AI call agent

    Returns structured data with the following fields:
    - offer_in_hand: Current salary/offer amount (number)
    - notice_period: Notice period duration (string)
    - expected_salary: Expected salary amount (number)
    - reason_to_switch: Primary reason for job change (string)
    - preferred_time_for_interview: Preferred interview time (string)
    - preferred_date_for_interview: Preferred interview date (string)
    """
    try:
        # Start timing
        start_time = time.time()

        # Get the summary text
        summary_text = request.summary

        # Generate a prompt for the AI model
        prompt = f"""You are an expert at extracting structured information from text.
Extract the following key information from this call summary/transcript:

1. offer_in_hand: The candidate's current salary/offer amount (as a number, convert to numeric value)
2. notice_period: The candidate's notice period duration (as a string)
3. expected_salary: The candidate's expected salary amount (as a number, convert to numeric value)
4. reason_to_switch: The candidate's primary reason for changing jobs (as a string)
5. preferred_time_for_interview: The candidate's preferred time for interview (as a string)
6. preferred_date_for_interview: The candidate's preferred date for interview (as a string)

For salary values:
- Convert text like "1 lakh" to 100000
- Convert text like "10 lakhs" to 1000000
- Convert text like "1 crore" to 10000000

If any information is not available in the text, return null for that field.
Return ONLY a valid JSON object with these fields, nothing else.

Call Summary/Transcript:
{summary_text}
"""

        # Generate response using the AI model
        response = get_response(prompt, timeout_seconds=60)

        # Parse the JSON response
        # Find JSON object in the response (it might be surrounded by markdown code blocks or other text)
        json_match = re.search(r'```(?:json)?\s*({[\s\S]*?})\s*```|({[\s\S]*?})', response)
        if json_match:
            json_str = json_match.group(1) or json_match.group(2)
            try:
                extracted_data = json.loads(json_str)

                # Create response object
                result = InterfixResponse(
                    offer_in_hand=extracted_data.get("offer_in_hand"),
                    notice_period=extracted_data.get("notice_period"),
                    expected_salary=extracted_data.get("expected_salary"),
                    reason_to_switch=extracted_data.get("reason_to_switch"),
                    preferred_time_for_interview=extracted_data.get("preferred_time_for_interview"),
                    preferred_date_for_interview=extracted_data.get("preferred_date_for_interview")
                )

                # Log metrics
                end_time = time.time()

                # Create a metrics object for this request
                metrics = RequestMetrics(endpoint="/interfix")

                # Log model metrics
                log_model_metrics(
                    request_metrics=metrics,
                    model_name=MODEL_NAME,
                    prompt_length=len(prompt),
                    response_length=len(response),
                    processing_time=end_time - start_time
                )

                logger.info(f"Successfully extracted information from call summary")
                return result
            except json.JSONDecodeError:
                logger.error("Failed to parse JSON from AI response")
                raise HTTPException(status_code=500, detail="Failed to parse structured data from the model response")
        else:
            logger.error("No JSON found in AI response")
            raise HTTPException(status_code=500, detail="Failed to extract structured data from the model response")

    except HTTPException as e:
        # Re-raise HTTP exceptions
        raise e
    except Exception as e:
        logger.error(f"Error extracting information from call summary: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An error occurred during extraction: {str(e)}")

@app.post("/intervet2", response_model=Dict, summary="Evaluate candidate-job fit from files", description="Upload resume and job description files to evaluate candidate-job fit")
async def evaluate_candidate_job_fit_from_files_endpoint(
    request: Request,
    resume_file: UploadFile = File(..., description="Resume file to parse (PDF or DOCX format)"),
    jd_file: UploadFile = File(..., description="Job description file to parse (PDF or DOCX format)")
):
    """Evaluate how well a candidate's resume matches a job description, taking both as file uploads.

    - **resume_file**: Upload a PDF or DOCX file containing a resume
    - **jd_file**: Upload a PDF or DOCX file containing a job description

    Returns a comprehensive evaluation with:
    - Overall match score (0-100)
    - Fit category (Excellent/Strong/Good/Moderate/Weak Match)
    - Summary of the match
    - Detailed scores for each evaluation criterion
    - Detailed rationale for each score

    The evaluation is based on multiple criteria:
    1. Skills matching (direct and from projects/experience)
    2. Years of experience
    3. Reliability (experience to job ratio)
    4. Location match
    5. Academic qualifications
    6. Alma mater prestige
    7. Relevant certifications

    Note: This endpoint combines the functionality of /resume, /jd_parser, and /intervet into a single endpoint.
    """
    try:
        logger.info("Starting candidate-job fit evaluation from files")

        # Get metrics tracker from request state
        metrics = getattr(request.state, "metrics", None)
        if metrics:
            metrics.add_metric("endpoint_type", "intervet2")

        # Process resume file
        resume_temp_file = None
        jd_temp_file = None

        try:
            # Process resume file
            resume_file_extension = os.path.splitext(resume_file.filename.lower())[1]
            if resume_file_extension == '.pdf':
                resume_file_type = "pdf"
                resume_suffix = '.pdf'
            elif resume_file_extension == '.docx':
                resume_file_type = "docx"
                resume_suffix = '.docx'
            else:
                logger.warning(f"Unsupported resume file format: {resume_file.filename}")
                raise HTTPException(status_code=400, detail="Only PDF and DOCX files are supported for resume.")

            logger.info(f"Processing resume file: {resume_file.filename} (type: {resume_file_type})")

            # Create a temporary file for the resume
            with tempfile.NamedTemporaryFile(delete=False, suffix=resume_suffix) as temp_file:
                # Write the uploaded file content to the temporary file
                resume_content = await resume_file.read()
                if not resume_content:
                    logger.error("Uploaded resume file is empty")
                    raise HTTPException(status_code=400, detail="The uploaded resume file is empty.")

                temp_file.write(resume_content)
                resume_temp_file = temp_file.name
                logger.info(f"Saved uploaded resume file to temporary location: {resume_temp_file}")

            # Process job description file
            jd_file_extension = os.path.splitext(jd_file.filename.lower())[1]
            if jd_file_extension == '.pdf':
                jd_file_type = "pdf"
                jd_suffix = '.pdf'
            elif jd_file_extension == '.docx':
                jd_file_type = "docx"
                jd_suffix = '.docx'
            else:
                logger.warning(f"Unsupported job description file format: {jd_file.filename}")
                raise HTTPException(status_code=400, detail="Only PDF and DOCX files are supported for job description.")

            logger.info(f"Processing job description file: {jd_file.filename} (type: {jd_file_type})")

            # Create a temporary file for the job description
            with tempfile.NamedTemporaryFile(delete=False, suffix=jd_suffix) as temp_file:
                # Write the uploaded file content to the temporary file
                jd_content = await jd_file.read()
                if not jd_content:
                    logger.error("Uploaded job description file is empty")
                    raise HTTPException(status_code=400, detail="The uploaded job description file is empty.")

                temp_file.write(jd_content)
                jd_temp_file = temp_file.name
                logger.info(f"Saved uploaded job description file to temporary location: {jd_temp_file}")

            # Extract text from both files
            start_resume_extraction_time = time.time()
            resume_text = extract_text_from_file(resume_temp_file, resume_file_type, request_metrics=metrics)
            resume_extraction_time = time.time() - start_resume_extraction_time
            logger.info(f"Extracted {len(resume_text)} characters from resume file in {resume_extraction_time:.2f} seconds")

            start_jd_extraction_time = time.time()
            jd_text = extract_text_from_file(jd_temp_file, jd_file_type, request_metrics=metrics)
            jd_extraction_time = time.time() - start_jd_extraction_time
            logger.info(f"Extracted {len(jd_text)} characters from job description file in {jd_extraction_time:.2f} seconds")

            # Log a preview of the extracted JD text for debugging
            jd_text_preview = jd_text[:300] + "..." if len(jd_text) > 300 else jd_text
            logger.info(f"JD text preview: {jd_text_preview}")

            if metrics:
                metrics.add_metric("resume_extraction_time", resume_extraction_time)
                metrics.add_metric("jd_extraction_time", jd_extraction_time)
                metrics.add_metric("resume_text_length", len(resume_text))
                metrics.add_metric("jd_text_length", len(jd_text))

            # Parse resume and job description
            start_resume_parsing_time = time.time()
            resume_data = parse_resume(resume_text, convert_skills_to_dict_format=True)
            resume_parsing_time = time.time() - start_resume_parsing_time
            logger.info(f"Parsed resume in {resume_parsing_time:.2f} seconds")

            # Log resume parsing results
            if "skills" in resume_data:
                if isinstance(resume_data["skills"], dict):
                    skills_count = len(resume_data["skills"])
                    logger.info(f"Extracted {skills_count} skills from resume")
                    if skills_count > 0:
                        skills_preview = ", ".join(list(resume_data["skills"].keys())[:5])
                        if skills_count > 5:
                            skills_preview += f", ... ({skills_count-5} more)"
                        logger.info(f"Resume skills preview: {skills_preview}")
                elif isinstance(resume_data["skills"], list):
                    skills_count = len(resume_data["skills"])
                    logger.info(f"Extracted {skills_count} skills from resume")

            start_jd_parsing_time = time.time()
            jd_data = parse_jd(jd_text)
            jd_parsing_time = time.time() - start_jd_parsing_time
            logger.info(f"Parsed job description in {jd_parsing_time:.2f} seconds")

            # Log JD parsing results
            if "required_skills" in jd_data:
                skills_count = len(jd_data["required_skills"])
                logger.info(f"Extracted {skills_count} required skills from JD")
                if skills_count > 0:
                    skills_preview = ", ".join(jd_data["required_skills"][:5])
                    if skills_count > 5:
                        skills_preview += f", ... ({skills_count-5} more)"
                    logger.info(f"JD required skills preview: {skills_preview}")

            if "education_requirements" in jd_data:
                edu_count = len(jd_data["education_requirements"])
                logger.info(f"Extracted {edu_count} education requirements from JD")
                if edu_count > 0:
                    logger.info(f"JD education requirements: {jd_data['education_requirements']}")

            if "education_details" in jd_data and isinstance(jd_data["education_details"], dict):
                logger.info(f"JD education details: {jd_data['education_details']}")

            if metrics:
                metrics.add_metric("resume_parsing_time", resume_parsing_time)
                metrics.add_metric("jd_parsing_time", jd_parsing_time)

            # Calculate the match score
            start_evaluation_time = time.time()
            result = calculate_candidate_job_fit(resume_data, jd_data)
            evaluation_time = time.time() - start_evaluation_time

            if metrics:
                metrics.add_metric("evaluation_time", evaluation_time)
                metrics.add_metric("total_score", result.get("total_score", 0))
                metrics.add_metric("fit_category", result.get("fit_category", "Unknown"))

            return result

        finally:
            # Clean up the temporary files
            if resume_temp_file and os.path.exists(resume_temp_file):
                logger.info(f"Removing temporary resume file: {resume_temp_file}")
                os.remove(resume_temp_file)

            if jd_temp_file and os.path.exists(jd_temp_file):
                logger.info(f"Removing temporary job description file: {jd_temp_file}")
                os.remove(jd_temp_file)

    except HTTPException as e:
        # Re-raise HTTP exceptions
        raise e
    except Exception as e:
        logger.error(f"Error in candidate-job fit evaluation from files: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An error occurred during evaluation: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
