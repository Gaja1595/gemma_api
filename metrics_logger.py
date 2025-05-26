import json
import logging
import os
import time
import uuid
from datetime import datetime
from typing import Dict, Any, Optional

# Configure metrics logger
metrics_logger = logging.getLogger("gemma-api-metrics")
metrics_logger.setLevel(logging.INFO)

# Create logs directory if it doesn't exist
try:
    # Try to create the logs directory with proper permissions
    logs_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
    os.makedirs(logs_dir, exist_ok=True)

    # Create a unique session ID for this run
    SESSION_ID = str(uuid.uuid4())
    SESSION_START_TIME = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # Create a file handler for the metrics log
    log_file_path = os.path.join(logs_dir, f"metrics_{SESSION_START_TIME}.log")
    metrics_file_handler = logging.FileHandler(log_file_path)
    metrics_file_handler.setLevel(logging.INFO)

    # Create a formatter for the metrics log
    metrics_formatter = logging.Formatter('%(message)s')
    metrics_file_handler.setFormatter(metrics_formatter)

    # Add the handler to the logger
    metrics_logger.addHandler(metrics_file_handler)

    # Log successful initialization
    print(f"Metrics logger initialized. Writing to {log_file_path}")
except Exception as e:
    # Fall back to console logging if file logging fails
    print(f"Warning: Could not set up file logging for metrics: {e}")
    print("Falling back to console logging for metrics")

    # Create a console handler for metrics
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('METRICS: %(message)s')
    console_handler.setFormatter(console_formatter)

    # Add the console handler to the logger
    metrics_logger.addHandler(console_handler)

    # Still create a session ID
    SESSION_ID = str(uuid.uuid4())
    SESSION_START_TIME = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

# Log session start
metrics_logger.info(json.dumps({
    "event": "session_start",
    "session_id": SESSION_ID,
    "timestamp": datetime.now().isoformat(),
    "message": "New API session started"
}))

class RequestMetrics:
    """Class to track metrics for a single request."""

    def __init__(self, endpoint: str, request_id: Optional[str] = None):
        self.endpoint = endpoint
        self.request_id = request_id or str(uuid.uuid4())
        self.start_time = time.time()
        self.first_byte_time = None
        self.end_time = None
        self.input_tokens = 0
        self.output_tokens = 0
        self.status_code = None
        self.error = None
        self.additional_metrics = {}

        # Log request start
        self._log_event("request_start", {
            "message": f"Request started for endpoint: {endpoint}"
        })

    def mark_first_byte(self):
        """Mark the time when the first byte of the response is sent."""
        self.first_byte_time = time.time()
        ttfb = self.first_byte_time - self.start_time

        self._log_event("first_byte", {
            "ttfb_seconds": ttfb,
            "message": f"Time to first byte: {ttfb:.4f}s"
        })

        return ttfb

    def mark_complete(self, status_code: int, error: Optional[str] = None):
        """Mark the request as complete."""
        self.end_time = time.time()
        self.status_code = status_code
        self.error = error

        total_time = self.end_time - self.start_time

        metrics = {
            "total_time_seconds": total_time,
            "status_code": status_code,
            "message": f"Request completed in {total_time:.4f}s with status {status_code}"
        }

        if error:
            metrics["error"] = error

        if self.first_byte_time:
            metrics["processing_time_seconds"] = self.end_time - self.first_byte_time

        self._log_event("request_complete", metrics)

        return total_time

    def set_token_counts(self, input_tokens: int, output_tokens: int):
        """Set the token counts for the request."""
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens

        self._log_event("token_usage", {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            "message": f"Token usage: {input_tokens} input, {output_tokens} output"
        })

    def add_metric(self, key: str, value: Any):
        """Add an additional metric to track."""
        self.additional_metrics[key] = value

        self._log_event("custom_metric", {
            key: value,
            "message": f"Custom metric: {key}={value}"
        })

    def _log_event(self, event_type: str, data: Dict[str, Any]):
        """Log an event with the given data."""
        log_data = {
            "event": event_type,
            "session_id": SESSION_ID,
            "request_id": self.request_id,
            "endpoint": self.endpoint,
            "timestamp": datetime.now().isoformat(),
            **data
        }

        metrics_logger.info(json.dumps(log_data))

def log_model_metrics(request_metrics: RequestMetrics, model_name: str, prompt_length: int,
                     response_length: int, processing_time: float, error: Optional[str] = None):
    """Log metrics specific to model generation."""

    # Estimate token counts (rough approximation)
    input_tokens = prompt_length // 4  # Rough estimate: ~4 chars per token
    output_tokens = response_length // 4

    # Set token counts in the request metrics
    request_metrics.set_token_counts(input_tokens, output_tokens)

    # Add model-specific metrics
    request_metrics.add_metric("model_name", model_name)
    request_metrics.add_metric("prompt_length", prompt_length)
    request_metrics.add_metric("response_length", response_length)
    request_metrics.add_metric("model_processing_time", processing_time)

    if error:
        request_metrics.add_metric("model_error", error)

def log_file_processing_metrics(request_metrics: RequestMetrics, file_type: str, file_size: int,
                              extraction_method: str, text_length: int, processing_time: float):
    """Log metrics specific to file processing."""

    request_metrics.add_metric("file_type", file_type)
    request_metrics.add_metric("file_size_bytes", file_size)
    request_metrics.add_metric("extraction_method", extraction_method)
    request_metrics.add_metric("extracted_text_length", text_length)
    request_metrics.add_metric("file_processing_time", processing_time)

def shutdown_metrics_logger():
    """Log session end and close the metrics logger."""
    try:
        metrics_logger.info(json.dumps({
            "event": "session_end",
            "session_id": SESSION_ID,
            "timestamp": datetime.now().isoformat(),
            "message": "API session ended"
        }))

        # Close handlers
        for handler in list(metrics_logger.handlers):  # Create a copy of the list to avoid modification during iteration
            try:
                handler.close()
                metrics_logger.removeHandler(handler)
            except Exception as e:
                print(f"Error closing log handler: {e}")
    except Exception as e:
        print(f"Error during metrics logger shutdown: {e}")
