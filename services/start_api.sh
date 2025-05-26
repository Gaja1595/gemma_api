#!/bin/bash

# This script ensures the Gemma API starts properly
# It checks for dependencies, creates necessary directories, and sets proper permissions

# Function to log messages
log_message() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

# Check if we're running as root
if [ "$EUID" -eq 0 ]; then
    log_message "Please run this script as the ubuntu user, not as root"
    exit 1
fi

# Set the working directory to the script's location
cd "$(dirname "$0")"
WORKING_DIR="$(pwd)"
log_message "Working directory: $WORKING_DIR"

# Create logs directory with proper permissions
log_message "Creating logs directory..."
mkdir -p "$WORKING_DIR/logs"
chmod 755 "$WORKING_DIR/logs"

# Check if virtual environment exists
if [ ! -d "$WORKING_DIR/venv" ]; then
    log_message "Virtual environment not found. Creating..."
    python3 -m venv venv
    log_message "Activating virtual environment..."
    source venv/bin/activate
    log_message "Installing dependencies..."
    pip install --upgrade pip
    pip install -r requirements.txt
else
    log_message "Activating existing virtual environment..."
    source venv/bin/activate
fi

# Check if Ollama is running
log_message "Checking if Ollama is running..."
if ! systemctl is-active --quiet ollama; then
    log_message "Ollama is not running. Starting Ollama..."
    sudo systemctl start ollama
    sleep 5  # Give Ollama time to start
fi

# Check if Gemma model is available
log_message "Checking if Gemma model is available..."
if ! ollama list | grep -q "gemma3:4b"; then
    log_message "Gemma model not found. Pulling model..."
    ollama pull gemma3:4b
fi

# Start the API
log_message "Starting Gemma API..."
python main.py
