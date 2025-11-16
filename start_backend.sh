#!/bin/bash
# Start the Flask backend server

echo "Starting BadgerBuild Backend Server..."
echo "Make sure you're in the virtual environment!"

# Activate virtual environment if it exists
if [ -d ".venv" ]; then
    source .venv/bin/activate
    echo "Virtual environment activated"
fi

# Check if Flask is installed
python -c "import flask" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "Flask not found. Installing dependencies..."
    pip install -r requirements.txt
fi

# Start the server
echo "Starting Flask server on http://localhost:5000"
python app.py

