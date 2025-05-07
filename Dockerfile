# Use the official Python image from Docker Hub
FROM python:3.11-slim

# Set the working directory inside the container
WORKDIR /app

# Copy all files from the current directory to the /app directory in the container
COPY . /app/

# Install required Python packages
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Download the SpaCy model
RUN python -m spacy download en_core_web_sm

# Expose the port your app runs on
EXPOSE 5000

# Command to run the Flask app
CMD ["python", "app.py"]
