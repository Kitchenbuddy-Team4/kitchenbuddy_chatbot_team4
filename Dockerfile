# Use an official lightweight Python image
FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all source code
COPY . .

# Expose the port that Flask/Gunicorn will run on
EXPOSE 8080

# Start Flask app using gunicorn
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:8080"]
