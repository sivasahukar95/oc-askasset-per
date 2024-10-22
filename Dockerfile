# Use the official Python 3.10 slim-buster image as the base image
FROM python:3.10-slim-buster

# Set environment variables to prevent Python from writing .pyc files to disk
# and to prevent buffering of the stdout and stderr streams
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Set the working directory inside the container to /app
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip to the latest version
RUN pip install --upgrade pip

# Copy the requirements.txt file from the local machine to the container
COPY requirements.txt .

# Install the required dependencies listed in requirements.txt, using --no-cache-dir
RUN pip install --no-cache-dir -r requirements.txt --trusted-host pypi.org --trusted-host pypi.python.org --trusted-host files.pythonhosted.org

# Copy the rest of the application code from the local machine to the container
COPY . .

# Ensure necessary directories are present and set proper permissions
RUN mkdir -p /.cache \
    && chmod -R 777 /app \
    && chmod -R 777 /.cache

# Expose port 8000 to allow external access to the FastAPI application
EXPOSE 8000

# Command to run the FastAPI application using Uvicorn, binding it to all network interfaces (0.0.0.0)
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
