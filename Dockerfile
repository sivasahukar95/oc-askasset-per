# Use the official Python 3.11 slim image from DockerHub as the base image
FROM python:3.11-slim

# Set the working directory inside the container to /app
WORKDIR /app

# Copy the requirements.txt file from the local machine to the container
COPY requirements.txt .

# Install the required dependencies listed in requirements.txt, 
# using --no-cache-dir to avoid caching and keep the image size small
RUN pip install --no-cache-dir -r requirements.txt

# Copy the .env file from the local machine to the container
# COPY .env .

# Copy the rest of the application code from the local machine to the container
COPY . .


WORKDIR /app/app

RUN chmod -R 777 /app/app
RUN mkdir -p /.cache
RUN chmod -R 777 /.cache

# Expose port 8000 to allow external access to the FastAPI application
EXPOSE 8000

# Command to run the FastAPI application using Uvicorn, 
# binding it to all network interfaces (0.0.0.0) on port 8000 with auto-reload enabled
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]