# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /code

# Copy the dependencies file to the working directory
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
# We also install ffmpeg which is a good practice for opencv projects
RUN apt-get update && apt-get install -y ffmpeg libsm6 libxext6 \
    && pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code to the working directory
COPY ./templates ./templates
COPY main.py .

# Expose the port the app runs on
EXPOSE 8080

# Define the command to run your app.
# We use 0.0.0.0 to bind to all network interfaces and port 8080.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]