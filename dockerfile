# Use an official Python image as the base
FROM python:3.13.7-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the NEW, smaller requirements file
COPY requirements.txt .

# Install ONLY the necessary Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy only the files needed to RUN the server
COPY main.py .
COPY index.html .
COPY static/ ./static/

# --- NEW STEP ---
# Copy your pre-built database directly into the image.
# This will make the image VERY large.
COPY ./chroma_db /app/chroma_db
# --- END OF CHANGE ---

# Expose the port that your app will run on
EXPOSE 8000

# The command to run your application
CMD ["python3", "main.py"]