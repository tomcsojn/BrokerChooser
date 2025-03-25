FROM python:3.9-slim

WORKDIR /app

# Copy the requirements file into the container
COPY deep-requirements.txt .

# Install required packages
RUN pip install --no-cache-dir -r ./deep-requirements.txt

# Copy the script into the container
COPY deep-translate.py .

# Default command
ENTRYPOINT ["python", "deep-translate.py"]
CMD ["--help"]