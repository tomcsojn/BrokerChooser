FROM python:3.9-slim

WORKDIR /app

# Copy the requirements file into the container
COPY transform-requirements.txt .

# Install required packages
RUN pip install --no-cache-dir -r ./transform-requirements.txt

# Copy the script into the container
COPY transformer-translate.py .

# Default command
ENTRYPOINT ["python", "transformer-translate.py"]
CMD ["--help"]