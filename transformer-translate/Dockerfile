FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

WORKDIR /app

# Copy the requirements file
COPY transform-requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r transform-requirements.txt

# Copy the translation script
COPY transformer-translate.py .

# Default command
ENTRYPOINT ["python", "transformer-translate.py"]
CMD ["--help"]