FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy project files
COPY requirement.txt .
RUN pip install --no-cache-dir -r requirement.txt

COPY . .

# Expose FastAPI port
EXPOSE 8000

# Run API with Uvicorn
CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]
