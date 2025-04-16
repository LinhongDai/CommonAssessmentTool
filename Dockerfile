# Use Python 3.11 image as base
FROM python:3.11

# Set working directory
WORKDIR /code

# Copy requirements first
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy entire app and rebuild script after installing dependencies
COPY . .

# Rebuild model inside Docker
RUN python scripts/rebuild_models.py

# Expose port
EXPOSE 8000

# Run the backend
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]