FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements-web.txt .
RUN pip install --no-cache-dir -r requirements-web.txt

# Copy python_layer since headless_sim imports fault_injector from there
COPY python_layer/ ./python_layer/

# Copy the web application files
COPY web_app/ ./web_app/

# Expose port
EXPOSE 8000

# Run the FastAPI server via uvicorn
CMD ["uvicorn", "web_app.server:app", "--host", "0.0.0.0", "--port", "8000"]
