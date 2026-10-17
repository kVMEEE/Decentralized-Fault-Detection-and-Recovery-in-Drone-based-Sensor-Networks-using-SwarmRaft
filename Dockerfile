FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements-web.txt .
RUN pip install --no-cache-dir -r requirements-web.txt

# Copy the C++ native engine (pre-compiled Linux .so or Windows DLL)
# Note: the .dll is Windows-only; on Linux, swarmraft_native.py falls back to pure Python.
COPY cpp_mesh/ ./cpp_mesh/

# Copy python_layer (swarmraft_native.py, consensus.py, trajectory.py, fault_injector.py, RL model)
COPY python_layer/ ./python_layer/

# Copy the web application
COPY web_app/ ./web_app/

# Expose port
EXPOSE 8000

# Run the FastAPI server via uvicorn
CMD ["uvicorn", "web_app.server:app", "--host", "0.0.0.0", "--port", "8000"]
