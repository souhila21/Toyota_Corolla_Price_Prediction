# Use a lightweight Python image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy and install dependencies
COPY requirements.txt .

# Pin scikit-learn to the trained version for pickle compatibility
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir scikit-learn==1.6.1

# Copy project files into the container
COPY . .

# Expose FastAPI (8000) and Streamlit (8501) ports
EXPOSE 8000 8501

# Run FastAPI backend and Streamlit frontend together
CMD bash -c "uvicorn app:app --host 0.0.0.0 --port 8000 & streamlit run streamlit_app.py --server.address 0.0.0.0 --server.port 8501"
