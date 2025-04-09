FROM tensorflow/tensorflow:2.10.0

WORKDIR /app

#necessary dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

#Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy model and application code
COPY hurricane_damage_model.h5 .
COPY app.py .

# Set environment variables
ENV MODEL_PATH=hurricane_damage_model.h5
ENV PORT=5000

# Expose port
EXPOSE 5000

# Start application
CMD ["python", "app.py"]