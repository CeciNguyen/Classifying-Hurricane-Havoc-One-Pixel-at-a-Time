FROM tensorflow/tensorflow:2.13.0

WORKDIR /app

# Install necessary dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY hurricane_damage_model.keras .
COPY app.py .

# Set environment variables
ENV MODEL_PATH=hurricane_damage_model.keras
ENV PORT=5000

EXPOSE 5000

CMD ["python", "app.py"]
