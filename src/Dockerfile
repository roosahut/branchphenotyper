# Use an official Python runtime as a parent image
FROM tensorflow/tensorflow:latest

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY backend/requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY backend/ .

COPY models/ models/

# Make port 8000 available to the world outside this container
EXPOSE 8000


# Run uvicorn server when the container launches
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]