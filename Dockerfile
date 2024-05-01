# Seting up Python runtime environment
FROM python:3.8-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY /. /app/

# Install the dependencies
RUN pip install -r requirements.txt

# Make port 8080 available to the world outside this container
EXPOSE 8080

# Define environment variable
ENV NAME World

# Run the command
CMD ["python", "model.py"]

