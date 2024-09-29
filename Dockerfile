# Use an official Python runtime as a parent image
FROM python:3.9.13-slim

# Set the working directory in the container
WORKDIR /usr/src/app

# Copy the requirements file into the container
COPY requirement.txt ./

# Install build dependencies
RUN apt-get update && apt-get install -y build-essential wget

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirement.txt

# Copy the current directory contents into the container at /usr/src/app
COPY . .

# Make port 8505 available to the world outside this container
EXPOSE 8505

# Define environment variable
ENV PORT 8505

# Run app.py when the container launches
CMD ["streamlit", "run", "app.py", "--server.port=8505"]

