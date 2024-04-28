# Use an official Python runtime as a parent image
FROM python:3.9

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install -r requirements.txt

# Expose the port the app runs on
EXPOSE 8080

# Run app.py when the container launches
CMD ["python", "app.py", "--host", "0.0.0.0", "--port", "8080", "--debug", "False"]