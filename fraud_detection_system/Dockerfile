# Use the official Ubuntu 22.04 base image
FROM ubuntu:22.04
ARG DEBIAN_FRONTEND=noninteractive

# Update the package lists and install Python and Pip
RUN apt-get update && apt-get install -y python3.9 python3-pip

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip3 install -r requirements.txt

# Make port 8000 available to the world outside this container
EXPOSE 8000

# Run app.py when the container launches
CMD ["python3", "main.py"]