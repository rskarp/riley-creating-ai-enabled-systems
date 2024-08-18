# Retrieval Augmented Generation

This project contains all of the code components necessary to run a retrieval augmented generation system.

## Instructions

## Instructions to run the Flask application locally

Here are step-by-step instructions to run the Flask application locally:

1. **Clone the Repository:**

   Clone this [repository](https://github.com/creating-ai-enabled-systems-summer-2024/karp-riley/tree/main).

   ```sh
   git clone https://github.com/creating-ai-enabled-systems-summer-2024/karp-riley.git
   cd karp-riley/retrieval_augmented_generation
   ```

2. **Install dependencies from the Requirements File:**

   Install all necessary Python libraries:

   ```sh
   pip install -r requirements.txt
   ```

3. **Run Flask application:**

   Start the application:

   ```sh
   python main.py
   ```

## Instructions to Run the Flask application with Docker:

Here are step-by-step instructions to run the Flask application with Docker:

1. **Install Docker:**

   Make sure Docker is installed on your machine. You can download and install Docker from the [official Docker website](https://www.docker.com/products/docker-desktop).

2. **Build the Docker Image:**

   Build the Docker image from the Dockerfile:

   ```sh
   docker build -t retrieval_augmented_generation:latest .
   ```

3. **Run the Docker Container:**

   Run the Docker container with the built image (Flask application on port 8000):

   ```sh
   docker run -it --rm -p 8000:8000 retrieval_augmented_generation
   ```
