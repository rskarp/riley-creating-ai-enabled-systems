# Visual Search System

This project contains all of the code components necessary to run a visual search system.

## Instructions

## Instructions to run the system pipeline locally

Here are step-by-step instructions to run the pipeline module locally:

1. **Clone the Repository:**

   Clone this [repository](https://github.com/creating-ai-enabled-systems-summer-2024/karp-riley/tree/main).

   ```sh
   git clone https://github.com/creating-ai-enabled-systems-summer-2024/karp-riley.git
   cd karp-riley/visual_search_system
   ```

2. **Install dependencies from the Requirements File:**

   Install all necessary Python libraries:

   ```sh
   pip install -r requirements.txt
   ```

3. **Run search pipeline:**

   Run search pipeline:

   ```sh
   python pipeline.py
   ```

## Instructions to Run the pipeline module with Docker:

Here are step-by-step instructions to run the pipeline module with Docker:

1. **Install Docker:**

   Make sure Docker is installed on your machine. You can download and install Docker from the [official Docker website](https://www.docker.com/products/docker-desktop).

2. **Build the Docker Image:**

   Build the Docker image from the Dockerfile:

   ```sh
   docker build -t visual_search_system:latest .
   ```

3. **Run the Docker Container:**

   Run the Docker container with the built image:

   ```sh
   docker run -it --rm -p 8000:8000 visual_search_system
   ```
