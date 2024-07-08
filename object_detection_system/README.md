# Object Detection System

This project contains all of the code components necessary to run an object detection (classification and localization) system.

## Instructions

## Instructions to Run the inference module locally

Here are step-by-step instructions to run the inference module locally:

1. **Clone the Repository:**

   Clone this [repository](https://github.com/creating-ai-enabled-systems-summer-2024/karp-riley/tree/main).

   ```sh
   git clone https://github.com/creating-ai-enabled-systems-summer-2024/karp-riley.git
   cd karp-riley/object_detection_system
   ```

2. **Create the Requirements File:**

   Install all necessary Python libraries:

   ```sh
   pip install -r requirements.txt
   ```

3. **Run inference pipeline:**

   Install all necessary Python libraries:

   ```sh
   python pipeline.py
   ```

4. **Stream video to port 23000:**

   Stream video to localhost port 23000:

   ```sh
   ffmpeg -re -i ./yolo_resources/test_videos/worker-zone-detection.mp4 -r 30 -vcodec mpeg4 -f mpegts udp://127.0.0.1:23000
   ```

## Instructions to Run the inference module with Docker:

Here are step-by-step instructions to run the inference module with Docker:

1. **Install Docker:**

   Make sure Docker is installed on your machine. You can download and install Docker from the [official Docker website](https://www.docker.com/products/docker-desktop).

2. **Build the Docker Image:**

   Build the Docker image from the Dockerfile:

   ```sh
   docker build -t object_detection_system:latest .
   ```

3. **Run the Docker Container:**

   Run the Docker container with the built image:

   ```sh
   docker run -it --rm -p 23000:23000/udp object_detection_system
   ```

4. **Stream video to port 23000:**

   Stream video to localhost port 23000:

   ```sh
   ffmpeg -re -i ./yolo_resources/test_videos/worker-zone-detection.mp4 -r 30 -vcodec mpeg4 -f mpegts udp://127.0.0.1:23000
   ```
