# Object Detection System

This project contains all of the code components necessary to run an object detection (classification and localization) system.

# Prerequisites

- Docker installed on your machine
- Basic knowledge of Docker and Flask
- (Optional) Postman installed on your machine and basic knowledge of Postman

# Contents

These are the files you can find in this repository.

- **[main.py](main.py)**: The main application script that starts the Flask server and defines the API routes.
- **[deployment.py](deployment.py)**: Contains the `Deployment` class, which handles the deployment logic for ML model and system.
- **[pipeline.py](pipeline.py)**: Contains the `InferenceService` class, which handles the inference pipeline.
- **[requirements.txt](requirements.txt)**: Lists the Python dependencies required for the project.
- **[Dockerfile](Dockerfile)**: Defines the Docker image configuration for the project.
- **[analysis/data_analysis.ipynb](analysis/data_analysis.ipynb)**: Contains exploratory data analysis plots and explanation of findings.
- **[analysis/design_considerations.ipynb](analysis/design_considerations.ipynb)**: Contains analysis of different models, parameters, and their performance.
- **[analysis/systems_report.ipynb](analysis/systems_report.ipynb)**: Contains analysis of entire system and project.
- **[src/metrics.py](src/metrics.py)**: Contains the `Metrics` and `Loss` classes, which are used to calculate metrics about the models.
- **[src/inference/non_maximal_suppression.py](src/inference/non_maximal_suppression.py)**: Contains the `NMS` class, which is used to perform Non-Maximal Suppression (NMS) on bounding boxes
- **[src/inference/object_detection.py](src/inference/object_detection.py)**: Contains the `YOLOObjectDetector` class, which is used to perform object detection using YOLO (You Only Look Once)
- **[src/inference/video_processing.py](src/inference/video_processing.py)**: Contains the `VideoProcessing` class, which is used to process video streams from a UDP source.
- **[src/rectification/hard_negative_mining.py](src/rectification/hard_negative_mining.py)**: Contains the `HardNegativeMiner` class, which is used to determine the hardest negative samples for the model to detect.
- **[assets/images/](assets/images/)**: Contains images used in [analysis/systems_report.ipynb](analysis/systems_report.ipynb).
- **[assets/EN.705.603.82 Case Study 7.postman_collection.json](assets/EN.705.603.82%20Case%20Study%207.postman_collection.json)**: Postman collection containing example requests for all endpoints.
- **[storages/training/](storages/training/)**: Contains images and annotation files from `logistics.zip` used in [analysis/design_considerations.ipynb](analysis/design_considerations.ipynb).
- **[storages/prediction/](storages/prediction/)**: Contains images and annotation files created from detections captured by running the Flask application and streaming video.
- **[yolo_resources/models/](yolo_resources/models/)**: Contains trained model weights for two trained YOLO detection models.
- **[yolo_resources/test_images/test_images.jpg](yolo_resources/test_images/test_images.jpg)**: An example image for testing the model.
- **[yolo_resources/test_videos/](yolo_resources/test_videos/)**: Contains example videos to stream to the system.
- **[yolo_resources/logistics.names](yolo_resources/logistics.names)**: Contains the names of the object classes.

## Instructions

## Instructions to Run the Flask application locally

Here are step-by-step instructions to run the Flask application locally:

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

3. **Run Flask application:**

   Start the application:

   ```sh
   python main.py
   ```

4. **Stream video to port 23000:**

   Stream video to localhost port 23000:

   ```sh
   ffmpeg -re -i ./yolo_resources/test_videos/worker-zone-detection.mp4 -r 30 -vcodec mpeg4 -f mpegts udp://127.0.0.1:23000
   ```

5. **Access the Flask Application:**

   Open your web browser and navigate to:

   ```
   http://localhost:8000
   ```

   You should see the "Welcome to Riley's Object Detection System!" message from the index route. Your flask server is now properly running!

6. **Using the API Endpoints:**

   **Supported Endpoints**

   - **GET /**: Returns a welcome message.
   - **GET /detections_list**: Get the list of detections identified by the system within the given frame range. Requires `start_frame` and `end_frame` parameters.
   - **GET /hard_negatives**: Get the list of top N hard negatives. Requires a `N` parameter.
   - **GET /predictions**: Get the image files with detections and associated prediction files. Requires `start_frame` and `end_frame` parameters.

   **Example Requests**
   Examples of using all of the API endpoints can be seen in the Postman collection saved to [assets/EN.705.603.82 Case Study 7.postman_collection.json](./assets/EN.705.603.82%20Case%20Study%207.postman_collection.json). You can download the collection and import into Postman to make the requests. Some very basic curl requests are also demonstrated below.

   - **Get Detections:**

     ```sh
     curl -X GET "http://localhost:8000/detections_list?start_frame=<START_FRAME>&end_frame=<END_FRAME>"
     ```

   - **Get Hard Negatives:**

     ```sh
     curl -X GET 'http://localhost:8000/hard_negatives?N=<N>'
     ```

   - **Get Predictions:**

     ```sh
     curl -X GET "http://localhost:8000/predictions?start_frame=<START_FRAME>&end_frame=<END_FRAME>"
     ```

## Instructions to Run the Flask application with Docker:

Here are step-by-step instructions to run the Flask application with Docker:

1. **Install Docker:**

   Make sure Docker is installed on your machine. You can download and install Docker from the [official Docker website](https://www.docker.com/products/docker-desktop).

2. **Build the Docker Image:**

   Build the Docker image from the Dockerfile:

   ```sh
   docker build -t object_detection_system:latest .
   ```

3. **Run the Docker Container:**

   Run the Docker container with the built image (video stream on port 23000 and Flask application on port 8000):

   ```sh
   docker run -it --rm -p 23000:23000/udp -p 8000:8000 object_detection_system
   ```

4. **Stream video to port 23000:**

   Stream video to localhost port 23000:

   ```sh
   ffmpeg -re -i ./yolo_resources/test_videos/worker-zone-detection.mp4 -r 30 -vcodec mpeg4 -f mpegts udp://127.0.0.1:23000
   ```
