# Fraud Detection System

This is a boiler plate code that you can use for your system project submission. You will need to modify this! Running the server will create a `resources` directory which will store logs and dataset files.

# Prerequisites

- Docker installed on your machine
- Basic knowledge of Docker and Flask (See Module 5 Lectures)

# Contents

These are the files you can find in this repository. They are intentionally incomplete.

- **`main.py`**: The main application script that starts the Flask server and defines the API routes.
- **`deployment_pipeline.py`**: Contains the `DeploymentPipeline` class, which handles the deployment logic for ML models.
- **`dataset.py`**: Contains the `DatasetConstructor` class, which handles dataset generation and loading.
- **`model.py`**: Contains the `Model` class, which defines the model architecture and prediction logic.
- **`metrics.py`**: Contains the `Metrics` class, which calculates evaluation metrics for the model.
- **`requirements.txt`**: Lists the Python dependencies required for the project.
- **`Dockerfile`**: Defines the Docker image configuration for the project.

# Instructions

### Instructions to Run the Flask Application locally

Here are step-by-step instructions to run the Flask application locally:

1. **Clone the Repository:**

   Clone this [repository](https://github.com/creating-ai-enabled-systems-summer-2024/karp-riley/tree/main).

   ```sh
   git clone https://github.com/creating-ai-enabled-systems-summer-2024/karp-riley.git
   cd karp-riley/fraud_detection_system
   ```

2. **Create the Requirements File:**

   Install all necessary Python libraries:

   ```sh
   pip install -r requirements.txt
   ```

3. **Run Flask application:**

   Install all necessary Python libraries:

   ```sh
   python main.py
   ```

4. **Access the Flask Application:**

   Open your web browser and navigate to:

   ```
   http://localhost:8000
   ```

   You should see the "Hello World!" message from the index route. Your flask server is now properly running!

5. **Using the API Endpoints:**

   **Supported Endpoints**

   - **GET /**: Returns a welcome message.
   - **PUT /generate_new_dataset**: Generates a new dataset. Requires a `version` parameter.
   - **GET /dataset_description**: Retrieves the description of a dataset. Requires a `version` parameter.
   - **POST /predict**: Makes a prediction based on input data.

   **Example Requests**

   - **Generate Dataset:**

     ```sh
     curl -X PUT "http://localhost:8000/generate_new_dataset?version=<DATASET_VERSION>"
     ```

     Note: Only implements `DATASET_VERSION` as a string.

   - **Get Dataset Description:**

     ```sh
     curl -X GET "http://localhost:8000/dataset_description?version=<DATASET_VERSION>"
     ```

     Note: Only implements random `DATASET_VERSION` (e.g., `3bcf57ff`). This retrieves the information in `logs/dataset/`

     - **Make a Prediction:**

     ```sh
     curl -X POST "http://localhost:8000/predict" -H "Content-Type: application/json" -d @<LOG_FILE>
     ```

     For example, you can run `curl -X POST "http://localhost:8000/predict" -H "Content-Type: application/json" -d @test_data.json`
     Note: Only implements `random` model

   Note: You will need to add more here...

   For more information regarding the difference between GET, PUT, and POST, see this [resource](https://blog.postman.com/what-are-http-methods/).

### Instructions to Run the Flask Application with Docker:

It is highly recommended that you develop locally before moving to docker. Here are step-by-step instructions to run the Flask application with Docker:

1. **Install Docker:**

   Make sure Docker is installed on your machine. You can download and install Docker from the [official Docker website](https://www.docker.com/products/docker-desktop).

2. **Build the Docker Image:**

   Build the Docker image from the Dockerfile:

   ```sh
   docker build -t fraud_detection_system:latest .
   ```

3. **Run the Docker Container:**

   Run the Docker container with the built image:

   ```sh
   docker run -it -v $(pwd)/resources:/app/resources -p 8000:8000 fraud_detection_system:latest
   ```
