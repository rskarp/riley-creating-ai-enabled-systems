# Visual Search System

This project contains all of the code components necessary to run a visual search system.

# Prerequisites

- Docker installed on your machine
- Basic knowledge of Docker and Flask
- (Optional) Postman installed on your machine and basic knowledge of Postman

# Contents

These are the files you can find in this repository.

- **[main.py](main.py)**: The main application script that starts the Flask server and defines the API routes.
- **[deployment.py](deployment.py)**: Contains the `Deployment` class, which handles the deployment logic for ML model and system.
- **[pipeline.py](pipeline.py)**: Contains the `Pipeline` class, which handles the extraction and search pipeline.
- **[requirements.txt](requirements.txt)**: Lists the Python dependencies required for the project.
- **[Dockerfile](Dockerfile)**: Defines the Docker image configuration for the project.
- **[analysis/data_analysis.ipynb](analysis/data_analysis.ipynb)**: Contains exploratory data analysis plots and explanation of findings.
- **[analysis/design_considerations.ipynb](analysis/design_considerations.ipynb)**: Contains analysis of different models, parameters, and their performance.
- **[analysis/systems_report.ipynb](analysis/systems_report.ipynb)**: Contains analysis of entire system and project.
- **[src/metrics.py](src/metrics.py)**: Contains the `RankingMetrics` class, which is used to calculate metrics about the models.
- **[src/extraction/model.py](src/extraction/model.py)**: Contains the `Model` class and related classes necessary to create a SimCLR model, which is used to calculate embeddings.
- **[src/extraction/preprocess.py](src/extraction/preprocess.py)**: Contains the `Preprocessing` class, which is used to perform resizing and scaling transformations on the images.
- **[src/search/indexing.py](src/search/indexing.py)**: Contains the `KDTree` class, which is used to create a KD Tree search index.
- **[src/search/search.py](src/search/search.py)**: Contains the `KDTreeSearch` and `Measure` classes, which are used to run a nearest neighbors search using a specified distance measure.
- **[assets/images/](assets/images/)**: Contains images used in [analysis/systems_report.ipynb](analysis/systems_report.ipynb).
- **[assets/EN.705.603.82 Case Study 9.postman_collection.json](assets/EN.705.603.82%20Case%20Study%209.postman_collection.json)**: Postman collection containing example requests for all endpoints.
- **[storage/embeddings/](storage/embeddings/)**: Contains `.npy` files containing embedding vectors for each image in the gallery.
- **[storage/gallery/](storage/gallery/)**: Contains images from the original single-image gallery.
- **[storage/multi_image_gallery/](storage/multi_image_gallery/)**: Contains images from the multi-image gallery.
- **[storage/logs/](storage/logs/)**: Contains image and json files that are captured as logs upon authentication attempts.
- **[simclr_resources/](simclr_resources/)**: Contains trained model weights for four trained SimCLR models.
- **[simclr_resources/probe](simclr_resources/probe)**: Contains example images to be used as probes during testing.

## Instructions

## Instructions to run the Flask application locally

Here are step-by-step instructions to run the Flask application locally:

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

3. **Run Flask application:**

   Start the application:

   ```sh
   python main.py
   ```

4. **Access the Flask Application:**

   Open your web browser and navigate to:

   ```
   http://localhost:8000
   ```

   You should see the "Welcome to Riley's Visual Search System!" message from the index route. Your flask server is now properly running!

5. **Using the API Endpoints:**

   **Supported Endpoints**

   - **GET /**: Returns a welcome message.
   - **POST /authenticate**: Trigger the authentication process for the given probe image. Requires `image` file parameter provided as form data.
   - **GET /identity**: Get the list of image files for an identity in the gallery. Requires a `full_name` parameter.
   - **PUT /add_identity**: Add an image identity to the gallery. Requires `full_name` URL parameter and `image` file parameter provided as form data.
   - **PUT /remove_identity**: Remove an image identity from the gallery. Requires `filename` parameter.
   - **GET /access_logs**: Get the list of access log history for a specific time period. Requires `start_time` and `end_time` parameters.
   - **POST /images**: Get the image files associated with the given filename. Requires `files` attribute in the JSON body.

   **Example Requests**
   Examples of using all of the API endpoints can be seen in the Postman collection saved to [assets/EN.705.603.82 Case Study 9.postman_collection.json](./assets/EN.705.603.82%20Case%20Study%209.postman_collection.json). You can download the collection and import into Postman to make the requests. Some very basic curl requests are also demonstrated below.

   - **Initiate authentication:**

     ```sh
     curl -X POST "http://localhost:8000/authenticate" --form 'image=@"<IMAGE_FILE_NAME>"'
     ```

   - **Get identity:**

     ```sh
     curl -X GET 'http://localhost:8000/identity?full_name=<FULL_NAME>'
     ```

   - **Add identity:**

     ```sh
     curl -X PUT "http://localhost:8000/add_identity?full_name=<FULL_NAME>" --form 'image=@"<IMAGE_FILE_NAME>"'
     ```

   - **Remove identity:**

     ```sh
     curl -X PUT "http://localhost:8000/remove_identity?filename=<FILENAME>"
     ```

   - **Get access logs:**

     ```sh
     curl -X GET "http://localhost:8000/access_logs?start_time=<START_TIME>&end_time=<END_TIME>"
     ```

   - **Download image files:**

     ```sh
     curl -X POST "http://localhost:8000/images" --header 'Content-Type: application/json' --data '{"files": [<FILENAMES>]}'
     ```

## Instructions to Run the Flask application with Docker:

Here are step-by-step instructions to run the Flask application with Docker:

1. **Install Docker:**

   Make sure Docker is installed on your machine. You can download and install Docker from the [official Docker website](https://www.docker.com/products/docker-desktop).

2. **Build the Docker Image:**

   Build the Docker image from the Dockerfile:

   ```sh
   docker build -t visual_search_system:latest .
   ```

3. **Run the Docker Container:**

   Run the Docker container with the built image (Flask application on port 8000):

   ```sh
   docker run -it --rm -p 8000:8000 visual_search_system
   ```
