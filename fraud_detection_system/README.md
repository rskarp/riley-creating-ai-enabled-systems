# Fraud Detection System

This project contains all of the code components necessary to run a fraud detection (classification) system and access it using a Flask REST API. This project also contains documentation regarding the system requirements and system design choices made throughoutthe project. Running the server will create a `resources` directory which will store logs and dataset files.

# Prerequisites

- Docker installed on your machine
- Basic knowledge of Docker and Flask
- Postman installed on your machine and basic knowledge of Postman (optional)

# Contents

These are the files you can find in this repository.

- **[main.py](main.py)**: The main application script that starts the Flask server and defines the API routes.
- **[deployment.py](deployment.py)**: Contains the `DeploymentPipeline` class, which handles the deployment logic for ML models.
- **[dataset.py](dataset.py)**: Contains the `Dataset` class, which handles dataset generation and loading.
- **[feature_engineering.py](feature_engineering.py)**: Contains the `FeatureEngineering` class, which handles data transformation and feature extraction.
- **[data_engineering.py](data_engineering.py)**: Contains the `DataEngineering` class, which handles data cleaning and validation.
- **[model.py](model.py)**: Contains the `Model` class, which defines the model architecture and prediction logic.
- **[metrics.py](metrics.py)**: Contains the `Metrics` class, which calculates evaluation metrics for the model.
- **[requirements.txt](requirements.txt)**: Lists the Python dependencies required for the project.
- **[Dockerfile](Dockerfile)**: Defines the Docker image configuration for the project.
- **[analysis/exploratory_data_analysis.ipynb](analysis/exploratory_data_analysis.ipynb)**: Contains exploratory data analysis plots and explanation of findings.
- **[analysis/model_performance_and_selection.ipynb](analysis/model_performance_and_selection.ipynb)**: Contains analysis of three different models and their performance with different hyperparameters.
- **[analysis/systems_report.ipynb](analysis/systems_report.ipynb)**: Contains analysis of entire system and project.
- **[requirements/elicitation.md](requirements/elicitation.md)**: Contains description of stakeholders and questions for requirements elicitation.
- **[requirements/requirements_analysis.md](requirements/requirements_analysis.md)**: Contains list of functional and non-functional requirements.
- **[requirements/requirements_specification.md](requirements/requirements_specification.md)**: Contains detailed descriptions of each requirement.
- **[requirements/validation_and_verification.md](requirements/validation_and_verification.md)**: DContains description of validation and verification process for requirements.
- **[assets/images/](assets/images/)**: Contains images used in [analysis/systems_report.ipynb](analysis/systems_report.ipynb).
- **[assets/EN.705.603.82 Case Study.postman_collection.json](assets/EN.705.603.82%20Case%20Study%205.postman_collection.json)**: Postman collection containing example requests for all endpoints.

# Instructions

After cloning this repository, but before running the Flask application, be sure to put the data source files into the `data` folder as indicated by the [put_data_here.txt](../data/put_data_here.txt) file.

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

   You should see the "Welcome to Riley's Fraud Detection System!" message from the index route. Your flask server is now properly running!

5. **Using the API Endpoints:**

   **Supported Endpoints**

   - **GET /**: Returns a welcome message.
   - **PUT /generate_new_dataset**: Generates a new dataset. Requires a `version` parameter. Optional parameters: `type`, `n_samples`, `sampling_type`, `random_state`, `generate_features`.
   - **PUT /generate_new_features**: Extracts features for an existing dataset. Requires a `version` parameter. Optional parameters: `run_smote`, `random_state`.
   - **GET /dataset_description**: Retrieves the description of a dataset. Requires a `version` parameter.
   - **GET /dataset_features_description**: Retrieves the description of features for a dataset. Requires a `version` parameter.
   - **GET /dataset_list**: Retrieves the list of available datasets.
   - **GET /feature_list**: Retrieves the list of available features files.
   - **PUT /train**: Makes a prediction based on input data. Requires `model_version`, `model_type`, and `dataset_version` parameters. JSON Body must be hyperparameters valid fo the selected `model_type`. Optional parameters: `random_state`.
   - **GET /model_list**: Retrieves the list of available trained models.
   - **POST /predict**: Makes a prediction based on input data. Requires a `version` parameter. Requires JSON body of a transaction. Optional parameters: `random_state`.
   - **GET /model_metrics**: Retrieves the list of available trained models. Requires `model_version` and `dataset_version` parameters.
   - **GET /system_metrics**: Retrieves the list of available trained models.

   **Example Requests**
   Examples of using all of the API endpoints can be seen in the Postman collection saved to [assets/EN.705.603.82 Case Study 5.postman_collection.json](../assets/EN.705.603.82%20Case%20Study%205.postman_collection.json). You can download the collection and import into Postman to make the requests. Some very basic curl requests are also demonstrated below.

   - **Generate Dataset:**

     ```sh
     curl -X PUT "http://localhost:8000/generate_new_dataset?version=<DATASET_VERSION>"
     ```

   - **Generate Features:**

     ```sh
     curl --location --request PUT 'http://localhost:8000/generate_new_features?version=<DATASET_VERSION>'
     ```

   - **Get Dataset Description:**

     ```sh
     curl -X GET "http://localhost:8000/dataset_description?version=<DATASET_VERSION>"
     ```

   - **Get Features Description:**

     ```sh
     curl -X GET "http://localhost:8000/dataset_features_description?version=<DATASET_VERSION>"
     ```

   - **Get Dataset List:**

     ```sh
     curl -X GET "http://localhost:8000/dataset_list"
     ```

   - **Get Features List:**

     ```sh
     curl -X GET "http://localhost:8000/feature_list"
     ```

   - **Make Trained Model:**

     ```sh
     curl -X PUT "http://localhost:8000/train?dataset_version=<DATASET_VERSION>&model_version=<MODEL_VERSION>&model_type=<MODEL_TYPE>" -H "Content-Type: application/json" -d @<HYPERPARAMETER_FILE>
     ```

   - **Get Models List:**

     ```sh
     curl -X GET "http://localhost:8000/model_list"
     ```

   - **Make a Prediction:**

     ```sh
     curl -X POST "http://localhost:8000/predict?version=<DATASET_VERSION>" -H "Content-Type: application/json" -d @<DATA_FILE>
     ```

   - **Get Model Metrics:**

     ```sh
     curl -X GET "http://localhost:8000/model_metrics?dataset_version=<DATASET_VERSION>&model_version=<MODEL_VERSION>"
     ```

   - **Get System Metrics:**

     ```sh
     curl -X GET "http://localhost:8000/system_metrics"
     ```

### Instructions to Run the Flask Application with Docker:

Here are step-by-step instructions to run the Flask application with Docker:

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
