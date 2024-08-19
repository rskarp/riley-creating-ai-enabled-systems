# Retrieval Augmented Generation

This project contains all of the code components necessary to run a retrieval augmented generation system.

# Prerequisites

- Docker installed on your machine
- Basic knowledge of Docker and Flask
- (Optional) Postman installed on your machine and basic knowledge of Postman

# Contents

These are the files you can find in this repository.

- **[main.py](main.py)**: The main application script that starts the Flask server and defines the API routes.
- **[deployment.py](deployment.py)**: Contains the `Deployment` class, which handles the deployment logic for ML model and system.
- **[pipeline.py](pipeline.py)**: Contains the `Pipeline` class, which handles the extraction, retrieval, and generation pipeline.
- **[requirements.txt](requirements.txt)**: Lists the Python dependencies required for the project.
- **[Dockerfile](Dockerfile)**: Defines the Docker image configuration for the project.
- **[analysis/data_analysis.ipynb](analysis/data_analysis.ipynb)**: Contains exploratory data analysis plots and explanation of findings.
- **[analysis/design_considerations.ipynb](analysis/design_considerations.ipynb)**: Contains analysis of different models, parameters, and their performance.
- **[analysis/systems_report.ipynb](analysis/systems_report.ipynb)**: Contains analysis of entire system and project.
- **[src/metrics.py](src/metrics.py)**: Contains the `Metrics_Automated` class, which is used to evaluate the RAG system.
- **[src/extraction/embedding.py](src/extraction/embedding.py)**: Contains the `Embedding` class used to create a SentenceTransformer model, which is used to calculate document embeddings.
- **[src/extraction/preprocessing.py](src/extraction/preprocessing.py)**: Contains the `DocumentProcessing` class, which is used to split the document into sentence chunks.
- **[src/retrieval/index.py](src/retrieval/index.py)**: Contains the `KDTree` class, which is used to create a KD Tree search index.
- **[src/retrieval/search.py](src/retrieval/search.py)**: Contains the `KDTreeSearch` and `Measure` classes, which are used to run a nearest neighbors search using a specified distance measure.
- **[src/generator/question_answering.py](src/generator/question_answering.py)**: Contains the `BERTQuestionAnswer` class, which is used to generate an answer to a question using a BERT model.
- **[assets/images/](assets/images/)**: Contains images used in [analysis/systems_report.ipynb](analysis/systems_report.ipynb).
- **[assets/EN.705.603.82 Case Study 12.postman_collection.json](assets/EN.705.603.82%20Case%20Study%2012.postman_collection.json)**: Postman collection containing example requests for all endpoints.
- **[storage/embeddings/](storage/embeddings/)**: Contains `.npy` files containing embedding vectors for each document in the corpus, organized by model and sentences per chunk.
- **[storage/corpus/](storage/corpus/)**: Contains documents that make up the corpus used to retrieve context.
- **[storage/logs/](storage/logs/)**: Contains json files that are captured as logs for each question asked using the Flask `/question` endpoint.
- **[qa_resources/questions.csv](qa_resources/questions.csv)**: Contains example questions used for validadtion and analysis.

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

4. **Access the Flask Application:**

   Open your web browser and navigate to:

   ```
   http://localhost:8000
   ```

   You should see the "Welcome to Riley's Retrieval Augmented Generation System!" message from the index route. Your flask server is now properly running!

5. **Using the API Endpoints:**

   **Supported Endpoints**

   - **GET /**: Returns a welcome message.
   - **POST /question**: Trigger the question answering process for the given question. Requires `question` string provided as an attribute in the JSON body.
   - **GET /documents**: Get the list of document files in the corpus.
   - **PUT /document**: Add a document to the corpus. Requires `document` file parameter provided as form data.
   - **DELETE /document**: Remove document from the corpus. Requires `filename` query parameter.
   - **GET /logs**: Get the list of question log history for a specific time period. Requires `start_time` and `end_time` query parameters.
   - **POST /document_files**: Get the document files associated with the given filenames. Requires `filenames` list attribute in the JSON body.

   **Example Requests**
   Examples of using all of the API endpoints can be seen in the Postman collection saved to [assets/EN.705.603.82 Case Study 12.postman_collection.json](./assets/EN.705.603.82%20Case%20Study%2012.postman_collection.json). You can download the collection and import into Postman to make the requests. Some very basic curl requests are also demonstrated below.

   - **Initiate question answering process:**

     ```sh
     curl -X POST "http://localhost:8000/question" --header 'Content-Type: application/json' --data '{"question": "<QUESTION_TEXT>"}'
     ```

   - **Get documents list:**

     ```sh
     curl -X GET 'http://localhost:8000/documents'
     ```

   - **Add document:**

     ```sh
     curl -X PUT "http://localhost:8000/document" --form 'document=@"<DOCUMENT_FILE_NAME>"'
     ```

   - **Remove document:**

     ```sh
     curl -X DELETE "http://localhost:8000/document?filename=<FILENAME>"
     ```

   - **Get question logs:**

     ```sh
     curl -X GET "http://localhost:8000/logs?start_time=<START_TIME>&end_time=<END_TIME>"
     ```

   - **Download document files:**

     ```sh
     curl -X POST "http://localhost:8000/document_files" --header 'Content-Type: application/json' --data '{"filenames": [<FILENAMES>]}'
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
