from glob import glob
from src.extraction.embedding import Embedding
from src.extraction.preprocessing import DocumentProcessing
from src.generator.question_answering import BERTQuestionAnswer
from src.retrieval.index import KDTree
from src.retrieval.search import Measure, KDTreeSearch
import numpy as np
import pandas as pd
from pathlib import Path
import os

# Location of storage
EMBEDDINGS_STORAGE = "storage/embeddings"
CORPUS_STORAGE = "storage/corpus"

parent_folder = str(Path(__file__).parent)


class Pipeline:
    def __init__(
        self,
        measure: Measure,
        embedding_model_name: str,
        qa_model_name: str,
        sentences_per_chunk: int,
    ):
        """
        Initialize the Pipeline class with all components and precompute corpus embeddings.

        Parameters:
            measure (Measure method): The distance measure to use for nearest neighbor search.
            embedding_model_name (str): The name of the embedding model to use.
            qa_model_name (str): The name of the question answering model to use.
            sentences_per_chunk (int): Number of sentences per text chunk.

        Returns:
            None
        """
        self.processing = DocumentProcessing()
        self.embedding_model_name = embedding_model_name
        self.embedding_model = Embedding(self.embedding_model_name)
        self.qa_model_name = qa_model_name
        self.qa_model = BERTQuestionAnswer(self.qa_model_name)
        self.index = None
        self.search = None
        self.measure = measure
        self.sentences_per_chunk = sentences_per_chunk
        self.__precompute()

    def __predict(self, text: str):
        """
        Extract the embedding vector output from a preprocessed text.

        Parameters:
            text (str): The input text string.

        Returns:
            ndarray: The embedding vector.
        """
        return self.embedding_model.encode(text)

    def __save_embeddings(
        self, document_filename: str, segment_number: int, embedding: np.ndarray
    ):
        """
        Store the embeddings in a numpy format.

        Parameters:
            document_filename (string): The name of the document file.
            embedding (ndarray): The embedding vector of the image.

        Returns:
            None
        """
        directory = f"{parent_folder}/{EMBEDDINGS_STORAGE}/{self.embedding_model_name}"
        outputFile = (
            f"{directory}/{os.path.basename(document_filename)}_{segment_number}.npy"
        )
        Path(directory).mkdir(parents=True, exist_ok=True)
        with open(outputFile, "wb") as f:
            np.save(f, embedding)

    def __precompute(self):
        """
        Precomputes the embeddings for all text chunks in storage/corpus/*.txt.clean and constructs a K-D Tree to organize the embeddings.
        """
        documents = glob(f"{parent_folder}/{CORPUS_STORAGE}/*.txt.clean")
        points = None
        metadata = []
        print("Calculating embeddings...")
        for document in documents:
            chunks = self.processing.split_document(
                document, sentences_per_chunk=self.sentences_per_chunk
            )
            for idx, chunk in enumerate(chunks):
                # Calculate Embedding
                embedding = self.__predict(chunk)
                points = (
                    embedding.T if points is None else np.vstack([points, embedding.T])
                )
                # Save embedding
                self.__save_embeddings(document, idx, embedding)
                # Get metadata
                meta = {
                    "sentences": chunk,
                    "chunk_number": idx,
                    "document": os.path.basename(document),
                }
                metadata.append(meta)

        print("Indexing embeddings...")
        self.index = KDTree(k=points.shape[1], points=points, metadata_list=metadata)
        self.search = KDTreeSearch(self.index, self.measure)

    def search_context(self, question: str, k: int):
        """
        Returns the nearest neighbors of a question (most relevant sentence chunks).
        """
        text = self.processing.trim_white_space(question)
        embedding = self.__predict(text)
        return self.search.find_nearest_neighbors(embedding, k)

    def set_search_measure(self, measure: Measure):
        """
        Sets the search measure used in the KNN KD search to the given measure.

        Parameters:
            measure (function): The measurement function to calculate the similarity between 2 points.

        Returns:
            None
        """
        self.measure = measure
        self.search = KDTreeSearch(self.index, self.measure)

    def get_search_measure(self):
        """
        Gets the search measure that's currently being used in the pipeline's KNN KD search.

        Parameters:
            None.

        Returns:
            str: Name of the similarity measure function.
        """
        return self.measure.__name__


if __name__ == "__main__":
    questions = pd.read_csv("qa_resources/questions.csv", delimiter="\t")
    pipeline = Pipeline(
        measure=Measure.euclidean,
        embedding_model_name="all-mpnet-base-v2",
        qa_model_name="google-bert/bert-large-cased-whole-word-masking-finetuned-squad",
        sentences_per_chunk=3,
    )
    qText = questions.loc[0, "Question"]
    neighbors = pipeline.search_context(qText, 3)
    for n in neighbors:
        print(n[1])
    print(f'Question: {qText}\nAnswer: {questions.loc[0,"Answer"]}')
