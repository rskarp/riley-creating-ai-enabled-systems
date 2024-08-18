from glob import glob
from typing import List
from pipeline import CORPUS_STORAGE, Pipeline
from src.retrieval.search import Measure
import os
from datetime import datetime
import json
from io import BytesIO
from zipfile import ZipFile


class Deployment:
    """
    Deployment class handles the deployment logic for the system components,
    including extraction, retrieval, and generation services.
    """

    def __init__(self):
        """
        Initialize the Deployment with necessary components.
        """
        self.sentences_per_chunk = 6
        self.embedding_model_name = "all-mpnet-base-v2"
        self.qa_model_name = "google-bert/bert-large-cased-whole-word-masking-finetuned-squad"

        self.pipeline = Pipeline(
            measure=Measure.euclidean,
            embedding_model_name=self.embedding_model_name,
            qa_model_name=self.qa_model_name,
            sentences_per_chunk=self.sentences_per_chunk,
        )

        self.k = 3
        self.log_date_format = "%Y%m%d_%H%M%S"

    def _save_access_log(self, question: str, answer: str, context: List):
        """
        Saves quesiton, answer, and context to .json file.

        Parameters:
            question (str): question asked.
            answer (str): answer generated.
            context (List): list of context chunks from corpus documents used.
        Returns:
            None
        """
        # Get timestamp and create log filename based on timestamp
        timestamp = datetime.now()
        filename = f'{timestamp.strftime(self.log_date_format)}'
        path = 'storage/logs'
        # Create folder if it doesn't exist
        os.makedirs(path, exist_ok=True)
        # Save log information in json file
        log_entry = {
            'question': question,
            'answer': answer,
            'timestamp': timestamp,
            'context': context
        }
        with open(f'{path}/{filename}.json', "w") as log:
            json.dump(log_entry, log, indent=4, default=str)

    def answer_question(self, question: str):
        """
        Get answer and context for given question.

        Parameters:
            question (str): Question to be answered.

        Returns:
            Tuple: answer and list of context used.
        """
        neighbors = self.pipeline.search_context(question, self.k)
        neighbor_metadata = [n[1] for n in neighbors]
        context = [n['sentences'] for n in neighbor_metadata]
        answer = self.pipeline.get_answer(question, context)
        self._save_access_log(question, answer, neighbor_metadata)
        return {'answer': answer, 'context': context}

    def get_documents(self):
        """
        Get document files in corpus.

        Returns:
            List: document file names in the corpus.
        """
        pattern = f'{CORPUS_STORAGE}/*'
        return glob(pattern)

    def add_document(self, filename: str, document):
        """
        Add document to the corpus.

        Parameters:
            filename (str): Name of the document.
            document (file): Document to add.

        Returns:
            str: Name given to the document file that was added to the corpus.
        """
        filename = f'{CORPUS_STORAGE}/{filename}'
        # Save file
        with open(filename, 'w') as fp:
            fp.write(document)
        # Add new document to KD Tree
        self.pipeline.add_document(filename)
        return filename

    def remove_document(self, filename: str):
        """
        Remove document from the corpus.

        Parameters:
            filename (str): Name of the document to remove.

        Returns:
            str: Name of the document file that was removed.
        """
        # Remove from KD tree and corpus
        self.pipeline.remove_document(filename, True)
        # Return removed filename
        return filename

    def get_logs(self, start_time: datetime, end_time: datetime):
        """
        Get the question log history of a specific time period.

        Parameters:
            start_time (datetime): Starting time of time range.
            end_time (datetime): Ending time of time range.

        Returns:
            List: question logs.
        """
        logs = []
        pattern = 'storage/logs/*.json'
        logFiles = glob(pattern)
        for f in logFiles:
            if start_time <= datetime.strptime(f.removeprefix('storage/logs/').removesuffix('.json'), self.log_date_format) <= end_time:
                with open(f, 'r') as file:
                    data = json.load(file)
                logs.append(data)
        return logs

    def get_document_files(self, files: List):
        """
        Get the document files.

        Parameters:
            filenames (List): list of document filenames.

        Returns:
            stream: Stream of a zip file containing the files of the documents.
        """
        stream = BytesIO()
        with ZipFile(stream, 'w') as zf:
            for documentPath in files:
                if os.path.isfile(documentPath):
                    zf.write(documentPath, os.path.basename(documentPath))
        stream.seek(0)
        return stream
