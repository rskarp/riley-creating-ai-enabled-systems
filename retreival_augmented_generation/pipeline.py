from src.extraction.embedding import Embedding
from src.extraction.preprocessing import Document_Processing
from src.generator.question_answering import BERT_Question_Answer
# from src.retrieval.index import ...
# from src.retrieval.search import ...


# Some useful code...
## To get questions and corresponding answers from `qa_resources/questions.csv`
import pandas as pd
questions = pd.read_csv("qa_resources/questions.csv", delimiter="\t")
print(questions[['Question', 'Answer']])

## To read all documents in `storage/corpus`
from glob import glob
documents = glob("storage/corpus/*.txt.clean")
processing = Document_Processing()
for document in documents:
    chunks = processing.split_document(document, sentences_per_chunk=3)
    print(chunks)



class Pipeline:
    ...