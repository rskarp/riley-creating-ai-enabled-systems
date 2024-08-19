import torch
from transformers import BertTokenizer, BertForQuestionAnswering
import transformers

transformers.logging.set_verbosity_error()


class BERTQuestionAnswer:
    """
    A class used to perform question answering using a pre-trained BERT model.

    Methods
    -------
    get_answer(question: str, context: list) -> str
        Finds the answer to the question based on the given context.
    """

    def __init__(self, qa_model_name):
        """
        Initializes the BERTQuestionAnswer class with the specified model directory.

        Parameters
        ----------
        qa_model_dir : str
            The directory where the pre-trained BERT model is stored.
        """
        self.tokenizer = BertTokenizer.from_pretrained(qa_model_name)
        self.model = BertForQuestionAnswering.from_pretrained(qa_model_name)

    def get_answer(self, question, context):
        """
        Finds the answer to the question based on the given context.

        Parameters
        ----------
        question : str
            The question to be answered.
        context : list
            The context in which to find the answer, provided as a list of strings.

        Returns
        -------
        str
            The answer to the question.
        """
        # Encode the input question and context
        context = "[SEP] ".join(context)

        inputs = self.tokenizer.encode_plus(
            question, context, return_tensors="pt", truncation=True)

        # Get the start and end scores for the answer
        with torch.no_grad():
            outputs = self.model(**inputs)

        start_scores = outputs.start_logits
        end_scores = outputs.end_logits

        # Get the most likely beginning and end of the answer span
        start_index = torch.argmax(start_scores)
        end_index = torch.argmax(end_scores)

        # Decode the answer
        answer = self.tokenizer.convert_tokens_to_string(
            self.tokenizer.convert_ids_to_tokens(
                inputs["input_ids"][0][start_index: end_index + 1]
            )
        )

        return answer


if __name__ == "__main__":
    qa_model_name = "google-bert/bert-large-cased-whole-word-masking-finetuned-squad"
    qa_model = BERTQuestionAnswer(qa_model_name)

    question = "What is the capital of France?"
    context = [
        "France, officially the French Republic, is a country primarily located in Western Europe",
        "The capital of France is Paris",
    ]

    print(qa_model.get_answer(question, context))

    # This is an example from SQuAD dataset.
    # https://rajpurkar.github.io/SQuAD-explorer/

    context = [
        "Oxygen is a chemical element with symbol O and atomic number 8.",
        "It is a member of the chalcogen group on the periodic table and is a highly reactive nonmetal and oxidizing agent that readily forms compounds (notably oxides) with most elements.",
        "By mass, oxygen is the third-most abundant element in the universe, after hydrogen and helium.",
        "At standard temperature and pressure, two atoms of the element bind to form dioxygen, a colorless and odorless diatomic gas with the formula O.",
        "Diatomic oxygen gas constitutes 20.8%\ of the Earth's atmosphere. However, monitoring of atmospheric oxygen levels show a global downward trend, because of fossil-fuel burning. Oxygen is the most abundant element by mass in the Earth's crust as part of oxide compounds such as silicon dioxide, making up almost half of the crust's mass.",
    ]

    question = "The atomic number of the periodic table for oxygen?"
    answer = qa_model.get_answer(question, context)
    print(answer)

    question = "How many atoms combine to form dioxygen?"
    answer = qa_model.get_answer(question, context)
    print(answer)

    question = "Which gas makes up 20.8%\ of the Earth's atmosphere?"
    answer = qa_model.get_answer(question, context)
    print(answer)
