# import libraries
import spacy

class IntentClassifier:

    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")

    def classify(self, request: str) -> str:
        pass


