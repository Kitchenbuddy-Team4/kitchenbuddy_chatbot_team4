# import libraries
import spacy
import json
import torch
from utilities.intent_net import IntentNet
from torch.serialization import safe_globals


class IntentClassifier:

    def __init__(self):
        # set Natural Language Processing (nlp) attribute for on Classifier object
        self.nlp = spacy.load("en_core_web_sm")

    def classify(self, request: str) -> str:
        # Load model and related assets only once
        if not hasattr(self, 'model'):
            # Safely load the full model
            with safe_globals({"IntentNet": IntentNet}):
                self.model = torch.load("../utilities/model/intent_model_full.pth", weights_only=False)

            self.model.eval()

            # Load vocab
            with open("model/vocab.json", "r") as f:
                self.vocab = json.load(f)

            # Load label index → intent name mapping
            with open("model/labels.json", "r") as f:
                label_data = json.load(f)
                self.idx2label = {int(k): v for k, v in label_data["idx2label"].items()}

        # === Preprocess the user input ===
        tokens = self.tokenize(request)
        input_ids = [self.vocab.get(token, 0) for token in tokens]  # Use 0 if OOV

        max_len = 10  # Must match training time
        if len(input_ids) < max_len:
            input_ids += [0] * (max_len - len(input_ids))
        else:
            input_ids = input_ids[:max_len]

        input_tensor = torch.tensor([input_ids], dtype=torch.long)

        # ✅ Match model's device (GPU or CPU)
        input_tensor = input_tensor.to(self.model.embedding.weight.device)

        # === Run inference ===
        with torch.no_grad():
            outputs = self.model(input_tensor)
            predicted_index = torch.argmax(outputs, dim=1).item()
            predicted_label = self.idx2label[predicted_index]

        return predicted_label

    def get_intents(self, intents_file: str) -> tuple:

        # local variable initialization
        patterns = []
        labels = []

        # open the files and load json data
        with open(intents_file, "r") as f:
            data = json.load(f)

        # store message and text with labels
        for intent in data["intents"]:
            tag = intent["tag"]
            for pattern in intent["patterns"]:
                patterns.append(pattern)
                labels.append(tag)

        return patterns, labels

    def tokenize(self, text: str) -> list:
        # convert the text to doc object to use nlp methods
        doc = self.nlp(text.lower())

        # generate a list of tokens for the text
        tokens = [token.lemma_ for token in doc
                  if not token.is_punct
                  and not token.is_space
                  and not token.is_stop]

        return tokens

    def build_vocab(self, patterns: list) -> dict:
        all_words = set()

        for sentence in patterns:
            tokens = self.tokenize(sentence)
            all_words.update(tokens)

        vocab = {word: id for id, word in enumerate(sorted(all_words))}
        return vocab

    def encode_labels(self, labels: list) -> tuple:
        unique_labels = sorted(set(labels))  # ensures consistent ordering
        label2idx = {label: id for id, label in enumerate(unique_labels)} # for training
        idx2label = {id: label for label, id in label2idx.items()} # for testing
        encoded_labels = [label2idx[label] for label in labels]

        return encoded_labels, label2idx, idx2label

    def encode_patterns(self, patterns: list, vocab: dict, max_len: int) -> list:
        encoded = []

        for sentence in patterns:
            tokens = self.tokenize(sentence)
            word_ids = [vocab[token] for token in tokens if token in vocab]

            # Pad or trim to max_len
            if len(word_ids) < max_len:
                word_ids += [0] * (max_len - len(word_ids))
            else:
                word_ids = word_ids[:max_len]

            encoded.append(word_ids)

        return encoded
