import torch
import torch.nn as nn
from flask import Blueprint, request, jsonify
import json
import requests
import os

from utilities.intent_net import IntentNet  # Ensure this is importable
import spacy

# Blueprint
chatbot_bp = Blueprint("chatbot", __name__)

# === Load Ingredients ===
with open(os.path.join("..", "resources", "ingredients", "unique_ingredients.json")) as f:
    all_ingredients = json.load(f)

# === Load Model and Metadata Once ===
nlp = spacy.load("en_core_web_sm")

with open("../utilities/model/vocab.json", "r") as f:
    vocab = json.load(f)

with open("../utilities/model/labels.json", "r") as f:
    label_data = json.load(f)
    idx2label = {int(k): v for k, v in label_data["idx2label"].items()}

# Match same architecture used in training
model = IntentNet(vocab_size=len(vocab), embedding_dim=50, hidden_dim=64, output_dim=len(idx2label))
model.load_state_dict(torch.load("../utilities/model/intent_model_state.pth", map_location=torch.device("cpu")))
model.eval()


# === Helper functions ===
def tokenize(text):
    doc = nlp(text.lower())
    return [token.lemma_ for token in doc if not token.is_stop and not token.is_punct and not token.is_space]

def extract_ingredients(message):
    message = message.lower()
    return [ingredient for ingredient in all_ingredients if ingredient.lower() in message]

def predict_intent(message):
    tokens = tokenize(message)
    input_ids = [vocab.get(token, 0) for token in tokens]

    max_len = 10
    if len(input_ids) < max_len:
        input_ids += [0] * (max_len - len(input_ids))
    else:
        input_ids = input_ids[:max_len]

    input_tensor = torch.tensor([input_ids], dtype=torch.long)
    with torch.no_grad():
        outputs = model(input_tensor)
        pred_idx = torch.argmax(outputs, dim=1).item()
        return idx2label[pred_idx]


# === Main chatbot route ===
@chatbot_bp.route("/chatbot/message", methods=["POST"])
def handle_message():
    data = request.get_json()
    message = data.get("message", "")

    if not message:
        return jsonify({"error": "Missing 'message'"}), 400

    intent = predict_intent(message)

    if intent == "greeting":
        return jsonify({
            "intent": "greeting",
            "response": "Hello! I'm Uncle Cheffington. Let me know what ingredients you have!"
        })

    elif intent == "recipe_request":
        ingredients = extract_ingredients(message)
        if not ingredients:
            return jsonify({
                "intent": "recipe_request",
                "response": "Hmm, I couldn't recognize any ingredients. Could you try again?"
            })

        try:
            ingredient_str = ",".join(ingredients)
            spring_response = requests.get(f"http://localhost:8080/recipes/match/{ingredient_str}")
            recipe = spring_response.json()

            return jsonify({
                "intent": "recipe_request",
                "response": f"How about this: {recipe['title']}"
            })
        except Exception:
            return jsonify({
                "intent": "recipe_request",
                "response": "Sorry, I couldn't find a matching recipe right now."
            }), 500

    return jsonify({
        "intent": intent,
        "response": "I'm still learning to handle that kind of message!"
    })
