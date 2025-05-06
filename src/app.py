from flask import Flask, request, jsonify
import requests
import nltk
from nltk.tokenize import word_tokenize
from services.chatbot_service import chatbot_bp
import os

nltk.download("punkt")

app = Flask(__name__)
app.register_blueprint(chatbot_bp)

# Health Check Route
@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({"status": "ok", "message": "KitchenBuddy API is running"}), 200

# Home Route
@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Welcome to KitchenBuddy API Gateway"}), 200

# Run app on appropriate host and port for deployment
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, host="0.0.0.0", port=port)
