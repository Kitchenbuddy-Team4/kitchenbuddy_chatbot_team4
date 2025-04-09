from flask import Flask, request, jsonify
import requests
import nltk
from nltk.tokenize import word_tokenize
from services.chatbot_service import chatbot_bp

nltk.download("punkt")

app = Flask(__name__)
app.register_blueprint(chatbot_bp)

# SPRING_BACKEND_URL = "http://localhost:8080/api/recipes"  # Update this based on your API

# Health Check Route
@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({"status": "ok", "message": "KitchenBuddy API is running"}), 200

# Future microservice integration points
@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Welcome to KitchenBuddy API Gateway"}), 200

if __name__ == "__main__":
    app.run(debug=True, host="localhost", port=5000)  # Main entry point