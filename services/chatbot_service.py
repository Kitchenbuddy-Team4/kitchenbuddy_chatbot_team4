from flask import Blueprint, request, jsonify

# Create a Flask Blueprint
chatbot_bp = Blueprint("chatbot", __name__)

# /chatbot/greet endpoint
@chatbot_bp.route("/chatbot/greet", methods=["GET"])
def greet():
    return jsonify({"response": "Greet endpoint"}), 200

# /chatbot/bye endpoint
@chatbot_bp.route("/chatbot/bye", methods=["GET"])
def bye():
    return jsonify({"response": "Bye endpoint"}), 200

# /chatbot/recipe-request endpoint
@chatbot_bp.route("/chatbot/recipe-request", methods=["POST"])
def recipe_request():
    return jsonify({"response": "Recipe request endpoint"}), 200
