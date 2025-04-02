from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route("/chatbot", methods=["POST"])
def chatbot():
    user_input = request.json.get("message", "")
    return jsonify({"response": f"You said: {user_input}"})

if __name__ == "__main__":
    app.run(debug=True, host="localhost", port=5000)