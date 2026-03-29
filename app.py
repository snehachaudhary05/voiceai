"""
NVIDIA Voice Assistant - Web App Backend
Flask server: browser handles ASR (Web Speech API) + TTS (speechSynthesis)
Backend handles: LLM (NVIDIA NIM) + serving the UI
"""

import os
from pathlib import Path
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).parent / ".env")

app = Flask(__name__)
CORS(app)

NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")

client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=NVIDIA_API_KEY,
)

conversation_history = [
    {
        "role": "system",
        "content": """You are a helpful, friendly voice assistant.
Always respond in the SAME language the user speaks.
Keep responses SHORT and conversational (2-4 sentences max) since they will be spoken aloud."""
    }
]


@app.after_request
def no_cache(response):
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response


@app.route("/")
def index():
    html_path = Path(__file__).parent / "templates" / "index.html"
    return send_file(html_path, mimetype="text/html")


@app.route("/chat", methods=["POST"])
def chat():
    """Receives user text → returns AI response text via NVIDIA LLM."""
    data = request.get_json()
    user_text = data.get("text", "").strip()

    if not user_text:
        return jsonify({"error": "Empty message"}), 400

    conversation_history.append({"role": "user", "content": user_text})

    try:
        response = client.chat.completions.create(
            model="meta/llama-3.3-70b-instruct",
            messages=conversation_history,
            temperature=0.7,
            max_tokens=300,
        )
        assistant_text = response.choices[0].message.content.strip()
        conversation_history.append({"role": "assistant", "content": assistant_text})

        if len(conversation_history) > 21:
            conversation_history[1:3] = []

        return jsonify({"text": assistant_text})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/reset", methods=["POST"])
def reset():
    """Clears conversation history."""
    global conversation_history
    conversation_history = [conversation_history[0]]
    return jsonify({"status": "reset"})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    print(f"\n Local URL: http://127.0.0.1:{port}\n")
    app.run(debug=False, host="0.0.0.0", port=port)
