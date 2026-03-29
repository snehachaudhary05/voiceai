"""
VoiceAI — Smart AI Assistant with Voice + Vision + Text
Powered by NVIDIA NIM Free APIs (Llama 3.3 70B + Qwen 3.5 Vision)
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

TEXT_MODEL = "meta/llama-3.3-70b-instruct"
VISION_MODEL = "qwen/qwen3.5-397b-a17b"

# Conversation history (text-only, in-memory)
conversation_history = [
    {
        "role": "system",
        "content": """You are VoiceAI, a helpful, friendly, and knowledgeable AI assistant.
You can analyze images, answer questions, write code, explain concepts, and help with anything.
Always respond in the SAME language the user speaks.
Format your responses beautifully using markdown:
- Use **bold** for key terms
- Use bullet points and numbered lists
- Use code blocks with language tags for code
- Use tables when comparing things
- Use headings for long answers
- Use emojis sparingly to make responses engaging
Keep voice responses SHORT (2-4 sentences). For text/image queries, be as detailed as needed."""
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
    """Handles both text-only and vision (image+text) requests."""
    data = request.get_json()
    user_text = data.get("text", "").strip()
    images = data.get("images", [])  # list of base64 data URIs

    if not user_text and not images:
        return jsonify({"error": "Empty message"}), 400

    try:
        if images:
            # Vision request — use Qwen 3.5 Vision model
            content = []
            if user_text:
                content.append({"type": "text", "text": user_text})
            for img_data in images:
                content.append({
                    "type": "image_url",
                    "image_url": {"url": img_data}
                })

            response = client.chat.completions.create(
                model=VISION_MODEL,
                messages=[{"role": "user", "content": content}],
                temperature=0.7,
                max_tokens=1024,
            )
        else:
            # Text-only request — use Llama 3.3 with conversation history
            conversation_history.append({"role": "user", "content": user_text})

            response = client.chat.completions.create(
                model=TEXT_MODEL,
                messages=conversation_history,
                temperature=0.7,
                max_tokens=1024,
            )

        assistant_text = response.choices[0].message.content.strip()

        # Only track history for text conversations (not vision)
        if not images:
            conversation_history.append({"role": "assistant", "content": assistant_text})
            if len(conversation_history) > 21:
                conversation_history[1:3] = []

        return jsonify({"text": assistant_text})

    except Exception as e:
        print(f"Chat error: {e}", flush=True)
        return jsonify({"error": str(e)}), 500


@app.route("/reset", methods=["POST"])
def reset():
    """Clears conversation history."""
    global conversation_history
    conversation_history = [conversation_history[0]]
    return jsonify({"status": "reset"})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    print(f"\n VoiceAI running at http://127.0.0.1:{port}\n")
    app.run(debug=False, host="0.0.0.0", port=port)
