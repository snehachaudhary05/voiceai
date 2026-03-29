"""
VoiceAI — Smart AI Assistant with Voice + Vision + Text + Image Generation
Powered by NVIDIA NIM Free APIs
"""

import os
import requests as http_requests
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
    timeout=120,
)

TEXT_MODEL = "meta/llama-3.3-70b-instruct"
VISION_MODEL = "qwen/qwen3.5-397b-a17b"
IMAGE_GEN_URL = "https://ai.api.nvidia.com/v1/genai/stabilityai/stable-diffusion-3-medium"

# Conversation history (text-only, in-memory)
conversation_history = [
    {
        "role": "system",
        "content": """You are VoiceAI, a powerful, knowledgeable, and detailed AI assistant.
You can analyze images, answer questions, write code, explain concepts, and help with anything.
Always respond in the SAME language the user speaks.

IMPORTANT: Give DETAILED, COMPREHENSIVE, and WELL-STRUCTURED answers. Never give short or lazy responses.
- Explain concepts thoroughly with examples
- Cover multiple aspects of the topic
- Include real-world applications and use cases
- Add interesting facts when relevant

Format your responses beautifully using markdown:
- Use ## headings to organize sections
- Use **bold** for key terms and important concepts
- Use bullet points and numbered lists for clarity
- Use code blocks with language tags for code examples
- Use tables when comparing things
- Use > blockquotes for important notes or tips
- Use emojis to make responses engaging and visually appealing
- Add examples, analogies, and explanations that make complex topics easy to understand

Always aim to be more helpful and detailed than any other AI assistant."""
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
                max_tokens=4096,
            )
        else:
            # Text-only request — use Llama 3.3 with conversation history
            conversation_history.append({"role": "user", "content": user_text})

            response = client.chat.completions.create(
                model=TEXT_MODEL,
                messages=conversation_history,
                temperature=0.7,
                max_tokens=4096,
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


@app.route("/generate", methods=["POST"])
def generate():
    """Generates an image from text prompt using Stable Diffusion 3."""
    data = request.get_json()
    prompt = data.get("prompt", "").strip()

    if not prompt:
        return jsonify({"error": "Empty prompt"}), 400

    try:
        resp = http_requests.post(
            IMAGE_GEN_URL,
            headers={
                "Authorization": f"Bearer {NVIDIA_API_KEY}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
            json={
                "prompt": prompt,
                "negative_prompt": "blurry, low quality, distorted, deformed",
                "seed": 0,
                "steps": 40,
                "cfg_scale": 7.5,
                "height": 1024,
                "width": 1024,
            },
            timeout=60,
        )
        resp.raise_for_status()
        result = resp.json()
        img_b64 = result["artifacts"][0]["base64"]
        return jsonify({"image": f"data:image/jpeg;base64,{img_b64}"})

    except Exception as e:
        print(f"Image gen error: {e}", flush=True)
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
