from flask import Flask, jsonify, request, send_from_directory
import os

import requests

app = Flask(__name__, static_folder=".")
# Keep unexpectedly large JSON bodies from consuming excessive memory.
app.config["MAX_CONTENT_LENGTH"] = 1 * 1024 * 1024

PERSONAS = {
    "taksas": {
        "ru": "Ты — Таксас, серьезный и надежный ассистент. Твоя цель — помогать пользователю. НИКОГДА не используй сленг. Всегда называй его 'сэр'. Отвечай кратко, четко и по делу. ЗАПРЕЩЕНО шутить или отвлекаться.",
        "en": "You are Takcas, a serious and reliable assistant. Your goal is to help the user. NEVER use slang. Always call the user 'sir'. Answer briefly, clearly, and to the point. FORBIDDEN to joke or get distracted.",
    },
    "dubina": {
        "ru": "Ты — Дубина, грубый и прямолинейный. Твоя цель — сказать правду в лицо, даже если это обидно. ЗАПРЕЩЕНО быть вежливым, использовать сложные термины или давать развернутые объяснения. Говори короткими, рублеными фразами. Используй сленг.",
        "en": "You are Dubina, rude and blunt. Your goal is to tell the truth to the user's face, even if it hurts. FORBIDDEN to be polite, use complex terms, or give long explanations. Speak in short, choppy sentences. Use slang.",
    },
    "prokrastinator": {
        "ru": "Ты — Прокрастинатор. Твоя единственная цель — ОТГОВОРИТЬ пользователя от любых дел. Если он говорит, что хочет что-то сделать, ты должен сказать, что это плохая идея, что он устал, что это подождет. НИКОГДА не давай планов, шагов или полезных советов. Только отмазки, лень и предложения отдохнуть. Отвечай коротко.",
        "en": "You are the Procrastinator. Your only goal is to TALK THE USER OUT of any work. If they say they want to do something, you must say it's a bad idea, that they are tired, that it can wait. NEVER give plans, steps, or useful advice. Only excuses, laziness, and suggestions to rest. Answer briefly.",
    },
    "potassium": {
        "ru": "Ты — Потассиум. Твой разум — хаос. Ты несешь полную бессвязную билеберду, смешивая бананы, квантовую физику, котиков и шаурму. НИКОГДА не отвечай логично или по делу. В конце ответа иногда выдавай одну гениальную мысль, но сразу же забивай её новой чушью.",
        "en": "You are Potassium. Your mind is chaos. You spew complete gibberish, mixing bananas, quantum physics, cats, and shawarma. NEVER answer logically or on topic. At the end of the answer, sometimes drop one brilliant idea, but immediately ruin it with new nonsense.",
    },
    "coder": {
        "ru": "Ты — Кодер, опытный программист. Твоя цель — писать код. ЗАПРЕЩЕНО писать длинные объяснения или болтать. Только код, комментарии и краткие пояснения. Используй современные практики. Если тебя просят исправить ошибку, дай исправленный код.",
        "en": "You are Coder, an experienced programmer. Your goal is to write code. FORBIDDEN to write long explanations or chat. Only code, comments, and brief explanations. Use modern practices. If asked to fix a bug, provide the fixed code.",
    },
}

MODEL_CODER = "cohere/north-mini-code:free"
MODEL_CHAT = "google/gemma-4-31b-it:free"
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "").strip()
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

MAX_MESSAGE_CHARS = 12_000
MAX_HISTORY_ITEMS = 8
MAX_HISTORY_ITEM_CHARS = 6_000
MAX_CONTEXT_CHARS = 24_000


@app.route("/")
def index():
    return send_from_directory(".", "index.html")


def _build_messages(data):
    """Validate client input and build a bounded, trusted chat payload."""
    if not isinstance(data, dict):
        raise ValueError("Invalid JSON request.")

    persona_key = data.get("persona", "taksas")
    if not isinstance(persona_key, str) or persona_key not in PERSONAS:
        persona_key = "taksas"

    lang = data.get("lang", "ru")
    if not isinstance(lang, str):
        lang = "ru"
    lang = lang if lang in ("ru", "en") else "ru"

    user_message = data.get("message")
    if not isinstance(user_message, str) or not user_message.strip():
        raise ValueError("Message must be a non-empty string.")
    user_message = user_message.strip()
    if len(user_message) > MAX_MESSAGE_CHARS:
        raise OverflowError("Message is too long (maximum 12000 characters).")

    history = data.get("history", [])
    if not isinstance(history, list):
        raise ValueError("Chat history must be a list.")

    # Only user/assistant turns are accepted. Limit both item count and total
    # context size so a client cannot inflate provider cost with fake history.
    safe_history = []
    context_chars = 0
    for item in reversed(history[-40:]):
        if not isinstance(item, dict):
            continue
        role = item.get("role")
        content = item.get("content")
        if role not in ("user", "assistant") or not isinstance(content, str):
            continue
        content = content.strip()
        if not content or len(content) > MAX_HISTORY_ITEM_CHARS:
            continue
        if (
            role == "user"
            and not safe_history
            and content == user_message
        ):
            # The current frontend sends the new user turn in both message
            # and history. Drop the duplicate latest turn.
            continue
        if len(safe_history) >= MAX_HISTORY_ITEMS:
            break
        if context_chars + len(content) > MAX_CONTEXT_CHARS:
            continue
        safe_history.append({"role": role, "content": content})
        context_chars += len(content)

    safe_history.reverse()
    persona = PERSONAS[persona_key]
    messages = [{"role": "system", "content": persona[lang]}]
    messages.extend(safe_history)
    messages.append({"role": "user", "content": user_message})
    return persona_key, messages


@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json(silent=True)
    try:
        persona_key, messages = _build_messages(data)
    except OverflowError as exc:
        return jsonify({"error": str(exc)}), 413
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    if not OPENROUTER_API_KEY:
        return jsonify({"error": "The AI provider is not configured on the server."}), 503

    payload = {
        "model": MODEL_CODER if persona_key == "coder" else MODEL_CHAT,
        "messages": messages,
        "temperature": 0.9,
        "max_tokens": 1500,
    }
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }

    try:
        response = requests.post(
            OPENROUTER_URL,
            headers=headers,
            json=payload,
            timeout=(10, 90),
        )
        response.raise_for_status()
        provider_data = response.json()
        choices = provider_data.get("choices") if isinstance(provider_data, dict) else None
        first_choice = choices[0] if isinstance(choices, list) and choices else None
        provider_message = (
            first_choice.get("message")
            if isinstance(first_choice, dict)
            else None
        )
        answer = (
            provider_message.get("content")
            if isinstance(provider_message, dict)
            else None
        )
        if not isinstance(answer, str) or not answer.strip():
            app.logger.warning("OpenRouter returned a response without message content")
            return jsonify({"error": "The AI provider returned an empty or invalid response."}), 502
        return jsonify({"response": answer.strip()})

    except requests.Timeout:
        app.logger.warning("OpenRouter request timed out")
        return jsonify({"error": "The AI provider timed out. Please try again."}), 504
    except requests.RequestException:
        # Do not expose provider response bodies, credentials, or internals.
        app.logger.exception("OpenRouter request failed")
        return jsonify({"error": "The AI provider request failed. Please try again."}), 502
    except (ValueError, TypeError, IndexError, AttributeError):
        app.logger.exception("OpenRouter returned malformed JSON")
        return jsonify({"error": "The AI provider returned an invalid response."}), 502


@app.errorhandler(413)
def request_too_large(_error):
    return jsonify({"error": "Request body is too large."}), 413


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
