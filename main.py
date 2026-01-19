from flask import Flask, request, jsonify, render_template, abort
from flask_cors import CORS
import shutil
import os
import uuid
from dotenv import load_dotenv

load_dotenv()

# --- Vision ---
from vision.inference import predict

# --- Pipeline ---
from pipeline.confidence_gate import confidence_gate

# --- RAG ---
from rag.retriever import retrieve_docs

# --- LLM ---
from llm.report_generator import generate_report
from llm.parser import parse_report
from llm.chatbot import chat_llm

# --- Chat memory ---
from chat.chat_memory import create_session, save_message, get_history

# -------------------------
# App Initialization
# -------------------------
app = Flask(
    __name__,
    template_folder="templates",
    static_folder="static"
)
CORS(app)

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# -------------------------
# GLOBAL STATE
# -------------------------
LATEST_REPORT = None
CURRENT_SESSION_ID = None

# -------------------------
# WEB UI ROUTES
# -------------------------
@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze_ui():
    global LATEST_REPORT, CURRENT_SESSION_ID

    if "file" not in request.files:
        abort(400, "No file uploaded")

    file = request.files["file"]

    if not file.filename.lower().endswith((".png", ".jpg", ".jpeg")):
        abort(400, "Invalid image format")

    file_id = str(uuid.uuid4())
    image_path = os.path.join(UPLOAD_DIR, f"{file_id}_{file.filename}")

    with open(image_path, "wb") as buffer:
        shutil.copyfileobj(file.stream, buffer)

    try:
        # 1️⃣ Vision
        probabilities = predict(image_path)

        # 2️⃣ Confidence gate
        facts = confidence_gate(probabilities)

        # SAFETY FALLBACK
        if not facts["positive_findings"] and not facts["indeterminate_findings"]:
            result = {
                "summary": "No confident abnormality detected.",
                "findings": ["No acute cardiopulmonary abnormality identified."],
                "impression": ["Normal chest radiograph."]
            }
        else:
            # 3️⃣ RAG
            rag_docs = retrieve_docs({
                "findings": facts["positive_findings"],
                "uncertain": facts["indeterminate_findings"],
                "normal": facts["normal"]
            })

            # 4️⃣ LLM
            raw_report = generate_report(facts, rag_docs)
            result = parse_report(raw_report)

        # ✅ STORE REPORT & CREATE CHAT SESSION
        LATEST_REPORT = result
        CURRENT_SESSION_ID = str(uuid.uuid4())
        create_session(CURRENT_SESSION_ID)

        return render_template(
            "result.html",
            result=result,
            facts=facts,
            probabilities=probabilities,
            session_id=CURRENT_SESSION_ID
        )

    finally:
        if os.path.exists(image_path):
            os.remove(image_path)

# -------------------------
# API ROUTES
# -------------------------
@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "message": "Radiology AI API running"
    })


@app.route("/generate-report", methods=["POST"])
def generate_report_api():
    global LATEST_REPORT, CURRENT_SESSION_ID

    if "file" not in request.files:
        abort(400, "No file uploaded")

    file = request.files["file"]

    if not file.filename.lower().endswith((".png", ".jpg", ".jpeg")):
        abort(400, "Invalid image format")

    file_id = str(uuid.uuid4())
    image_path = os.path.join(UPLOAD_DIR, f"{file_id}_{file.filename}")

    with open(image_path, "wb") as buffer:
        shutil.copyfileobj(file.stream, buffer)

    try:
        probabilities = predict(image_path)
        facts = confidence_gate(probabilities)

        if not facts["positive_findings"] and not facts["indeterminate_findings"]:
            report = {
                "summary": "No confident abnormality detected.",
                "findings": ["No acute cardiopulmonary abnormality identified."],
                "impression": ["Normal chest radiograph."]
            }
        else:
            rag_docs = retrieve_docs({
                "findings": facts["positive_findings"],
                "uncertain": facts["indeterminate_findings"],
                "normal": facts["normal"]
            })

            raw_report = generate_report(facts, rag_docs)
            report = parse_report(raw_report)

        # ✅ STORE REPORT & CREATE CHAT SESSION
        LATEST_REPORT = report
        CURRENT_SESSION_ID = str(uuid.uuid4())
        create_session(CURRENT_SESSION_ID)

        return jsonify({
            "probabilities": probabilities,
            "facts": facts,
            "report": report,
            "session_id": CURRENT_SESSION_ID
        })

    finally:
        if os.path.exists(image_path):
            os.remove(image_path)

# -------------------------
# CHATBOT ROUTE
# -------------------------
@app.route("/chat", methods=["POST"])
def chat_endpoint():
    global LATEST_REPORT

    if LATEST_REPORT is None:
        abort(400, "Generate report first")

    data = request.get_json()
    if not data:
        abort(400, "Invalid JSON")

    session_id = data.get("session_id")
    question = data.get("question")

    if not session_id or not question:
        abort(400, "session_id and question are required")

    # Save user message
    save_message(session_id, "user", question)

    history = get_history(session_id)
    history_text = "\n".join(
        [f"{m['role']}: {m['content']}" for m in history]
    )

    prompt = f"""
Radiology Report:
{LATEST_REPORT}

Conversation History:
{history_text}

User Question:
{question}

INSTRUCTIONS:
- Answer ONLY from the report
- If information is insufficient, say so
- Do NOT diagnose
- Do NOT suggest treatment
"""

    answer = chat_llm(prompt)

    # Save assistant reply
    save_message(session_id, "assistant", answer)

    return jsonify({"answer": answer})

# -------------------------
# Run Server
# -------------------------
if __name__ == "__main__":
    app.run(debug=True)
