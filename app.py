import hashlib
import json
import math
import os
import re
import uuid
from collections import Counter
from datetime import datetime, timezone

from dotenv import load_dotenv
from flask import Flask, jsonify, request
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from groq import Groq

load_dotenv()

app = Flask(__name__)

limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=["100 per hour"],
    storage_uri="memory://",
)

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

submissions = {}
appeals = {}
audit_log = []


# ---------------------------------------------------------------------------
# Signal 1: Statistical Text Analysis
# ---------------------------------------------------------------------------

AI_FILLER_PHRASES = [
    "it is important to", "it should be noted", "in conclusion",
    "furthermore", "moreover", "additionally", "in order to",
    "it is worth", "as a whole", "in today's", "in the realm of",
    "plays a crucial role", "it is essential", "a wide range of",
    "in various", "on the other hand", "taken as a whole",
    "increasingly important", "several key", "is important to note",
]


def analyze_statistical(text):
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    words = re.findall(r'\b\w+\b', text.lower())

    if len(words) < 10 or len(sentences) < 2:
        return {"score": 0.5, "detail": "Text too short for reliable statistical analysis"}

    lengths = [len(re.findall(r'\b\w+\b', s)) for s in sentences]
    mean_len = sum(lengths) / len(lengths)
    variance = sum((l - mean_len) ** 2 for l in lengths) / len(lengths)
    std_dev = math.sqrt(variance)
    cv = std_dev / mean_len if mean_len > 0 else 0
    burstiness_score = max(0.0, min(1.0, 1.0 - (cv / 0.8)))

    text_lower = text.lower()
    filler_hits = sum(1 for phrase in AI_FILLER_PHRASES if phrase in text_lower)
    filler_density = filler_hits / len(sentences)
    filler_score = max(0.0, min(1.0, filler_density / 1.5))

    avg_sentence_len = mean_len
    length_uniformity = 1.0 - min(1.0, cv)
    starts = [s.strip().split()[0].lower() if s.strip().split() else "" for s in sentences]
    unique_starts = len(set(starts)) / len(starts) if starts else 1.0
    structural_score = max(0.0, min(1.0,
        (length_uniformity * 0.5) + ((1.0 - unique_starts) * 0.5)
    ))

    combined = (burstiness_score * 0.35 + filler_score * 0.35 + structural_score * 0.3)

    return {
        "score": round(combined, 4),
        "detail": {
            "burstiness": round(burstiness_score, 4),
            "filler_phrases": round(filler_score, 4),
            "structural_patterns": round(structural_score, 4),
        },
    }


# ---------------------------------------------------------------------------
# Signal 2: LLM-Based Classification
# ---------------------------------------------------------------------------

def analyze_llm(text):
    prompt = (
        "You are an expert AI-generated text detector. Your job is to score how likely "
        "the following text was written by an AI language model (like ChatGPT, Claude, etc.) "
        "versus a human.\n\n"
        "SCORING GUIDE — use the full range:\n"
        "- 0.0-0.2: Clearly human. Messy, informal, has typos/slang, personal voice, "
        "emotional rawness, irregular structure.\n"
        "- 0.3-0.4: Probably human. Well-written but has human quirks — unexpected word "
        "choices, uneven pacing, genuine perspective.\n"
        "- 0.5: Truly uncertain. Could go either way.\n"
        "- 0.6-0.7: Probably AI. Suspiciously polished, uses hedging language "
        "(\"it is important to note\"), balanced both-sides framing, generic examples.\n"
        "- 0.8-1.0: Clearly AI. Formulaic transitions (Furthermore/Moreover/Additionally), "
        "no personal voice, reads like a template, uniform sentence structure.\n\n"
        "STRONG AI INDICATORS: \"It is important to note\", \"Furthermore\", \"Moreover\", "
        "\"In conclusion\", \"a wide range of\", \"various sectors\", parallel list structure, "
        "lack of specific personal details, absence of contractions.\n\n"
        "STRONG HUMAN INDICATORS: Contractions, slang, sentence fragments, emotional "
        "language, specific personal anecdotes, irregular punctuation, humor/sarcasm.\n\n"
        "Respond with ONLY valid JSON (no markdown, no extra text):\n"
        '{"score": <float 0.0-1.0>, "rationale": "<one sentence>"}\n\n'
        f"Text to analyze:\n\"\"\"\n{text[:3000]}\n\"\"\""
    )

    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=200,
        )
        raw = response.choices[0].message.content.strip()
        # Strip markdown fences if present
        if raw.startswith("```"):
            raw = re.sub(r'^```\w*\n?', '', raw)
            raw = re.sub(r'\n?```$', '', raw)
        result = json.loads(raw)
        score = float(result.get("score", 0.5))
        score = max(0.0, min(1.0, score))
        return {"score": round(score, 4), "rationale": result.get("rationale", "")}
    except Exception as e:
        return {"score": 0.5, "rationale": f"LLM analysis unavailable: {str(e)}"}


# ---------------------------------------------------------------------------
# Confidence Scoring
# ---------------------------------------------------------------------------

STATISTICAL_WEIGHT = 0.4
LLM_WEIGHT = 0.6

def compute_confidence(stat_score, llm_score):
    weighted = stat_score * STATISTICAL_WEIGHT + llm_score * LLM_WEIGHT

    disagreement = abs(stat_score - llm_score)
    if disagreement > 0.3:
        pull_factor = min(1.0, (disagreement - 0.3) / 0.4)
        weighted = weighted + (0.5 - weighted) * pull_factor * 0.5

    return round(max(0.0, min(1.0, weighted)), 4)


# ---------------------------------------------------------------------------
# Transparency Labels
# ---------------------------------------------------------------------------

def generate_label(score):
    if score >= 0.75:
        return {
            "category": "ai_generated",
            "text": (
                f"This content has been flagged as likely AI-generated. "
                f"Our analysis detected patterns consistent with AI writing tools. "
                f"Confidence: {round(score * 100)}%. "
                f"If you believe this is incorrect, you can submit an appeal."
            ),
        }
    elif score <= 0.25:
        return {
            "category": "human_written",
            "text": (
                f"This content appears to be human-written. "
                f"Our analysis found patterns consistent with original human authorship. "
                f"Confidence: {round((1 - score) * 100)}%."
            ),
        }
    else:
        return {
            "category": "uncertain",
            "text": (
                "We couldn't determine with confidence whether this content is "
                "AI-generated or human-written. This may reflect mixed authorship, "
                "an unusual writing style, or heavy editing. "
                "The creator can provide additional context through an appeal."
            ),
        }


# ---------------------------------------------------------------------------
# Audit Logging
# ---------------------------------------------------------------------------

def log_event(event_type, data):
    entry = {
        "id": str(uuid.uuid4()),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "event_type": event_type,
        **data,
    }
    audit_log.append(entry)
    return entry


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/submit", methods=["POST"])
@limiter.limit("10 per minute;50 per hour")
def submit():
    data = request.get_json()
    if not data:
        return jsonify({"error": "Missing JSON body"}), 400

    content = data.get("text") or data.get("content")
    if not content:
        return jsonify({"error": "Missing required field: text"}), 400

    creator_id = data.get("creator_id", "anonymous")

    if len(content.strip()) < 20:
        return jsonify({"error": "Content must be at least 20 characters"}), 400

    if len(content) > 50000:
        return jsonify({"error": "Content must be under 50,000 characters"}), 400

    stat_result = analyze_statistical(content)
    llm_result = analyze_llm(content)

    confidence = compute_confidence(stat_result["score"], llm_result["score"])
    label = generate_label(confidence)

    content_id = str(uuid.uuid4())
    content_hash = hashlib.sha256(content.encode()).hexdigest()[:16]

    submission = {
        "content_id": content_id,
        "creator_id": creator_id,
        "content_hash": content_hash,
        "content_preview": content[:100] + ("..." if len(content) > 100 else ""),
        "attribution": label["category"],
        "confidence_score": confidence,
        "signals": {
            "statistical": stat_result,
            "llm": llm_result,
        },
        "label": label["text"],
        "status": "classified",
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    submissions[content_id] = submission

    log_event("classification", {
        "content_id": content_id,
        "creator_id": creator_id,
        "content_hash": content_hash,
        "statistical_score": stat_result["score"],
        "llm_score": llm_result["score"],
        "combined_score": confidence,
        "attribution": label["category"],
        "label_text": label["text"],
    })

    return jsonify({
        "content_id": content_id,
        "attribution": label["category"],
        "confidence_score": confidence,
        "label": label["text"],
        "signals": {
            "statistical": stat_result["score"],
            "llm": llm_result["score"],
        },
    }), 200


@app.route("/appeal", methods=["POST"])
@limiter.limit("5 per hour")
def appeal():
    data = request.get_json()
    if not data or "content_id" not in data:
        return jsonify({"error": "Missing required field: content_id"}), 400

    content_id = data["content_id"]
    if content_id not in submissions:
        return jsonify({"error": "Submission not found"}), 404

    reasoning = data.get("creator_reasoning") or data.get("reason")
    if not reasoning:
        return jsonify({"error": "Missing required field: creator_reasoning"}), 400

    submission = submissions[content_id]

    if submission.get("status") == "under_review":
        return jsonify({"error": "An appeal is already under review for this submission"}), 409

    appeal_id = str(uuid.uuid4())
    appeal_record = {
        "appeal_id": appeal_id,
        "content_id": content_id,
        "creator_id": submission["creator_id"],
        "appeal_reasoning": reasoning,
        "original_attribution": submission["attribution"],
        "original_confidence": submission["confidence_score"],
        "status": "under_review",
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    appeals[appeal_id] = appeal_record
    submissions[content_id]["status"] = "under_review"

    log_event("appeal_submitted", {
        "appeal_id": appeal_id,
        "content_id": content_id,
        "creator_id": submission["creator_id"],
        "appeal_reasoning": reasoning,
        "original_attribution": submission["attribution"],
        "original_confidence": submission["confidence_score"],
        "status": "under_review",
    })

    return jsonify({
        "appeal_id": appeal_id,
        "content_id": content_id,
        "status": "under_review",
        "message": (
            "Your appeal has been received and the content's classification "
            "has been placed under review. A moderator will evaluate your "
            "submission with the additional context you provided."
        ),
    }), 200


@app.route("/status/<content_id>", methods=["GET"])
def status(content_id):
    if content_id not in submissions:
        return jsonify({"error": "Submission not found"}), 404

    submission = submissions[content_id]
    related_appeals = [a for a in appeals.values() if a["content_id"] == content_id]

    return jsonify({
        "content_id": content_id,
        "attribution": submission["attribution"],
        "confidence_score": submission["confidence_score"],
        "status": submission["status"],
        "label": submission["label"],
        "appeals": related_appeals,
    }), 200


@app.route("/log", methods=["GET"])
def get_log():
    return jsonify({"entries": audit_log, "total": len(audit_log)}), 200


@app.route("/", methods=["GET"])
def index():
    return jsonify({
        "service": "Provenance Guard",
        "description": "AI content attribution and transparency system",
        "endpoints": {
            "POST /submit": "Submit content for attribution analysis",
            "POST /appeal": "Appeal a classification decision",
            "GET /status/<content_id>": "Check submission status",
            "GET /log": "View audit log",
        },
    })


if __name__ == "__main__":
    app.run(debug=True, port=5001)
