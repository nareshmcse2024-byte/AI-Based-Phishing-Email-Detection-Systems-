"""Privacy-preserving text helpers for email triage."""
from __future__ import annotations
import re
from .url_analysis import extract_urls

URGENCY_PATTERNS = [
    ("Urgency or deadline language", re.compile(r"\b(?:urgent|immediately|act now|expires today|within 24 hours|final notice|suspended|will be closed)\b", re.I)),
    ("Credential or account access request", re.compile(r"\b(?:password|sign in|log in|login|credentials|verify your account|account verification)\b", re.I)),
    ("OTP or authentication-code request", re.compile(r"\b(?:otp|one[- ]time code|verification code|authentication code|security code)\b", re.I)),
    ("Payment or financial request", re.compile(r"\b(?:payment|wire transfer|bank details|gift card|invoice|refund|crypto(?:currency)?)\b", re.I)),
    ("Sensitive information request", re.compile(r"\b(?:social security|ssn|date of birth|credit card|personal information|identity document)\b", re.I)),
    ("Threat or unusual consequence language", re.compile(r"\b(?:legal action|penalty|lose access|deactivated|unauthorized activity)\b", re.I)),
]

def normalize_text(text: str | None) -> str:
    text = (text or "").replace("\x00", " ")
    return re.sub(r"\s+", " ", text).strip()

def detect_text_signals(text: str) -> list[str]:
    normalized = normalize_text(text)
    found = [label for label, pattern in URGENCY_PATTERNS if pattern.search(normalized)]
    urls = extract_urls(normalized)
    if urls: found.append(f"Contains {len(urls)} visible URL(s)")
    return found

def redact_for_model(text: str) -> str:
    """Reduce memorization of identifiers/secrets before they enter the TF-IDF vocabulary."""
    text = normalize_text(text)
    text = re.sub(r"(?i)\b(?:https?://|www\.)[^\s<>\"']+", " urltoken ", text)
    text = re.sub(r"(?i)\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", " emailtoken ", text)
    text = re.sub(r"(?i)\b(password|passcode|otp|one[- ]time code|verification code|security code)\s*(?:is|:|=)\s*[^\s,;]+", r"\1 redactedvalue", text)
    text = re.sub(r"\b\d{4,}\b", " numbertoken ", text)
    return re.sub(r"\s+", " ", text).strip()

def compose_email_text(subject: str, body: str, sender: str = "", urls: list[str] | None = None) -> str:
    """Build text-only model input; identifiers and raw URLs are not features or persisted vocabulary."""
    return f"subject: {redact_for_model(subject)}\nbody: {redact_for_model(body)}"
