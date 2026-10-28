"""Local prediction using a saved sklearn pipeline; no external calls."""
from __future__ import annotations
from pathlib import Path
import joblib
from .preprocessing import compose_email_text, detect_text_signals
from .url_analysis import analyze_urls, analyze_sender, extract_urls
from config import ARTIFACT_PATH, LABELS, MAX_EMAIL_CHARS, MAX_URLS, MAX_URL_CHARS

class ModelUnavailable(RuntimeError): pass

def load_bundle(path: str | Path = ARTIFACT_PATH) -> dict:
    try:
        bundle = joblib.load(path)
        if not isinstance(bundle, dict) or "model" not in bundle: raise ValueError("Unexpected model artifact format")
        return bundle
    except Exception as exc:
        raise ModelUnavailable(f"Could not load local model: {exc}. Train it with `python -m src.train_model`.") from exc

def predict_email(subject: str, body: str, sender: str = "", urls: list[str] | None = None, bundle: dict | None = None) -> dict:
    subject, body, sender = subject or "", body or "", sender or ""
    if not subject.strip() and not body.strip(): raise ValueError("Enter an email subject or body before analyzing.")
    if any(len(v) > MAX_EMAIL_CHARS for v in (subject, body, sender)):
        raise ValueError(f"Input is too long. Each text field must be at most {MAX_EMAIL_CHARS:,} characters.")
    bundle = bundle or load_bundle()
    urls = list(urls or [])
    urls += [u for u in extract_urls(subject + " " + body) if u not in urls]
    if len(urls) > MAX_URLS: raise ValueError(f"Too many visible URLs. Limit: {MAX_URLS}.")
    if any(len(u) > MAX_URL_CHARS for u in urls): raise ValueError(f"A URL is too long. Limit: {MAX_URL_CHARS} characters per URL.")
    model_text = compose_email_text(subject, body, sender, urls)
    model = bundle["model"]
    try:
        predicted = str(model.predict([model_text])[0])
        scores = model.predict_proba([model_text])[0]
        classes = list(model.classes_)
        score_map = {str(c): float(s) for c, s in zip(classes, scores)}
        score = score_map.get(predicted, max(score_map.values(), default=0.0))
    except Exception as exc:
        raise ModelUnavailable(f"Prediction failed: {exc}") from exc
    combined = subject + " " + body
    text_signals = detect_text_signals(combined)
    url_result = analyze_urls(urls)
    sender_result = analyze_sender(sender)
    signals = list(dict.fromkeys(text_signals + url_result["signals"] + sender_result["signals"]))
    if not signals: signals = ["No common heuristic warning patterns were detected; this does not prove the email is safe."]
    if score >= .75: band = "Higher model score"
    elif score >= .5: band = "Moderate model score"
    else: band = "Low model score"
    action = {
        "Likely Phishing": "Do not click links, open attachments, reply, or provide information. Report it through your organization’s approved channel.",
        "Suspicious": "Pause and verify the request using an independently obtained official contact method. Do not use contact details or links in the message.",
        "Benign": "No strong phishing classification was produced, but remain cautious. Verify unexpected sensitive requests independently.",
    }.get(predicted, "Review the message carefully and verify sensitive requests independently.")
    explanation = f"The local model classified this message as {predicted.lower()} with a {band.lower()} ({score:.0%} model score). This score is not a guarantee or a calibrated probability. "
    if signals: explanation += "Signals below are indicators for review, not proof."
    return {"category": predicted if predicted in LABELS else "Suspicious", "score": score, "score_band": band, "signals": signals, "sender": sender_result, "urls": url_result, "explanation": explanation, "recommended_action": action}
