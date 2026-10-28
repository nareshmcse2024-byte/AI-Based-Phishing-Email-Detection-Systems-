"""Offline-only URL and sender text inspection; nothing is fetched or resolved."""
from __future__ import annotations
import ipaddress
import re
from urllib.parse import urlsplit

URL_RE = re.compile(r"(?i)\b(?:https?://|www\.)[^\s<>\"']+")

def extract_urls(text: str) -> list[str]:
    return [m.group(0).rstrip(".,;:!?)]}>") for m in URL_RE.finditer(text or "")]

def analyze_url(url: str) -> dict:
    raw = (url or "").strip()
    candidate = raw if "://" in raw else "http://" + raw
    try:
        parsed = urlsplit(candidate)
        host = (parsed.hostname or "").lower().rstrip(".")
        if not host:
            raise ValueError("missing host")
        try:
            ipaddress.ip_address(host)
            ip_host = True
        except ValueError:
            ip_host = False
        labels = [x for x in host.split(".") if x]
        path = parsed.path or ""
        query = parsed.query or ""
        signals = []
        if parsed.scheme.lower() != "https": signals.append("Link does not use HTTPS")
        if ip_host: signals.append("Host is an IP address rather than a domain name")
        if len(labels) >= 5: signals.append("Unusually deep subdomain structure")
        if "@" in parsed.netloc: signals.append("Contains @ in the URL authority")
        if len(raw) > 100: signals.append("Unusually long URL")
        if re.search(r"%[0-9a-f]{2}", raw, re.I): signals.append("Contains percent-encoded characters")
        if re.search(r"(?:login|verify|secure|account|update|password|payment)", raw, re.I):
            signals.append("Path or query contains account-action terms")
        if host.startswith("xn--") or ".xn--" in host: signals.append("Internationalized/punycode hostname")
        return {"host": host, "scheme": parsed.scheme.lower(), "https": parsed.scheme.lower() == "https", "ip_host": ip_host, "subdomain_count": max(0, len(labels)-2), "length": len(raw), "signals": signals}
    except Exception:
        return {"host": "unparsed", "scheme": "", "https": False, "ip_host": False, "subdomain_count": 0, "length": len(raw), "signals": ["URL could not be parsed"]}

def analyze_urls(urls: list[str]) -> dict:
    rows = [analyze_url(u) for u in urls if u.strip()]
    return {"count": len(rows), "https_count": sum(r["https"] for r in rows), "ip_host_count": sum(r["ip_host"] for r in rows), "details": rows, "signals": list(dict.fromkeys(s for r in rows for s in r["signals"]))}

def analyze_sender(sender: str) -> dict:
    value = (sender or "").strip().lower()
    signals = []
    if not value: return {"domain": "not provided", "valid_shape": False, "signals": ["Sender address not provided"]}
    if "@" not in value or value.count("@") != 1:
        return {"domain": "unparsed", "valid_shape": False, "signals": ["Sender address has an unexpected format"]}
    local, domain = value.rsplit("@", 1)
    if not local or "." not in domain or " " in value:
        signals.append("Sender address appears malformed")
    if domain in {"gmail.com", "yahoo.com", "outlook.com", "hotmail.com", "icloud.com"}:
        signals.append("Sender uses a public email provider; this alone is not evidence of phishing")
    if re.search(r"\d{5,}", local): signals.append("Sender local part contains a long digit sequence")
    if domain.startswith("xn--") or ".xn--" in domain: signals.append("Sender domain uses punycode")
    return {"domain": domain, "valid_shape": not bool(re.search(r"\s", value)) and "." in domain, "signals": signals}

def compare_display_target(display_text: str, target_url: str) -> list[str]:
    """Only compares text; it never follows the target."""
    display = (display_text or "").strip().lower().removeprefix("www.")
    target = analyze_url(target_url)["host"].removeprefix("www.")
    if not display or target == "unparsed": return []
    # Compare when display text itself looks like a hostname/domain.
    if re.fullmatch(r"[a-z0-9.-]+\.[a-z]{2,}", display) and display != target:
        return ["Displayed link text/domain differs from the actual target host"]
    return []
