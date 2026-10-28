from __future__ import annotations
import csv
import pandas as pd
import pytest
from src.preprocessing import normalize_text, detect_text_signals, compose_email_text
from src.url_analysis import analyze_url, analyze_urls, analyze_sender, compare_display_target, extract_urls
from src.train_model import load_dataset
from src.predict import predict_email
from src.database import init_db, save_analysis, get_history, save_review, delete_history
from src.report_export import sanitized_report, sanitize_training_csv
from io import BytesIO


def test_normalization_and_signals():
    assert normalize_text("\nHello\t  world  ") == "Hello world"
    hits = detect_text_signals("Urgent: send your password and OTP today")
    assert any("Urgency" in s for s in hits)
    assert any("Credential" in s for s in hits)
    assert any("OTP" in s for s in hits)

def test_empty_malformed_and_urls_are_text_only():
    assert extract_urls("Go to https://example.invalid/x.") == ["https://example.invalid/x"]
    assert analyze_url("http://203.0.113.4/login")["ip_host"] is True
    assert "Link does not use HTTPS" in analyze_url("http://example.invalid")["signals"]
    assert analyze_sender("not-an-email")["valid_shape"] is False
    assert compare_display_target("bank.example", "https://other.example/path")
    assert analyze_urls([])["count"] == 0
    with pytest.raises(ValueError): predict_email("", " ", bundle={})

def test_dataset_label_validation_and_deduplication(tmp_path):
    p = tmp_path / "input.csv"
    rows = [{"subject":"A","body":"x","label":"Benign"}, {"subject":"A","body":"x","label":"Benign"}]
    rows += [{"subject":f"b{i}","body":f"example b {i}","label":"Benign"} for i in range(4)]
    rows += [{"subject":f"s{i}","body":f"example s {i}","label":"Suspicious"} for i in range(5)]
    rows += [{"subject":f"p{i}","body":f"example p {i}","label":"Likely Phishing"} for i in range(5)]
    pd.DataFrame(rows).to_csv(p,index=False)
    df=load_dataset(p)
    assert len(df)==15
    bad=tmp_path/"bad.csv"
    pd.DataFrame([{"subject":"x","body":"y","label":"danger"}]).to_csv(bad,index=False)
    with pytest.raises(ValueError, match="Invalid labels"): load_dataset(bad)
    missing=tmp_path/"missing.csv"
    pd.DataFrame([{"subject":"x","body":"y"}]).to_csv(missing,index=False)
    with pytest.raises(ValueError, match="missing required"): load_dataset(missing)

def test_csv_upload_limits():
    f=BytesIO(b"subject,body,label\na,b,Benign\n")
    assert len(sanitize_training_csv(f, 1000, 10))==1
    with pytest.raises(ValueError, match="exceeds"): sanitize_training_csv(BytesIO(b"1234567890"),5,10)

def test_database_review_and_no_content_storage(tmp_path):
    path=tmp_path/"history.sqlite"
    init_db(path)
    result={"category":"Suspicious","score":.66,"sender":{"domain":"private.example"},"urls":{"count":1},"signals":["Urgency or deadline language"]}
    rid=save_analysis(result,path)
    rows=get_history(path)
    assert len(rows)==1 and "private.example" == rows[0]["sender_domain"]
    assert not any("raw body" in str(v) for v in rows[0].values())
    save_review(rid,"Benign","Reviewed; contact a.person@example.invalid at https://private.invalid",path)
    saved=get_history(path)[0]
    assert saved["reviewed_category"]=="Benign"
    assert "a.person@example.invalid" not in saved["review_note"] and "private.invalid" not in saved["review_note"]
    assert delete_history(path)==1

def test_sanitized_export_omits_sender_domain_and_raw_fields():
    output=sanitized_report([{"id":1,"category":"Benign","score":.9,"sender_domain":"private.example","body":"secret","url":"https://private.invalid","review_note":"=HYPERLINK(\"https://private.invalid\")"}]).decode()
    assert "private.example" not in output and "secret" not in output and "private.invalid" not in output and "HYPERLINK" not in output
    assert "category" in output


def test_model_loading_failure_and_prediction(tmp_path):
    from src.predict import load_bundle, ModelUnavailable
    with pytest.raises(ModelUnavailable): load_bundle(tmp_path / "missing.joblib")
    from config import ARTIFACT_PATH
    if ARTIFACT_PATH.exists():
        result = predict_email("Urgent account verification", "Please verify your password today.", bundle=load_bundle(ARTIFACT_PATH))
        assert result["category"] in {"Likely Phishing", "Suspicious", "Benign"}
        assert 0 <= result["score"] <= 1
        assert result["signals"]

def test_model_text_redacts_identifiers_and_url_values():
    text=compose_email_text("Contact a.person@example.com", "Password: Secret123 and visit https://private.example/path")
    assert "a.person@example.com" not in text
    assert "private.example" not in text
    assert "secret123" not in text.lower()
    details=analyze_url("https://private.example/path")
    assert "url" not in details


def test_conflicting_duplicate_labels_rejected(tmp_path):
    records=[]
    for label in ("Likely Phishing", "Suspicious", "Benign"):
        for i in range(5): records.append({"subject":f"{label} {i}","body":f"unique body {label} {i}","label":label})
    records.append({"subject":"Likely Phishing 0","body":"unique body Likely Phishing 0","sender":"different@example.invalid","label":"Benign"})
    p=tmp_path/"conflict.csv"; pd.DataFrame(records).to_csv(p,index=False)
    with pytest.raises(ValueError, match="conflicting labels"): load_dataset(p)


def test_url_input_limits():
    with pytest.raises(ValueError, match="Too many visible URLs"):
        predict_email("subject", "body", urls=["https://example.invalid"] * 51, bundle={})
    with pytest.raises(ValueError, match="URL is too long"):
        predict_email("subject", "body", urls=["https://x.invalid/" + "a" * 2100], bundle={})
