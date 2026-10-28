# SignalShield — AI-Based Phishing Email Detection System

A student-friendly, local-first Streamlit application for triaging email text into **Likely Phishing**, **Suspicious**, or **Benign**. It combines a TF-IDF text model with transparent offline URL/sender checks, explains warning signals, records analyst decisions separately, and exports sanitized metadata. It is a decision-support demonstration—not a guarantee that any message is safe or malicious.

![SignalShield local architecture](architecture.png)

## Abstract

Phishing messages often use urgency, credential requests, financial prompts, and deceptive links. SignalShield demonstrates how a lightweight supervised machine-learning pipeline and explainable heuristic checks can support an analyst reviewing an email. The app accepts a subject, body, sender address, and visible URLs; classifies the text into three review-oriented categories; and presents model score, signals, URL/sender observations, uncertainty, and recommended next steps. It runs on a student laptop and makes no network requests to analyze messages or URLs.

## Problem statement and objectives

Manual email review is inconsistent and users may overlook urgency, credential requests, or unusual URL structure. The project aims to provide a reproducible baseline classifier, separate uncertain messages into a review category, explain observations in plain English, offer human review and local history, and protect input privacy by not storing raw email content.

## Existing-system limitations

Simple keyword filters overreact to ordinary language and can miss novel attacks. Blacklists require up-to-date feeds and may disclose submitted URLs to an external service. Generic machine-learning classifiers may not explain their output and can perform poorly when trained on unrepresentative data. SignalShield avoids external URL reputation lookups and uses its own three-class demonstration model, but it inherits the dataset and generalization limitations described below.

## Proposed system and architecture

1. Streamlit validates and accepts message text, sender information, and visible URLs.
2. Text preprocessing normalizes whitespace and detects common warning phrases.
3. URL/sender analysis parses strings only. It never resolves a domain, opens a link, downloads a file, or executes content.
4. A saved scikit-learn TF-IDF pipeline predicts one of three labels. The top class score is displayed as a **model score**, not as a calibrated probability or guarantee.
5. The result explains the classification, presents heuristic signals separately, and recommends safe verification actions.
6. SQLite stores only prediction metadata, sender domain (not full address), URL count, signals, and analyst review notes. It does not store subject, body, or raw URL values. Human feedback never triggers automatic retraining.
7. An explicit local training workflow deduplicates first, uses a stratified validation split to select between Logistic Regression and Random Forest, then reports both candidates on a separate untouched test split and saves the selected pipeline and evaluation report.

Diagram source: [`architecture.mmd`](architecture.mmd). The rendered diagram is [`architecture.png`](architecture.png).

## Project structure

```text
phishing-email-detector/
├── .env.example                    # No secrets required by the local demo
├── .gitignore
├── app.py                         # Streamlit UI and dashboard
├── architecture.mmd / .png
├── config.py                      # Paths, labels and input limits
├── pytest.ini
├── requirements.txt
├── README.md
├── data/
│   └── sample_synthetic_emails.csv # 60 labeled, invented demo examples
├── models/
│   ├── model.joblib                # Trained demo pipeline
│   └── metrics.json                # Actual synthetic holdout metrics
├── src/
│   ├── database.py
│   ├── predict.py
│   ├── preprocessing.py
│   ├── report_export.py
│   ├── train_model.py
│   └── url_analysis.py
└── tests/
    └── test_core.py
```

## Dataset and licensing

The included `sample_synthetic_emails.csv` contains **60 hand-written synthetic examples** (20 per class), invented for demonstration; URLs use reserved `.invalid` names or documentation IP ranges and do not refer to real targets. It is not a public corpus and carries no third-party dataset license. Do not interpret its metrics as production or real-world performance. For meaningful evaluation, supply a properly licensed, representative, independently collected labeled dataset, document its provenance and license, remove duplicates, and evaluate on a leakage-resistant holdout set. Training supports a CSV with `subject`, `body`, and `label`; `sender` and `urls` are optional. Labels must exactly be `Likely Phishing`, `Suspicious`, or `Benign`.

## Algorithms and features

- **Text:** word-level TF-IDF unigram/bigram features with sublinear term frequency.
- **Input redaction:** before TF-IDF vectorization, raw URLs, email-address strings, obvious inline credential values, and long numeric sequences are replaced; sender and separately supplied URL fields are analyzed outside the text model and excluded from its vocabulary.
- **Baselines:** class-weighted Logistic Regression and Random Forest. Both produce sklearn `predict_proba` scores; the displayed score is not calibrated and must not be interpreted as a true probability.
- **Selection and evaluation:** deduplication happens before stratified train/validation/test splits (60%/20%/20%, seed 42). Candidate models are selected by validation macro F1; both candidates are compared on the untouched test set using accuracy, macro precision, macro recall, macro F1, per-class report, and confusion matrix. The selected model is refit on all deduplicated rows after evaluation.
- **Heuristics:** urgency, credentials, OTPs, payment/sensitive-data requests, URL count, HTTPS use, IP-host URL, deep subdomains, suspicious structures, punycode, and sender-shape/domain observations. Signals are contextual indicators, never proof by themselves.
- **Uncertainty:** a three-class classifier gives a separate `Suspicious` class. Score display bands are defined in `src/predict.py` (≥75% higher, ≥50% moderate, otherwise low); they are display bands on uncalibrated scores, not policy thresholds or calibrated confidence.

## Measured demonstration results

The included artifact was generated from the synthetic dataset using seed 42, a stratified validation split for selection, and an untouched 20% test split. These are the actual run outputs for that synthetic split, **not real-world performance estimates**:

| Model | Accuracy | Macro precision | Macro recall | Macro F1 |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.917 | 0.933 | 0.917 | 0.915 |
| Random Forest | 0.833 | 0.889 | 0.833 | 0.838 |

Logistic Regression confusion matrix (actual rows, predicted columns; order: Likely Phishing, Suspicious, Benign):

| | Likely Phishing | Suspicious | Benign |
|---|---:|---:|---:|
| **Likely Phishing** | 3 | 1 | 0 |
| **Suspicious** | 0 | 4 | 0 |
| **Benign** | 0 | 0 | 4 |

The validation macro F1 scores were 1.000 for Logistic Regression and 0.915 for Random Forest, so Logistic Regression was selected. The test split has only 12 messages; these results primarily demonstrate that the evaluation pipeline runs, and high validation scores on tiny synthetic data are not meaningful evidence of efficacy. The persisted source of truth is `models/metrics.json`, and the app displays it with this caveat.

## Install and run

Requires Python 3.10 or newer. From this project directory:

```bash
python -m venv .venv
# macOS/Linux:
source .venv/bin/activate
# Windows PowerShell:
# .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m src.train_model            # optional: retrain included synthetic demo model
streamlit run app.py
```

Open the local URL printed by Streamlit. To test without changing the bundled model, keep the supplied artifact and start the app directly. Tests:

```bash
python -m pytest -q
```

## Using the application

- **Analyze email:** enter subject/body, optionally sender and visible URLs, and click **Analyze locally**. A displayed-link/target comparison is also available; it compares text only. Results show a category, model score and band, independent warning signals, URL/sender observations, and next-step guidance.
- **Dashboard & history:** inspect class counts, distribution, score averages, and metadata filters; download the sanitized CSV; delete local history when no longer needed.
- **Human review:** select an analysis, save a corrected label and optional note, and export reviewed records. Reviews do not retrain the model.
- **Model evaluation:** compare the two candidate baselines, read the confusion matrix and classification report, and review the dataset caveat.
- **Optional approved data:** in the sidebar, upload a labeled CSV (maximum 2 MB and 5,000 rows) and explicitly select **Validate and train local model**. This replaces `models/model.joblib` and `models/metrics.json`; use a backup or version-control checkpoint if you need to preserve the demo artifact. Unverified human review data is never included automatically.

Example result shape (illustrative; the prediction depends on the trained artifact):

```text
Category: Suspicious
Model score: 64% (Moderate model score; not a calibrated probability)
Signals: urgency language; one visible URL; link does not use HTTPS
Action: Verify through an independently obtained official contact method.
```

## Privacy, retention and safety

- Email inputs are processed locally by the Python app; no email content or URL is sent to a third-party service by this project. Do not paste live credentials, OTPs, or unnecessary sensitive information.
- The app does **not** persist analyzed subject/body or raw URLs in its history or prediction results. SQLite is created at `data/analysis_history.sqlite3` and stores timestamp, predicted class, model score, sender domain, URL count, warning-signal text, and optional review label/note. URL/email/obvious credential values in analyst notes are redacted, but notes should still not contain email text or personal details. The database is local and not encrypted by this demo; protect access to the laptop and filesystem.
- The sanitized CSV export excludes sender-domain detail and freeform review notes; it contains aggregate warning text and analyst labels/timestamps. Use **Delete all local analysis history** in the dashboard or remove `data/analysis_history.sqlite3` to clear history and review notes. Exported reports are separate files and must be deleted separately.
- CSV training uploads are size/row/label validated, written briefly to local disk only while training, and removed on success or failure. The serialized TF-IDF model necessarily retains a learned vocabulary from remaining training text (though URL/email strings and obvious inline credential values are redacted from model text). Use only synthetic/public/explicitly approved, appropriately sanitized data; never train on secrets or confidential messages. Retain or back up any model artifact you need before replacing it.
- Never click a suspicious link because of this app. For password, OTP, payment, or sensitive-data requests, independently contact the organization through an official number/site you already trust. Legitimate-looking messages can still be malicious; false positives and false negatives are expected.

## Tests and error handling

The test suite covers normalization and warning phrases, malformed/empty input, offline URL parsing, label validation, duplicate removal, CSV size validation, metadata-only database operations, reviews, deletion, and sanitized export. Each training class must have at least five unique examples for stratified splits. Bad model artifacts produce a friendly training instruction. CSV training is limited to 2 MB and 5,000 rows; individual text fields are limited to 20,000 characters; visible URL input is limited to 50 URLs with a maximum of 2,048 characters each. This is a local project baseline, not a hardened multi-user service.

## Limitations and future enhancements

The synthetic dataset is small and cannot represent language, culture, organization, and attack drift. Sender authentication headers (SPF, DKIM, DMARC) are not parsed; displayed-link checking only works when the analyst supplies both strings. URL checks are syntactic and do not provide reputation or redirect analysis. The model score is not calibrated; validation should include a larger, representative, temporally separated dataset, calibration analysis, subgroup/error analysis, and measured operational thresholds. Additional extensions could include a verified public dataset with provenance/licensing, improved local feature explanations, analyst roles, encrypted-at-rest storage, configurable retention, accessibility testing, and safe integration with a mail gateway.

## Optional deployment guidance

The app is designed for local execution. A Streamlit-hosted deployment may be possible, but it changes the privacy boundary and may have ephemeral storage or resource limits. Before deployment, verify the provider's current requirements, privacy/retention rules, and package constraints; do not upload confidential emails. Exclude local SQLite history and secrets from Git. The included `.gitignore` excludes local databases, virtual environments, and temporary uploads. No API keys or external-service credentials are required.

## College presentation demonstration (3–5 minutes)

1. Explain the objective and show the architecture diagram; point out that URLs are parsed as text and never visited.
2. In **Analyze email**, use an invented sample such as subject “Urgent: verify your account today” and a non-routable `example.invalid` URL. Call out the category, non-calibrated score, and distinct heuristic signals.
3. Show how to verify requests using an independently obtained official channel. Emphasize the model can be wrong.
4. Open **Dashboard & history** and **Human review** to demonstrate metadata-only persistence, correction, and sanitized export.
5. Open **Model evaluation** to compare both algorithms and explain that 60 synthetic examples and a small holdout only prove the pipeline works—not real-world efficacy.
6. Show the train command, unit-test command, and documented future work.
