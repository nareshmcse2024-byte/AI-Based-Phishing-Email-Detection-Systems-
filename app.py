from __future__ import annotations
import json
import pandas as pd
import streamlit as st
from config import APP_TITLE, ARTIFACT_PATH, METRICS_PATH, SYNTHETIC_DATA_PATH, DATABASE_PATH, LABELS, MAX_CSV_ROWS, MAX_UPLOAD_BYTES, MAX_EMAIL_CHARS, MAX_URLS, MAX_URL_CHARS
from src.predict import predict_email, load_bundle, ModelUnavailable
from src.url_analysis import compare_display_target
from src.database import init_db, save_analysis, get_history, save_review, delete_history
from src.report_export import sanitized_report, sanitize_training_csv
from src.train_model import load_dataset, train_and_save

st.set_page_config(page_title=APP_TITLE, page_icon="🛡", layout="wide")
st.markdown("""<style>
.block-container {max-width: 1300px; padding-top: 1.4rem;}
div[data-testid="stMetric"] {background: #f3f6fa; padding: 14px; border-radius: 12px; border: 1px solid #e3eaf2;}
.signal {padding: .55rem .8rem; margin: .35rem 0; border-radius: 8px; background: #fff5e6; border-left: 4px solid #e9a23b;}
.small-note {color: #536273; font-size: .92rem;}
</style>""", unsafe_allow_html=True)
init_db()

@st.cache_resource
def get_bundle(path: str):
    return load_bundle(path)

@st.cache_data
def read_metrics(path: str):
    try: return json.loads(open(path, encoding="utf-8").read())
    except Exception: return None

def load_dataset_from_frame(df):
    # Reuse identical label validation and deduplication rules with an in-memory CSV round-trip.
    import io
    return load_dataset(io.BytesIO(df.to_csv(index=False).encode("utf-8")))

st.title("SignalShield")
st.caption("Local AI-assisted phishing email triage · offline URL inspection · analyst review")
st.warning("This is a decision-support demo, not a security guarantee. Messages are processed locally. Analysis history stores prediction metadata only—not email subject/body, full sender address, or raw URLs. Synthetic demo metrics do not predict real-world performance.")

with st.sidebar:
    st.header("Model & privacy")
    st.write("Model artifact:", "Ready" if ARTIFACT_PATH.exists() else "Not trained")
    st.caption("No email or URL is sent to an external service. Never open suspicious links from this app.")
    with st.expander("Train using an approved labeled CSV"):
        st.caption("Training replaces the local model. Use only public, synthetic, or explicitly approved samples. Expected columns: subject, body, label; optional sender, urls. Labels: Likely Phishing, Suspicious, Benign.")
        training_file = st.file_uploader("Approved labeled CSV", type=["csv"], key="training_csv")
        if st.button("Validate and train local model", disabled=training_file is None, use_container_width=True):
            temp_path = SYNTHETIC_DATA_PATH.parent / "uploaded_training_local.csv"
            try:
                df = sanitize_training_csv(training_file, MAX_UPLOAD_BYTES, MAX_CSV_ROWS)
                checked = load_dataset_from_frame(df)
                checked.to_csv(temp_path, index=False)
                metrics = train_and_save(temp_path, dataset_type="user-provided / source not verified")
                get_bundle.clear(); read_metrics.clear()
                st.success(f"Training finished. Selected: {metrics['selected_model']}. Holdout metrics are dataset-specific; review the Model tab.")
            except Exception as exc:
                st.error(f"Training failed: {exc}")
            finally:
                temp_path.unlink(missing_ok=True)

try:
    bundle = get_bundle(str(ARTIFACT_PATH))
    model_error = None
except ModelUnavailable as exc:
    bundle, model_error = None, str(exc)
if bundle is None:
    st.error(model_error or "Model is not available.")
    if st.button("Train the bundled synthetic demo model"):
        try:
            train_and_save()
            get_bundle.clear(); read_metrics.clear()
            st.rerun()
        except Exception as exc: st.exception(exc)


def render_record(result: dict):
    color = {"Likely Phishing": "#b42318", "Suspicious": "#b54708", "Benign": "#027a48"}.get(result["category"], "#475467")
    st.markdown(f"### Prediction: <span style='color:{color}'>{result['category']}</span>", unsafe_allow_html=True)
    a, b, c = st.columns(3)
    a.metric("Model score", f"{result['score']:.0%}")
    b.metric("Score band", result["score_band"])
    c.metric("Visible URLs", result["urls"]["count"])
    st.caption(result["explanation"])
    st.markdown("**Signals to review**")
    for signal in result["signals"]: st.warning(signal)
    with st.expander("Sender and URL observations"):
        st.write("Sender domain:", result["sender"]["domain"])
        if result["urls"]["details"]:
            st.dataframe(pd.DataFrame([{k: v for k, v in d.items() if k != "url"} for d in result["urls"]["details"]]), use_container_width=True, hide_index=True)
        else: st.write("No URLs detected or entered.")
    st.info("Recommended next step: " + result["recommended_action"])


tab_analyze, tab_dashboard, tab_review, tab_model = st.tabs(["Analyze email", "Dashboard & history", "Human review", "Model evaluation"])
with tab_analyze:
    st.subheader("Analyze a message")
    st.caption("Inputs stay in this session and are not stored. Submitted text is used only to create an on-device prediction.")
    with st.form("email_analysis"):
        left, right = st.columns([2, 1])
        with left:
            subject = st.text_input("Email subject", max_chars=MAX_EMAIL_CHARS)
            body = st.text_area("Email body", height=220, max_chars=MAX_EMAIL_CHARS)
        with right:
            sender = st.text_input("Sender email address (optional)", max_chars=320)
            urls_raw = st.text_area("Visible URLs (one per line, optional)", height=120, max_chars=MAX_URLS * MAX_URL_CHARS)
            display_domain = st.text_input("Displayed link text/domain (optional)", max_chars=MAX_URL_CHARS)
            target_for_compare = st.text_input("Actual target URL to compare (optional)", max_chars=MAX_URL_CHARS)
        submit = st.form_submit_button("Analyze locally", type="primary", use_container_width=True)
    if submit:
        if bundle is None: st.error("Train or load a model before analyzing.")
        elif not subject.strip() and not body.strip(): st.error("Enter an email subject or body.")
        else:
            try:
                urls = [line.strip() for line in urls_raw.splitlines() if line.strip()]
                result = predict_email(subject, body, sender, urls, bundle)
                comparison = compare_display_target(display_domain, target_for_compare)
                result["signals"] = list(dict.fromkeys(result["signals"] + comparison))
                record_id = save_analysis(result)
                st.session_state["latest_result"] = result
                st.session_state["latest_record_id"] = record_id
            except Exception as exc: st.error(f"Could not analyze message: {exc}")
    if st.session_state.get("latest_result"):
        render_record(st.session_state["latest_result"])

with tab_dashboard:
    st.subheader("Analysis overview")
    records = get_history()
    df = pd.DataFrame(records)
    if df.empty:
        st.info("No analyses yet. Analyze a message to populate this dashboard.")
    else:
        cls = st.multiselect("Filter by automatic class", list(LABELS), default=list(LABELS))
        min_score, max_score = st.slider("Filter by model score", 0.0, 1.0, (0.0, 1.0), 0.01)
        filtered = df[df.category.isin(cls) & df.score.between(min_score, max_score)]
        counts = df.category.value_counts()
        k1,k2,k3,k4 = st.columns(4)
        k1.metric("Total analyzed", len(df)); k2.metric("Likely Phishing", int(counts.get("Likely Phishing", 0))); k3.metric("Suspicious", int(counts.get("Suspicious", 0))); k4.metric("Benign", int(counts.get("Benign", 0)))
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Class distribution**")
            st.bar_chart(counts.reindex(LABELS, fill_value=0))
        with col2:
            st.markdown("**Scores by class**")
            st.bar_chart(filtered.groupby("category").score.mean().reindex(LABELS).dropna())
        st.markdown("**Filtered analysis history**")
        show_cols = [c for c in ["id", "created_at", "category", "score", "sender_domain", "url_count", "reviewed_category", "reviewed_at"] if c in filtered]
        st.dataframe(filtered[show_cols], use_container_width=True, hide_index=True)
        st.download_button("Download sanitized CSV", sanitized_report(filtered.to_dict("records")), "signalshield_sanitized_report.csv", "text/csv")
        st.caption("Export omits sender-domain details, freeform review notes, and any raw message or URL content. Stored rows can be deleted below.")
        if st.button("Delete all local analysis history", type="secondary"):
            count = delete_history(); st.success(f"Deleted {count} local records."); st.rerun()

with tab_review:
    st.subheader("Analyst review")
    records = get_history()
    if not records: st.info("No records to review yet.")
    else:
        options = {f"#{r['id']} · {r['created_at']} · predicted {r['category']} · {r['score']:.0%}": r for r in records}
        selected_label = st.selectbox("Choose an analysis record", list(options))
        selected = options[selected_label]
        st.write("Automatic signals:", selected["signals"] or "None recorded")
        if selected["reviewed_category"]: st.success(f"Previously reviewed as {selected['reviewed_category']} on {selected['reviewed_at']}")
        with st.form("review_form"):
            reviewed_category = st.selectbox("Analyst decision", list(LABELS), index=LABELS.index(selected["reviewed_category"] or selected["category"]))
            review_note = st.text_area("Review note (avoid secrets or personal information)", value=selected["review_note"] or "", max_chars=1000)
            save = st.form_submit_button("Save human review")
        if save:
            try: save_review(selected["id"], reviewed_category, review_note); st.success("Review saved separately from automatic prediction; model was not retrained."); st.rerun()
            except Exception as exc: st.error(f"Could not save review: {exc}")
        st.download_button("Export reviewed records (sanitized)", sanitized_report([r for r in records if r.get("reviewed_category")]), "signalshield_reviewed.csv", "text/csv")

with tab_model:
    st.subheader("Model evaluation")
    metrics = read_metrics(str(METRICS_PATH)) if METRICS_PATH.exists() else None
    if not metrics:
        st.info("No saved evaluation metrics yet. Train a local model first.")
    else:
        st.warning(f"Dataset: {metrics.get('dataset_type', 'unspecified')}. {metrics.get('evaluation_note', '')}")
        st.write(f"Selected baseline: **{metrics.get('selected_model', 'unknown')}** · {metrics.get('rows_after_deduplication', '?')} unique rows · test fraction {metrics.get('test_size', '?')} · seed {metrics.get('random_state', '?')}")
        summary = []
        for name, m in metrics.get("models", {}).items():
            summary.append({"Model": name, "Accuracy": m["accuracy"], "Macro precision": m["macro_precision"], "Macro recall": m["macro_recall"], "Macro F1": m["macro_f1"]})
        st.dataframe(pd.DataFrame(summary).set_index("Model").style.format("{:.3f}"), use_container_width=True)
        model_name = st.selectbox("Confusion matrix for", list(metrics.get("models", {})))
        matrix = metrics["models"][model_name]["confusion_matrix"]
        label_order = metrics.get("label_order", LABELS)
        st.write("Rows = actual class; columns = predicted class; order: " + ", ".join(label_order))
        st.dataframe(pd.DataFrame(matrix, index=label_order, columns=label_order), use_container_width=True)
        st.markdown("**Aggregate error analysis**")
        errors = []
        for i, label in enumerate(label_order):
            support = sum(matrix[i])
            false_negatives = support - matrix[i][i]
            false_positives = sum(row[i] for row in matrix) - matrix[i][i]
            errors.append({"Class": label, "Test support": support, "False negatives": false_negatives, "False positives": false_positives, "Recall": (matrix[i][i] / support) if support else 0.0})
        st.dataframe(pd.DataFrame(errors).set_index("Class").style.format({"Recall": "{:.1%}"}), use_container_width=True)
        report = metrics["models"][model_name].get("classification_report", {})
        if report: st.dataframe(pd.DataFrame(report).T, use_container_width=True)
        st.caption("These holdout metrics are from the provided dataset only. Synthetic demonstration data is intentionally small and is not a substitute for representative, independently sourced evaluation data.")
