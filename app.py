import io
import base64
import re
from email import policy
from email.parser import BytesParser, Parser

import pandas as pd
import streamlit as st

from classifier import classify_email
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from streamlit_oauth import OAuth2Component

st.set_page_config(page_title="MailMind AI", page_icon="📧", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
.block-container {padding-top: 2rem; max-width: 1200px;}
[data-testid="stMetric"] {background: rgba(128,128,128,.08); padding: 18px; border-radius: 12px;}
.stButton > button {border-radius: 10px;}
</style>
""", unsafe_allow_html=True)

if "results" not in st.session_state:
    st.session_state.results = []
if "last_profile" not in st.session_state:
    st.session_state.last_profile = None

with st.sidebar:
    st.title("📧 MailMind AI")
    st.caption("Personalized, uncertainty-aware email triage")
    profile_name = st.selectbox("Personalization Profile", ["Ananya - AI Student", "Rohan - Mechanical Student"])
    profile_id = "ananya_ai" if profile_name == "Ananya - AI Student" else "rohan_mech"
    if st.session_state.last_profile != profile_id:
        st.session_state.results = []
        st.session_state.last_profile = profile_id
    st.info("Uses DistilBERT, MiniLM and personalized decision rules.")
    if st.button("Clear Inbox", use_container_width=True):
        st.session_state.results = []
        st.rerun()
    st.caption("Gmail Beta requires approved Google OAuth test access. Avoid using sensitive messages for early tests.")

st.title("📧 MailMind AI — Your Smart Inbox")
st.write("Paste emails or drag and drop files. MailMind AI extracts available details and categorizes each message as Priority, Review or Spam.")


def parse_email_text(text):
    """Parse a complete copied email; tolerate missing headers and plain body text."""
    text = text.strip()
    if not text:
        raise ValueError("Empty email")
    # Treat leading RFC-style headers as metadata, not arbitrary body lines.
    has_headers = bool(re.match(r"(?i)^(from|sender|subject|to|date):\s*", text))
    if has_headers:
        msg = Parser(policy=policy.default).parsestr(text)
        sender = str(msg.get("From") or msg.get("Sender") or "Unknown sender")
        subject = str(msg.get("Subject") or "(No subject)")
        if msg.is_multipart():
            body_part = msg.get_body(preferencelist=("plain",))
            body = body_part.get_content() if body_part else ""
        else:
            body = msg.get_content()
        # If headers are incomplete/malformed, preserve original content.
        if not str(body).strip():
            body = text
    else:
        sender, subject, body = "Unknown sender", "(No subject)", text
    return {"sender": sender.strip(), "subject": subject.strip(), "body": str(body).strip()}


def parse_eml(raw_bytes):
    msg = BytesParser(policy=policy.default).parsebytes(raw_bytes)
    sender = str(msg.get("From") or "Unknown sender")
    subject = str(msg.get("Subject") or "(No subject)")
    if msg.is_multipart():
        part = msg.get_body(preferencelist=("plain",))
        body = part.get_content() if part else ""
    else:
        body = msg.get_content()
    if not str(body).strip():
        raise ValueError("No readable plain-text email body")
    return {"sender": sender, "subject": subject, "body": str(body)}


def analyze_email(email):
    result = classify_email(email, profile_id)
    return {
        "sender": email["sender"], "subject": email["subject"], "body": email["body"],
        "category": str(result["category"]).upper(),
        "relevance": result["relevance_score"],
        "confidence": result["priority_confidence"],
        "spam_probability": result["features"]["ai_spam_probability"],
        "explanation": result["explanation"], "features": result["features"],
    }


def run_batch(emails):
    if not emails:
        st.warning("No emails found to analyze.")
        return
    if len(emails) > 100:
        st.error("Maximum 100 emails per batch. Please split larger batches.")
        return
    progress = st.progress(0)
    records, failures = [], []
    for i, email in enumerate(emails):
        try:
            if not str(email.get("body", "")).strip():
                raise ValueError("Missing email body")
            records.append(analyze_email(email))
        except Exception as exc:
            failures.append(f"Email {i + 1}: {exc}")
        progress.progress((i + 1) / len(emails))
    st.session_state.results = records + st.session_state.results
    st.success(f"Classified {len(records)} of {len(emails)} emails.")
    if failures:
        with st.expander(f"View {len(failures)} processing errors"):
            for error in failures:
                st.write(error)


paste_tab, upload_tab, gmail_tab = st.tabs(["📋 Copy & Paste", "📂 Drag & Drop Files", "📨 Gmail Beta"])
with paste_tab:
    st.caption("Paste a full email (including From: and Subject: if available). For multiple emails, put a line containing only ===EMAIL=== between messages.")
    pasted = st.text_area("Paste one or more emails", height=240, placeholder="From: professor@college.edu\nSubject: Assignment Deadline\n\nSubmit your assignment by tomorrow.\n\n===EMAIL===\n\nFrom: events@college.edu\nSubject: AI Workshop\n\nJoin our workshop on Python and machine learning.")
    if st.button("Analyze Pasted Emails", type="primary", use_container_width=True):
        chunks = [chunk.strip() for chunk in re.split(r"(?m)^\s*===EMAIL===\s*$", pasted) if chunk.strip()]
        if not chunks:
            st.warning("Paste at least one email first.")
        else:
            run_batch([parse_email_text(chunk) for chunk in chunks])

with upload_tab:
    st.caption("Drag and drop .eml, .txt or .csv files. You can select multiple files. CSV needs a body column; sender and subject are optional.")
    uploaded_files = st.file_uploader("Drop email files here", type=["eml", "txt", "csv"], accept_multiple_files=True)
    if uploaded_files and st.button("Analyze Uploaded Emails", type="primary", use_container_width=True):
        emails, import_errors = [], []
        for uploaded in uploaded_files:
            try:
                suffix = uploaded.name.rsplit(".", 1)[-1].lower()
                content = uploaded.getvalue()
                if suffix == "eml":
                    emails.append(parse_eml(content))
                elif suffix == "txt":
                    text = content.decode("utf-8-sig")
                    chunks = [x.strip() for x in re.split(r"(?m)^\s*===EMAIL===\s*$", text) if x.strip()]
                    emails.extend(parse_email_text(chunk) for chunk in chunks)
                elif suffix == "csv":
                    df = pd.read_csv(io.BytesIO(content), dtype=str).fillna("")
                    df.columns = df.columns.str.strip().str.lower()
                    if "body" not in df.columns:
                        raise ValueError("CSV must contain a 'body' column")
                    for row in df.to_dict("records"):
                        emails.append({"sender": row.get("sender", "") or "Unknown sender", "subject": row.get("subject", "") or "(No subject)", "body": row["body"]})
            except Exception as exc:
                import_errors.append(f"{uploaded.name}: {exc}")
        if import_errors:
            with st.expander(f"Could not import {len(import_errors)} file(s)"):
                for error in import_errors:
                    st.write(error)
        run_batch(emails)


with gmail_tab:
    st.subheader("Connect your Gmail (Beta)")
    st.caption("For Google OAuth-approved test users only. Gmail is read-only in this version; no messages are changed or deleted.")
    try:
        settings = st.secrets["google_oauth"]
        oauth = OAuth2Component(
            settings["client_id"],
            settings["client_secret"],
            "https://accounts.google.com/o/oauth2/v2/auth",
            "https://oauth2.googleapis.com/token",
            "https://oauth2.googleapis.com/token",
            "https://oauth2.googleapis.com/revoke",
        )
        # This component uses a dedicated OAuth callback endpoint, NOT the homepage.
        redirect_uri = "https://mailmind-triage-ai.streamlit.app/component/streamlit_oauth.authorize_button"
        if "gmail_token" not in st.session_state:
            auth_result = oauth.authorize_button(
                "Connect Gmail",
                redirect_uri=redirect_uri,
                scope="https://www.googleapis.com/auth/gmail.modify",
                key="gmail_oauth",
                pkce="S256",
                use_container_width=True,
                extras_params={"access_type": "offline", "prompt": "consent"},
            )
            if auth_result and auth_result.get("token"):
                st.session_state.gmail_token = auth_result["token"]
                st.rerun()
        else:
            st.success("Gmail authorized for this browser session.")
            if st.button("Disconnect Gmail", key="disconnect_gmail"):
                # Clear this browser session; revocation can be added later.
                st.session_state.pop("gmail_token", None)
                st.session_state.results = [r for r in st.session_state.results if r.get("source") != "gmail"]
                st.rerun()
            if st.button("Fetch and classify 15 Gmail messages", type="primary"):
                try:
                    token = oauth.refresh_token(st.session_state.gmail_token)
                    st.session_state.gmail_token = token
                    credentials = Credentials(token=token["access_token"])
                    gmail = build("gmail", "v1", credentials=credentials, cache_discovery=False)
                    messages = gmail.users().messages().list(userId="me", labelIds=["INBOX"], maxResults=15).execute().get("messages", [])
                    parsed = []
                    ids = []
                    for item in messages:
                        payload = gmail.users().messages().get(userId="me", id=item["id"], format="raw").execute()
                        raw = base64.urlsafe_b64decode(payload["raw"] + "=" * (-len(payload["raw"]) % 4))
                        try:
                            parsed.append(parse_eml(raw))
                            ids.append(item["id"])
                        except ValueError:
                            continue
                    if not parsed:
                        st.warning("No readable plain-text emails found in the latest 15 Inbox messages.")
                    else:
                        progress = st.progress(0)
                        records = []
                        failures = 0
                        for index, (email, gmail_id) in enumerate(zip(parsed, ids)):
                            try:
                                record = analyze_email(email)
                                record["source"] = "gmail"
                                record["gmail_id"] = gmail_id
                                records.append(record)
                            except Exception:
                                failures += 1
                            progress.progress((index + 1) / len(parsed))
                        st.session_state.results = records + [r for r in st.session_state.results if r.get("source") != "gmail"]
                        st.success(f"Classified {len(records)} Gmail emails. {failures} could not be classified.")
                except Exception as exc:
                    st.error(f"Unable to fetch Gmail messages: {type(exc).__name__}. Check OAuth permissions and deployment logs.")
    except KeyError:
        st.warning("Google OAuth credentials are missing from Streamlit Secrets.")

st.divider()
st.subheader("Inbox Overview")
results = st.session_state.results
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Emails", len(results))
c2.metric("🔴 Priority", sum(r["category"] == "PRIORITY" for r in results))
c3.metric("🟡 Review", sum(r["category"] == "REVIEW" for r in results))
c4.metric("🚫 Spam", sum(r["category"] == "SPAM" for r in results))

st.subheader("Categorized Inbox")
filter_choice = st.radio("Filter Emails", ["All", "PRIORITY", "REVIEW", "SPAM"], horizontal=True)
filtered = [r for r in results if filter_choice == "All" or r["category"] == filter_choice]
if not filtered:
    st.info("Paste or upload emails to see classification results here.")
for record in filtered:
    category = record["category"]
    icon = {"PRIORITY": "🔴", "REVIEW": "🟡", "SPAM": "🚫"}.get(category, "✉️")
    with st.expander(f"{icon} {record['subject']} — {record['sender']} [{category}]"):
        st.write(f"**From:** {record['sender']}")
        st.write(f"**Subject:** {record['subject']}")
        st.write(record["body"])
        a, b, c = st.columns(3)
        a.metric("Relevance", f"{record['relevance']:.1%}")
        b.metric("Priority Score", f"{record['confidence']:.1%}")
        c.metric("Spam Probability", f"{record['spam_probability']:.1%}")
        st.write("**AI Explanation**")
        st.info(record["explanation"])
        with st.expander("Detailed Feature Scores"):
            st.json(record["features"])

if results:
    cols = ["sender", "subject", "body", "category", "relevance", "confidence", "spam_probability", "explanation"]
    export_df = pd.DataFrame([{key: record[key] for key in cols} for record in results])
    st.download_button("⬇️ Download Classified Emails", data=export_df.to_csv(index=False), file_name="mailmind_classified_emails.csv", mime="text/csv", use_container_width=True)

st.divider()
st.caption("MailMind AI | Personalized, uncertainty-aware email triage prototype. Scores are experimental and not calibrated probabilities.")
