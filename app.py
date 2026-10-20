import io
import json
import base64
import re
import requests
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

# All database calls run on the Streamlit server. Never send the secret key to the browser.
@st.cache_resource
def db_settings():
    cfg = st.secrets["supabase"]
    return str(cfg["url"]).rstrip("/"), str(cfg["key"])


def db_request(method, table, *, params=None, payload=None, prefer=None):
    url, key = db_settings()
    headers = {"apikey": key, "Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    if prefer:
        headers["Prefer"] = prefer
    response = requests.request(method, f"{url}/rest/v1/{table}", headers=headers,
                                params=params, json=payload, timeout=20)
    response.raise_for_status()
    return response.json() if response.content else None


def load_user_profile(user_id):
    rows = db_request("GET", "user_profiles", params={"user_id": f"eq.{user_id}", "select": "*", "limit": "1"})
    if not rows:
        return None
    row = rows[0]
    profile = {field: row.get(field) or [] for field in (
        "interests", "important_topics", "important_senders", "important_domains", "low_priority_topics")}
    profile.update({"name": row.get("name") or "", "role": row.get("role") or "", "course": row.get("role") or ""})
    profile["learned_senders"] = load_learned_senders(user_id)
    return profile


def load_learned_senders(user_id):
    # Replay saved feedback to recover learned preferences across sessions.
    rows = db_request("GET", "user_feedback", params={
        "user_id": f"eq.{user_id}", "select": "sender,corrected_category",
        "order": "created_at.asc,id.asc", "limit": "10000"}) or []
    learned = {}
    for row in rows:
        sender = (row.get("sender") or "").lower().strip()
        category = (row.get("corrected_category") or "").upper()
        if sender and category in ("PRIORITY", "REVIEW", "SPAM"):
            learned.setdefault(sender, []).append(1 if category == "PRIORITY" else -1 if category == "SPAM" else 0)
    return learned


def save_user_profile(user_id, profile):
    fields = ("interests", "important_topics", "important_senders", "important_domains", "low_priority_topics")
    payload = {"user_id": user_id, "name": profile.get("name", ""), "role": profile.get("role", ""),
               **{field: profile.get(field, []) for field in fields}}
    db_request("POST", "user_profiles", params={"on_conflict": "user_id"},
               payload=payload, prefer="resolution=merge-duplicates,return=minimal")


def save_feedback(user_id, record, predicted, corrected):
    db_request("POST", "user_feedback", payload={
        "user_id": user_id, "sender": record["sender"], "subject": record["subject"],
        "predicted_category": predicted, "corrected_category": corrected}, prefer="return=minimal")


# Authenticate first: do not let users choose another person's database identity.
try:
    google_settings = st.secrets["google_oauth"]
    oauth = OAuth2Component(
        google_settings["client_id"], google_settings["client_secret"],
        "https://accounts.google.com/o/oauth2/v2/auth",
        "https://oauth2.googleapis.com/token",
        "https://oauth2.googleapis.com/token",
        "https://oauth2.googleapis.com/revoke",
    )
    db_settings()
except Exception:
    st.error("Google OAuth or Supabase settings are missing/invalid in Streamlit Secrets.")
    st.stop()

st.title("📧 MailMind AI — Your Smart Inbox")
if "gmail_token" not in st.session_state:
    st.info("Connect Gmail first. Your verified Gmail account identifies your saved personalization profile.")
    auth_result = oauth.authorize_button(
        "Connect Gmail",
        redirect_uri="https://mailmind-triage-ai.streamlit.app/component/streamlit_oauth.authorize_button",
        scope="https://www.googleapis.com/auth/gmail.modify",
        key="gmail_oauth", pkce="S256", use_container_width=True,
        extras_params={"access_type": "offline", "prompt": "consent"},
    )
    if auth_result and auth_result.get("token"):
        st.session_state.gmail_token = auth_result["token"]
        st.rerun()
    st.stop()

try:
    token = oauth.refresh_token(st.session_state.gmail_token)
    st.session_state.gmail_token = token
    credentials = Credentials(token=token["access_token"])
    gmail_identity_service = build("gmail", "v1", credentials=credentials, cache_discovery=False)
    user_id = gmail_identity_service.users().getProfile(userId="me").execute()["emailAddress"].strip().lower()
except Exception:
    st.error("Could not verify your Gmail account. Disconnect and reconnect Gmail.")
    if st.button("Disconnect Gmail"):
        st.session_state.clear()
        st.rerun()
    st.stop()

# Clear all user-specific state if the signed-in account changes.
if st.session_state.get("profile_user_id") != user_id:
    st.session_state.profile_user_id = user_id
    st.session_state.pop("personal_profile", None)
    st.session_state.pop("gmail_next_page", None)
    st.session_state.results = []

if "personal_profile" not in st.session_state:
    try:
        st.session_state.personal_profile = load_user_profile(user_id)
    except Exception as e:
        st.error("Unable to load your saved profile.")
        st.error(f"Error type: {type(e).__name__}")
        st.error(f"Error details: {str(e)[:300]}")
        st.stop()

with st.sidebar:
    st.title("📧 MailMind AI")
    st.caption(f"Signed in: {user_id}")
    st.info("DistilBERT + MiniLM + persistent personalization")
    if st.button("Disconnect Gmail", key="disconnect_account"):
        st.session_state.clear()
        st.rerun()
    if st.button("Clear Inbox", use_container_width=True):
        st.session_state.results = []
        st.rerun()
    st.caption("Your profile and corrections are saved in Supabase; inbox results are session-only.")


def terms(text):
    return [x.strip() for x in re.split(r"[,\n]", text) if x.strip()]


def profile_editor():
    st.subheader("👤 Personalize MailMind AI")
    st.caption("Create once; edit anytime. Your changes are saved across logins.")
    current = st.session_state.personal_profile or {}
    with st.form("profile_form"):
        name = st.text_input("Profile name", value=current.get("name", ""))
        role = st.text_input("Your role / course", value=current.get("role", ""))
        interests = st.text_area("Interests (comma-separated)", value=", ".join(current.get("interests", [])))
        topics = st.text_area("Important topics", value=", ".join(current.get("important_topics", [])))
        senders = st.text_area("Important senders", value=", ".join(current.get("important_senders", [])))
        domains = st.text_area("Important email domains", value=", ".join(current.get("important_domains", [])))
        low = st.text_area("Less relevant topics", value=", ".join(current.get("low_priority_topics", [])))
        submitted = st.form_submit_button("Save personalization", type="primary")
        if submitted:
            if not name.strip() or not (interests.strip() or topics.strip()):
                st.error("Enter your name and at least one interest or important topic.")
            else:
                new_profile = {
                    "name": name.strip(), "role": role.strip(), "course": role.strip(),
                    "interests": terms(interests), "important_topics": terms(topics),
                    "important_senders": terms(senders), "important_domains": terms(domains),
                    "low_priority_topics": terms(low),
                    "learned_senders": current.get("learned_senders", {}),
                }
                try:
                    save_user_profile(user_id, new_profile)
                except Exception:
                    st.error("Database save failed. Your old profile is unchanged; please retry.")
                else:
                    st.session_state.personal_profile = new_profile
                    st.session_state.results = []
                    st.rerun()
    if st.session_state.personal_profile:
        st.success("Your saved profile is active. New emails use these preferences.")


profile_editor()
if not st.session_state.personal_profile:
    st.info("Create your personalization profile to unlock classification and Gmail sync.")
    st.stop()

profile_id = st.session_state.personal_profile


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
    st.subheader("Gmail Inbox (Beta)")
    st.caption("Connected as " + user_id + ". Gmail is read-only in this version.")
    try:
            st.caption("Sync your entire Inbox in pages of up to 100 messages. No total limit; each click continues where the last batch ended.")
            if st.button("Start a fresh Inbox sync"):
                st.session_state.pop("gmail_next_page", None)
                st.rerun()
            if st.button("Sync next Gmail batch", type="primary"):
                try:
                    token = oauth.refresh_token(st.session_state.gmail_token)
                    st.session_state.gmail_token = token
                    credentials = Credentials(token=token["access_token"])
                    gmail = build("gmail", "v1", credentials=credentials, cache_discovery=False)
                    # Gmail returns at most 500 message IDs per API page. Use smaller
                    # pages to keep Streamlit responsive; continue until all pages are read.
                    page_token = st.session_state.get("gmail_next_page")
                    request = {"userId": "me", "labelIds": ["INBOX"], "maxResults": 100}
                    if page_token:
                        request["pageToken"] = page_token
                    page = gmail.users().messages().list(**request).execute()
                    messages = page.get("messages", [])
                    st.session_state.gmail_next_page = page.get("nextPageToken")
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
                        st.warning("No readable plain-text emails found in this Gmail batch.")
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
                        # Preserve previous batches and update any emails seen again.
                        existing = [r for r in st.session_state.results if r.get("gmail_id") not in {x["gmail_id"] for x in records}]
                        st.session_state.results = records + existing
                        st.success(f"Classified {len(records)} emails in this batch. {failures} could not be classified.")
                    if st.session_state.get("gmail_next_page"):
                        st.info("More Gmail messages remain. Click 'Sync next Gmail batch' to continue. There is no total email limit.")
                    else:
                        st.success("Reached the end of your Gmail Inbox. To check for new messages, click 'Start a fresh Inbox sync'.")
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
for idx, record in enumerate(filtered):
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
        if st.checkbox("Show Detailed Feature Scores", key=f"features_{idx}"):
            st.json(record["features"])
        st.caption("Was this classification right? Correct it to teach MailMind your preferences.")
        correction = st.selectbox("Correct category", ["PRIORITY", "REVIEW", "SPAM"],
                                  index=["PRIORITY", "REVIEW", "SPAM"].index(category),
                                  key=f"correction_{idx}")
        if st.button("Save correction", key=f"feedback_{idx}"):
            if correction != category:
                try:
                    save_feedback(user_id, record, category, correction)
                except Exception:
                    st.error("Feedback could not be saved to the database. Please retry.")
                else:
                    record["category"] = correction
                    record["explanation"] = "Corrected by user (saved in Supabase)."
                    sender_key = record["sender"].lower().strip()
                    history = st.session_state.personal_profile.setdefault("learned_senders", {})
                    events = history.setdefault(sender_key, [])
                    events.append(1 if correction == "PRIORITY" else -1 if correction == "SPAM" else 0)
                    st.success("Correction saved permanently. Repeated feedback influences future messages.")
                    st.rerun()
            else:
                st.info("No category change was made.")

if results:
    cols = ["sender", "subject", "body", "category", "relevance", "confidence", "spam_probability", "explanation"]
    export_df = pd.DataFrame([{key: record[key] for key in cols} for record in results])
    st.download_button("⬇️ Download Classified Emails", data=export_df.to_csv(index=False), file_name="mailmind_classified_emails.csv", mime="text/csv", use_container_width=True)

st.divider()
st.caption("MailMind AI | Profiles and feedback persist in Supabase. Classification scores are experimental.")
