
import io

import pandas as pd
import streamlit as st

from classifier import classify_email


st.set_page_config(
    page_title="MailMind AI",
    page_icon="📧",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.block-container {
    padding-top: 2rem;
    max-width: 1200px;
}
[data-testid="stMetric"] {
    background: rgba(128, 128, 128, 0.08);
    padding: 18px;
    border-radius: 12px;
}
.stButton > button {
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# Application state
# --------------------------------------------------

if "results" not in st.session_state:
    st.session_state.results = []

if "last_profile" not in st.session_state:
    st.session_state.last_profile = None


# --------------------------------------------------
# Sidebar
# --------------------------------------------------

with st.sidebar:
    st.title("📧 MailMind AI")
    st.caption("Your intelligent email assistant")

    st.divider()

    profile_name = st.selectbox(
        "Personalization Profile",
        [
            "Ananya - AI Student",
            "Rohan - Mechanical Student",
        ],
    )

    profile_id = (
        "ananya_ai"
        if profile_name == "Ananya - AI Student"
        else "rohan_mech"
    )

    if st.session_state.last_profile != profile_id:
        st.session_state.results = []
        st.session_state.last_profile = profile_id

    st.divider()

    st.info(
        "MailMind AI uses DistilBERT, MiniLM, "
        "and personalized decision rules."
    )

    if st.button("Clear Inbox", use_container_width=True):
        st.session_state.results = []
        st.rerun()

    st.caption("Prototype mode: no email account connected.")


# --------------------------------------------------
# Header
# --------------------------------------------------

st.title("Your Smart Inbox")
st.write(
    "Understand what matters. Let AI organize your "
    "emails into Priority, Review, and Spam."
)

st.divider()


# --------------------------------------------------
# Analysis functions
# --------------------------------------------------

def analyze_email(sender, subject, body):
    email = {
        "sender": str(sender),
        "subject": str(subject),
        "body": str(body),
    }

    result = classify_email(email, profile_id)

    return {
        "sender": email["sender"],
        "subject": email["subject"],
        "body": email["body"],
        "category": result["category"],
        "relevance": result["relevance_score"],
        "confidence": result["priority_confidence"],
        "spam_probability": result["features"][
            "ai_spam_probability"
        ],
        "explanation": result["explanation"],
        "features": result["features"],
    }


# --------------------------------------------------
# Input tabs
# --------------------------------------------------

single_tab, bulk_tab = st.tabs(
    ["✉️ Analyze One Email", "📂 Bulk Email Upload"]
)

with single_tab:
    with st.form("single_email_form"):
        sender = st.text_input(
            "Sender",
            placeholder="professor@college.edu",
        )

        subject = st.text_input(
            "Subject",
            placeholder="Assignment Deadline",
        )

        body = st.text_area(
            "Email Message",
            placeholder="Submit by tomorrow.",
            height=140,
        )

        submitted = st.form_submit_button(
            "Analyze Email",
            type="primary",
            use_container_width=True,
        )

    if submitted:
        if not all([
            sender.strip(),
            subject.strip(),
            body.strip(),
        ]):
            st.warning("Please complete all email fields.")
        else:
            with st.spinner("MailMind AI is analyzing..."):
                try:
                    record = analyze_email(
                        sender, subject, body
                    )
                    st.session_state.results.insert(
                        0, record
                    )
                    st.success(
                        f"Email classified as "
                        f"{record['category']}."
                    )
                except Exception as exc:
                    st.error(
                        f"Classification failed: {exc}"
                    )


with bulk_tab:
    st.write(
        "Upload a CSV containing sender, subject, "
        "and body columns."
    )

    template = pd.DataFrame([
        {
            "sender": "professor@college.edu",
            "subject": "Assignment Deadline",
            "body": "Submit by tomorrow.",
        },
        {
            "sender": "events@college.edu",
            "subject": "Technology Workshop",
            "body": "Join our optional workshop.",
        },
    ])

    st.download_button(
        "Download Sample CSV",
        data=template.to_csv(index=False),
        file_name="mailmind_sample.csv",
        mime="text/csv",
    )

    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type=["csv"],
    )

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)

            required = {"sender", "subject", "body"}

            if not required.issubset(df.columns):
                st.error(
                    "CSV must contain sender, "
                    "subject, and body columns."
                )
            elif df.empty:
                st.warning("The CSV is empty.")
            else:
                st.write(
                    f"**{len(df)} emails found**"
                )

                st.dataframe(
                    df.head(10),
                    use_container_width=True,
                )

                if st.button(
                    "Analyze All Emails",
                    type="primary",
                    use_container_width=True,
                ):
                    if len(df) > 100:
                        st.error(
                            "Please upload a maximum of "
                            "100 emails per batch."
                        )
                    else:
                        progress = st.progress(0)
                        new_records = []
                        failures = []

                        for index, row in enumerate(
                            df.to_dict("records")
                        ):
                            try:
                                record = analyze_email(
                                    row["sender"],
                                    row["subject"],
                                    row["body"],
                                )
                                new_records.append(record)
                            except Exception as exc:
                                failures.append(
                                    f"Row {index + 2}: {exc}"
                                )

                            progress.progress(
                                (index + 1) / len(df)
                            )

                        st.session_state.results = (
                            new_records
                            + st.session_state.results
                        )

                        st.success(
                            f"Analyzed {len(new_records)} "
                            f"emails successfully."
                        )

                        if failures:
                            st.warning(
                                f"{len(failures)} emails "
                                "could not be analyzed."
                            )
                            with st.expander(
                                "View processing errors"
                            ):
                                for error in failures:
                                    st.write(error)

        except Exception as exc:
            st.error(f"Unable to read CSV: {exc}")


# --------------------------------------------------
# Dashboard statistics
# --------------------------------------------------

results = st.session_state.results

st.divider()
st.subheader("Inbox Overview")

total = len(results)

priority_count = sum(
    r["category"] == "PRIORITY" for r in results
)
review_count = sum(
    r["category"] == "REVIEW" for r in results
)
spam_count = sum(
    r["category"] == "SPAM" for r in results
)

col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Emails", total)
col2.metric("🔴 Priority", priority_count)
col3.metric("🟡 Review", review_count)
col4.metric("🚫 Spam", spam_count)


# --------------------------------------------------
# Smart inbox
# --------------------------------------------------

st.subheader("Categorized Inbox")

filter_choice = st.radio(
    "Filter Emails",
    ["All", "PRIORITY", "REVIEW", "SPAM"],
    horizontal=True,
)

filtered = [
    record
    for record in results
    if filter_choice == "All"
    or record["category"] == filter_choice
]

if not filtered:
    st.info(
        "No emails to display. Analyze an email "
        "or upload a CSV to get started."
    )

for index, record in enumerate(filtered):
    category = record["category"]

    label = {
        "PRIORITY": "🔴",
        "REVIEW": "🟡",
        "SPAM": "🚫",
    }.get(category, "✉️")

    with st.expander(
        f"{label} {record['subject']} "
        f"— {record['sender']} [{category}]"
    ):
        st.write(f"**From:** {record['sender']}")
        st.write(f"**Subject:** {record['subject']}")
        st.write(record["body"])

        st.divider()

        a, b, c = st.columns(3)

        a.metric(
            "Relevance",
            f"{record['relevance']:.1%}"
        )
        b.metric(
            "Priority Score",
            f"{record['confidence']:.1%}"
        )
        c.metric(
            "Spam Probability",
            f"{record['spam_probability']:.1%}"
        )

        st.write("**AI Explanation**")
        st.info(record["explanation"])

        with st.expander(
            "Detailed Feature Scores"
        ):
            st.json(record["features"])


# --------------------------------------------------
# Export results
# --------------------------------------------------

if results:
    export_columns = [
        "sender",
        "subject",
        "body",
        "category",
        "relevance",
        "confidence",
        "spam_probability",
        "explanation",
    ]

    export_df = pd.DataFrame([
        {
            key: record[key]
            for key in export_columns
        }
        for record in results
    ])

    st.download_button(
        "⬇️ Download Classified Emails",
        data=export_df.to_csv(index=False),
        file_name="mailmind_classified_emails.csv",
        mime="text/csv",
        use_container_width=True,
    )


st.divider()

st.caption(
    "MailMind AI | Personalized, uncertainty-aware "
    "email triage prototype. Scores are experimental "
    "and not calibrated probabilities."
)
