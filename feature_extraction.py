"""

feature_extraction.py

Extract interpretable features from an email for the Personalized
AI Email Triage System.

Initial prototype:
    - Personal relevance
    - Sender importance
    - Semantic/topic relevance
    - Urgency
    - Action required
    - Historical preference
    - Spam score

All scores are normalized to 0..1.

This is the INITIAL rule-based feature extractor. Later phases can replace
or augment these rules with trained ML/NLP models.
"""
from semantic_ai import semantic_similarity, semantic_priority_score
import json
import re
from pathlib import Path

from config import DEMO_PROFILES_JSON
from distilbert_ai import predict_spam


# --------------------------------------------------------------------------
# Utility functions
# --------------------------------------------------------------------------

def clamp(value):
    """Keep a score between 0.0 and 1.0."""
    return max(0.0, min(1.0, value))


def normalize_text(text):
    """Convert text to lowercase and remove unnecessary spacing."""
    if text is None:
        return ""

    text = str(text).lower()
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def count_matches(text, terms):
    """
    Count how many configured terms occur in the supplied text.

    Each configured term is counted once.
    """
    text = normalize_text(text)

    matches = 0

    for term in terms:
        if normalize_text(term) in text:
            matches += 1

    return matches


# --------------------------------------------------------------------------
# Load user profiles
# --------------------------------------------------------------------------

def load_profiles():
    """Load demonstration personalization profiles."""

    path = Path(DEMO_PROFILES_JSON)

    if not path.exists():
        raise FileNotFoundError(
            f"Profile file not found: {path}"
        )

    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def get_profile(profile_id):
    """Return one user profile."""

    profiles = load_profiles()

    if profile_id not in profiles:
        raise ValueError(
            f"Unknown profile '{profile_id}'. "
            f"Available profiles: {list(profiles.keys())}"
        )

    return profiles[profile_id]


# --------------------------------------------------------------------------
# Personal relevance
# --------------------------------------------------------------------------

def personal_relevance_score(subject, body, profile):
    """
    Estimate how closely the email matches the user's interests
    and important topics.
    """

    text = normalize_text(f"{subject} {body}")

    interests = profile.get("interests", [])
    important_topics = profile.get("important_topics", [])
    low_priority_topics = profile.get("low_priority_topics", [])

    interest_matches = count_matches(text, interests)
    important_matches = count_matches(text, important_topics)
    low_priority_matches = count_matches(text, low_priority_topics)

    score = 0.10

    score += min(interest_matches * 0.18, 0.45)
    score += min(important_matches * 0.20, 0.50)

    score -= min(low_priority_matches * 0.10, 0.30)

    return clamp(score)


# --------------------------------------------------------------------------
# Sender importance
# --------------------------------------------------------------------------

def sender_importance_score(sender, profile):
    """
    Estimate whether the sender belongs to an important person,
    organization or domain.
    """

    sender = normalize_text(sender)

    important_senders = profile.get("important_senders", [])
    important_domains = profile.get("important_domains", [])

    score = 0.10

    for item in important_senders:
        if normalize_text(item) in sender:
            score += 0.40

    for domain in important_domains:
        if normalize_text(domain) in sender:
            score += 0.60

    return clamp(score)


# --------------------------------------------------------------------------
# Semantic/topic relevance
# --------------------------------------------------------------------------


def semantic_relevance_score(subject, body, profile):
    """Measure semantic relevance using MiniLM."""

    email_text = f"{subject}\n{body}"

    interests = (
        profile.get("interests", [])
        + profile.get("important_topics", [])
    )

    if not interests:
        return 0.0

    similarity = semantic_similarity(email_text, interests)

    return round(max(0.0, min(1.0, similarity)), 4)
   


# --------------------------------------------------------------------------
# Urgency
# --------------------------------------------------------------------------

URGENCY_TERMS = (
    "urgent",
    "immediately",
    "today",
    "tomorrow",
    "deadline",
    "last date",
    "expires",
    "closing",
    "closes",
    "before",
    "scheduled",
)


def urgency_score(subject, body):
    """Estimate time sensitivity."""

    text = normalize_text(f"{subject} {body}")

    matches = count_matches(text, URGENCY_TERMS)

    if matches == 0:
        return 0.10

    return clamp(0.25 + matches * 0.18)


# --------------------------------------------------------------------------
# Action required
# --------------------------------------------------------------------------


ACTION_TERMS = (
    "submit",
    "register",
    "reply",
    "confirm",
    "attend",
    "complete",
    "approve",
    "review",
    "respond",
    "apply",
    "action required",
    "exam",
    "examination",
    "assessment",
    "scheduled",
)



def action_required_score(subject, body):
    """Estimate whether the recipient is expected to take action."""

    text = normalize_text(f"{subject} {body}")

    matches = count_matches(text, ACTION_TERMS)

    if matches == 0:
        return 0.10

    return clamp(0.25 + matches * 0.20)


# --------------------------------------------------------------------------
# Historical preference
# --------------------------------------------------------------------------

def historical_preference_score():
    """
    Initial neutral historical-preference score.

    We do not yet have user interaction history.

    Later feedback.py will replace this neutral value with a score learned
    from actions such as:
        Review -> Priority
        Review -> Spam
        Mark sender important
        Mark topic relevant
    """

    return 0.50


# --------------------------------------------------------------------------
# Spam estimation
# --------------------------------------------------------------------------

SPAM_TERMS = (
    "you have won",
    "winner",
    "congratulations",
    "click immediately",
    "bank details",
    "limited offer",
    "act now",
    "free money",
    "claim now",
    "guaranteed",
    "prize",
)


def spam_score(sender, subject, body):
    """
    Estimate spam likelihood using transparent prototype rules.

    This is NOT the final research spam model.
    """

    text = normalize_text(f"{sender} {subject} {body}")

    matches = count_matches(text, SPAM_TERMS)

    score = 0.05

    score += matches * 0.18

    # Excessive exclamation marks can be a weak spam indicator.
    exclamation_count = text.count("!")

    if exclamation_count >= 3:
        score += 0.15

    # Excessive uppercase words can also provide weak evidence.
    original_text = f"{subject} {body}"

    uppercase_words = re.findall(
        r"\b[A-Z]{4,}\b",
        original_text
    )

    if len(uppercase_words) >= 2:
        score += 0.10

    return clamp(score)


# --------------------------------------------------------------------------
# Complete email feature extraction
# --------------------------------------------------------------------------

def extract_features(email, profile_id):
    """
    Extract all interpretable features for one email.

    Parameters
    ----------
    email : dict
        Expected keys:
            sender
            subject
            body

    profile_id : str
        Example:
            ananya_ai
            rohan_mech

    Returns
    -------
    dict
        Feature scores in the range 0..1.
    """

    profile = get_profile(profile_id)

    sender = email.get("sender", "")
    subject = email.get("subject", "")
    body = email.get("body", "")

    features = {
        "personal_relevance":
            personal_relevance_score(
                subject,
                body,
                profile
            ),

        "sender_importance":
            sender_importance_score(
                sender,
                profile
            ),

        "semantic_relevance":
            semantic_relevance_score(
                subject,
                body,
                profile
            ),
            
        "semantic_priority":
            semantic_priority_score(
                f"{subject}. {body}"
            ),


        "urgency":
            urgency_score(
                subject,
                body
            ),

        "action_required":
            action_required_score(
                subject,
                body
            ),

        "historical_preference":
            historical_preference_score(),

                "spam_score":
            spam_score(
                sender,
                subject,
                body
            ),

        "ai_spam_probability":
            predict_spam(
                f"{subject}\n{body}"
            )["spam_probability"],
    }

    return features