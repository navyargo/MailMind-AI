
import truststore
truststore.inject_into_ssl()

import json
from functools import lru_cache
from pathlib import Path

from sentence_transformers import SentenceTransformer, util

# Load MiniLM once
model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

BASE_DIR = Path(__file__).resolve().parent
PROFILE_FILE = BASE_DIR / "data" / "demo_profiles.json"

with open(PROFILE_FILE, "r", encoding="utf-8") as f:
    profiles = json.load(f)


# Examples of emails that require attention
PRIORITY_EXAMPLES = [
    "Submit your assignment before the deadline.",
    "Please complete the required task by tomorrow.",
    "Your examination is scheduled for tomorrow.",
    "Attend your scheduled academic assessment.",
    "Complete your course registration today.",
    "Your internship application deadline is approaching.",
    "An important project submission is due soon.",
    "Please respond to this urgent academic request.",
]

# Examples of informative, non-urgent emails
REVIEW_EXAMPLES = [
    "You are invited to an optional technology workshop.",
    "Read the latest artificial intelligence newsletter.",
    "A guest lecture is being organized this weekend.",
    "Explore upcoming engineering conferences.",
    "Here are this week's general updates.",
]

# Compute reference embeddings once
priority_embeddings = model.encode(
    PRIORITY_EXAMPLES,
    convert_to_tensor=True,
    normalize_embeddings=True
)

review_embeddings = model.encode(
    REVIEW_EXAMPLES,
    convert_to_tensor=True,
    normalize_embeddings=True
)


@lru_cache(maxsize=512)
def encode_text(text):
    return model.encode(
        text,
        convert_to_tensor=True,
        normalize_embeddings=True
    )


@lru_cache(maxsize=128)
def encode_interests(interests):
    return model.encode(
        list(interests),
        convert_to_tensor=True,
        normalize_embeddings=True
    )


def semantic_similarity(email_text, interests):
    """Measure email relevance to a user's interests."""
    if not interests:
        return 0.0

    email_embedding = encode_text(email_text)
    interest_embeddings = encode_interests(tuple(interests))

    similarities = util.cos_sim(
        email_embedding,
        interest_embeddings
    )[0]

    return float(similarities.max())


def semantic_priority_score(email_text):
    """
    Estimate whether an email resembles an important task
    more than an optional informational message.

    This is a semantic similarity heuristic, not a
    calibrated probability.
    """
    email_embedding = encode_text(email_text)

    priority_similarity = float(
        util.cos_sim(
            email_embedding,
            priority_embeddings
        ).max()
    )

    review_similarity = float(
        util.cos_sim(
            email_embedding,
            review_embeddings
        ).max()
    )

    # Compare important-task and informational similarity.
    margin = priority_similarity - review_similarity

    score = 0.5 + margin

    return max(0.0, min(1.0, score))
