
from sentence_transformers import SentenceTransformer, util
import json
from pathlib import Path

# Load pretrained MiniLM model
model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

# Load user profiles
BASE_DIR = Path(__file__).resolve().parent
PROFILE_FILE = BASE_DIR / "data" / "demo_profiles.json"

with open(PROFILE_FILE, "r", encoding="utf-8") as f:
    profiles = json.load(f)


def semantic_similarity(email_text, interests):
    if not interests:
        return 0.0

    email_embedding = model.encode(
        email_text, convert_to_tensor=True
    )

    interest_embeddings = model.encode(
        interests, convert_to_tensor=True
    )

    similarities = util.cos_sim(
        email_embedding, interest_embeddings
    )[0]

    return float(similarities.max())


# Test email
email = (
    "IEEE workshop on Generative AI. "
    "Registration closes Friday."
)

# Compare two different user profiles
for user_id in ["ananya_ai", "rohan_mech"]:
    interests = profiles[user_id]["interests"]

    score = semantic_similarity(email, interests)

    print(f"{user_id}: Semantic similarity = {score:.4f}")
