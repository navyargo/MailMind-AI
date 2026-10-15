
from transformers import pipeline

print("Loading DistilBERT model...")

# Pretrained sentiment classifier used to verify
# DistilBERT inference, NOT for spam classification.
classifier = pipeline(
    task="text-classification",
    model="distilbert/distilbert-base-uncased-finetuned-sst-2-english",
    device=-1
)

emails = [
    "Congratulations! You have won a prize. Claim it now!",
    "Please submit your assignment before tomorrow.",
    "Thank you for your excellent work on the project."
]

for email in emails:
    result = classifier(email)

    print("\nEmail:", email)
    print("DistilBERT result:", result)
