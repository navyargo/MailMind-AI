from pathlib import Path
from transformers import pipeline

# Path to our fine-tuned DistilBERT model
MODEL_PATH = Path(__file__).resolve().parent / "model"

print("Loading MailMind AI trained DistilBERT model...")

spam_classifier = pipeline(
    task="text-classification",
    model=str(MODEL_PATH),
    tokenizer=str(MODEL_PATH),
    device=-1
    )

 def predict_spam(email):
    results = spam_classifier(
        email,
        top_k=None,
        truncation=True,
        max_length=256,
        return_token_type_ids=False
    )

    scores = {
        result["label"]: result["score"]
        for result in results
    }

    spam_probability = scores["LABEL_1"]

    return {
        "prediction": "SPAM" if spam_probability >= 0.5 else "HAM",
        "spam_probability": round(spam_probability, 4)
    }


if __name__ == "__main__":
    emails = [
        "Congratulations! You have won a prize. Claim it now!",
        "Please submit your assignment before tomorrow.",
        "Your professor has scheduled a meeting at 10 AM.",
        "Get 90% discount on all products today!"
    ]

    for email in emails:
        result = predict_spam(email)

        print("\nEmail:", email)
        print("Prediction:", result["prediction"])
        print("Spam probability:", result["spam_probability"])

