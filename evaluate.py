
import csv
from pathlib import Path
from collections import Counter
from classifier import classify_email

DATA_FILE = Path(__file__).parent / "data" / "evaluation_emails.csv"

total = 0
correct = 0
results = []
counts = Counter()

with open(DATA_FILE, newline="", encoding="utf-8-sig") as file:
    reader = csv.DictReader(file)

    for row in reader:
        email = {
            "sender": row["sender"],
            "subject": row["subject"],
            "body": row["body"],
        }

        expected = row["expected_category"].strip().upper()
        profile_id = row["profile_id"].strip()

        result = classify_email(email, profile_id)
        predicted = result["category"].upper()

        total += 1
        correct += predicted == expected
        counts[predicted] += 1

        results.append({
            **row,
            "predicted_category": predicted,
            "correct": predicted == expected,
            "relevance_score": round(result["relevance_score"], 4),
            "priority_confidence": round(result["priority_confidence"], 4),
            "spam_confidence": round(result["spam_confidence"], 4),
        })

        print(
            f"{total:02d}. {row['subject'][:30]:30} "
            f"Expected: {expected:8} "
            f"Predicted: {predicted:8}"
        )

output_file = Path(__file__).parent / "data" / "evaluation_results.csv"

with open(output_file, "w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(file, fieldnames=results[0].keys())
    writer.writeheader()
    writer.writerows(results)

print("\n========== EVALUATION SUMMARY ==========")
print(f"Total emails: {total}")
print(f"Correct: {correct}")
print(f"Accuracy: {correct / total * 100:.2f}%")
print(f"Predicted categories: {dict(counts)}")
print(f"Results saved to: {output_file}")
