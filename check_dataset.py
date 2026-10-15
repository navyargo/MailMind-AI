
from pathlib import Path
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent / "data"

train = pd.read_csv(DATA_DIR / "train_final.csv")
val = pd.read_csv(DATA_DIR / "validation.csv")
test = pd.read_csv(DATA_DIR / "test_clean.csv")

# Convert email texts into sets
train_texts = set(train["text"])
val_texts = set(val["text"])
test_texts = set(test["text"])

# Check whether identical emails occur in different datasets
print("Training vs Validation:",
      len(train_texts & val_texts))

print("Training vs Testing:",
      len(train_texts & test_texts))

print("Validation vs Testing:",
      len(val_texts & test_texts))

# Check whether identical messages have conflicting labels
all_data = pd.concat([train, val, test], ignore_index=True)
conflicts = all_data.groupby("text")["label"].nunique()
print("Emails with conflicting labels:",
      (conflicts > 1).sum())
