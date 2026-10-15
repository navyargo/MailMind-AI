
from pathlib import Path
import pandas as pd

# Locate our project folders
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

# Find the downloaded Parquet files
TRAIN_FILE = next(DATA_DIR.glob("train-*.parquet"))
TEST_FILE = next(DATA_DIR.glob("test-*.parquet"))

def clean_data(df):
    # Keep the two required columns
    df = df[["text", "label"]].copy()

    # Remove missing or empty emails
    df = df.dropna()
    df["text"] = df["text"].astype(str).str.strip()
    df = df[df["text"] != ""]

    # Convert labels from words to numbers
    df["label"] = df["label"].map({
        "not_spam": 0,
        "spam": 1
    })

    # Remove invalid labels and duplicate emails
    df = df.dropna()
    df = df.drop_duplicates(subset=["text"])

    df["label"] = df["label"].astype(int)

    return df

# Read both datasets
train_df = pd.read_parquet(TRAIN_FILE)
test_df = pd.read_parquet(TEST_FILE)

# Clean both datasets
train_df = clean_data(train_df)
test_df = clean_data(test_df)

# Remove emails from testing that also appear in training
test_df = test_df[~test_df["text"].isin(train_df["text"])]

# Save cleaned datasets
train_df.to_csv(DATA_DIR / "train_clean.csv", index=False)
test_df.to_csv(DATA_DIR / "test_clean.csv", index=False)

print("Dataset preparation completed!")

print("\nTraining dataset:")
print("Total:", len(train_df))
print("Legitimate:", (train_df["label"] == 0).sum())
print("Spam:", (train_df["label"] == 1).sum())

print("\nTesting dataset:")
print("Total:", len(test_df))
print("Legitimate:", (test_df["label"] == 0).sum())
print("Spam:", (test_df["label"] == 1).sum())

print("\nFiles saved successfully.")
