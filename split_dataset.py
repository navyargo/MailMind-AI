
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

# Find the data folder
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

# Read the cleaned training dataset
df = pd.read_csv(DATA_DIR / "train_clean.csv")

# Divide into training and validation
train_df, val_df = train_test_split(
    df,
    test_size=0.15,
    random_state=42,
    stratify=df["label"]
)

# Save the new datasets
train_df.to_csv(DATA_DIR / "train_final.csv", index=False)
val_df.to_csv(DATA_DIR / "validation.csv", index=False)

# Display the results
print("Dataset split completed!")

print("\nTraining:", len(train_df))
print("Validation:", len(val_df))

print("\nTraining labels:")
print(train_df["label"].value_counts())

print("\nValidation labels:")
print(val_df["label"].value_counts())

print("\nFiles saved successfully!")
