from pathlib import Path
from urllib.request import urlretrieve

MODEL_DIR = Path(__file__).resolve().parent / "model"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

MODEL_FILE = MODEL_DIR / "model.safetensors"

MODEL_URL = (
    "https://github.com/navyargo/MailMind-AI/"
    "releases/download/v1.0-model/model.safetensors"
)


def ensure_model():
    if MODEL_FILE.exists() and MODEL_FILE.stat().st_size > 100_000_000:
        print("Trained DistilBERT model already available.")
        return

    print("Downloading trained DistilBERT model...")
    temporary_file = MODEL_FILE.with_suffix(".download")

    try:
        urlretrieve(MODEL_URL, temporary_file)

        if temporary_file.stat().st_size < 100_000_000:
            raise RuntimeError("Downloaded model file is unexpectedly small.")

        temporary_file.replace(MODEL_FILE)
        print("Model downloaded successfully!")

    finally:
        temporary_file.unlink(missing_ok=True)


if __name__ == "__main__":
    ensure_model()
