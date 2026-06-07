import os
import shutil
from huggingface_hub import hf_hub_download

# Download the Base Kakuna Model
REPO_ID = "jakegrigsby/metamon"
FILENAME = "kakuna/ckpts/latest/policy.pt"
OUTPUT_FILE = "kakunatest.pt"

def download_weights(output_file=OUTPUT_FILE):
    HERE = os.path.dirname(os.path.abspath(__file__))
    WEIGHTS_DIR = os.path.normpath(os.path.join(HERE, "..", "weights"))
    os.makedirs(WEIGHTS_DIR, exist_ok=True)

    print(f"Downloading {FILENAME} from {REPO_ID}...")

    cached_path = hf_hub_download(
        repo_id=REPO_ID,
        filename=FILENAME,
        repo_type="model",
    )

    cached_size = os.path.getsize(cached_path)
    print(f"Downloaded to cache: {cached_path} ({cached_size:,} bytes)")

    output_path = os.path.join(WEIGHTS_DIR, output_file)
    shutil.copy2(cached_path, output_path)
    print(f"Copied to: {output_path} ({os.path.getsize(output_path):,} bytes)")

    print(f"Done. Weight file ready at {output_path}")
    return output_path

if __name__ == "__main__":
    download_weights()
