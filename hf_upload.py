import os
import glob
from dotenv import load_dotenv
from huggingface_hub import HfApi

load_dotenv(override=True)

def main():
    hf_token = os.environ.get("HF_TOKEN")
    hf_username = os.environ.get("HF_USERNAME")

    if not hf_token:
        raise ValueError("HF_TOKEN is missing in .env")
    if not hf_username:
        raise ValueError("HF_USERNAME is missing in .env")

    repo_id = f"{hf_username}/oneinbox-qwen3.5-4b-lora"
    api = HfApi(token=hf_token)

    # Locate the folder under weights/
    if not os.path.exists("weights"):
        raise FileNotFoundError("weights directory not found")

    subdirs = [
        os.path.join("weights", d)
        for d in os.listdir("weights")
        if os.path.isdir(os.path.join("weights", d))
    ]
    if not subdirs:
        raise FileNotFoundError("No subfolder found under weights/")

    weights_folder = subdirs[0]
    print(f"Creating / ensuring repo exists: {repo_id}...")
    api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)

    print(f"Uploading adapter files from {weights_folder} to {repo_id}...")
    api.upload_folder(
        folder_path=weights_folder,
        repo_id=repo_id,
        repo_type="model",
        ignore_patterns=["checkpoint_complete"]
    )

    if os.path.exists("MODEL_CARD.md"):
        print(f"Uploading MODEL_CARD.md as README.md to {repo_id}...")
        api.upload_file(
            path_or_fileobj="MODEL_CARD.md",
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="model"
        )

    print(f"Repo URL: https://huggingface.co/{repo_id}")
    files = api.list_repo_files(repo_id=repo_id, repo_type="model")
    print(f"Files in repo: {files}")

if __name__ == "__main__":
    main()
