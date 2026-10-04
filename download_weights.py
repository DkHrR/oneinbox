import sys
import subprocess

from dotenv import load_dotenv
load_dotenv(override=True)

def main():
    try:
        with open("checkpoint.txt", "r") as f:
            ckpt_path = f.read().strip()
    except FileNotFoundError:
        print("checkpoint.txt not found. Train the model first.")
        sys.exit(1)

    print(f"Downloading weights from {ckpt_path} to ./weights ...")
    subprocess.run(["tinker", "checkpoint", "download", ckpt_path, "--output", "./weights", "--force"], check=True)
    print("Done.")

if __name__ == "__main__":
    main()
