# -*- coding: utf-8 -*-
import os
import sys

# Force UTF-8
os.environ["PYTHONIOENCODING"] = "utf-8"
os.environ["PYTHONUTF8"] = "1"
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from dotenv import load_dotenv
load_dotenv()

from huggingface_hub import HfApi, create_repo

REPO_ID = "crow1234des/lpg-inspector"
TOKEN   = os.getenv("HF_TOKEN")

api = HfApi()

# Create the Space
print(f"Creating Space: {REPO_ID}")
create_repo(
    repo_id   = REPO_ID,
    repo_type = "space",
    space_sdk = "docker",
    token     = TOKEN,
    exist_ok  = True,
)
print("Space created.")

# Files to upload
files_to_upload = [
    "inference.py",
    "openenv.yaml",
    "README.md",
    "pyproject.toml",
    "requirements.txt",
    "uv.lock",
    "models.py",
    "client.py",
    "__init__.py",
    "server/__init__.py",
    "server/app.py",
    "server/environment.py",
    "server/data_generator.py",
    "server/graders.py",
    "server/Dockerfile",
]

print("Uploading files...")
for filepath in files_to_upload:
    if os.path.exists(filepath):
        api.upload_file(
            path_or_fileobj = filepath,
            path_in_repo    = filepath,
            repo_id         = REPO_ID,
            repo_type       = "space",
            token           = TOKEN,
        )
        print(f"  Uploaded: {filepath}")
    else:
        print(f"  MISSING:  {filepath}")

print()
print(f"Done! Space URL: https://huggingface.co/spaces/{REPO_ID}")
print(f"API endpoint:   https://crow1234des-lpg-inspector.hf.space")