# -*- coding: utf-8 -*-
from dotenv import load_dotenv
load_dotenv()
import os
from huggingface_hub import HfApi

api = HfApi()
token = os.getenv("HF_TOKEN")

# Upload Dockerfile to root of Space
with open("server/Dockerfile", "rb") as f:
    api.upload_file(
        path_or_fileobj = f,
        path_in_repo    = "Dockerfile",
        repo_id         = "crow1234des/lpg-inspector",
        repo_type       = "space",
        token           = token,
    )
print("Dockerfile uploaded to root.")

# Also make sure server/Dockerfile is there too
with open("server/Dockerfile", "rb") as f:
    api.upload_file(
        path_or_fileobj = f,
        path_in_repo    = "server/Dockerfile",
        repo_id         = "crow1234des/lpg-inspector",
        repo_type       = "space",
        token           = token,
    )
print("server/Dockerfile uploaded.")