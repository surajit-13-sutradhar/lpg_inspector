# -*- coding: utf-8 -*-
from dotenv import load_dotenv
load_dotenv()
import os
import io
from huggingface_hub import HfApi

api = HfApi()
token = os.getenv("HF_TOKEN")

dockerfile_content = b"""FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 7860

CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "4"]
"""

api.upload_file(
    path_or_fileobj = io.BytesIO(dockerfile_content),
    path_in_repo    = "Dockerfile",
    repo_id         = "crow1234des/lpg-inspector",
    repo_type       = "space",
    token           = token,
)
print("Dockerfile updated — port 7860.")