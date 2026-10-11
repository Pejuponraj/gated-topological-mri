"""
deploy_static.py  -  upload the 'site' folder as a FREE Hugging Face Static Space.
Before: hf auth login        Run:  python deploy_static.py YOUR_HF_USERNAME
"""
import sys
from huggingface_hub import HfApi

repo = f"{sys.argv[1]}/gated-topological-mri-demo"
api = HfApi()
api.create_repo(repo, repo_type="space", space_sdk="static", exist_ok=True)
api.upload_folder(repo_id=repo, repo_type="space", folder_path="site",
                  commit_message="Deploy static demo")
print(f"Uploaded. Your Space: https://huggingface.co/spaces/{repo}")
print("A static Space is ready in about a minute.")
