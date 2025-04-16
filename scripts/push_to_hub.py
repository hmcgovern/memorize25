import os
from huggingface_hub import HfApi, HfFolder
from pathlib import Path
import argparse as ap

def upload_model_to_huggingface(model_dir: str, repo_name: str):
    # Get the list of checkpoint directories in the model directory
    checkpoints = [f for f in os.listdir(model_dir) if os.path.isdir(os.path.join(model_dir, f))]
    
    # Sort the checkpoints by name assuming the latest checkpoint is the one with the highest number
    latest_checkpoint = max(checkpoints, key=lambda x: int(x.split('-')[-1]) if x.split('-')[-1].isdigit() else 0)

    # Set the path to the latest checkpoint
    model_path = os.path.join(model_dir, latest_checkpoint)
    
    # Initialize the Hugging Face API client
    api = HfApi()

    # Check if the repo exists, if not, create it
    try:
        api.create_repo(repo_id=repo_name, private=True)
        print(f"Created repository {repo_name} and set it to private.")
    except Exception as e:
        if "already exists" in str(e):
            print(f"Repository {repo_name} already exists.")
        else:
            print(f"Error creating repository: {e}")
            return

    # Now push the model to Hugging Face
    try:
        # Upload the model to the newly created or existing repo
        api.upload_folder(
            folder_path=model_path, 
            path_in_repo="", 
            repo_id=repo_name,
            repo_type="model"
        )
        print(f"Model {repo_name} uploaded successfully.")
    except Exception as e:
        print(f"Error uploading model: {e}")

if __name__ == "__main__":
    parser = ap.ArgumentParser()
    parser.add_argument('--model_dir')
    parser.add_argument('--repo_name')

    args = parser.parse_args()
    upload_model_to_huggingface(args.model_dir, args.repo_name)
    # model_directory = "path/to/your/model/directory"  # Set this to your model's folder
    # repository_name = "your-username/your-model-repo"  # Your Hugging Face repo name