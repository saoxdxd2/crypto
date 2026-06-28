import os
import json
import logging
from pathlib import Path
from src.core.sys_events import push_sys_event
import subprocess
import sys

logger = logging.getLogger(__name__)

KAGGLE_DIR = Path("kaggle_build")

def build_kernel_metadata():
    metadata = {
        "id": f"sao/lobert-auto-trainer",
        "title": "LOBERT Auto Trainer",
        "code_file": "kaggle_train_script.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": "true",
        "enable_gpu": "true",
        "enable_internet": "true",
        "dataset_sources": [],
        "competition_sources": [],
        "kernel_sources": []
    }
    KAGGLE_DIR.mkdir(parents=True, exist_ok=True)
    (KAGGLE_DIR / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2))
    
def build_training_script():
    script_content = '''
import os
import subprocess
print("Cloning repository...")
subprocess.run(["git", "clone", "https://github.com/sao/crypto-research.git"], check=True)
os.chdir("crypto-research")
print("Installing requirements...")
subprocess.run(["pip", "install", "-r", "requirements.txt"], check=True)
print("Starting training...")
subprocess.run(["python", "-m", "src.cloud.train_lobert", "--push_to_hub", "--epochs", "10", "--batch_size", "128"], check=True)
print("Training complete.")
'''
    (KAGGLE_DIR / "kaggle_train_script.py").write_text(script_content.strip())

def submit_training_job():
    logger.info("Initializing Kaggle Cloud Training Job...")
    
    # Check if access_token or KAGGLE_API_TOKEN exists
    kaggle_creds = Path.home() / ".kaggle" / "access_token"
    if not kaggle_creds.exists() and not os.environ.get("KAGGLE_API_TOKEN"):
        # Fallback check for old kaggle.json format
        old_creds = Path.home() / ".kaggle" / "kaggle.json"
        if not old_creds.exists():
            logger.error(f"Kaggle credentials not found at {kaggle_creds} and KAGGLE_API_TOKEN not set. Training aborted.")
            push_sys_event("ERROR", "Kaggle API Token missing. Training failed.")
            return
        
    try:
        build_kernel_metadata()
        build_training_script()
        
        logger.info("Pushing headless script to Kaggle API...")
        
        # Auto-install kaggle if missing
        try:
            from kaggle.api.kaggle_api_extended import KaggleApi
        except ImportError:
            logger.info("Kaggle library not found. Installing automatically...")
            subprocess.run([sys.executable, "-m", "pip", "install", "kaggle>=1.6.14"], check=True)
            from kaggle.api.kaggle_api_extended import KaggleApi
            
        # Native API call bypasses Windows PATH issues ([WinError 2])
        api = KaggleApi()
        api.authenticate()
        api.kernels_push(str(KAGGLE_DIR))
        
        logger.info("Kaggle Kernel pushed successfully! It is now queued for GPU execution.")
        push_sys_event("SYSTEM", "Kaggle Cloud Training job submitted successfully.", progress=0.2)
        
    except Exception as e:
        logger.error(f"Failed to submit Kaggle training job: {e}")
        push_sys_event("ERROR", f"Cloud Training failed to start: {e}")

if __name__ == "__main__":
    submit_training_job()
