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
subprocess.run(["git", "clone", "https://github.com/saoxdxd2/crypto.git"], check=True)
os.chdir("crypto")
print("Installing requirements...")
subprocess.run(["pip", "install", "-r", "requirements.txt"], check=True)
print("Starting training...")
subprocess.run(["python", "-m", "src.cloud.train_lobert", "--push_to_hub", "--resume_from_hub", "--epochs", "10", "--batch_size", "128"], check=True)
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
            subprocess.run(["python", "-m", "pip", "install", "kaggle>=1.6.14"], check=True)
            from kaggle.api.kaggle_api_extended import KaggleApi
            
        # Native API call bypasses Windows PATH issues ([WinError 2])
        api = KaggleApi()
        api.authenticate()
        api.kernels_push(str(KAGGLE_DIR))
        
        logger.info("Kaggle Kernel pushed successfully! It is now queued for GPU execution.")
        push_sys_event("SYSTEM", "Kaggle Cloud Training job submitted successfully. Awaiting allocation...", progress=0.2)
        
        import time
        kernel_slug = "sao/lobert-auto-trainer"
        last_status = ""
        
        # Give Kaggle a few seconds to register the push before polling
        time.sleep(5)
        
        while True:
            try:
                res = api.kernel_status(kernel_slug)
                # The Kaggle API returns a dict or an object depending on the version
                status = getattr(res, "status", None)
                if not status and isinstance(res, dict):
                    status = res.get("status")
                    
                if not status:
                    status = "unknown"
                
                status_lower = str(status).lower()
                
                if status_lower != last_status:
                    logger.info(f"Kaggle Status Update: {status_lower}")
                    if status_lower == "queued":
                        push_sys_event("SYSTEM", "Kaggle Cloud Status: QUEUED (Waiting for GPU)", progress=0.3)
                    elif status_lower == "running":
                        push_sys_event("SYSTEM", "Kaggle Cloud Status: RUNNING (Training LOBERT on GPU)", progress=0.5)
                    elif status_lower == "complete":
                        push_sys_event("SYSTEM", "Kaggle Cloud Status: COMPLETE (Training Finished)", progress=1.0)
                        break
                    elif status_lower in ["error", "failed", "cancel"]:
                        push_sys_event("ERROR", f"Kaggle Cloud Status: {status_lower.upper()} (Check Kaggle logs)")
                        break
                    else:
                        push_sys_event("SYSTEM", f"Kaggle Cloud Status: {status_lower.upper()}", progress=0.4)
                        
                    last_status = status_lower
                    
            except Exception as poll_e:
                logger.warning(f"Error polling Kaggle status: {poll_e}")
                
            time.sleep(30)
        
    except Exception as e:
        error_msg = str(e)
        if "409" in error_msg and "Conflict" in error_msg:
            logger.info("Kaggle Kernel is already running. Push ignored.")
            push_sys_event("SYSTEM", "Cloud Training is already in progress. Request ignored.")
        else:
            logger.error(f"Failed to submit Kaggle training job: {e}")
            push_sys_event("ERROR", f"Cloud Training failed to start: {e}")

if __name__ == "__main__":
    submit_training_job()
