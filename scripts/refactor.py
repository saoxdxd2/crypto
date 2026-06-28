import os
import shutil
import re
from pathlib import Path

SRC_DIR = Path("C:/Users/sao/Documents/crypto/src")
LEGACY_DIR = SRC_DIR / "crypto_research"

# Mapping of file/folder -> new domain
MAPPING = {
    # ML Models (Cloud)
    "rl_agent.py": "cloud",
    "pattern_model.py": "cloud",
    "lob_encoder.py": "cloud",
    "data_loaders.py": "cloud",
    "dataset_inventory.py": "cloud",
    "baselines": "cloud",

    # Data Ingestion
    "binance_public.py": "data",
    "l2_collector.py": "data",
    "ingest.py": "data",
    "news_extractor.py": "data",
    "hft_replay.py": "data",
    "ws_sync_server.py": "data",

    # Core Utilities
    "schemas.py": "core",
    "paths.py": "core",
    "duckdb_queries.py": "core",
    "hashing.py": "core",
    "rpc_client.py": "core",

    # Mission Control (UI / Brain)
    "runner.py": "mission_control",
    "cli.py": "mission_control",
    "gui": "mission_control",
    "governor.py": "mission_control",
    "flaml_optimizer.py": "mission_control",
    "deep_reasoning.py": "mission_control",
    "langgraph_automation.py": "mission_control",
    "decision.py": "mission_control",
    "forecast.py": "mission_control",
    "monitoring.py": "mission_control",
}

def move_files():
    print("Moving files...")
    for item in LEGACY_DIR.iterdir():
        if item.name == "__pycache__" or item.name == "__init__.py":
            continue
            
        if item.name in MAPPING:
            target_domain = MAPPING[item.name]
            target_dir = SRC_DIR / target_domain
            target_dir.mkdir(parents=True, exist_ok=True)
            
            target_path = target_dir / item.name
            print(f"Moving {item.name} -> {target_domain}/")
            shutil.move(str(item), str(target_path))
        else:
            print(f"WARNING: Unknown file {item.name}, moving to core/")
            target_dir = SRC_DIR / "core"
            shutil.move(str(item), str(target_dir / item.name))

def refactor_imports():
    print("Refactoring imports globally...")
    # Build a regex that matches `from crypto_research.xyz import`
    # or `import crypto_research.xyz`
    
    # Let's just create a global map of module_name -> new_domain
    # e.g., 'lob_encoder' -> 'cloud.lob_encoder'
    
    module_to_domain = {}
    for name, domain in MAPPING.items():
        mod_name = name.replace('.py', '')
        module_to_domain[mod_name] = f"src.{domain}.{mod_name}"

    # Also map the root package
    module_to_domain['gui'] = 'src.mission_control.gui'
    module_to_domain['baselines'] = 'src.cloud.baselines'

    python_files = list(SRC_DIR.rglob("*.py"))
    
    for py_file in python_files:
        with open(py_file, "r", encoding="utf-8") as f:
            content = f.read()

        new_content = content
        
        # Replace absolute imports `crypto_research.module` -> `src.domain.module`
        for mod, new_mod in module_to_domain.items():
            new_content = re.sub(
                fr'from\s+crypto_research\.{mod}\s+import',
                f'from {new_mod} import',
                new_content
            )
            new_content = re.sub(
                fr'import\s+crypto_research\.{mod}',
                f'import {new_mod}',
                new_content
            )
            
        # Catch any remaining `crypto_research` that might just be `from crypto_research import ...`
        # which is harder. But usually it's `from crypto_research.schemas import ...`
        
        # Replace `from src.core.logger import logger` with `from src.core.logger import logger`
        # (already correct in some places, but let's ensure paths are correct).
        
        if new_content != content:
            print(f"Updated imports in {py_file.name}")
            with open(py_file, "w", encoding="utf-8") as f:
                f.write(new_content)

if __name__ == "__main__":
    move_files()
    refactor_imports()
    print("Migration Complete.")
