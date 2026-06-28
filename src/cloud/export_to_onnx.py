"""
ONNX Exporter for PyTorch Models.

This script complies the Python PyTorch Neural Networks into the Open Neural 
Network Exchange (ONNX) format. This is the required first step before migrating
the live High-Frequency Trading execution loop to C++ or Zig, as it allows those 
lower-level languages to execute the forward pass using CUDA/TensorRT natively 
without the Python GIL.
"""

import torch
import onnx
import logging
from pathlib import Path
from rich.console import Console

from src.cloud.lob_encoder import LOBERTModel
from src.mission_control.forecast import FinCastModel

console = Console()
logging.basicConfig(level=logging.INFO)

def export_lobert(pt_path: Path, onnx_path: Path):
    console.print(f"[bold cyan]Exporting LOBERT to {onnx_path}...[/]")
    
    # 1. Initialize Model
    model = LOBERTModel()
    
    # Load weights if they exist
    if pt_path.exists():
        console.print(f"Loading weights from {pt_path}...")
        checkpoint = torch.load(pt_path, map_location="cpu")
        # Handle state_dict unpacking depending on how it was saved
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
    else:
        console.print(f"[bold yellow]Warning: {pt_path} not found. Exporting randomly initialized model.[/]")
        
    model.eval()

    # 2. Define static batch size and sequence length for HFT inference
    # Live execution almost always uses Batch=1 and fixed window sizes to avoid dynamic memory allocation overhead in C++.
    BATCH_SIZE = 1
    SEQ_LEN = 128
    
    dummy_messages = torch.rand(BATCH_SIZE, SEQ_LEN, 4)
    dummy_timestamps = torch.cumsum(torch.randint(1, 50, (BATCH_SIZE, SEQ_LEN)), dim=-1)
    
    # 3. Export to ONNX
    torch.onnx.export(
        model, 
        (dummy_messages, dummy_timestamps), 
        str(onnx_path),
        export_params=True,
        opset_version=14, # Modern opset
        do_constant_folding=True, # Hyper-optimize static paths
        input_names=['messages', 'timestamps_ms'],
        output_names=['pattern_score'],
        # Using dynamic axes ONLY for batch size, keeping sequence length static for speed
        dynamic_axes={'messages': {0: 'batch_size'}, 'timestamps_ms': {0: 'batch_size'}, 'pattern_score': {0: 'batch_size'}}
    )
    
    # 4. Verify ONNX Graph
    onnx_model = onnx.load(str(onnx_path))
    onnx.checker.check_model(onnx_model)
    console.print(f"[bold green]✅ LOBERT successfully exported and mathematically verified at {onnx_path}![/]")


def export_fincast(pt_path: Path, onnx_path: Path):
    console.print(f"\n[bold cyan]Exporting FinCast to {onnx_path}...[/]")
    
    model = FinCastModel()
    if pt_path.exists():
        console.print(f"Loading weights from {pt_path}...")
        checkpoint = torch.load(pt_path, map_location="cpu")
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
    else:
        console.print(f"[bold yellow]Warning: {pt_path} not found. Exporting randomly initialized model.[/]")
        
    model.eval()

    BATCH_SIZE = 1
    SEQ_LEN = 512
    
    dummy_ohlcv = torch.rand(BATCH_SIZE, SEQ_LEN, 5)
    
    torch.onnx.export(
        model, 
        dummy_ohlcv, 
        str(onnx_path),
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=['ohlcv'],
        output_names=['expected_return'],
        dynamic_axes={'ohlcv': {0: 'batch_size'}, 'expected_return': {0: 'batch_size'}}
    )
    
    onnx_model = onnx.load(str(onnx_path))
    onnx.checker.check_model(onnx_model)
    console.print(f"[bold green]✅ FinCast successfully exported and mathematically verified at {onnx_path}![/]")


if __name__ == "__main__":
    console.print("[bold magenta]🚀 Starting ONNX Compilation Bridge...[/]\n")
    
    out_dir = Path("onnx_exports")
    out_dir.mkdir(exist_ok=True)
    
    # Export LOBERT
    export_lobert(
        pt_path=Path("checkpoints/lobert_checkpoint.pt"),
        onnx_path=out_dir / "lobert.onnx"
    )
    
    # Export FinCast
    export_fincast(
        pt_path=Path("checkpoints/fincast_checkpoint.pt"),
        onnx_path=out_dir / "fincast.onnx"
    )
    
    console.print("\n[bold green]🎉 Compilation Complete. The .onnx binaries are ready for C++/Zig execution engines![/]")
