# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['src\\mission_control\\runner.py'],
    pathex=['.'],
    binaries=[('src\\go_orchestrator\\orchestrator.exe', 'src/go_orchestrator'), ('C:\\onnxruntime\\lib\\onnxruntime.dll', 'src/execution/zig-out/bin'), ('C:\\onnxruntime\\lib\\onnxruntime_providers_shared.dll', 'src/execution/zig-out/bin')],
    datas=[('src\\data\\tls_proxy.py', 'src/data'), ('onnx_exports', 'onnx_exports')],
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['torch._inductor', 'torch.distributed', 'tensorboard'],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='MissionControl',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='MissionControl',
)
