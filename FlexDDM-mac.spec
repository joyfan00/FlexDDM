# -*- mode: python ; coding: utf-8 -*-

from PyInstaller.utils.hooks import collect_submodules


hiddenimports = collect_submodules("scipy")
hiddenimports += collect_submodules("webview")

a = Analysis(
    ["flexddm_desktop.py"],
    pathex=[],
    binaries=[],
    datas=[("flexddm", "flexddm"), ("data/ulrich2015.csv", "data"), ("setup.py", ".")],
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="FlexDDM",
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
    name="FlexDDM",
)
app = BUNDLE(
    coll,
    name="FlexDDM.app",
    icon=None,
    bundle_identifier="edu.uchicago.flexddm",
)
