# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['emoji_manager.py'],
    pathex=[],
    binaries=[],
    datas=[],
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
def _drop_bins(binaries, drop_names):
    out = []
    for (src, dest, typ) in binaries:
        name = dest.replace("\\", "/").split("/")[-1].lower()
        if name in drop_names:
            continue
        out.append((src, dest, typ))
    return out

DROP_DLLS = {
    "qt6qml.dll",
    "qt6quick.dll",
    "qt6pdf.dll",
    "qt6opengl.dll",
    "opengl32sw.dll",
}

a.binaries = _drop_bins(a.binaries, DROP_DLLS)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='MmEmojiManager',
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
    name='MmEmojiManager',
)
