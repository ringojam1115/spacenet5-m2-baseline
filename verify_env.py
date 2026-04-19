"""
環境確認スクリプト。
インストール後に `python verify_env.py` で全ライブラリの動作を確認する。
"""

import sys

def check(label: str, fn):
    try:
        result = fn()
        print(f"  [OK] {label}: {result}")
    except Exception as e:
        print(f"  [NG] {label}: {e}")

print(f"\nPython: {sys.version}\n")

print("=== PyTorch (MPS) ===")
check("import torch", lambda: __import__("torch").__version__)
check("MPS available", lambda: str(__import__("torch").backends.mps.is_available()))
check("MPS built", lambda: str(__import__("torch").backends.mps.is_built()))

def mps_tensor_check():
    import torch
    t = torch.tensor([1.0, 2.0]).to("mps")
    return f"tensor on mps: {t}"
check("MPS tensor round-trip", mps_tensor_check)

print("\n=== Segmentation Models PyTorch ===")
def smp_check():
    import segmentation_models_pytorch as smp
    model = smp.Unet(encoder_name="resnet34", encoder_weights=None, classes=8)
    return f"Unet(resnet34, classes=8) OK — params: {sum(p.numel() for p in model.parameters()):,}"
check("SMP Unet(resnet34)", smp_check)

print("\n=== Geo stack ===")
check("rasterio", lambda: __import__("rasterio").__version__)
check("geopandas", lambda: __import__("geopandas").__version__)
check("shapely", lambda: __import__("shapely").__version__)
check("pyproj", lambda: __import__("pyproj").__version__)

print("\n=== 画像処理 ===")
check("scikit-image", lambda: __import__("skimage").__version__)
check("opencv", lambda: __import__("cv2").__version__)
check("albumentations", lambda: __import__("albumentations").__version__)

print("\n=== グラフ / データ ===")
check("networkx", lambda: __import__("networkx").__version__)
check("numpy", lambda: __import__("numpy").__version__)
check("pandas", lambda: __import__("pandas").__version__)

print("\n=== 可視化 ===")
check("matplotlib", lambda: __import__("matplotlib").__version__)
check("folium", lambda: __import__("folium").__version__)

print()