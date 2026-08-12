import os
import sys
import time


user_site = os.path.expanduser("~/.local/lib/python3.13/site-packages")
if user_site not in sys.path and os.path.exists(user_site):
    sys.path.insert(0, user_site)

def export_yolo11(variant="n", imgsz=320):
    model_name = f"yolo11{variant}"
    print("=" * 60)
    print(f"EXPORTING STATE-OF-THE-ART {model_name.upper()} MODEL TO ONNX (RPi 5)")
    print("=" * 60)

    try:
        from ultralytics import YOLO

        models_dir = os.path.dirname(os.path.abspath(__file__))
        pt_path = os.path.join(models_dir, f"{model_name}.pt")
        onnx_target = os.path.join(models_dir, f"{model_name}.onnx")

        print(f"1. Loading PyTorch weights for {model_name}...")
        model = YOLO(f"{model_name}.pt")

        print(f"2. Exporting to ARM NEON-optimized ONNX format (Resolution: {imgsz}x{imgsz})...")
        onnx_file = model.export(format="onnx", imgsz=imgsz, simplify=True, dynamic=False)

        if os.path.exists(onnx_file):
            size_mb = os.path.getsize(onnx_file) / (1024 * 1024)
            print(f"SUCCESS! Model exported to: {onnx_file} ({size_mb:.2f} MB)")
            return onnx_file
    except Exception as e:
        print(f"Export failed: {e}")
        return None

if __name__ == "__main__":
    export_yolo11(variant="n", imgsz=320)