import os
import sys
import logging

user_site = os.path.expanduser("~/.local/lib/python3.13/site-packages")
if user_site not in sys.path and os.path.exists(user_site):
    sys.path.insert(0, user_site)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("Quantizer")

def quantize_onnx(model_name="yolo11n.onnx"):
    models_dir = os.path.dirname(os.path.abspath(__file__))
    src_path = os.path.join(models_dir, model_name)

    if not os.path.exists(src_path):
        logger.error(f"Source ONNX model not found: {src_path}")
        return None

    base_name, ext = os.path.splitext(model_name)
    dst_name = f"{base_name}_int8{ext}"
    dst_path = os.path.join(models_dir, dst_name)

    logger.info(f"Quantizing {model_name} -> {dst_name} (INT8 Dynamic Weight Quantization)...")

    try:
        from onnxruntime.quantization import quantize_dynamic, QuantType
        quantize_dynamic(src_path, dst_path, weight_type=QuantType.QUInt8)

        src_mb = os.path.getsize(src_path) / (1024 * 1024)
        dst_mb = os.path.getsize(dst_path) / (1024 * 1024)
        logger.info(f"Quantization complete! Original: {src_mb:.2f} MB -> Quantized: {dst_mb:.2f} MB ({((src_mb-dst_mb)/src_mb)*100:.1f}% reduction)")
        return dst_path
    except Exception as e:
        logger.error(f"Quantization failed: {e}")
        return None

if __name__ == "__main__":
    quantize_onnx("yolo11n.onnx")