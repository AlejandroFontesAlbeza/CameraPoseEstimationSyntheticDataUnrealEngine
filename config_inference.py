from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent

### INFERENCE PATHS
INFERENCE_VIDEO_PATH = ROOT_DIR / "data" / "tennisMatch" / "clips" / "clip1.mp4"
INFERENCE_MODEL_PATH = ROOT_DIR / "models" / "unet_modelV1.pth"

OUTPUT_DIR = ROOT_DIR / "data" / "inference_info"

