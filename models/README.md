# Models Directory

This directory should contain the YOLO model files used for object detection.

## Required Models

The application uses YOLOv8 models from Ultralytics. The models will be automatically downloaded when first run, but you can also manually download them.

## Available Models

Choose based on your speed vs. accuracy requirements:

- **yolov8n.pt** - Nano (fastest, ~3MB, good for real-time on CPU)
- **yolov8s.pt** - Small (~11MB, balanced performance)
- **yolov8m.pt** - Medium (~26MB, better accuracy)
- **yolov8l.pt** - Large (~44MB, high accuracy)
- **yolov8x.pt** - Extra Large (~68MB, highest accuracy, slowest)

## Manual Download

If you want to manually download models:

```bash
# Using Python
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"

# Or download directly from:
# https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt
```

## Auto-Download

When you first run any of the applications, YOLO will automatically download the required model if it's not present. The model will be saved to the Ultralytics cache directory, typically at:
- Linux: `~/.config/Ultralytics/`
- macOS: `~/Library/Application Support/Ultralytics/`
- Windows: `%APPDATA%\Ultralytics\`

## Custom Models

To use a custom-trained YOLO model:

1. Place your `.pt` model file in this directory
2. Update the `MODEL_PATH` in your `.env` file or pass it as a parameter
3. Ensure your model is compatible with the YOLOv8 architecture

## Notes

- Model files are typically large (3-68MB) and are excluded from git via `.gitignore`
- First-run model download requires an internet connection
- GPU acceleration recommended for larger models (yolov8l, yolov8x)
