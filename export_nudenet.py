"""Exports NudeNet 320n and 640m to Core ML with NMS, FP16, and prints the resulting sizes."""
import os
import shutil
from ultralytics import YOLO

for weights, size in (("320n.pt", 320), ("640m.pt", 640)):
    out = YOLO(weights).export(format="coreml", imgsz=size, nms=True, half=True)
    target = f"NudeNet{weights.split('.')[0]}.mlpackage"
    if os.path.exists(target):
        shutil.rmtree(target)
    shutil.move(out, target)
    total = sum(os.path.getsize(os.path.join(d, f)) for d, _, files in os.walk(target) for f in files)
    print("EXPORTED", target, f"{total / 1e6:.1f} MB")
