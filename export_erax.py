"""Exports EraX YOLO11 s and m to Core ML with NMS, FP16, and prints sizes and a sanity check."""
import os
import shutil
from huggingface_hub import hf_hub_download
from ultralytics import YOLO

DIR = "/Users/mohammed/.claude/jobs/e44c6aff/tmp/models/"
for size in ("m", "s"):
    pt = hf_hub_download("erax-ai/EraX-Anti-NSFW-V1.1", f"erax-anti-nsfw-yolo11{size}-v1.1.pt")
    print(f"EraX-{size} .pt {os.path.getsize(pt) / 1e6:.1f} MB")
    out = YOLO(pt).export(format="coreml", imgsz=640, nms=True, half=True)
    target = DIR + f"EraX{size.upper()}.mlpackage"
    if os.path.exists(target):
        shutil.rmtree(target)
    shutil.move(out, target)
    total = sum(os.path.getsize(os.path.join(d, f)) for d, _, fs in os.walk(target) for f in fs)
    print(f"EXPORTED {target} {total / 1e6:.1f} MB")
    check = YOLO(target, task="detect").predict(DIR + "frame-28d81b8c.png", imgsz=640, conf=0.1, verbose=False)[0]
    print("  core ml check:", [(check.names[int(c)], round(float(p), 2)) for c, p in zip(check.boxes.cls, check.boxes.conf)][:3])
