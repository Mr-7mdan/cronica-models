"""Converts Freepik/nsfw_image_detector (timm EVA02, 448px) to Core ML and checks it on the test frames.

Output: a 4-way softmax, ordered as the model card lists: neutral, low, medium, high.
"""
import os
import numpy as np
import timm
import torch
import coremltools as ct
from PIL import Image

model = timm.create_model("hf_hub:Freepik/nsfw_image_detector", pretrained=True).eval()
cfg = timm.data.resolve_data_config({}, model=model)
print("config:", cfg)
print("labels:", getattr(model, "pretrained_cfg", {}).get("label_names"))


class Wrapped(torch.nn.Module):
    """Takes 0-255 RGB, normalises inside the model, returns probabilities."""

    def __init__(self, inner, mean, std):
        super().__init__()
        self.inner = inner
        self.register_buffer("mean", torch.tensor(mean).view(1, 3, 1, 1) * 255)
        self.register_buffer("std", torch.tensor(std).view(1, 3, 1, 1) * 255)

    def forward(self, x):
        return torch.softmax(self.inner((x - self.mean) / self.std), dim=-1)


size = cfg["input_size"][1]
wrapped = Wrapped(model, cfg["mean"], cfg["std"]).eval()
example = torch.rand(1, 3, size, size) * 255
traced = torch.jit.trace(wrapped, example)
mlmodel = ct.convert(
    traced,
    inputs=[ct.ImageType(name="image", shape=(1, 3, size, size), color_layout=ct.colorlayout.RGB)],
    outputs=[ct.TensorType(name="probabilities")],
    compute_precision=ct.precision.FLOAT16,
    minimum_deployment_target=ct.target.iOS17,
)
mlmodel.short_description = "Freepik nsfw_image_detector (MIT): neutral, low, medium, high"
mlmodel.save("FreepikNSFW.mlpackage")
total = sum(os.path.getsize(os.path.join(d, f)) for d, _, fs in os.walk("FreepikNSFW.mlpackage") for f in fs)
print(f"EXPORTED FreepikNSFW.mlpackage {total / 1e6:.1f} MB")

U = "/Users/mohammed/.claude/uploads/e44c6aff-0c89-433a-a258-e2cb1dacfcb6/"
for name in ["15443a45-image.png", "49742aa4-image.png", "28d81b8c-image.png", "6ff38d16-image.png",
             "24bfc20f-image.png", "2b9741bc-image.png", "d2a360b8-image.png"]:
    image = Image.open(U + name).convert("RGB").resize((size, size))
    probs = np.array(mlmodel.predict({"image": image})["probabilities"]).ravel()
    print("FREEPIK", name[:8], "neutral/low/medium/high =", np.round(probs, 2))
