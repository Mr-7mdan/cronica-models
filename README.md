# Cronica on-device models

Core ML conversions of open nudity-detection models, used by [Cronica](https://github.com/Mr-7mdan)'s
player to warn about, hide or skip sensitive scenes. Everything runs on the device; no frame ever
leaves it.

The app bundles the small models and downloads the large ones from this repository's
[releases](../../releases) on request, then compiles them on the device with
`MLModel.compileModel(at:)`.

## Models

| Model | What it answers | Input | Core ML size | Zip | In the app | Licence |
|---|---|---|---|---|---|---|
| lovoo NSFW | One NSFW probability for the whole picture | 299 px | 17 kB | – | Bundled | BSD-3-Clause |
| NudeNet 320n | Boxes around exposed body parts | 320 px | 6.2 MB | 5.6 MB | Bundled | AGPL-3.0 |
| NudeNet 640m | Boxes around exposed body parts, larger model | 640 px | 51.9 MB | 48.0 MB | Download | AGPL-3.0 |
| Freepik NSFW | Severity: neutral, low, medium, high | 448 px | 173.2 MB | 133.3 MB | Download | MIT |
| EraX Anti-NSFW (YOLO11m) | Boxes: anus, sexual activity, nipple, penis, vagina | 640 px | 40.3 MB | 37.2 MB | Download | Apache-2.0 |

Release `v1` files and SHA-256:

| File | SHA-256 |
|---|---|
| `NudeNet320n.mlpackage.zip` | `be32c8c17dcaf85ff26935490641934912e25591b65ec86ff67656154893544d` |
| `NudeNet640m.mlpackage.zip` | `18b44b39799bea1c4b911b68ddc8abdcae5822f89b3a757a194bf8b1bea81138` |
| `FreepikNSFW.mlpackage.zip` | `97208fbbc77fb2c2a22c90ec7fd0f1bd13f5d38768d1da8e61e3d6117f01eb8c` |
| `EraXM.mlpackage.zip` | `d0bc08d1837d6483737fa5195293a726b88027df2a6df7fae981b9fc0d6ac012` |

## Where they come from

- **lovoo NSFW** – [github.com/lovoo/NSFWDetector](https://github.com/lovoo/NSFWDetector). A Create ML
  classifier on Apple's VisionFeaturePrint. Not converted here; the app ships the original.
- **NudeNet 320n / 640m** – [github.com/notAI-tech/NudeNet](https://github.com/notAI-tech/NudeNet),
  weights from release [`v3.4-weights`](https://github.com/notAI-tech/NudeNet/releases/tag/v3.4-weights)
  (`320n.pt`, `640m.pt`). YOLOv8 detectors, 18 classes; the app counts `FEMALE_BREAST_EXPOSED`,
  `FEMALE_GENITALIA_EXPOSED`, `MALE_GENITALIA_EXPOSED`, `BUTTOCKS_EXPOSED` and `ANUS_EXPOSED`.
- **Freepik NSFW** – [huggingface.co/Freepik/nsfw_image_detector](https://huggingface.co/Freepik/nsfw_image_detector).
  An EVA02 base vision transformer; the app counts medium and high.

- **EraX Anti-NSFW (YOLO11m)** – [huggingface.co/erax-ai/EraX-Anti-NSFW-V1.1](https://huggingface.co/erax-ai/EraX-Anti-NSFW-V1.1)
  (`erax-anti-nsfw-yolo11m-v1.1.pt`). The only model here with a sexual-activity class (`make_love`),
  which is also its least reliable: it fired on clothed and swimwear scenes in testing.

## How they were converted

- NudeNet: `export_nudenet.py` – Ultralytics `export(format="coreml", nms=True, half=True)` at the
  model's own size. Vision should letterbox the frame (`imageCropAndScaleOption = .scaleFit`).
- EraX: `export_erax.py` – the same Ultralytics export as NudeNet, at 640.
- Freepik: `convert_freepik.py` – timm → TorchScript → coremltools, FP16, iOS 17 target. The image is
  squashed to 448 × 448, as the model was trained; normalisation and softmax are inside the model.

Tool versions: torch 2.14, ultralytics 8.4.150, coremltools 9.0, timm 1.0.29.

## Licences

Each model keeps its original licence; see `LICENSES/`. NudeNet and its Ultralytics YOLOv8 base are
AGPL-3.0: the converted weights here are redistributed under the same terms, with the conversion
script as the source of the change.
