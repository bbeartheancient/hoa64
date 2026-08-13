"""Detector → sphere boxes adapter for the vision tower.

Supports:
  * Precomputed boxes (JSON / dict list) — always available
  * YOLO-format label files (class x_c y_c w h normalized)
  * Optional torchvision detection (Faster R-CNN MobileNet) if weights load

No ultralytics/cv2 required. Image I/O via PIL + numpy.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Union

import numpy as np

PathLike = Union[str, Path]

# COCO class names subset for friendlier labels (optional)
_COCO_NAMES = None


def _coco_names() -> list[str]:
    global _COCO_NAMES
    if _COCO_NAMES is not None:
        return _COCO_NAMES
    # Minimal common set; unknown ids → class_{id}
    _COCO_NAMES = [
        "__background__", "person", "bicycle", "car", "motorcycle", "airplane",
        "bus", "train", "truck", "boat", "traffic light", "fire hydrant",
        "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse",
        "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
        "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis",
        "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
        "skateboard", "surfboard", "tennis racket", "bottle", "wine glass",
        "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich",
        "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
        "chair", "couch", "potted plant", "bed", "dining table", "toilet",
        "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
        "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
        "scissors", "teddy bear", "hair drier", "toothbrush",
    ]
    return _COCO_NAMES


@dataclass
class Detection:
    """Axis-aligned box in normalized image coords (x,y center or corners)."""

    # normalized [0,1] image: x right, y down; origin top-left
    x1: float
    y1: float
    x2: float
    y2: float
    score: float = 1.0
    label: str = ""
    class_id: int = -1

    @property
    def cx(self) -> float:
        return 0.5 * (self.x1 + self.x2)

    @property
    def cy(self) -> float:
        return 0.5 * (self.y1 + self.y2)

    @property
    def w(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def h(self) -> float:
        return max(0.0, self.y2 - self.y1)

    def to_sphere_box(
        self,
        *,
        hfov_deg: float = 90.0,
        vfov_deg: float = 60.0,
        sigma_scale: float = 1.0,
    ) -> dict:
        """Map image box → HOA vision box (az/el degrees, Ambix convention)."""
        # center: image x=0 left → +az, x=1 right → -az (matches vision._as_box)
        az = (0.5 - self.cx) * hfov_deg
        el = (0.5 - self.cy) * vfov_deg
        w_deg = max(2.0, self.w * hfov_deg * sigma_scale)
        h_deg = max(2.0, self.h * vfov_deg * sigma_scale)
        return {
            "az": float(az),
            "el": float(el),
            "w_deg": float(w_deg),
            "h_deg": float(h_deg),
            "weight": float(self.score),
            "label": self.label or (f"class_{self.class_id}" if self.class_id >= 0 else "det"),
            "kind": "box",
        }


def detections_to_sphere_boxes(
    dets: Sequence[Detection],
    *,
    hfov_deg: float = 90.0,
    vfov_deg: float = 60.0,
    min_score: float = 0.25,
) -> list[dict]:
    out = []
    for d in dets:
        if d.score < min_score:
            continue
        out.append(d.to_sphere_box(hfov_deg=hfov_deg, vfov_deg=vfov_deg))
    return out


def load_boxes_json(path: PathLike) -> list[dict]:
    """Load sphere boxes or detections JSON.

    Accepts:
      [{"az":..., "el":...}, ...]
      {"boxes": [...]}
      {"detections": [{"x1","y1","x2","y2",...}, ...]}  # normalized
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, list):
        if data and ("x1" in data[0] or "bbox" in data[0]):
            dets = []
            for item in data:
                if "bbox" in item:
                    x1, y1, x2, y2 = item["bbox"]
                else:
                    x1, y1, x2, y2 = item["x1"], item["y1"], item["x2"], item["y2"]
                dets.append(
                    Detection(
                        float(x1), float(y1), float(x2), float(y2),
                        score=float(item.get("score", item.get("confidence", 1.0))),
                        label=str(item.get("label", item.get("class", ""))),
                        class_id=int(item.get("class_id", -1)),
                    )
                )
            return detections_to_sphere_boxes(dets)
        return list(data)
    if isinstance(data, dict):
        if "boxes" in data:
            return list(data["boxes"])
        if "detections" in data:
            return load_boxes_json_from_obj(data["detections"])
    raise ValueError(f"unrecognized boxes JSON shape in {path}")


def load_boxes_json_from_obj(obj: Any) -> list[dict]:
    path_like = Path("/tmp/_unused")
    # reuse logic
    if isinstance(obj, list) and obj and "x1" in obj[0]:
        dets = [
            Detection(
                float(i["x1"]), float(i["y1"]), float(i["x2"]), float(i["y2"]),
                score=float(i.get("score", 1.0)),
                label=str(i.get("label", "")),
            )
            for i in obj
        ]
        return detections_to_sphere_boxes(dets)
    if isinstance(obj, list):
        return list(obj)
    raise ValueError("bad detections object")


def load_yolo_labels(
    path: PathLike,
    *,
    class_names: Optional[Sequence[str]] = None,
) -> list[Detection]:
    """YOLO txt: class x_center y_center width height (all normalized)."""
    dets: list[Detection] = []
    text = Path(path).read_text(encoding="utf-8").strip()
    if not text:
        return dets
    for line in text.splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        cid = int(float(parts[0]))
        cx, cy, w, h = map(float, parts[1:5])
        score = float(parts[5]) if len(parts) > 5 else 1.0
        x1, y1 = cx - w / 2, cy - h / 2
        x2, y2 = cx + w / 2, cy + h / 2
        label = ""
        if class_names and 0 <= cid < len(class_names):
            label = class_names[cid]
        else:
            label = f"class_{cid}"
        dets.append(Detection(x1, y1, x2, y2, score=score, label=label, class_id=cid))
    return dets


def detect_torchvision(
    image_path: PathLike,
    *,
    score_thresh: float = 0.5,
    device: Optional[str] = None,
    max_dets: int = 32,
) -> list[Detection]:
    """Run torchvision Faster R-CNN MobileNet on an image.

    First call may download weights (~50MB). Uses XPU if available.
    """
    from PIL import Image
    import torch
    import torchvision
    from torchvision.transforms import functional as F

    img = Image.open(image_path).convert("RGB")
    w, h = img.size
    tensor = F.to_tensor(img)

    if device is None:
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            device = "xpu"
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"

    weights = torchvision.models.detection.FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT
    model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(weights=weights)
    model.eval()
    model.to(device)
    with torch.no_grad():
        out = model([tensor.to(device)])[0]

    names = _coco_names()
    boxes = out["boxes"].detach().cpu().numpy()
    scores = out["scores"].detach().cpu().numpy()
    labels = out["labels"].detach().cpu().numpy()
    dets: list[Detection] = []
    for box, sc, lab in zip(boxes, scores, labels):
        if sc < score_thresh:
            continue
        x1, y1, x2, y2 = box
        cid = int(lab)
        label = names[cid] if cid < len(names) else f"class_{cid}"
        dets.append(
            Detection(
                x1 / w, y1 / h, x2 / w, y2 / h,
                score=float(sc),
                label=label,
                class_id=cid,
            )
        )
        if len(dets) >= max_dets:
            break
    return dets


def detect_to_sphere(
    image_path: PathLike,
    *,
    backend: str = "auto",
    score_thresh: float = 0.5,
    hfov_deg: float = 90.0,
    vfov_deg: float = 60.0,
) -> list[dict]:
    """Image → sphere boxes.

    backend: auto | torchvision | none
    auto tries torchvision, falls back to empty with a note if unavailable.
    """
    if backend in ("auto", "torchvision"):
        try:
            dets = detect_torchvision(image_path, score_thresh=score_thresh)
            return detections_to_sphere_boxes(
                dets, hfov_deg=hfov_deg, vfov_deg=vfov_deg, min_score=score_thresh
            )
        except Exception as e:
            if backend == "torchvision":
                raise
            return []
    return []


def write_demo_image_with_box(
    path: PathLike,
    *,
    size: tuple[int, int] = (640, 480),
    box_xyxy_norm: tuple[float, float, float, float] = (0.35, 0.35, 0.55, 0.65),
) -> Detection:
    """Create a simple synthetic image + known detection (no model)."""
    from PIL import Image, ImageDraw

    w, h = size
    img = Image.new("RGB", (w, h), (30, 30, 40))
    draw = ImageDraw.Draw(img)
    x1, y1, x2, y2 = box_xyxy_norm
    px = [x1 * w, y1 * h, x2 * w, y2 * h]
    draw.rectangle(px, outline=(0, 255, 80), width=4)
    draw.ellipse(
        [px[0] + 10, px[1] + 10, px[2] - 10, px[3] - 10],
        fill=(200, 80, 80),
    )
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    img.save(path)
    return Detection(x1, y1, x2, y2, score=0.99, label="demo_object", class_id=0)
