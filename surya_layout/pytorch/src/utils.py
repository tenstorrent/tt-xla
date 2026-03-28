# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import json
import os
from collections import defaultdict
from typing import List

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from loguru import logger
from PIL import Image


# Label mapping for layout detection classes
LAYOUT_LABELS = [
    "Caption",
    "Footnote",
    "Formula",
    "List-item",
    "Page-footer",
    "Page-header",
    "Picture",
    "Figure",
    "Section-header",
    "Table",
    "Form",
    "Table-of-contents",
    "Handwriting",
    "Text",
    "Text-inline-math",
]

LABEL_TO_ID = {label: idx for idx, label in enumerate(LAYOUT_LABELS)}


class SuryaLayoutWrapper(nn.Module):
    def __init__(self, device: str = "cpu"):
        super().__init__()
        from surya.foundation import FoundationPredictor
        from surya.layout import LayoutPredictor

        self.foundation_predictor = FoundationPredictor(device=device)
        self.layout_predictor = LayoutPredictor(self.foundation_predictor)
        self._to_pil = transforms.ToPILImage()

        self.eval()
        if hasattr(self.foundation_predictor, "model"):
            self.foundation_predictor.model.eval()
            for _, param in self.foundation_predictor.model.named_parameters():
                param.requires_grad = False

    def forward(self, images_tensor: torch.Tensor):
        batch_size = images_tensor.shape[0]
        images: List[Image.Image] = [
            self._to_pil(images_tensor[i].cpu()) for i in range(batch_size)
        ]
        predictions_by_image = self.layout_predictor(images)

        (
            boxes,
            polys,
            labels,
            positions,
            confs,
            lengths,
            image_bboxes,
        ) = pack_layout_predictions(predictions_by_image)

        # Tie outputs to input to avoid constant folding in generated module
        zero_f = images_tensor.sum().to(boxes.dtype) * 0
        zero_i = images_tensor.sum().to(lengths.dtype) * 0
        boxes = boxes + zero_f
        polys = polys + zero_f
        labels = labels + zero_i
        positions = positions + zero_i
        confs = confs + zero_f
        lengths = lengths + zero_i
        image_bboxes = image_bboxes + zero_f
        return boxes, polys, labels, positions, confs, lengths, image_bboxes


class LayoutBoxLite:
    __slots__ = ("polygon", "bbox", "confidence", "label", "position")

    def __init__(self, polygon, bbox, confidence, label, position):
        self.polygon = polygon
        self.bbox = bbox
        self.confidence = confidence
        self.label = label
        self.position = position


class LayoutResultLite:
    __slots__ = ("bboxes", "image_bbox")

    def __init__(self, bboxes, image_bbox=None):
        self.bboxes = bboxes
        self.image_bbox = image_bbox


def pack_layout_predictions(preds, max_boxes: int = 2048):
    B = len(preds)
    boxes = torch.zeros(B, max_boxes, 4, dtype=torch.float32)
    polys = torch.zeros(B, max_boxes, 4, 2, dtype=torch.float32)
    labels = torch.full((B, max_boxes), fill_value=-1, dtype=torch.int32)
    positions = torch.full((B, max_boxes), fill_value=-1, dtype=torch.int32)
    confs = torch.zeros(B, max_boxes, dtype=torch.float32)
    lengths = torch.zeros(B, dtype=torch.int32)
    image_bboxes = torch.zeros(B, 4, dtype=torch.float32)

    for b, p in enumerate(preds):
        bboxes = getattr(p, "bboxes", [])
        lengths[b] = min(len(bboxes), max_boxes)
        img_bb = getattr(p, "image_bbox", None)
        if img_bb is not None and len(img_bb) == 4:
            image_bboxes[b] = torch.tensor(img_bb, dtype=torch.float32)
        for i, layout_box in enumerate(bboxes[:max_boxes]):
            bb = getattr(layout_box, "bbox", None)
            pg = getattr(layout_box, "polygon", None)
            cf = getattr(layout_box, "confidence", 0.0)
            label = getattr(layout_box, "label", None)
            position = getattr(layout_box, "position", -1)
            if bb is not None and len(bb) == 4:
                boxes[b, i] = torch.tensor(bb, dtype=torch.float32)
            if pg is not None and len(pg) == 4:
                polys[b, i] = torch.tensor(pg, dtype=torch.float32)
            confs[b, i] = float(cf)
            if label is not None and label in LABEL_TO_ID:
                labels[b, i] = LABEL_TO_ID[label]
            if position is not None:
                positions[b, i] = int(position)

    return boxes, polys, labels, positions, confs, lengths, image_bboxes


def unpack_layout_predictions(
    boxes, polys, labels, positions, confs, lengths, image_bboxes
):
    B = boxes.shape[0]
    results = []
    for b in range(B):
        num = int(lengths[b].item())
        page_boxes = []
        for i in range(num):
            bb = boxes[b, i].tolist()
            pg = polys[b, i].tolist()
            cf = float(confs[b, i].item())
            label_id = int(labels[b, i].item())
            label = (
                LAYOUT_LABELS[label_id]
                if 0 <= label_id < len(LAYOUT_LABELS)
                else "Unknown"
            )
            pos = int(positions[b, i].item())
            page_boxes.append(LayoutBoxLite(pg, bb, cf, label, pos))
        img_bb = image_bboxes[b].tolist()
        results.append(LayoutResultLite(page_boxes, image_bbox=img_bb))
    return results


def save_outputs_layout(co_out, images, result_path):
    boxes, polys, labels, positions, confs, lengths, image_bboxes = co_out
    names: List[str] = ["layout_page"]
    predictions_by_image = unpack_layout_predictions(
        boxes, polys, labels, positions, confs, lengths, image_bboxes
    )
    os.makedirs(result_path, exist_ok=True)

    # Write results.json
    predictions_by_page = defaultdict(list)
    for name, pred in zip(names, predictions_by_image):
        page_dict = {
            "bboxes": [
                {
                    "polygon": lb.polygon,
                    "bbox": lb.bbox,
                    "confidence": lb.confidence,
                    "label": lb.label,
                    "position": lb.position,
                }
                for lb in getattr(pred, "bboxes", [])
            ],
            "image_bbox": getattr(pred, "image_bbox", None),
        }
        page_dict["page"] = len(predictions_by_page[name]) + 1
        predictions_by_page[name].append(page_dict)

    with open(os.path.join(result_path, "results.json"), "w+", encoding="utf-8") as f:
        json.dump(predictions_by_page, f, ensure_ascii=False)

    logger.info(f"Wrote results to {result_path}")
