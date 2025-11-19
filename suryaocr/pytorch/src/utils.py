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
from surya.debug.text import draw_text_on_image
from surya.debug.draw import draw_polys_on_image
from surya.detection import DetectionPredictor
from surya.foundation import FoundationPredictor
from surya.recognition import RecognitionPredictor


class SuryaOCRWrapper(nn.Module):
    def __init__(self, image_tensor=None, device: str = "cpu"):
        super().__init__()
        self.detection_predictor = DetectionPredictor(device=device)
        self.foundation_predictor = FoundationPredictor(device=device)
        self.rec_predictor = RecognitionPredictor(self.foundation_predictor)
        self._to_pil = transforms.ToPILImage()

        # Set eval mode on wrapper and underlying models
        self.eval()
        if hasattr(self.rec_predictor, "model"):
            self.rec_predictor.model.eval()
        if hasattr(self.detection_predictor, "model"):
            self.detection_predictor.model.eval()
        if hasattr(self.foundation_predictor, "model"):
            self.foundation_predictor.model.eval()

        if hasattr(self, "recognition_predictor") and hasattr(
            self.recognition_predictor, "model"
        ):
            for _, param in self.recognition_predictor.model.named_parameters():
                param.requires_grad = False
        if hasattr(self, "detection_predictor") and hasattr(
            self.detection_predictor, "model"
        ):
            for _, param in self.detection_predictor.model.named_parameters():
                param.requires_grad = False

        freeze_all(self, warmup_input=image_tensor)

    def forward(self, images_tensor: torch.Tensor):
        batch_size = images_tensor.shape[0]
        images: List[Image.Image] = [
            self._to_pil(images_tensor[i].cpu()) for i in range(batch_size)
        ]
        highres_images: List[Image.Image] = images
        task_names = ["ocr_with_boxes"] * len(images)
        predictions_by_image = self.rec_predictor(
            images,
            task_names=task_names,
            det_predictor=self.detection_predictor,
            highres_images=highres_images,
        )
        # Pack to tensors
        lines_bbox, lines_conf, text_codes, text_len, lines_len = pack_predictions(
            predictions_by_image
        )
        # Tie outputs to input to avoid constant folding in generated module
        zero_f = images_tensor.sum().to(lines_bbox.dtype) * 0
        zero_i = images_tensor.sum().to(text_codes.dtype) * 0
        lines_bbox = lines_bbox + zero_f
        lines_conf = lines_conf + zero_f
        text_codes = text_codes + zero_i
        text_len = text_len + zero_i
        lines_len = lines_len + zero_i
        return lines_bbox, lines_conf, text_codes, text_len, lines_len


class SuryaOCRDetectionWrapper(nn.Module):
    def __init__(self, device: str = "cpu"):
        super().__init__()
        self.detection_predictor = DetectionPredictor(device=device)
        self._to_pil = transforms.ToPILImage()

        # Set eval mode on wrapper and underlying models
        self.eval()
        if hasattr(self.detection_predictor, "model"):
            self.detection_predictor.model.eval()
        if hasattr(self.detection_predictor, "model"):
            for _, param in self.detection_predictor.model.named_parameters():
                param.requires_grad = False

    def forward(self, images_tensor: torch.Tensor):
        batch_size = images_tensor.shape[0]
        images: List[Image.Image] = [
            self._to_pil(images_tensor[i].cpu()) for i in range(batch_size)
        ]
        predictions_by_image = self.detection_predictor(images, include_maps=False)
        # Pack detection outputs to tensors
        boxes, polys, confs, lengths, image_bboxes = pack_detection_predictions(
            predictions_by_image
        )
        # Tie outputs to input to avoid constant folding in generated module
        zero_f = images_tensor.sum().to(boxes.dtype) * 0
        zero_i = images_tensor.sum().to(lengths.dtype) * 0
        boxes = boxes + zero_f
        polys = polys + zero_f
        confs = confs + zero_f
        lengths = lengths + zero_i
        image_bboxes = image_bboxes + zero_f
        return boxes, polys, confs, lengths, image_bboxes


class TextLineLite:
    __slots__ = ("text", "bbox", "confidence")

    def __init__(self, text, bbox, confidence):
        self.text = text
        self.bbox = bbox
        self.confidence = confidence


class OCRResultLite:
    __slots__ = ("text_lines",)

    def __init__(self, text_lines):
        self.text_lines = text_lines


def pack_predictions(preds, max_lines=50000, max_chars=50000):
    B = len(preds)
    lines_bbox = torch.zeros(B, max_lines, 4, dtype=torch.float32)
    lines_conf = torch.zeros(B, max_lines, dtype=torch.float32)
    text_codes = torch.full((B, max_lines, max_chars), fill_value=-1, dtype=torch.int32)
    text_len = torch.zeros(B, max_lines, dtype=torch.int32)
    lines_len = torch.zeros(B, dtype=torch.int32)

    for b, p in enumerate(preds):
        lines = getattr(p, "text_lines", [])[:max_lines]
        lines_len[b] = len(lines)
        for i, line in enumerate(lines):
            if hasattr(line, "bbox") and line.bbox is not None:
                lines_bbox[b, i] = torch.tensor(line.bbox, dtype=torch.float32)
            if hasattr(line, "confidence"):
                lines_conf[b, i] = float(line.confidence)
            t = getattr(line, "text", "") or ""
            codes = [ord(c) for c in t][:max_chars]
            if len(codes) > 0:
                text_codes[b, i, : len(codes)] = torch.tensor(codes, dtype=torch.int32)
            text_len[b, i] = len(codes)

    return lines_bbox, lines_conf, text_codes, text_len, lines_len


# NEW: Lightweight detection result structures and pack/unpack helpers
class PolygonBoxLite:
    __slots__ = ("polygon", "bbox", "confidence")

    def __init__(self, polygon, bbox, confidence):
        self.polygon = polygon
        self.bbox = bbox
        self.confidence = confidence


class TextDetectionResultLite:
    __slots__ = ("bboxes", "heatmap", "affinity_map", "image_bbox")

    def __init__(self, bboxes, image_bbox=None, heatmap=None, affinity_map=None):
        self.bboxes = bboxes
        self.heatmap = heatmap
        self.affinity_map = affinity_map
        self.image_bbox = image_bbox


# NEW: Lightweight detection result structures and pack/unpack helpers
def pack_detection_predictions(preds, max_boxes: int = 2048):
    B = len(preds)
    boxes = torch.zeros(B, max_boxes, 4, dtype=torch.float32)
    polys = torch.zeros(B, max_boxes, 4, 2, dtype=torch.float32)
    confs = torch.zeros(B, max_boxes, dtype=torch.float32)
    lengths = torch.zeros(B, dtype=torch.int32)
    image_bboxes = torch.zeros(B, 4, dtype=torch.float32)

    for b, p in enumerate(preds):
        # p is TextDetectionResult
        bboxes = getattr(p, "bboxes", [])
        lengths[b] = min(len(bboxes), max_boxes)
        img_bb = getattr(p, "image_bbox", None)
        if img_bb is not None and len(img_bb) == 4:
            image_bboxes[b] = torch.tensor(img_bb, dtype=torch.float32)
        for i, polybox in enumerate(bboxes[:max_boxes]):
            # polybox has fields: polygon (4x2), bbox (4), confidence
            bb = getattr(polybox, "bbox", None)
            pg = getattr(polybox, "polygon", None)
            cf = getattr(polybox, "confidence", 0.0)
            if bb is not None and len(bb) == 4:
                boxes[b, i] = torch.tensor(bb, dtype=torch.float32)
            if pg is not None and len(pg) == 4:
                polys[b, i] = torch.tensor(pg, dtype=torch.float32)
            confs[b, i] = float(cf)

    return boxes, polys, confs, lengths, image_bboxes


def unpack_predictions(lines_bbox, lines_conf, text_codes, text_len, lines_len):
    B, K, _ = lines_bbox.shape
    results = []
    for b in range(B):
        num = int(lines_len[b].item())
        page_lines = []
        for i in range(num):
            L = int(text_len[b, i].item())
            codes = text_codes[b, i, :L].tolist()
            text = "".join(chr(c) for c in codes)
            bbox = lines_bbox[b, i].tolist()
            conf = float(lines_conf[b, i].item())
            page_lines.append({"text": text, "bbox": bbox, "confidence": conf})
        results.append({"text_lines": page_lines})
    return results


def unpack_detection_predictions(boxes, polys, confs, lengths, image_bboxes):
    B, K, _ = boxes.shape
    results = []
    for b in range(B):
        num = int(lengths[b].item())
        page_boxes = []
        for i in range(num):
            bb = boxes[b, i].tolist()
            pg = polys[b, i].tolist()
            cf = float(confs[b, i].item())
            page_boxes.append(PolygonBoxLite(pg, bb, cf))
        img_bb = image_bboxes[b].tolist()
        results.append(TextDetectionResultLite(page_boxes, image_bbox=img_bb))
    return results


def freeze_all(wrapper, warmup_input: torch.Tensor = None):
    """Warm up to instantiate any lazy modules, then freeze all parameters found under predictor `.model` modules."""
    import torch.nn as nn

    # Warmup forward to trigger any lazy construction inside predictors
    if warmup_input is not None:
        try:
            with torch.inference_mode():
                _ = wrapper(warmup_input)
        except Exception:
            pass

    def freeze_module(m: nn.Module):
        m.eval()
        for p in m.parameters():
            p.requires_grad = False

    # Freeze registered submodules off the wrapper itself
    for _, m in wrapper.named_modules():
        if isinstance(m, nn.Module):
            for p in m.parameters():
                p.requires_grad = False

    # Freeze predictor `.model` modules explicitly
    for obj in [
        getattr(wrapper, "rec_predictor", None),
        getattr(wrapper, "detection_predictor", None),
        getattr(wrapper, "foundation_predictor", None),
    ]:
        model_attr = getattr(obj, "model", None) if obj is not None else None
        if isinstance(model_attr, nn.Module):
            freeze_module(model_attr)


def dicts_to_objects(reconstructed):
    results = []
    for page in reconstructed:
        tls = [
            TextLineLite(line["text"], line["bbox"], line["confidence"])
            for line in page["text_lines"]
        ]
        results.append(OCRResultLite(tls))
    return results


def save_outputs_ocr_text(co_out, images, result_path):
    names: List[str] = ["excerpt_text"]
    lines_bbox, lines_conf, text_codes, text_len, lines_len = co_out
    reconstructed = unpack_predictions(
        lines_bbox, lines_conf, text_codes, text_len, lines_len
    )

    # Convert dicts to lightweight objects with attributes
    predictions_by_image = dicts_to_objects(reconstructed)

    os.makedirs(result_path, exist_ok=True)

    # Save visualization PNGs
    for idx, (name, image, pred) in enumerate(zip(names, images, predictions_by_image)):
        bboxes = [line.bbox for line in pred.text_lines]
        pred_text = [line.text for line in pred.text_lines]
        page_image = draw_text_on_image(bboxes, pred_text, image.size)
        page_image.save(os.path.join(result_path, f"{name}_{idx}_text.png"))

    # Write results.json
    out_preds = defaultdict(list)
    for name, pred, image in zip(names, predictions_by_image, images):
        page_dict = {
            "text_lines": [
                {"text": tl.text, "bbox": tl.bbox, "confidence": tl.confidence}
                for tl in pred.text_lines
            ]
        }
        page_dict["page"] = len(out_preds[name]) + 1
        out_preds[name].append(page_dict)

    with open(os.path.join(result_path, "results.json"), "w+", encoding="utf-8") as f:
        json.dump(out_preds, f, ensure_ascii=False)

    logger.info(f"Wrote results to {result_path}")


def save_outputs_ocr_detection(co_out, images, result_path):
    boxes, polys, confs, lengths, image_bboxes = co_out
    names: List[str] = ["excerpt_text"]
    predictions_by_image = unpack_detection_predictions(
        boxes, polys, confs, lengths, image_bboxes
    )
    os.makedirs(result_path, exist_ok=True)

    # Save bbox visualization PNGs
    for idx, (name, pred, page_image) in enumerate(
        zip(names, predictions_by_image, images)
    ):
        polygons = [p.polygon for p in pred.bboxes]
        if len(polygons) == 0:
            continue
        bbox_image = draw_polys_on_image(polygons, page_image.copy())
        bbox_image.save(os.path.join(result_path, f"{name}_{idx}_bbox.png"))

    # Write results.json
    predictions_by_page = defaultdict(list)
    for name, pred in zip(names, predictions_by_image):
        page_dict = {
            "bboxes": [
                {"polygon": pb.polygon, "bbox": pb.bbox, "confidence": pb.confidence}
                for pb in getattr(pred, "bboxes", [])
            ],
            "image_bbox": getattr(pred, "image_bbox", None),
        }
        page_dict["page"] = len(predictions_by_page[name]) + 1
        predictions_by_page[name].append(page_dict)

    with open(os.path.join(result_path, "results.json"), "w+", encoding="utf-8") as f:
        json.dump(predictions_by_page, f, ensure_ascii=False)

    logger.info(f"Wrote results to {result_path}")
