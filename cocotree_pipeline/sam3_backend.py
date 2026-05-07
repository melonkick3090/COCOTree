from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
try:
    import pycocotools.mask as coco_mask
except Exception:  # pragma: no cover - runtime dependency in actual SAM3 env
    coco_mask = None
from PIL import Image


def resolve_models_dir(models_dir: Optional[str] = None) -> str:
    if models_dir:
        return models_dir
    env = os.environ.get("SAM3_MODELS_DIR", "").strip()
    if env:
        return env
    return os.path.abspath("models")


class Sam3Backend:
    """Thin SAM3 wrapper for server-side use.

    - No crop encoder reuse.
    - A crop is encoded once inside run_on_crop() via processor.set_image().
      Prompts for that crop reuse the returned processor state.
    """

    _model = None
    _processor = None
    _model_key: Optional[Tuple[str, str]] = None

    def __init__(
        self,
        *,
        models_dir: Optional[str] = None,
        checkpoint_path: Optional[str] = None,
    ) -> None:
        self.models_dir = resolve_models_dir(models_dir)
        self.sam3_dir = os.path.join(self.models_dir, "sam3")
        os.makedirs(self.sam3_dir, exist_ok=True)
        self.checkpoint_path = checkpoint_path or os.path.join(self.sam3_dir, "sam3.pt")

    def load(self, confidence_threshold: float):
        import sam3
        from sam3 import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor

        sam3_root = os.path.dirname(sam3.__file__)
        bpe_path = os.path.join(sam3_root, "assets", "bpe_simple_vocab_16e6.txt.gz")
        model_key = (bpe_path, self.checkpoint_path)

        if self.__class__._model is None or self.__class__._model_key != model_key:
            print("[SAM3] Loading SAM3 model...")
            if os.path.exists(self.checkpoint_path):
                model = build_sam3_image_model(
                    bpe_path=bpe_path,
                    checkpoint_path=self.checkpoint_path,
                    load_from_HF=False,
                )
            else:
                model = build_sam3_image_model(bpe_path=bpe_path)
            self.__class__._model = model
            self.__class__._model_key = model_key
            self.__class__._processor = None

        if self.__class__._processor is None:
            self.__class__._processor = Sam3Processor(
                self.__class__._model,
                confidence_threshold=float(confidence_threshold),
            )
        else:
            self.__class__._processor.confidence_threshold = float(confidence_threshold)

        return self.__class__._processor

    @staticmethod
    def decode_rle_masks(counts_list: List[str], h: int, w: int) -> List[np.ndarray]:
        if coco_mask is None:
            raise ImportError("pycocotools is required to decode SAM3 output RLEs")
        out: List[np.ndarray] = []
        for counts in counts_list or []:
            rle = {"size": [int(h), int(w)], "counts": counts}
            mask = coco_mask.decode(rle)
            if mask.ndim == 3:
                mask = mask[:, :, 0]
            out.append((mask > 0).astype(np.uint8))
        return out

    @staticmethod
    def encode_mask_to_rle(mask_u8: np.ndarray) -> Dict[str, Any]:
        mb = (mask_u8 > 0).astype(np.uint8)
        if coco_mask is None:
            flat = mb.flatten(order="F")
            counts: List[int] = []
            prev = 0
            run = 0
            for val in flat:
                val = int(val)
                if val == prev:
                    run += 1
                else:
                    counts.append(int(run))
                    run = 1
                    prev = val
            counts.append(int(run))
            return {"format": "simple_rle_v1", "size": [int(mask_u8.shape[0]), int(mask_u8.shape[1])], "counts": counts}

        rle = coco_mask.encode(np.asfortranarray(mb))
        counts = rle["counts"]
        if isinstance(counts, bytes):
            counts = counts.decode("utf-8")
        return {"format": "coco_rle", "size": [int(mask_u8.shape[0]), int(mask_u8.shape[1])], "counts": counts}

    def run_on_crop(self, processor, pil_crop: Image.Image, label_list: List[str]) -> Dict[str, Any]:
        orig_w, orig_h = pil_crop.size
        label_results: Dict[str, Dict[str, Any]] = {}

        try:
            base_state = processor.set_image(pil_crop)
        except Exception as exc:
            print(f"[SAM3] set_image error: {exc}")
            for label in label_list or []:
                label_results[str(label)] = {"rles": [], "scores": []}
            return {"orig_img_h": orig_h, "orig_img_w": orig_w, "label_results": label_results}

        for label in label_list or []:
            label = str(label)
            try:
                state_in = dict(base_state)
                state = processor.set_text_prompt(prompt=label, state=state_in)

                rles: List[str] = []
                scores: List[float] = []
                if state.get("masks") is not None and len(state["masks"]) > 0:
                    masks = state["masks"].squeeze(1).cpu()
                    score_tensor = state["scores"].cpu()
                    for i in range(len(masks)):
                        mask_np = masks[i].numpy().astype(np.uint8)
                        rle = coco_mask.encode(np.asfortranarray(mask_np))
                        counts = rle["counts"]
                        if isinstance(counts, bytes):
                            counts = counts.decode("utf-8")
                        rles.append(counts)
                        scores.append(float(score_tensor[i].item()))
                label_results[label] = {"rles": rles, "scores": scores}
            except Exception as exc:
                print(f"[SAM3] prompt='{label}' failed: {exc}")
                label_results[label] = {"rles": [], "scores": []}

        return {"orig_img_h": orig_h, "orig_img_w": orig_w, "label_results": label_results}
