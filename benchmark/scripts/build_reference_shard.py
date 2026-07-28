#!/usr/bin/env python3
"""Build a deterministic gzip JSONL shard from canonical per-image trees."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_ids(path: Path) -> list[int]:
    image_ids: list[int] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        text = line.strip()
        if not text or text.startswith("#"):
            continue
        value = json.loads(text)["image_id"] if text.startswith("{") else text
        try:
            image_ids.append(int(value))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{path}:{line_number}: invalid image_id") from exc
    if len(image_ids) != len(set(image_ids)):
        raise ValueError("Manifest contains duplicate image IDs")
    return image_ids


def main() -> None:
    args = parse_args()
    output = args.output.resolve()
    source = args.input_root.resolve()
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"Output exists; pass --overwrite: {output}")
    if output == source or source in output.parents:
        raise ValueError("Output must not be inside the input tree")

    image_ids = read_ids(args.manifest)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    raw_digest = hashlib.sha256()
    node_count = 0

    try:
        with temporary.open("wb") as raw_handle:
            with gzip.GzipFile(
                filename="",
                mode="wb",
                compresslevel=9,
                fileobj=raw_handle,
                mtime=0,
            ) as gzip_handle:
                for image_id in image_ids:
                    path = source / f"{image_id:012d}" / "prediction.json"
                    payload = json.loads(path.read_text(encoding="utf-8"))
                    if int(payload["image_id"]) != image_id:
                        raise ValueError(f"Image ID mismatch: {path}")
                    canonical = {
                        "schema": "cocotree_prediction_v1",
                        "image_id": image_id,
                        "nodes": payload["nodes"],
                    }
                    encoded = (
                        json.dumps(
                            canonical,
                            ensure_ascii=False,
                            sort_keys=True,
                            separators=(",", ":"),
                        )
                        + "\n"
                    ).encode("utf-8")
                    raw_digest.update(encoded)
                    gzip_handle.write(encoded)
                    node_count += len(canonical["nodes"])
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)

    compressed_digest = hashlib.sha256(output.read_bytes()).hexdigest()
    print(
        json.dumps(
            {
                "output": str(output),
                "images": len(image_ids),
                "mask_instance_nodes": node_count,
                "uncompressed_sha256": raw_digest.hexdigest(),
                "compressed_sha256": compressed_digest,
                "compressed_bytes": output.stat().st_size,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
